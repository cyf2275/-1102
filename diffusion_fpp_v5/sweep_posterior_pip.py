"""Validation-selected posterior guidance sweep for PIP-DiffFPP."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.dataset_pip import create_pip_loaders
from diffusion_pip import PIPDiffusion
from models import ConditionalUNet, PointwisePhaseProjectionHead
from train_pip_lite import METRIC_KEYS, HARD_TEST_SAMPLES, save_rows, summarize, write_eval_outputs
from utils.metrics import compute_metrics
from utils.visualization import save_comparison


def parse_list(text, cast=float):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def load_phase_head(path, device):
    ckpt = torch.load(path, map_location=device)
    args = ckpt.get("args", {})
    head = PointwisePhaseProjectionHead(
        hidden_dim=int(args.get("hidden_dim", 64)),
        num_layers=int(args.get("num_layers", 4)),
    ).to(device)
    head.load_state_dict(ckpt["model_state_dict"])
    head.phase_depth_input = str(args.get("depth_input", "height_norm"))
    head.raw_depth_center = float(args.get("raw_depth_center", 0.0))
    head.raw_depth_scale = float(args.get("raw_depth_scale", 1.0) or 1.0)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


@torch.no_grad()
def eval_loader(diffusion, loader, device, height_scale, ddim_steps, ensemble, guidance=None,
                split_name="val", out_dir=None, save_images=False):
    rows = []
    if save_images and out_dir is not None:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
        (out_dir / "hard_samples").mkdir(parents=True, exist_ok=True)
    for idx, batch in enumerate(tqdm(loader, desc=f"eval {split_name}")):
        pred = diffusion.sample_ddim(batch, steps=ddim_steps, ensemble_size=ensemble, guidance=guidance)
        pred_mm = torch.clamp((pred + 1.0) * 0.5, 0.0, 1.0) * height_scale
        target_raw = batch["height_raw"].to(device, non_blocking=True)
        metrics = compute_metrics(pred_mm, target_raw)
        hard = int(split_name == "test" and idx in HARD_TEST_SAMPLES)
        rows.append({"sample": idx, "hard": hard, **metrics})
        if save_images and out_dir is not None:
            fringe = batch["fringe"].to(device, non_blocking=True)
            if idx < 8:
                save_comparison(fringe, target_raw, pred_mm, out_dir / "samples" / f"sample_{idx:02d}.png",
                                title=f"PIP posterior RMSE {metrics['rmse']:.2f}mm")
            if hard:
                save_comparison(fringe, target_raw, pred_mm, out_dir / "hard_samples" / f"sample_{idx:02d}.png",
                                title=f"PIP posterior hard RMSE {metrics['rmse']:.2f}mm")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/root/diffusion_fpp_v5/data")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/diffusion_fpp_pip_cache")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--phase_head", required=True)
    parser.add_argument("--out_dir", default="/root/diffusion_fpp_v5/results/pip_posterior_sweep")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_h", type=int, default=480)
    parser.add_argument("--image_w", type=int, default=640)
    parser.add_argument("--weights", default="0,0.005,0.01,0.02,0.05")
    parser.add_argument("--starts", default="0.5,0.7,0.85")
    parser.add_argument("--clips", default="0.03,0.05,0.1")
    parser.add_argument("--ddim_steps", default="20,50")
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--require_cache", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    include_ftp = bool(ckpt.get("include_ftp", ckpt_args.get("include_ftp", False)))
    base_channels = int(ckpt_args.get("base_channels", 48))
    timesteps = int(ckpt_args.get("timesteps", 200))

    loaders = create_pip_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        require_cache=args.require_cache,
        include_ftp=include_ftp,
        image_h=args.image_h,
        image_w=args.image_w,
    )
    model = ConditionalUNet(cond_channels=loaders["cond_channels"], base_ch=base_channels,
                            ch_mult=(1, 2, 4, 8), dropout=0.05).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    phase_head = load_phase_head(args.phase_head, device)
    diffusion = PIPDiffusion(
        model,
        timesteps=timesteps,
        image_h=args.image_h,
        image_w=args.image_w,
        device=device,
        phase_head=phase_head,
    )
    height_scale = loaders["height_scale"]
    rows = []
    best = None
    for weight, start, clip, steps in product(
        parse_list(args.weights, float),
        parse_list(args.starts, float),
        parse_list(args.clips, float),
        parse_list(args.ddim_steps, int),
    ):
        guidance = None if weight <= 0 else {
            "weight": weight,
            "apply_start_ratio": start,
            "grad_clip": clip,
            "eta": 1.0,
            "k": 8.0,
            "tau": 0.4,
        }
        val_rows = eval_loader(
            diffusion, loaders["val"], device, height_scale,
            ddim_steps=steps, ensemble=args.ensemble, guidance=guidance, split_name="val")
        summary = summarize(val_rows)
        rec = {
            "weight": weight,
            "apply_start_ratio": start,
            "grad_clip": clip,
            "ddim_steps": steps,
            "ensemble": args.ensemble,
        }
        rec.update({key: summary[key]["mean"] for key in METRIC_KEYS})
        rows.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
        if best is None or rec["rmse"] < best["rmse"]:
            best = rec
    with open(out_dir / "val_sweep.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["weight", "apply_start_ratio", "grad_clip", "ddim_steps", "ensemble"] + METRIC_KEYS)
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "best_posterior_config.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    best_guidance = None if best["weight"] <= 0 else {
        "weight": best["weight"],
        "apply_start_ratio": best["apply_start_ratio"],
        "grad_clip": best["grad_clip"],
        "eta": 1.0,
        "k": 8.0,
        "tau": 0.4,
    }
    test_dir = out_dir / "test_best"
    test_rows = eval_loader(
        diffusion, loaders["test"], device, height_scale,
        ddim_steps=int(best["ddim_steps"]), ensemble=int(best["ensemble"]),
        guidance=best_guidance, split_name="test", out_dir=test_dir, save_images=True)
    class SimpleArgs:
        ddim_steps = int(best["ddim_steps"])
        ensemble = int(best["ensemble"])
    write_eval_outputs(test_rows, test_dir, height_scale, args.checkpoint, SimpleArgs)
    print("Best posterior config:")
    print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
