"""Unified evaluation for existing v3, v5, and v3.5 checkpoints."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data import create_loaders
from data.dataset_v35 import create_v35_loaders
from diffusion import PhysicsConditionedDiffusion
from diffusion_v35 import PhaseEdgeDiffusion
from models import CoarsePredictor, ConditionalUNet
from utils.metrics import compute_metrics
from utils.visualization import save_comparison


METRIC_KEYS = ["rmse", "mae", "edge_rmse", "normal_deg", "ssim"]
HARD_TEST_SAMPLES = {18, 19, 32, 33, 34, 35}


def mean_std(rows, key):
    vals = np.array([r[key] for r in rows], dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)


def summarize(rows):
    return {key: {"mean": mean_std(rows, key)[0], "std": mean_std(rows, key)[1]} for key in METRIC_KEYS}


def write_outputs(rows, out_dir, height_scale, checkpoint, extra):
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["sample", "hard"] + METRIC_KEYS
    with open(out_dir / "per_sample_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in keys})
    summary = summarize(rows)
    summary["n"] = len(rows)
    summary["height_scale"] = height_scale
    summary["checkpoint"] = str(checkpoint)
    summary.update(extra)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    hard_rows = [r for r in rows if r["hard"]]
    hard_summary = summarize(hard_rows) if hard_rows else {}
    hard_summary["n"] = len(hard_rows)
    hard_summary["hard_samples"] = sorted(HARD_TEST_SAMPLES)
    with open(out_dir / "hard_sample_summary.json", "w", encoding="utf-8") as f:
        json.dump(hard_summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


@torch.no_grad()
def eval_v35(args, device):
    ckpt = torch.load(args.ckpt, map_location=device)
    saved_args = ckpt.get("args", {})
    height_scale = float(ckpt["height_scale"])
    image_h = int(saved_args.get("image_h", 480))
    image_w = int(saved_args.get("image_w", 640))
    base_channels = int(saved_args.get("base_channels", 48))
    timesteps = int(saved_args.get("timesteps", 200))
    loaders = create_v35_loaders(
        args.data_dir, batch_size=1, num_workers=args.num_workers, height_scale=height_scale,
        cache_dir=args.cache_dir, require_cache=True, image_h=image_h, image_w=image_w)
    model = ConditionalUNet(cond_channels=8, base_ch=base_channels, ch_mult=(1, 2, 4, 8), dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    diffusion = PhaseEdgeDiffusion(model, timesteps=timesteps, image_h=image_h, image_w=image_w, device=device)
    return eval_loader(diffusion, loaders["test"], device, height_scale, args, "v35")


@torch.no_grad()
def eval_v5(args, device):
    ckpt = torch.load(args.ckpt, map_location=device)
    saved_args = ckpt.get("args", {})
    height_scale = float(ckpt["height_scale"])
    image_h = int(saved_args.get("image_h", 480))
    image_w = int(saved_args.get("image_w", 640))
    base_channels = int(saved_args.get("base_channels", 48))
    timesteps = int(saved_args.get("timesteps", 200))
    loaders = create_loaders(args.data_dir, batch_size=1, num_workers=args.num_workers,
                             height_scale=height_scale, image_h=image_h, image_w=image_w,
                             cache_dir=args.cache_dir, require_cache=True)
    cond_channels = 7
    coarse_model = None
    if ckpt.get("coarse_state_dict") is not None:
        coarse_model = CoarsePredictor(in_channels=7, base_ch=32).to(device)
        coarse_model.load_state_dict(ckpt["coarse_state_dict"])
        coarse_model.eval()
        cond_channels = 8
    model = ConditionalUNet(cond_channels=cond_channels, base_ch=base_channels,
                            ch_mult=(1, 2, 4, 8), dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    diffusion = PhysicsConditionedDiffusion(model, timesteps=timesteps, image_h=image_h, image_w=image_w, device=device)
    return eval_loader(diffusion, loaders["test"], device, height_scale, args, "v5", coarse_model=coarse_model)


@torch.no_grad()
def eval_v3(args, device):
    def load_symbol(module_name, file_path, symbol):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return getattr(module, symbol)

    V3Diffusion = load_symbol("v3_diffusion_module", "/root/diffusion_fpp_v3/diffusion.py", "GaussianDiffusion")
    V3UNet = load_symbol("v3_unet_module", "/root/diffusion_fpp_v3/unet.py", "ConditionalUNet")

    class V3TestDataset(torch.utils.data.Dataset):
        def __init__(self, data_dir, height_scale):
            data_dir = Path(data_dir)
            self.fringe = np.load(str(data_dir / "X_test_fringe.npy"), mmap_mode="r")
            self.height = np.load(str(data_dir / "Z_test.npy"), mmap_mode="r")
            self.height_scale = float(height_scale)

        def __len__(self):
            return int(self.fringe.shape[0])

        def __getitem__(self, idx):
            f = np.transpose(np.asarray(self.fringe[idx]), (2, 0, 1)).astype(np.float32)
            h_raw = np.transpose(np.asarray(self.height[idx]), (2, 0, 1)).astype(np.float32)
            h01 = np.clip(h_raw / self.height_scale, 0.0, 1.0).astype(np.float32)
            return torch.from_numpy(f.copy()).float(), torch.from_numpy(h01.copy()).float()

    ckpt = torch.load(args.ckpt, map_location=device)
    height_scale = float(ckpt.get("height_scale", 204.13104270935037))
    base_channels = int(ckpt.get("base_channels", 48))
    timesteps = int(ckpt.get("timesteps", 200))
    ds = V3TestDataset(args.v3_data_dir, height_scale)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = V3UNet(base_ch=base_channels, ch_mult=[1, 2, 4, 8], dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    diffusion = V3Diffusion(model, timesteps=timesteps, image_h=480, image_w=640, device=device, loss_type="l1_l2")
    rows = []
    out_dir = Path(args.out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "hard_samples").mkdir(parents=True, exist_ok=True)
    for idx, (fringe, height01) in enumerate(tqdm(loader, desc="eval v3")):
        fringe = fringe.to(device, non_blocking=True)
        height01 = height01.to(device, non_blocking=True)
        pred = diffusion.sample_ddim(fringe, steps=args.ddim_steps, ensemble_size=args.ensemble, progress=False)
        pred_mm = torch.clamp((pred + 1.0) * 0.5, 0.0, 1.0) * height_scale
        target_mm = height01 * height_scale
        metrics = compute_metrics(pred_mm, target_mm)
        hard = int(idx in HARD_TEST_SAMPLES)
        rows.append({"sample": idx, "hard": hard, **metrics})
        if idx < 8:
            save_comparison(fringe, target_mm, pred_mm, out_dir / "samples" / f"sample_{idx:02d}.png",
                            title=f"v3 RMSE {metrics['rmse']:.2f}mm")
        if hard:
            save_comparison(fringe, target_mm, pred_mm, out_dir / "hard_samples" / f"sample_{idx:02d}.png",
                            title=f"v3 hard RMSE {metrics['rmse']:.2f}mm")
    return rows, height_scale


@torch.no_grad()
def eval_loader(diffusion, loader, device, height_scale, args, label, coarse_model=None):
    out_dir = Path(args.out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "hard_samples").mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, batch in enumerate(tqdm(loader, desc=f"eval {label}")):
        cond = batch["cond"].to(device, non_blocking=True)
        fringe = batch["fringe"].to(device, non_blocking=True)
        target_raw = batch["height_raw"].to(device, non_blocking=True)
        coarse = None
        if coarse_model is not None:
            coarse = coarse_model(cond)
            cond = torch.cat([cond, coarse], dim=1)
            pred = diffusion.sample_ddim(cond, steps=args.ddim_steps, ensemble_size=args.ensemble,
                                         coarse=coarse, start_ratio=args.start_ratio, progress=False)
        else:
            pred = diffusion.sample_ddim(cond, steps=args.ddim_steps, ensemble_size=args.ensemble, progress=False)
        pred_mm = torch.clamp((pred + 1.0) * 0.5, 0.0, 1.0) * height_scale
        metrics = compute_metrics(pred_mm, target_raw)
        hard = int(idx in HARD_TEST_SAMPLES)
        rows.append({"sample": idx, "hard": hard, **metrics})
        if idx < 8:
            save_comparison(fringe, target_raw, pred_mm, out_dir / "samples" / f"sample_{idx:02d}.png",
                            title=f"{label} RMSE {metrics['rmse']:.2f}mm")
        if hard:
            save_comparison(fringe, target_raw, pred_mm, out_dir / "hard_samples" / f"sample_{idx:02d}.png",
                            title=f"{label} hard RMSE {metrics['rmse']:.2f}mm")
    return rows, height_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["v3", "v5", "v35"], required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--data_dir", default="/root/diffusion_fpp_v5/data")
    parser.add_argument("--v3_data_dir", default="/root/diffusion_fpp_v3/data")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/diffusion_fpp_v35_cache")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ensemble", type=int, default=3)
    parser.add_argument("--start_ratio", type=float, default=0.85)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if args.kind == "v3":
        rows, height_scale = eval_v3(args, device)
    elif args.kind == "v5":
        rows, height_scale = eval_v5(args, device)
    else:
        rows, height_scale = eval_v35(args, device)
    write_outputs(rows, Path(args.out_dir), height_scale, args.ckpt,
                  {"kind": args.kind, "sampling": {"ddim_steps": args.ddim_steps, "ensemble": args.ensemble}})


if __name__ == "__main__":
    main()
