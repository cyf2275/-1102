"""Sweep DDIM sampling parameters on full validation set, then test once.

This script intentionally tunes only on the validation split. The test split is
evaluated exactly once with the best validation setting.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data import create_loaders
from diffusion import PhysicsConditionedDiffusion
from models import CoarsePredictor, ConditionalUNet
from utils.metrics import compute_metrics
from utils.visualization import save_comparison


METRIC_KEYS = ["rmse", "mae", "edge_rmse", "normal_deg", "ssim"]


def mean_std(rows, key):
    vals = np.array([r[key] for r in rows], dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)


def summarize(rows):
    summary = {}
    for key in METRIC_KEYS:
        mean, std = mean_std(rows, key)
        summary[key] = {"mean": mean, "std": std}
    summary["n"] = len(rows)
    return summary


def write_per_sample(rows, path):
    keys = ["sample", "start_ratio", "ddim_steps", "ensemble"] + METRIC_KEYS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in keys})


def write_sweep(rows, path):
    keys = ["split", "start_ratio", "ddim_steps", "ensemble"] + METRIC_KEYS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in keys})


@torch.no_grad()
def evaluate_split(loaders, split, diffusion, coarse_model, device, height_scale,
                   start_ratio, ddim_steps, ensemble, out_dir=None, save_samples=False):
    diffusion.model.eval()
    if coarse_model is not None:
        coarse_model.eval()

    rows = []
    sample_dir = None
    if save_samples and out_dir is not None:
        sample_dir = out_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

    desc = f"{split} sr={start_ratio:.2f} steps={ddim_steps} ens={ensemble}"
    for idx, batch in enumerate(tqdm(loaders[split], desc=desc)):
        cond = batch["cond"].to(device, non_blocking=True)
        fringe = batch["fringe"].to(device, non_blocking=True)
        target_raw = batch["height_raw"].to(device, non_blocking=True)

        coarse = None
        if coarse_model is not None:
            coarse = coarse_model(cond)
            cond = torch.cat([cond, coarse], dim=1)

        pred = diffusion.sample_ddim(
            cond,
            steps=ddim_steps,
            ensemble_size=ensemble,
            coarse=coarse,
            start_ratio=start_ratio,
            progress=False,
        )
        pred_mm = torch.clamp((pred + 1.0) * 0.5, 0.0, 1.0) * height_scale
        metrics = compute_metrics(pred_mm, target_raw)
        row = {
            "sample": idx,
            "start_ratio": start_ratio,
            "ddim_steps": ddim_steps,
            "ensemble": ensemble,
            **metrics,
        }
        rows.append(row)

        if sample_dir is not None and idx < 8:
            save_comparison(
                fringe,
                target_raw,
                pred_mm,
                sample_dir / f"sample_{idx:02d}.png",
                title=f"v5 RMSE {metrics['rmse']:.2f}mm",
            )

    return rows


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/root/diffusion_fpp_v5/data")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/diffusion_fpp_v5_cache")
    parser.add_argument("--ckpt", default="/root/diffusion_fpp_v5/results/fringe_physics/checkpoints/best.pt")
    parser.add_argument("--out_dir", default="/root/diffusion_fpp_v5/results/fringe_physics/sampling_sweep")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--start_ratios", type=float, nargs="+",
                        default=[0.15, 0.30, 0.45, 0.55, 0.70, 0.85])
    parser.add_argument("--ddim_steps", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--ensembles", type=int, nargs="+", default=[1, 3])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    saved_args = ckpt.get("args", {})
    image_h = int(saved_args.get("image_h", 480))
    image_w = int(saved_args.get("image_w", 640))
    base_channels = int(saved_args.get("base_channels", 48))
    timesteps = int(saved_args.get("timesteps", 200))
    lambda_grad = float(saved_args.get("lambda_grad", 0.2))
    lambda_fft = float(saved_args.get("lambda_fft", 0.05))
    height_scale = float(ckpt["height_scale"])

    loaders = create_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        height_scale=height_scale,
        image_h=image_h,
        image_w=image_w,
        cache_dir=args.cache_dir,
        require_cache=True,
    )

    coarse_model = None
    cond_channels = 7
    if ckpt.get("coarse_state_dict") is not None:
        coarse_model = CoarsePredictor(in_channels=7, base_ch=32).to(device)
        coarse_model.load_state_dict(ckpt["coarse_state_dict"])
        coarse_model.eval()
        cond_channels = 8

    model = ConditionalUNet(
        cond_channels=cond_channels,
        base_ch=base_channels,
        ch_mult=(1, 2, 4, 8),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    diffusion = PhysicsConditionedDiffusion(
        model,
        timesteps=timesteps,
        image_h=image_h,
        image_w=image_w,
        device=device,
        lambda_grad=lambda_grad,
        lambda_fft=lambda_fft,
    )

    val_sweep = []
    for start_ratio in args.start_ratios:
        for steps in args.ddim_steps:
            for ensemble in args.ensembles:
                rows = evaluate_split(
                    loaders,
                    "val",
                    diffusion,
                    coarse_model,
                    device,
                    height_scale,
                    start_ratio,
                    steps,
                    ensemble,
                )
                summary = summarize(rows)
                record = {
                    "split": "val",
                    "start_ratio": start_ratio,
                    "ddim_steps": steps,
                    "ensemble": ensemble,
                }
                record.update({k: summary[k]["mean"] for k in METRIC_KEYS})
                val_sweep.append(record)
                write_sweep(val_sweep, out_dir / "val_sweep.csv")
                print(json.dumps(record, ensure_ascii=False))

    best = min(val_sweep, key=lambda row: row["rmse"])
    with open(out_dir / "best_sampling.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    print("Best validation sampling:")
    print(json.dumps(best, indent=2, ensure_ascii=False))

    test_dir = out_dir / "test_best"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_rows = evaluate_split(
        loaders,
        "test",
        diffusion,
        coarse_model,
        device,
        height_scale,
        float(best["start_ratio"]),
        int(best["ddim_steps"]),
        int(best["ensemble"]),
        out_dir=test_dir,
        save_samples=True,
    )
    write_per_sample(test_rows, test_dir / "per_sample_metrics.csv")
    test_summary = summarize(test_rows)
    test_summary["height_scale"] = height_scale
    test_summary["checkpoint"] = str(args.ckpt)
    test_summary["selected_by"] = "full_val_rmse"
    test_summary["sampling"] = {
        "start_ratio": float(best["start_ratio"]),
        "ddim_steps": int(best["ddim_steps"]),
        "ensemble": int(best["ensemble"]),
    }
    with open(test_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2, ensure_ascii=False)
    print("Final test with best validation sampling:")
    print(json.dumps(test_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
