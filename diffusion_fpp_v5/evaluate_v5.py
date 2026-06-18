"""Evaluate v5 on all 36 test samples with unified metrics."""
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


def mean_std(rows, key):
    vals = np.array([r[key] for r in rows], dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/root/diffusion_fpp_v5/data")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/diffusion_fpp_v5_cache")
    parser.add_argument("--ckpt", default="/root/diffusion_fpp_v5/results/fringe_physics/checkpoints/best.pt")
    parser.add_argument("--out_dir", default="/root/diffusion_fpp_v5/results/fringe_physics/evaluation")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ensemble", type=int, default=5)
    parser.add_argument("--start_ratio", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    saved_args = ckpt.get("args", {})
    image_h = int(saved_args.get("image_h", 480))
    image_w = int(saved_args.get("image_w", 640))
    start_ratio = float(args.start_ratio if args.start_ratio is not None else saved_args.get("start_ratio", 0.55))
    base_channels = int(saved_args.get("base_channels", 48))
    timesteps = int(saved_args.get("timesteps", 200))
    lambda_grad = float(saved_args.get("lambda_grad", 0.2))
    lambda_fft = float(saved_args.get("lambda_fft", 0.05))
    height_scale = float(ckpt["height_scale"])

    loaders = create_loaders(args.data_dir, batch_size=args.batch_size,
                             num_workers=args.num_workers, height_scale=height_scale,
                             image_h=image_h, image_w=image_w, cache_dir=args.cache_dir,
                             require_cache=True)

    has_coarse = ckpt.get("coarse_state_dict") is not None
    coarse_model = None
    cond_channels = 7
    if has_coarse:
        coarse_model = CoarsePredictor(in_channels=7, base_ch=32).to(device)
        coarse_model.load_state_dict(ckpt["coarse_state_dict"])
        coarse_model.eval()
        cond_channels = 8

    model = ConditionalUNet(cond_channels=cond_channels, base_ch=base_channels,
                            ch_mult=(1, 2, 4, 8), dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    diffusion = PhysicsConditionedDiffusion(model, timesteps=timesteps, image_h=image_h,
                                            image_w=image_w, device=device,
                                            lambda_grad=lambda_grad, lambda_fft=lambda_fft)

    rows = []
    for idx, batch in enumerate(tqdm(loaders["test"], desc="test")):
        cond = batch["cond"].to(device, non_blocking=True)
        fringe = batch["fringe"].to(device, non_blocking=True)
        target_raw = batch["height_raw"].to(device, non_blocking=True)
        coarse = None
        if coarse_model is not None:
            coarse = coarse_model(cond)
            cond = torch.cat([cond, coarse], dim=1)
        pred = diffusion.sample_ddim(cond, steps=args.ddim_steps, ensemble_size=args.ensemble,
                                     coarse=coarse, start_ratio=start_ratio, progress=False)
        pred_mm = torch.clamp((pred + 1.0) * 0.5, 0.0, 1.0) * height_scale
        metrics = compute_metrics(pred_mm, target_raw)
        metrics["sample"] = idx
        rows.append(metrics)
        if idx < 8:
            save_comparison(fringe, target_raw, pred_mm, out_dir / "samples" / f"sample_{idx:02d}.png",
                            title=f"v5 RMSE {metrics['rmse']:.2f}mm")

    keys = ["sample", "rmse", "mae", "edge_rmse", "normal_deg", "ssim"]
    with open(out_dir / "per_sample_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in keys})

    summary = {}
    for key in keys[1:]:
        m, s = mean_std(rows, key)
        summary[key] = {"mean": m, "std": s}
    summary["n"] = len(rows)
    summary["height_scale"] = height_scale
    summary["checkpoint"] = str(args.ckpt)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
