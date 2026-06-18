from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.dataset_fpp_phase import create_fpp_phase_loaders
from precompute_fpp_phase_diffusion_predictions import build_model_from_checkpoint
from train_fpp_phase_diffusion import angle_from_sincos, phase_metrics_tensor, phase_target


def wrap_angle(x):
    return np.angle(np.exp(1j * x))


def aligned_phase_arrays(pred, target, mask):
    pred_ang = angle_from_sincos(pred).detach().cpu().numpy()[0, 0]
    target_ang = angle_from_sincos(target).detach().cpu().numpy()[0, 0]
    valid = mask.detach().cpu().numpy()[0, 0] > 0.5
    delta = target_ang[valid] - pred_ang[valid]
    offset = math.atan2(float(np.sin(delta).mean()), float(np.cos(delta).mean())) if valid.any() else 0.0
    aligned_pred = wrap_angle(pred_ang + offset)
    aligned_err = np.abs(wrap_angle(aligned_pred - target_ang))
    raw_err = np.abs(wrap_angle(pred_ang - target_ang))
    aligned_err = np.where(valid, aligned_err, np.nan)
    raw_err = np.where(valid, raw_err, np.nan)
    return target_ang, aligned_pred, raw_err, aligned_err, offset


def save_visual(batch, pred, target, mask, path, title):
    import matplotlib.pyplot as plt

    fringe = batch["fringe"][0, 0].detach().cpu().numpy()
    target_ang, aligned_pred, raw_err, aligned_err, offset = aligned_phase_arrays(pred, target, mask)
    aligned_mae = float(np.nanmean(aligned_err))
    raw_mae = float(np.nanmean(raw_err))

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)
    axes[0].imshow(fringe, cmap="gray")
    axes[0].set_title("single fringe")
    axes[1].imshow(target_ang, cmap="twilight", vmin=-math.pi, vmax=math.pi)
    axes[1].set_title("GT wrapped")
    axes[2].imshow(aligned_pred, cmap="twilight", vmin=-math.pi, vmax=math.pi)
    axes[2].set_title("pred aligned")
    im3 = axes[3].imshow(raw_err, cmap="magma", vmin=0.0, vmax=math.pi)
    axes[3].set_title("raw error")
    im4 = axes[4].imshow(aligned_err, cmap="magma", vmin=0.0, vmax=1.0)
    axes[4].set_title("aligned error")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    fig.suptitle(f"{title} | aligned MAE {aligned_mae:.3f} rad | raw MAE {raw_mae:.3f} rad | offset {offset:.3f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_phase_cache_960")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--ensemble", type=int, default=5)
    parser.add_argument("--sample_start_from", choices=["noise", "ftp", "hilbert"], default="ftp")
    parser.add_argument("--sample_start_ratio", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    diffusion, _, _ = build_model_from_checkpoint(ckpt, device)
    loaders = create_fpp_phase_loaders(
        base_cache_dir=args.base_cache_dir,
        phase_cache_dir=args.phase_cache_dir,
        batch_size=1,
        eval_batch_size=1,
        num_workers=args.num_workers,
        image_h=args.image_size,
        image_w=args.image_size,
        require_cache=True,
    )
    out_dir = Path(args.save_dir)
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        loader = loaders["train_eval" if split == "train" else split]
        for idx, batch in enumerate(tqdm(loader, desc=f"aligned visuals {split}")):
            if idx >= args.limit:
                break
            pred = diffusion.sample_ddim(
                batch,
                steps=args.ddim_steps,
                ensemble_size=args.ensemble,
                start_from=args.sample_start_from,
                start_ratio=args.sample_start_ratio,
                progress=False,
            )
            target = phase_target(batch, device, target_channels=diffusion.target_channels)
            mask = batch["mask"].to(device, non_blocking=True)
            metrics = phase_metrics_tensor(pred, target, mask=mask)
            save_visual(
                batch,
                pred,
                target,
                mask,
                out_dir / split / f"{split}_{idx:02d}_aligned.png",
                title=f"{split} sample {idx} metric aligned {metrics['phase_aligned_mae_rad']:.3f}",
            )


if __name__ == "__main__":
    main()
