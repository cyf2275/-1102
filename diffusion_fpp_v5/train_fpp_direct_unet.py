"""Deterministic FPP-ML-Bench sanity baseline.

This script is intentionally not a diffusion model. It checks whether the
FPP-ML-Bench A0 official split, mask handling, and denormalization pipeline can
support a plain supervised model before spending time on diffusion training.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from models import ConditionalUNet
from utils.metrics import compute_metrics
from utils.visualization import save_comparison


METRIC_KEYS = ["rmse", "mae", "edge_rmse", "normal_deg", "ssim"]


def masked_mean(x, mask=None):
    if mask is None:
        return x.mean()
    mask = torch.clamp(mask.to(device=x.device, dtype=x.dtype), 0.0, 1.0)
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def charbonnier(pred, target, eps=1e-3, mask=None):
    return masked_mean(torch.sqrt((pred - target) ** 2 + eps * eps), mask=mask)


def masked_mse(pred, target, mask=None):
    return masked_mean((pred - target) ** 2, mask=mask)


def gradient_loss(pred, target, mask=None):
    pdx = pred[..., :, 1:] - pred[..., :, :-1]
    tdx = target[..., :, 1:] - target[..., :, :-1]
    pdy = pred[..., 1:, :] - pred[..., :-1, :]
    tdy = target[..., 1:, :] - target[..., :-1, :]
    if mask is None:
        return torch.mean(torch.abs(pdx - tdx)) + torch.mean(torch.abs(pdy - tdy))
    mx = mask[..., :, 1:] * mask[..., :, :-1]
    my = mask[..., 1:, :] * mask[..., :-1, :]
    return masked_mean(torch.abs(pdx - tdx), mx) + masked_mean(torch.abs(pdy - tdy), my)


def prediction_to_mm(pred, batch):
    pred_01 = torch.clamp((pred + 1.0) * 0.5, 0.0, 1.0)
    minmax = batch["depth_minmax"].to(pred.device, non_blocking=True)
    depth_min = minmax[:, 0].view(-1, 1, 1, 1)
    depth_max = minmax[:, 1].view(-1, 1, 1, 1)
    return pred_01 * (depth_max - depth_min).clamp(min=1e-6) + depth_min


def mean_std(rows, key):
    vals = np.array([r[key] for r in rows], dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)


def summarize(rows):
    return {key: {"mean": mean_std(rows, key)[0], "std": mean_std(rows, key)[1]} for key in METRIC_KEYS}


def save_rows(rows, path):
    keys = ["sample"] + METRIC_KEYS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in keys})


def forward_model(model, batch, device):
    cond = batch["cond"].to(device, non_blocking=True)
    zeros = torch.zeros((cond.shape[0], 1, cond.shape[2], cond.shape[3]), device=device)
    t = torch.zeros((cond.shape[0],), device=device, dtype=torch.long)
    return torch.tanh(model(zeros, t, cond))


@torch.no_grad()
def evaluate(model, loader, device, split_name, args, out_dir=None, save_images=False):
    model.eval()
    rows = []
    if save_images and out_dir is not None:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc=f"eval {split_name}"):
        pred = forward_model(model, batch, device)
        pred_mm = prediction_to_mm(pred, batch)
        fringe = batch["fringe"].to(device, non_blocking=True)
        target_raw = batch["height_raw"].to(device, non_blocking=True)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        for j in range(pred.shape[0]):
            sample_idx = len(rows)
            single_mask = mask[j:j + 1] if mask is not None else None
            metrics = compute_metrics(pred_mm[j:j + 1], target_raw[j:j + 1], mask=single_mask)
            rows.append({"sample": sample_idx, **metrics})
            if save_images and out_dir is not None and sample_idx < 8:
                save_comparison(fringe[j:j + 1], target_raw[j:j + 1], pred_mm[j:j + 1],
                                out_dir / "samples" / f"sample_{sample_idx:02d}.png",
                                title=f"Direct UNet RMSE {metrics['rmse']:.2f}mm",
                                mask=single_mask)
    return rows


def write_eval_outputs(rows, out_dir, checkpoint, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rows(rows, out_dir / "per_sample_metrics.csv")
    summary = summarize(rows)
    summary["n"] = len(rows)
    summary["checkpoint"] = str(checkpoint)
    summary["args"] = vars(args)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_480")
    parser.add_argument("--save_dir", default="/root/autodl-tmp/diffusion_fpp_v5/results/fpp480_direct_unet_a0")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--train_epoch_repeats", type=int, default=1,
                        help="Repeat tiny A0 training split within one epoch using replacement sampling.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--lambda_grad", type=float, default=0.10)
    parser.add_argument("--image_h", type=int, default=480)
    parser.add_argument("--image_w", type=int, default=480)
    parser.add_argument("--include_ftp", action="store_true")
    parser.add_argument("--require_cache", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    save_dir = Path(args.save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    loaders = create_fpp_ml_bench_loaders(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        require_cache=args.require_cache,
        include_ftp=args.include_ftp,
        image_h=args.image_h,
        image_w=args.image_w,
        train_epoch_repeats=args.train_epoch_repeats,
    )
    cond_channels = loaders["cond_channels"]
    print(f"Device: {device}")
    print(f"Cond channels: {cond_channels}")
    print(f"Train {len(loaders['train'].dataset)} | Val {len(loaders['val'].dataset)} | Test {len(loaders['test'].dataset)}")

    model = ConditionalUNet(
        cond_channels=cond_channels,
        base_ch=args.base_channels,
        ch_mult=(1, 2, 4, 8),
        dropout=0.05,
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=(device.type == "cuda"))
    best = float("inf")
    history = []
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            sch.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        best = float(ckpt.get("best_val_rmse", best))
        history = ckpt.get("history", history)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for ep in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"direct-unet {ep}/{args.epochs}"):
            target = batch["height"].to(device, non_blocking=True)
            mask = batch.get("mask")
            if mask is not None:
                mask = torch.clamp(mask.to(device, non_blocking=True), 0.0, 1.0)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                pred = forward_model(model, batch, device)
                loss = charbonnier(pred, target, mask=mask) + 0.5 * masked_mse(pred, target, mask=mask)
                if args.lambda_grad > 0:
                    loss = loss + args.lambda_grad * gradient_loss(pred, target, mask=mask)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.item())
            seen += 1
            if args.max_train_batches and seen >= args.max_train_batches:
                break
        sch.step()

        log = {
            "epoch": ep,
            "train_loss": total / max(1, seen),
            "lr": sch.get_last_lr()[0],
            "seconds": time.time() - t0,
        }
        if ep == 1 or ep % args.eval_every == 0:
            val_rows = evaluate(model, loaders["val"], device, "val", args)
            val_summary = summarize(val_rows)
            log.update({f"val_{k}": val_summary[k]["mean"] for k in METRIC_KEYS})
            val_rmse = val_summary["rmse"]["mean"]
            if val_rmse < best:
                best = val_rmse
                ckpt = {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_val_rmse": best,
                    "cond_channels": cond_channels,
                    "include_ftp": args.include_ftp,
                }
                torch.save(ckpt, save_dir / "checkpoints" / "best.pt")
                first = next(iter(loaders["val"]))
                pred = forward_model(model, first, device)
                pred_mm = prediction_to_mm(pred, first)
                first_mask = first.get("mask")
                if first_mask is not None:
                    first_mask = first_mask.to(device, non_blocking=True)
                save_comparison(first["fringe"].to(device), first["height_raw"].to(device), pred_mm,
                                save_dir / "visualizations" / f"val_ep{ep:03d}.png",
                                title=f"Direct UNet val RMSE {best:.2f}mm",
                                mask=first_mask)
                print(f"  -> best val RMSE {best:.3f}mm")

        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        if args.save_every > 0 and (ep == 1 or ep == args.epochs or ep % args.save_every == 0):
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sch.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "args": vars(args),
                "best_val_rmse": best,
                "history": history,
            }, save_dir / "checkpoints" / "latest.pt")

    best_path = save_dir / "checkpoints" / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_rows = evaluate(model, loaders["test"], device, "test", args,
                             out_dir=save_dir / "evaluation", save_images=True)
        summary = write_eval_outputs(test_rows, save_dir / "evaluation", best_path, args)
        print("Final test:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
