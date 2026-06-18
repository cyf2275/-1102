"""Deterministic residual refiner on top of cached FPP base predictions."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from diffusion_pip import (
    charbonnier,
    confidence_edge_loss,
    masked_mse,
    normal_loss,
    oriented_gradient_loss,
)
from models import ConditionalUNetAdapter
from train_fpp_official_style_unet import METRIC_KEYS, channel_names, parse_channel_spec, summarize
from train_pip_lite import prediction_to_mm, zero_initialize_prediction_head
from utils.metrics import compute_metrics
from utils.visualization import save_comparison


def select_cond(batch, device, physics_channels):
    cond = batch["cond"].to(device, non_blocking=True)
    if physics_channels is not None:
        cond = cond[:, physics_channels]
    return cond


def residual_target(height, base, residual_scale):
    return torch.clamp((height - base) / float(residual_scale), -1.0, 1.0)


def residual_to_depth(residual, base, residual_scale):
    return torch.clamp(base + residual * float(residual_scale), -1.0, 1.0)


def resolve_residual_scale(args):
    if args.residual_scale > 0:
        return float(args.residual_scale)
    stats_path = Path(args.cache_dir) / f"{args.base_prefix}_stats.json"
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return float(stats.get("p99_abs_residual", stats.get("residual_scale", 0.0)))


def save_rows(rows, path):
    keys = ["sample"] + METRIC_KEYS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in keys})


def refiner_loss(model, batch, device, physics_channels, residual_scale, args):
    base = batch["base_height"].to(device, non_blocking=True)
    height = batch["height"].to(device, non_blocking=True)
    cond = select_cond(batch, device, physics_channels)
    mask = torch.clamp(batch["mask"].to(device, non_blocking=True), 0.0, 1.0)
    t = torch.zeros((base.shape[0],), device=device, dtype=torch.long)
    pred_residual = torch.clamp(model(base, t, cond), -1.0, 1.0)
    target_residual = residual_target(height, base, residual_scale)
    pred_depth = residual_to_depth(pred_residual, base, residual_scale)
    loss = charbonnier(pred_residual, target_residual, mask=mask)
    loss = loss + 0.5 * masked_mse(pred_residual, target_residual, mask=mask)
    loss = loss + 0.5 * charbonnier(pred_depth, height, mask=mask)
    if args.lambda_oriented > 0:
        loss = loss + args.lambda_oriented * oriented_gradient_loss(
            pred_depth,
            height,
            batch["phase_sin"].to(device, non_blocking=True),
            batch["phase_cos"].to(device, non_blocking=True),
            batch["phase_conf"].to(device, non_blocking=True),
            mask=mask,
        )
    if args.lambda_edge > 0:
        loss = loss + args.lambda_edge * confidence_edge_loss(
            pred_depth,
            height,
            batch["edge_score"].to(device, non_blocking=True),
            batch["phase_conf"].to(device, non_blocking=True),
            mask=mask,
        )
    if args.lambda_normal > 0:
        loss = loss + args.lambda_normal * normal_loss(pred_depth, height, mask=mask)
    return loss


@torch.no_grad()
def evaluate(model, loader, device, physics_channels, residual_scale, out_dir=None, save_images=False):
    model.eval()
    rows = []
    base_rows = []
    if save_images and out_dir is not None:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc="eval refiner"):
        base = batch["base_height"].to(device, non_blocking=True)
        cond = select_cond(batch, device, physics_channels)
        t = torch.zeros((base.shape[0],), device=device, dtype=torch.long)
        pred_residual = torch.clamp(model(base, t, cond), -1.0, 1.0)
        pred = residual_to_depth(pred_residual, base, residual_scale)
        pred_mm = prediction_to_mm(pred, batch, 1.0)
        base_mm = prediction_to_mm(base, batch, 1.0)
        target_raw = batch["height_raw"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        fringe = batch["fringe"].to(device, non_blocking=True)
        for j in range(pred.shape[0]):
            sample_idx = len(rows)
            single_mask = mask[j:j + 1]
            rows.append({
                "sample": sample_idx,
                **compute_metrics(pred_mm[j:j + 1], target_raw[j:j + 1], mask=single_mask),
            })
            base_rows.append({
                "sample": sample_idx,
                **compute_metrics(base_mm[j:j + 1], target_raw[j:j + 1], mask=single_mask),
            })
            if save_images and out_dir is not None and sample_idx < 8:
                save_comparison(
                    fringe[j:j + 1],
                    target_raw[j:j + 1],
                    pred_mm[j:j + 1],
                    out_dir / "samples" / f"sample_{sample_idx:02d}.png",
                    title=f"residual refiner RMSE {rows[-1]['rmse']:.2f}mm",
                    mask=single_mask,
                )
    return rows, base_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--save_dir", default="/root/autodl-tmp/diffusion_fpp_v5/results/fpp_residual_refiner")
    parser.add_argument("--base_prefix", default="base_c4_adapter")
    parser.add_argument("--include_ftp", action="store_true")
    parser.add_argument("--physics_channels", default="1,2,3,4,5,6")
    parser.add_argument("--residual_scale", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=48)
    parser.add_argument("--adapter_hidden", type=int, default=32)
    parser.add_argument("--image_h", type=int, default=480)
    parser.add_argument("--image_w", type=int, default=480)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--lambda_oriented", type=float, default=0.08)
    parser.add_argument("--lambda_edge", type=float, default=0.03)
    parser.add_argument("--lambda_normal", type=float, default=0.01)
    parser.add_argument("--require_cache", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=0)
    args = parser.parse_args()

    args.physics_channel_indices = parse_channel_spec(args.physics_channels, args.include_ftp)
    args.physics_channel_names = channel_names(args.physics_channel_indices)
    args.resolved_residual_scale = resolve_residual_scale(args)

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
        include_ftp=args.include_ftp,
        image_h=args.image_h,
        image_w=args.image_w,
        require_cache=args.require_cache,
        base_prefix=args.base_prefix,
    )
    cond_channels = len(args.physics_channel_indices) if args.physics_channel_indices is not None else loaders["cond_channels"]
    model = ConditionalUNetAdapter(
        cond_channels=cond_channels,
        base_ch=args.base_channels,
        ch_mult=(1, 2, 4, 8),
        dropout=0.05,
        adapter_hidden=args.adapter_hidden,
    ).to(device)
    zero_initialize_prediction_head(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    best = float("inf")
    history = []

    print(f"Device: {device}")
    print(f"Base prefix: {args.base_prefix} | residual_scale={args.resolved_residual_scale:.6f}")
    print(f"Physics channels: {args.physics_channel_indices} | {args.physics_channel_names}")
    print(f"Train {len(loaders['train'].dataset)} | Val {len(loaders['val'].dataset)} | Test {len(loaders['test'].dataset)}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"residual refiner {ep}/{args.epochs}"):
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                loss = refiner_loss(model, batch, device, args.physics_channel_indices,
                                    args.resolved_residual_scale, args)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.item())
            seen += 1
            if args.max_train_batches and seen >= args.max_train_batches:
                break
        scheduler.step()
        log = {
            "epoch": ep,
            "train_loss": total / max(1, seen),
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - t0,
        }
        if ep == 1 or ep % args.eval_every == 0:
            rows, base_rows = evaluate(model, loaders["val"], device, args.physics_channel_indices,
                                       args.resolved_residual_scale)
            summary = summarize(rows)
            base_summary = summarize(base_rows)
            log.update({f"val_{k}": summary[k]["mean"] for k in METRIC_KEYS})
            log.update({f"base_val_{k}": base_summary[k]["mean"] for k in METRIC_KEYS})
            if summary["rmse"]["mean"] < best:
                best = summary["rmse"]["mean"]
                torch.save({
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_val_rmse": best,
                }, save_dir / "checkpoints" / "best.pt")
                print(f"  -> best val RMSE {best:.3f}mm (base {base_summary['rmse']['mean']:.3f}mm)")
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        if args.save_every > 0 and (ep == 1 or ep == args.epochs or ep % args.save_every == 0):
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "args": vars(args),
                "best_val_rmse": best,
                "history": history,
            }, save_dir / "checkpoints" / "latest.pt")

    best_path = save_dir / "checkpoints" / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        rows, base_rows = evaluate(model, loaders["test"], device, args.physics_channel_indices,
                                   args.resolved_residual_scale,
                                   out_dir=save_dir / "evaluation", save_images=True)
        summary = summarize(rows)
        summary["base"] = summarize(base_rows)
        summary["n"] = len(rows)
        save_rows(rows, save_dir / "evaluation" / "per_sample_metrics.csv")
        with open(save_dir / "evaluation" / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("Final test:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
