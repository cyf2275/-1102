"""Depth residual refiner conditioned on phase-restoration diffusion output."""
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

from data.dataset_fpp_phase import create_fpp_phase_loaders
from diffusion_pip import charbonnier, confidence_edge_loss, masked_mse, normal_loss, oriented_gradient_loss
from models import ConditionalUNetAdapter
from train_fpp_official_style_unet import METRIC_KEYS, summarize
from train_fpp_phase_diffusion import parse_channel_spec, parse_ch_mult
from train_pip_lite import prediction_to_mm, zero_initialize_prediction_head
from utils.metrics import compute_metrics
from utils.visualization import save_comparison


def resolve_residual_scale(args):
    if args.residual_scale > 0:
        return float(args.residual_scale)
    stats_path = Path(args.base_cache_dir) / f"{args.base_prefix}_stats.json"
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    for key in ("p99_abs_residual", "residual_scale"):
        val = float(stats.get(key, 0.0))
        if val > 0:
            return val
    raise ValueError(f"no positive residual scale in {stats_path}")


def residual_target(height, base, residual_scale):
    return torch.clamp((height - base) / float(residual_scale), -1.0, 1.0)


def residual_to_depth(residual, base, residual_scale):
    return torch.clamp(base + residual * float(residual_scale), -1.0, 1.0)


def select_cond(batch, device, phase_channels, use_phase_pred=True):
    cond = batch["cond"].to(device, non_blocking=True)
    if phase_channels is not None:
        cond = cond[:, phase_channels]
    if use_phase_pred:
        if "phase_pred" not in batch:
            raise KeyError("use_phase_pred=True requires phase_pred_prefix cache")
        phase_pred = batch["phase_pred"].to(device, non_blocking=True)
        cond = torch.cat([cond, phase_pred], dim=1)
    return cond


def save_rows(rows, path):
    keys = ["sample"] + METRIC_KEYS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in keys})


def refiner_loss(model, batch, device, phase_channels, residual_scale, args):
    base = torch.clamp(batch["base_height"].to(device, non_blocking=True), -1.0, 1.0)
    height = batch["height"].to(device, non_blocking=True)
    cond = select_cond(batch, device, phase_channels, use_phase_pred=args.use_phase_pred)
    mask = torch.clamp(batch["mask"].to(device, non_blocking=True), 0.0, 1.0)
    t = torch.zeros((base.shape[0],), device=device, dtype=torch.long)
    pred_residual = torch.clamp(model(base, t, cond), -1.0, 1.0)
    target_residual = residual_target(height, base, residual_scale)
    pred_depth = residual_to_depth(pred_residual, base, residual_scale)
    loss = charbonnier(pred_residual, target_residual, mask=mask)
    loss = loss + 0.5 * masked_mse(pred_residual, target_residual, mask=mask)
    loss = loss + 0.5 * charbonnier(pred_depth, height, mask=mask)
    if args.lambda_oriented > 0:
        phase_sin = batch["phase_target"][:, 0:1].to(device, non_blocking=True)
        phase_cos = batch["phase_target"][:, 1:2].to(device, non_blocking=True)
        phase_conf = torch.clamp(batch["cond"][:, 8:9].to(device, non_blocking=True), 0.0, 1.0)
        loss = loss + args.lambda_oriented * oriented_gradient_loss(
            pred_depth, height, phase_sin, phase_cos, phase_conf, mask=mask
        )
    if args.lambda_edge > 0:
        cond_full = batch["cond"].to(device, non_blocking=True)
        edge = torch.clamp(0.5 * cond_full[:, 9:10] + 0.5 * cond_full[:, 10:11], 0.0, 1.0)
        conf = torch.clamp(cond_full[:, 8:9], 0.0, 1.0)
        loss = loss + args.lambda_edge * confidence_edge_loss(pred_depth, height, edge, conf, mask=mask)
    if args.lambda_normal > 0:
        loss = loss + args.lambda_normal * normal_loss(pred_depth, height, mask=mask)
    return loss


@torch.no_grad()
def evaluate(model, loader, device, phase_channels, residual_scale, args, out_dir=None, save_images=False):
    model.eval()
    rows = []
    base_rows = []
    if save_images and out_dir is not None:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc="eval phase-pred refiner"):
        base = torch.clamp(batch["base_height"].to(device, non_blocking=True), -1.0, 1.0)
        cond = select_cond(batch, device, phase_channels, use_phase_pred=args.use_phase_pred)
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
                    title=f"phase-pred refiner RMSE {rows[-1]['rmse']:.2f}mm",
                    mask=single_mask,
                )
    return rows, base_rows


def checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best, history):
    return {
        "epoch": ep,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": vars(args),
        "best_val_rmse": best,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_phase_cache_960")
    parser.add_argument("--save_dir", default="/root/autodl-tmp/diffusion_fpp_v5/results/fpp960_phase_pred_refiner")
    parser.add_argument("--base_prefix", default="base_c4_adapter")
    parser.add_argument("--phase_pred_prefix", default="")
    parser.add_argument("--phase_channels", default="1-10")
    parser.add_argument("--use_phase_pred", action="store_true")
    parser.add_argument("--residual_scale", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--ch_mult", default="1,2,4,8,8")
    parser.add_argument("--adapter_hidden", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lambda_oriented", type=float, default=0.04)
    parser.add_argument("--lambda_edge", type=float, default=0.02)
    parser.add_argument("--lambda_normal", type=float, default=0.005)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=3)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()
    args.ch_mult_tuple = parse_ch_mult(args.ch_mult)
    args.resolved_residual_scale = resolve_residual_scale(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    loaders = create_fpp_phase_loaders(
        base_cache_dir=args.base_cache_dir,
        phase_cache_dir=args.phase_cache_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        image_h=args.image_size,
        image_w=args.image_size,
        base_prefix=args.base_prefix,
        phase_pred_prefix=(args.phase_pred_prefix if args.use_phase_pred else None),
        require_cache=True,
    )
    max_channel = loaders["cond_channels"] - 1
    args.phase_channel_indices = parse_channel_spec(args.phase_channels, max_channel=max_channel)
    cond_channels = len(args.phase_channel_indices) if args.phase_channel_indices is not None else loaders["cond_channels"]
    if args.use_phase_pred:
        cond_channels += 2

    save_dir = Path(args.save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    model = ConditionalUNetAdapter(
        in_channels=1,
        cond_channels=cond_channels,
        out_channels=1,
        base_ch=args.base_channels,
        ch_mult=args.ch_mult_tuple,
        dropout=0.05,
        adapter_hidden=args.adapter_hidden,
    ).to(device)
    zero_initialize_prediction_head(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))

    print(f"Device: {device}")
    print(f"Base prefix: {args.base_prefix} | residual_scale={args.resolved_residual_scale:.6f}")
    print(f"Phase channels: {args.phase_channel_indices} | use_phase_pred={args.use_phase_pred} prefix={args.phase_pred_prefix}")
    print(f"Train {len(loaders['train'].dataset)} | Val {len(loaders['val'].dataset)} | Test {len(loaders['test'].dataset)}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    history = []
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"phase-pred refiner {ep}/{args.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                loss = refiner_loss(model, batch, device, args.phase_channel_indices, args.resolved_residual_scale, args)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
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
            rows, base_rows = evaluate(model, loaders["val"], device, args.phase_channel_indices, args.resolved_residual_scale, args)
            summary = summarize(rows)
            base_summary = summarize(base_rows)
            log.update({f"val_{k}": summary[k]["mean"] for k in METRIC_KEYS})
            log.update({f"base_val_{k}": base_summary[k]["mean"] for k in METRIC_KEYS})
            if summary["rmse"]["mean"] < best:
                best = summary["rmse"]["mean"]
                torch.save(
                    checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best, history),
                    save_dir / "checkpoints" / "best.pt",
                )
                print(f"  -> best val RMSE {best:.3f}mm (base {base_summary['rmse']['mean']:.3f}mm)")
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        if args.save_every > 0 and (ep == 1 or ep == args.epochs or ep % args.save_every == 0):
            torch.save(
                checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best, history),
                save_dir / "checkpoints" / "latest.pt",
            )

    best_path = save_dir / "checkpoints" / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        rows, base_rows = evaluate(
            model,
            loaders["test"],
            device,
            args.phase_channel_indices,
            args.resolved_residual_scale,
            args,
            out_dir=save_dir / "evaluation",
            save_images=True,
        )
        eval_dir = save_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        save_rows(rows, eval_dir / "per_sample_metrics.csv")
        save_rows(base_rows, eval_dir / "base_per_sample_metrics.csv")
        summary = summarize(rows)
        base_summary = summarize(base_rows)
        summary["base"] = base_summary
        summary["n"] = len(rows)
        summary["checkpoint"] = str(best_path)
        summary["args"] = vars(args)
        with open(eval_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("Final test:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
