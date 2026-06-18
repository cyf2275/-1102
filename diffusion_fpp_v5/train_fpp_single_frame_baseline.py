from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from models.single_frame_baselines import build_single_frame_baseline
from train_fpp_official_style_unet import (
    HybridL1Loss,
    METRIC_KEYS,
    prediction_to_mm,
    summarize,
)
from utils.metrics import compute_metrics
from utils.visualization import save_comparison


def save_rows(rows, path):
    keys = ["sample"] + METRIC_KEYS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in keys})


def checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best_val_loss, best_val_rmse, history):
    return {
        "epoch": ep,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": vars(args),
        "best_val_loss": best_val_loss,
        "best_val_rmse": best_val_rmse,
        "history": history,
    }


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    seen = 0
    for batch in tqdm(loader, desc="val loss", leave=False):
        fringe = batch["fringe"].to(device, non_blocking=True)
        target = batch["height_01"].to(device, non_blocking=True)
        total += float(criterion(model(fringe), target).item())
        seen += 1
    return total / max(1, seen)


@torch.no_grad()
def evaluate_metrics(model, loader, device, args, out_dir=None, save_images=False):
    model.eval()
    rows = []
    if save_images and out_dir is not None:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc="eval metrics"):
        fringe = batch["fringe"].to(device, non_blocking=True)
        pred = model(fringe)
        pred_mm = prediction_to_mm(pred, batch)
        target = batch["height_raw"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        for j in range(pred.shape[0]):
            sample_idx = len(rows)
            metrics = compute_metrics(pred_mm[j:j + 1], target[j:j + 1], mask=mask[j:j + 1])
            rows.append({"sample": sample_idx, **metrics})
            if save_images and out_dir is not None and sample_idx < 8:
                save_comparison(
                    fringe[j:j + 1],
                    target[j:j + 1],
                    pred_mm[j:j + 1],
                    out_dir / "samples" / f"sample_{sample_idx:02d}.png",
                    title=f"{args.arch} RMSE {metrics['rmse']:.2f}mm",
                    mask=mask[j:j + 1],
                )
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
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--arch", choices=["resunet", "attention_unet", "nested_unet"], required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--base_channels", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_metrics_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--require_cache", action="store_true")
    parser.add_argument("--preload_ram", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=2300)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    loaders = create_fpp_ml_bench_loaders(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        image_h=args.image_size,
        image_w=args.image_size,
        require_cache=args.require_cache,
    )

    save_dir = Path(args.save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    model = build_single_frame_baseline(
        args.arch,
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels or None,
        dropout_rate=args.dropout,
    ).to(device)
    criterion = HybridL1Loss(alpha=args.alpha)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10, min_lr=1e-6
    )
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))

    print(f"Device: {device}")
    print(f"Arch: {args.arch} | Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Train {len(loaders['train'].dataset)} | Val {len(loaders['val'].dataset)} | Test {len(loaders['test'].dataset)}")

    history = []
    best_val_loss = float("inf")
    best_val_rmse = float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"{args.arch} {ep}/{args.epochs}"):
            fringe = batch["fringe"].to(device, non_blocking=True)
            target = batch["height_01"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                loss = criterion(model(fringe), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item())
            seen += 1
            if args.max_train_batches and seen >= args.max_train_batches:
                break
        train_loss = total / max(1, seen)
        val_loss = None
        if ep == 1 or ep == args.epochs or ep % max(1, args.eval_every) == 0:
            val_loss = evaluate_loss(model, loaders["val"], criterion, device)
            scheduler.step(val_loss)

        log = {
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        }
        improved_rmse = False
        if ep == 1 or ep == args.epochs or ep % max(1, args.eval_metrics_every) == 0:
            val_rows = evaluate_metrics(model, loaders["val"], device, args)
            val_summary = summarize(val_rows)
            log.update({f"val_{k}": val_summary[k]["mean"] for k in METRIC_KEYS})
            if val_summary["rmse"]["mean"] < best_val_rmse:
                best_val_rmse = val_summary["rmse"]["mean"]
                improved_rmse = True

        history.append(log)
        if improved_rmse:
            torch.save(
                checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best_val_loss, best_val_rmse, history),
                save_dir / "checkpoints" / "best_rmse.pt",
            )
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best_val_loss, best_val_rmse, history),
                save_dir / "checkpoints" / "best.pt",
            )
        if args.save_every > 0 and (ep == 1 or ep == args.epochs or ep % args.save_every == 0):
            torch.save(
                checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best_val_loss, best_val_rmse, history),
                save_dir / "checkpoints" / "latest.pt",
            )
        print(json.dumps(log, ensure_ascii=False))
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    best_path = save_dir / "checkpoints" / "best_rmse.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        rows = evaluate_metrics(model, loaders["test"], device, args, out_dir=save_dir / "evaluation", save_images=True)
        summary = write_eval_outputs(rows, save_dir / "evaluation", best_path, args)
        print("Final test:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Persistent dataloader workers can otherwise keep queue scripts alive.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
