"""Official-style fringe UNet with zero-initialized physics adapters.

This is a controlled follow-up to concat-channel ablations. The raw fringe
path can be initialized from the A-control checkpoint, while physics features
are injected through lightweight zero adapters.
"""
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
from models import OfficialUNetFPPAdapter
from train_fpp_official_style_unet import (
    HybridL1Loss,
    METRIC_KEYS,
    channel_names,
    parse_channel_spec,
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


def select_cond(batch, device, physics_channels):
    cond = batch["cond"].to(device, non_blocking=True)
    if physics_channels is not None:
        cond = cond[:, physics_channels]
    return cond


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


def resolve_final_checkpoint(save_dir, choice):
    candidates = {
        "best_rmse": save_dir / "checkpoints" / "best_rmse.pt",
        "best_loss": save_dir / "checkpoints" / "best.pt",
        "latest": save_dir / "checkpoints" / "latest.pt",
    }
    path = candidates[choice]
    if path.exists():
        return path
    for fallback in (candidates["best_loss"], candidates["latest"], candidates["best_rmse"]):
        if fallback.exists():
            return fallback
    return path


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, physics_channels):
    model.eval()
    total = 0.0
    seen = 0
    for batch in tqdm(loader, desc="val loss", leave=False):
        fringe = batch["fringe"].to(device, non_blocking=True)
        cond = select_cond(batch, device, physics_channels)
        target = batch["height_01"].to(device, non_blocking=True)
        pred = model(fringe, cond)
        loss = criterion(pred, target)
        total += float(loss.item())
        seen += 1
    return total / max(1, seen)


@torch.no_grad()
def evaluate_metrics(model, loader, device, physics_channels, out_dir=None, save_images=False):
    model.eval()
    rows = []
    if save_images and out_dir is not None:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc="eval metrics"):
        fringe = batch["fringe"].to(device, non_blocking=True)
        cond = select_cond(batch, device, physics_channels)
        pred = model(fringe, cond)
        pred_mm = prediction_to_mm(pred, batch)
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
                save_comparison(
                    fringe[j:j + 1],
                    target_raw[j:j + 1],
                    pred_mm[j:j + 1],
                    out_dir / "samples" / f"sample_{sample_idx:02d}.png",
                    title=f"adapter UNet RMSE {metrics['rmse']:.2f}mm",
                    mask=single_mask,
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
    parser.add_argument("--save_dir", default="/root/autodl-tmp/diffusion_fpp_v5/results/fpp960_c4_adapter_no_xy")
    parser.add_argument("--base_checkpoint", default="")
    parser.add_argument("--init_checkpoint", default="",
                        help="Optional full OfficialUNetFPPAdapter checkpoint for initialization only.")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--include_ftp", action="store_true")
    parser.add_argument("--physics_channels", default="1,2,3,4,5,6")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--adapter_hidden", type=int, default=32)
    parser.add_argument("--eval_metrics_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", default="")
    parser.add_argument("--eval_initial", action="store_true",
                        help="Evaluate and save the initialized model before the first fine-tuning epoch.")
    parser.add_argument("--require_cache", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--final_checkpoint", choices=["best_rmse", "best_loss", "latest"], default="best_rmse")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.physics_channel_indices = parse_channel_spec(args.physics_channels, args.include_ftp)
    args.physics_channel_names = channel_names(args.physics_channel_indices)

    if args.freeze_backbone and not args.base_checkpoint and not args.resume:
        raise ValueError("--freeze_backbone requires --base_checkpoint or --resume")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
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
        image_h=args.image_size,
        image_w=args.image_size,
        require_cache=args.require_cache,
    )

    cond_channels = len(args.physics_channel_indices) if args.physics_channel_indices is not None else loaders["cond_channels"]
    model = OfficialUNetFPPAdapter(
        cond_channels=cond_channels,
        out_channels=1,
        dropout_rate=args.dropout,
        adapter_hidden=args.adapter_hidden,
    ).to(device)

    if args.init_checkpoint and not args.resume:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"Initialized full adapter model from {args.init_checkpoint}")
    elif args.base_checkpoint and not args.resume:
        ckpt = torch.load(args.base_checkpoint, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_backbone_state_dict(state, strict=True)
        print(f"Loaded backbone from {args.base_checkpoint}")

    if args.freeze_backbone:
        model.freeze_backbone()

    trainable = [p for p in model.parameters() if p.requires_grad]
    criterion = HybridL1Loss(alpha=args.alpha)
    optimizer = torch.optim.RMSprop(trainable, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10, min_lr=1e-6
    )
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))

    history = []
    best_val_loss = float("inf")
    best_val_rmse = float("inf")
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        history = ckpt.get("history", history)
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        best_val_rmse = float(ckpt.get("best_val_rmse", best_val_rmse))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}")
    print(f"Physics channels: {args.physics_channel_indices} | {args.physics_channel_names}")
    print(f"Train {len(loaders['train'].dataset)} | Val {len(loaders['val'].dataset)} | Test {len(loaders['test'].dataset)}")
    print(f"Params: {total_params / 1e6:.2f}M | trainable {trainable_params / 1e6:.2f}M")

    if args.eval_initial and start_epoch == 1:
        val_loss = evaluate_loss(model, loaders["val"], criterion, device, args.physics_channel_indices)
        val_rows = evaluate_metrics(model, loaders["val"], device, args.physics_channel_indices)
        val_summary = summarize(val_rows)
        best_val_loss = val_loss
        best_val_rmse = val_summary["rmse"]["mean"]
        log = {
            "epoch": 0,
            "train_loss": None,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": 0.0,
            **{f"val_{k}": val_summary[k]["mean"] for k in METRIC_KEYS},
        }
        history.append(log)
        torch.save(
            checkpoint_state(0, model, optimizer, scheduler, scaler, args, best_val_loss, best_val_rmse, history),
            save_dir / "checkpoints" / "best_rmse.pt",
        )
        torch.save(
            checkpoint_state(0, model, optimizer, scheduler, scaler, args, best_val_loss, best_val_rmse, history),
            save_dir / "checkpoints" / "best.pt",
        )
        print(json.dumps(log, ensure_ascii=False))
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    for ep in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"adapter UNet {ep}/{args.epochs}"):
            fringe = batch["fringe"].to(device, non_blocking=True)
            cond = select_cond(batch, device, args.physics_channel_indices)
            target = batch["height_01"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                pred = model(fringe, cond)
                loss = criterion(pred, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item())
            seen += 1
            if args.max_train_batches and seen >= args.max_train_batches:
                break

        train_loss = total / max(1, seen)
        val_loss = evaluate_loss(model, loaders["val"], criterion, device, args.physics_channel_indices)
        scheduler.step(val_loss)
        log = {
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        }
        improved_rmse = False
        if ep == 1 or ep % args.eval_metrics_every == 0:
            val_rows = evaluate_metrics(model, loaders["val"], device, args.physics_channel_indices)
            val_summary = summarize(val_rows)
            log.update({f"val_{k}": val_summary[k]["mean"] for k in METRIC_KEYS})
            if val_summary["rmse"]["mean"] < best_val_rmse:
                best_val_rmse = val_summary["rmse"]["mean"]
                improved_rmse = True
                first = next(iter(loaders["val"]))
                fringe = first["fringe"].to(device, non_blocking=True)
                cond = select_cond(first, device, args.physics_channel_indices)
                pred = model(fringe, cond)
                pred_mm = prediction_to_mm(pred, first)
                first_mask = first.get("mask")
                if first_mask is not None:
                    first_mask = first_mask.to(device, non_blocking=True)
                save_comparison(
                    fringe,
                    first["height_raw"].to(device),
                    pred_mm,
                    save_dir / "visualizations" / f"val_ep{ep:03d}.png",
                    title=f"adapter val RMSE {val_summary['rmse']['mean']:.2f}mm",
                    mask=first_mask,
                )

        history.append(log)
        if improved_rmse:
            torch.save(
                checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best_val_loss, best_val_rmse, history),
                save_dir / "checkpoints" / "best_rmse.pt",
            )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best_val_loss, best_val_rmse, history),
                save_dir / "checkpoints" / "best.pt",
            )
        print(json.dumps(log, ensure_ascii=False))
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        if args.save_every > 0 and (ep == 1 or ep == args.epochs or ep % args.save_every == 0):
            torch.save(
                checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best_val_loss, best_val_rmse, history),
                save_dir / "checkpoints" / "latest.pt",
            )
        if optimizer.param_groups[0]["lr"] <= 1e-6:
            print("Learning rate reached minimum. Stopping.")
            break

    best_path = resolve_final_checkpoint(save_dir, args.final_checkpoint)
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_rows = evaluate_metrics(
            model,
            loaders["test"],
            device,
            args.physics_channel_indices,
            out_dir=save_dir / "evaluation",
            save_images=True,
        )
        summary = write_eval_outputs(test_rows, save_dir / "evaluation", best_path, args)
        print("Final test:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
