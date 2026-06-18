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
from models.official_unet import OfficialUNetFPP
from train_fpp_official_style_unet import HybridL1Loss, METRIC_KEYS, prediction_to_mm, summarize
from utils.metrics import compute_metrics
from utils.visualization import save_comparison


def make_input(batch, device, mode):
    cond = batch["cond"].to(device, non_blocking=True)
    xy = cond[:, 11:13]
    pieces = []
    if mode.endswith("_plus_fringe"):
        pieces.append(batch["fringe"].to(device, non_blocking=True))
    if mode.startswith("gt_phase"):
        pieces.append(batch["phase_target"].to(device, non_blocking=True))
    elif mode.startswith("phase_pred"):
        phase = batch["phase_pred"].to(device, non_blocking=True)
        if phase.shape[1] < 3:
            raise ValueError(f"phase_pred input requires at least 3 channels, got {phase.shape[1]}")
        pieces.append(phase[:, :3])
    else:
        raise ValueError(f"unknown input mode: {mode}")
    pieces.append(xy)
    return torch.cat(pieces, dim=1)


def input_channels(mode):
    return 6 if mode.endswith("_plus_fringe") else 5


def save_rows(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample"] + METRIC_KEYS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in ["sample"] + METRIC_KEYS})


def load_expanded_state(model, checkpoint_path, raw_channel=0):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    source = ckpt.get("model_state_dict", ckpt)
    target = model.state_dict()
    loaded = {}
    expanded = {}
    skipped = []
    for key, value in source.items():
        dst_key = key
        if dst_key.startswith("backbone."):
            dst_key = dst_key[len("backbone."):]
        if dst_key not in target:
            skipped.append(key)
            continue
        dst = target[dst_key]
        if tuple(dst.shape) == tuple(value.shape):
            loaded[dst_key] = value
            continue
        if dst_key == "down1.conv.conv.0.weight" and value.ndim == 4 and dst.ndim == 4:
            if value.shape[0] == dst.shape[0] and value.shape[2:] == dst.shape[2:]:
                new_value = dst.clone()
                new_value.zero_()
                if value.shape[1] == 1 and 0 <= raw_channel < dst.shape[1]:
                    new_value[:, raw_channel:raw_channel + 1] = value
                    expanded[dst_key] = new_value
                    continue
                if value.shape[1] <= dst.shape[1]:
                    new_value[:, : value.shape[1]] = value
                    expanded[dst_key] = new_value
                    continue
        skipped.append(key)
    target.update(loaded)
    target.update(expanded)
    model.load_state_dict(target)
    return {"loaded": len(loaded), "expanded": len(expanded), "skipped_examples": skipped[:20]}


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
def evaluate_loss(model, loader, criterion, device, mode):
    model.eval()
    total = 0.0
    seen = 0
    for batch in tqdm(loader, desc="val loss", leave=False):
        x = make_input(batch, device, mode)
        target = batch["height_01"].to(device, non_blocking=True)
        total += float(criterion(model(x), target).item())
        seen += 1
    return total / max(1, seen)


@torch.no_grad()
def evaluate_metrics(model, loader, device, mode, out_dir=None, save_images=False):
    model.eval()
    rows = []
    if save_images and out_dir is not None:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc="eval metrics"):
        x = make_input(batch, device, mode)
        pred = model(x)
        pred_mm = prediction_to_mm(pred, batch)
        target_raw = batch["height_raw"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        fringe = batch["fringe"].to(device, non_blocking=True)
        for j in range(pred.shape[0]):
            sample_idx = len(rows)
            metrics = compute_metrics(pred_mm[j:j + 1], target_raw[j:j + 1], mask=mask[j:j + 1])
            rows.append({"sample": sample_idx, **metrics})
            if save_images and out_dir is not None and sample_idx < 8:
                save_comparison(
                    fringe[j:j + 1],
                    target_raw[j:j + 1],
                    pred_mm[j:j + 1],
                    out_dir / "samples" / f"sample_{sample_idx:02d}.png",
                    title=f"{mode} phase2depth RMSE {metrics['rmse']:.2f}mm",
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
    parser.add_argument("--base_cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_phase_cache_960")
    parser.add_argument("--phase_pred_prefix", default=None)
    parser.add_argument("--save_dir", default="/root/autodl-tmp/diffusion_fpp_v5/results/fpp960_phase2depth_unet")
    parser.add_argument(
        "--input_mode",
        choices=["gt_phase", "phase_pred", "gt_phase_plus_fringe", "phase_pred_plus_fringe"],
        default="gt_phase",
    )
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--train_crop_h", type=int, default=0)
    parser.add_argument("--train_crop_w", type=int, default=0)
    parser.add_argument("--train_epoch_repeats", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval_metrics_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--eval_initial", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
        train_crop_h=args.train_crop_h,
        train_crop_w=args.train_crop_w,
        train_epoch_repeats=args.train_epoch_repeats,
        phase_pred_prefix=args.phase_pred_prefix,
        require_cache=True,
    )
    if args.input_mode.startswith("phase_pred") and int(loaders.get("phase_pred_channels", 0)) < 3:
        raise ValueError("phase_pred input requires --phase_pred_prefix with at least 3 channels")

    save_dir = Path(args.save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    model = OfficialUNetFPP(
        in_channels=input_channels(args.input_mode),
        out_channels=1,
        dropout_rate=args.dropout,
    ).to(device)
    load_info = None
    if args.init_checkpoint:
        load_info = load_expanded_state(model, args.init_checkpoint)

    criterion = HybridL1Loss(alpha=args.alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))

    print(f"Device: {device}")
    print(f"Input mode: {args.input_mode} | channels={input_channels(args.input_mode)}")
    print(f"Phase pred prefix: {args.phase_pred_prefix} | channels={loaders.get('phase_pred_channels', 0)}")
    print(f"Init load: {load_info}")
    print(f"Train {len(loaders['train'].dataset)} | Val {len(loaders['val'].dataset)} | Test {len(loaders['test'].dataset)}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    history = []
    best_val_loss = float("inf")
    best_val_rmse = float("inf")
    if args.eval_initial:
        val_loss = evaluate_loss(model, loaders["val"], criterion, device, args.input_mode)
        val_rows = evaluate_metrics(model, loaders["val"], device, args.input_mode)
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
        torch.save(checkpoint_state(0, model, optimizer, scheduler, scaler, args, best_val_loss, best_val_rmse, history), save_dir / "checkpoints" / "best_rmse.pt")
        torch.save(checkpoint_state(0, model, optimizer, scheduler, scaler, args, best_val_loss, best_val_rmse, history), save_dir / "checkpoints" / "best.pt")
        print(json.dumps(log, ensure_ascii=False))

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"phase2depth {ep}/{args.epochs}"):
            x = make_input(batch, device, args.input_mode)
            target = batch["height_01"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                loss = criterion(model(x), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item())
            seen += 1
            if args.max_train_batches and seen >= args.max_train_batches:
                break
        train_loss = total / max(1, seen)
        val_loss = evaluate_loss(model, loaders["val"], criterion, device, args.input_mode)
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
            val_rows = evaluate_metrics(model, loaders["val"], device, args.input_mode)
            val_summary = summarize(val_rows)
            log.update({f"val_{k}": val_summary[k]["mean"] for k in METRIC_KEYS})
            if val_summary["rmse"]["mean"] < best_val_rmse:
                best_val_rmse = val_summary["rmse"]["mean"]
                improved_rmse = True
        history.append(log)
        if improved_rmse:
            torch.save(checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best_val_loss, best_val_rmse, history), save_dir / "checkpoints" / "best_rmse.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best_val_loss, best_val_rmse, history), save_dir / "checkpoints" / "best.pt")
        print(json.dumps(log, ensure_ascii=False))
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        if args.save_every > 0 and (ep == 1 or ep == args.epochs or ep % args.save_every == 0):
            torch.save(checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best_val_loss, best_val_rmse, history), save_dir / "checkpoints" / "latest.pt")

    best_path = save_dir / "checkpoints" / "best_rmse.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        rows = evaluate_metrics(model, loaders["test"], device, args.input_mode, out_dir=save_dir / "evaluation", save_images=True)
        summary = write_eval_outputs(rows, save_dir / "evaluation", best_path, args)
        print("Final test:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
