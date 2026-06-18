"""Official-style FPP-ML-Bench UNet with optional physics-instruction input.

This script keeps the public FPP-ML-Bench UNet/loss/optimizer protocol close,
while allowing the input to be swapped from the single A0 fringe to the cached
PIP physics instruction tensor. It is intended for the B1 control experiment.
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
from models import OfficialUNetFPP
from physics_features_pip import FEATURE_ORDER
from utils.metrics import compute_metrics
from utils.visualization import save_comparison


METRIC_KEYS = ["rmse", "mae", "edge_rmse", "normal_deg", "ssim"]


def parse_channel_spec(spec, include_ftp):
    if spec is None:
        return None
    text = str(spec).strip().lower()
    if text in ("", "default", "none"):
        return None
    max_channel = 10 if include_ftp else 8
    if text == "all":
        return list(range(max_channel + 1))

    selected = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_text, hi_text = part.split("-", 1)
            lo, hi = int(lo_text), int(hi_text)
            if hi < lo:
                raise ValueError(f"invalid channel range: {part}")
            selected.extend(range(lo, hi + 1))
        else:
            selected.append(int(part))

    deduped = []
    for idx in selected:
        if idx not in deduped:
            deduped.append(idx)
    invalid = [idx for idx in deduped if idx < 0 or idx > max_channel]
    if invalid:
        raise ValueError(
            f"invalid physics channel(s) {invalid}; max channel is {max_channel}. "
            "Use --include_ftp before selecting channels 9 or 10."
        )
    return deduped


def channel_names(indices):
    if indices is None:
        return None
    return [FEATURE_ORDER[idx] for idx in indices]


class HybridL1Loss(torch.nn.Module):
    """Official HybridL1: alpha * masked L1 + (1-alpha) * global L1."""

    def __init__(self, alpha=0.7, eps=1e-8):
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, pred, target):
        mask = (target > 0).to(dtype=pred.dtype)
        masked_l1 = (torch.abs(pred - target) * mask).sum() / mask.sum().clamp(min=self.eps)
        global_l1 = torch.abs(pred - target).mean()
        return self.alpha * masked_l1 + (1.0 - self.alpha) * global_l1


def make_input(batch, mode, device, physics_channels=None):
    fringe = batch["fringe"].to(device, non_blocking=True)
    if mode == "fringe":
        return fringe
    cond = batch["cond"].to(device, non_blocking=True)
    if mode == "physics":
        if physics_channels is not None:
            return cond[:, physics_channels]
        return cond
    if mode == "fringe_plus_physics":
        if physics_channels is not None:
            selected = [idx for idx in physics_channels if idx != 0]
            if selected:
                return torch.cat([fringe, cond[:, selected]], dim=1)
            return fringe
        # cond channel 0 is the raw fringe, so remove it to avoid exact duplication.
        return torch.cat([fringe, cond[:, 1:]], dim=1)
    raise ValueError(f"unknown input mode: {mode}")


def prediction_to_mm(pred_norm, batch):
    pred_01 = torch.clamp(pred_norm, 0.0, 1.0)
    minmax = batch["depth_minmax"].to(pred_norm.device, non_blocking=True)
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


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, input_mode, physics_channels=None):
    model.eval()
    total = 0.0
    seen = 0
    for batch in tqdm(loader, desc="val loss", leave=False):
        x = make_input(batch, input_mode, device, physics_channels)
        target = batch["height_01"].to(device, non_blocking=True)
        pred = model(x)
        loss = criterion(pred, target)
        total += float(loss.item())
        seen += 1
    return total / max(1, seen)


@torch.no_grad()
def evaluate_metrics(model, loader, device, input_mode, physics_channels=None, out_dir=None, save_images=False):
    model.eval()
    rows = []
    if save_images and out_dir is not None:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc="eval metrics"):
        x = make_input(batch, input_mode, device, physics_channels)
        pred = model(x)
        pred_mm = prediction_to_mm(pred, batch)
        target_raw = batch["height_raw"].to(device, non_blocking=True)
        fringe = batch["fringe"].to(device, non_blocking=True)
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
                    title=f"{input_mode} UNet RMSE {metrics['rmse']:.2f}mm",
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


def input_channels(input_mode, include_ftp, physics_channels=None):
    cond_channels = 11 if include_ftp else 9
    if input_mode == "fringe":
        return 1
    if input_mode == "physics":
        if physics_channels is not None:
            return len(physics_channels)
        return cond_channels
    if input_mode == "fringe_plus_physics":
        if physics_channels is not None:
            return 1 + len([idx for idx in physics_channels if idx != 0])
        return cond_channels
    raise ValueError(f"unknown input mode: {input_mode}")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--save_dir", default="/root/autodl-tmp/diffusion_fpp_v5/results/fpp960_b1_physics_unet")
    parser.add_argument("--input_mode", choices=["fringe", "physics", "fringe_plus_physics"], default="physics")
    parser.add_argument("--include_ftp", action="store_true")
    parser.add_argument(
        "--physics_channels",
        default="",
        help="Comma/range channel spec in PIP feature order, e.g. 0,1,2,3,4 or 0-6. Empty means default.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval_metrics_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", default="")
    parser.add_argument("--require_cache", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--final_checkpoint", choices=["best_rmse", "best_loss", "latest"], default="best_rmse")
    args = parser.parse_args()
    args.physics_channel_indices = parse_channel_spec(args.physics_channels, args.include_ftp)
    args.physics_channel_names = channel_names(args.physics_channel_indices)

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
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
    in_ch = input_channels(args.input_mode, args.include_ftp, args.physics_channel_indices)
    model = OfficialUNetFPP(in_channels=in_ch, out_channels=1, dropout_rate=args.dropout).to(device)
    criterion = HybridL1Loss(alpha=args.alpha)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-5)
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

    print(f"Device: {device}")
    print(f"Input mode: {args.input_mode} | channels={in_ch}")
    if args.physics_channel_indices is not None:
        print(f"Physics channels: {args.physics_channel_indices} | {args.physics_channel_names}")
    print(f"Train {len(loaders['train'].dataset)} | Val {len(loaders['val'].dataset)} | Test {len(loaders['test'].dataset)}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    for ep in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"{args.input_mode} UNet {ep}/{args.epochs}"):
            x = make_input(batch, args.input_mode, device, args.physics_channel_indices)
            target = batch["height_01"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                pred = model(x)
                loss = criterion(pred, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item())
            seen += 1
            if args.max_train_batches and seen >= args.max_train_batches:
                break

        train_loss = total / max(1, seen)
        val_loss = evaluate_loss(model, loaders["val"], criterion, device, args.input_mode, args.physics_channel_indices)
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
            val_rows = evaluate_metrics(model, loaders["val"], device, args.input_mode, args.physics_channel_indices)
            val_summary = summarize(val_rows)
            log.update({f"val_{k}": val_summary[k]["mean"] for k in METRIC_KEYS})
            if val_summary["rmse"]["mean"] < best_val_rmse:
                best_val_rmse = val_summary["rmse"]["mean"]
                improved_rmse = True
                first = next(iter(loaders["val"]))
                pred = model(make_input(first, args.input_mode, device, args.physics_channel_indices))
                pred_mm = prediction_to_mm(pred, first)
                first_mask = first.get("mask")
                if first_mask is not None:
                    first_mask = first_mask.to(device, non_blocking=True)
                save_comparison(
                    first["fringe"].to(device),
                    first["height_raw"].to(device),
                    pred_mm,
                    save_dir / "visualizations" / f"val_ep{ep:03d}.png",
                    title=f"{args.input_mode} val RMSE {val_summary['rmse']['mean']:.2f}mm",
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
        test_rows = evaluate_metrics(model, loaders["test"], device, args.input_mode, args.physics_channel_indices,
                                     out_dir=save_dir / "evaluation", save_images=True)
        summary = write_eval_outputs(test_rows, save_dir / "evaluation", best_path, args)
        print("Final test:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
