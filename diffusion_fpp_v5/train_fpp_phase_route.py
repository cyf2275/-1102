"""Deterministic phase-route baseline for FPP-ML-Bench.

The model keeps the official UNet raw fringe backbone, injects corrected
phase instructions through zero adapters, and predicts depth plus supervised
phase targets. This is E1 in the reset plan.
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

from data.dataset_fpp_phase import create_fpp_phase_loaders
from models import OfficialUNetFPPAdapter
from train_fpp_official_style_unet import METRIC_KEYS, HybridL1Loss, prediction_to_mm, summarize
from utils.metrics import compute_metrics
from utils.visualization import save_comparison


PHASE_METRIC_KEYS = ["phase_mae_rad", "phase_rmse_rad", "uph_mae"]


def parse_channel_spec(spec: str | None, max_channel: int):
    if spec is None:
        return None
    text = str(spec).strip().lower()
    if text in ("", "default", "none"):
        return None
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
            selected.extend(range(lo, hi + 1))
        else:
            selected.append(int(part))
    deduped = []
    for idx in selected:
        if idx not in deduped:
            deduped.append(idx)
    invalid = [idx for idx in deduped if idx < 0 or idx > max_channel]
    if invalid:
        raise ValueError(f"invalid phase instruction channels: {invalid}; max={max_channel}")
    return deduped


def select_cond(batch, device, phase_channels):
    cond = batch["cond"].to(device, non_blocking=True)
    if phase_channels is not None:
        cond = cond[:, phase_channels]
    return cond


def normalize_sincos(x):
    sin = x[:, 0:1]
    cos = x[:, 1:2]
    norm = torch.sqrt(sin * sin + cos * cos).clamp(min=1e-6)
    return torch.cat([sin / norm, cos / norm], dim=1)


def phase_metrics(pred_phase_raw, target_phase, mask=None):
    pred_sc = normalize_sincos(pred_phase_raw[:, 0:2])
    target_sc = target_phase[:, 0:2]
    pred_angle = torch.atan2(pred_sc[:, 0:1], pred_sc[:, 1:2])
    target_angle = torch.atan2(target_sc[:, 0:1], target_sc[:, 1:2])
    diff = torch.atan2(torch.sin(pred_angle - target_angle), torch.cos(pred_angle - target_angle))
    uph_diff = torch.abs(torch.clamp(pred_phase_raw[:, 2:3], 0.0, 1.0) - target_phase[:, 2:3])
    if mask is not None:
        valid = mask.to(device=diff.device, dtype=torch.bool)
        if valid.any():
            diff_v = diff[valid]
            uph_v = uph_diff[valid]
        else:
            diff_v = diff.flatten()
            uph_v = uph_diff.flatten()
    else:
        diff_v = diff.flatten()
        uph_v = uph_diff.flatten()
    return {
        "phase_mae_rad": float(torch.mean(torch.abs(diff_v)).item()),
        "phase_rmse_rad": float(torch.sqrt(torch.mean(diff_v * diff_v)).item()),
        "uph_mae": float(torch.mean(uph_v).item()),
    }


class PhaseRouteLoss(torch.nn.Module):
    def __init__(self, depth_alpha=0.7, phase_weight=0.25, uph_weight=0.10, unit_weight=0.02):
        super().__init__()
        self.depth_loss = HybridL1Loss(alpha=depth_alpha)
        self.phase_weight = float(phase_weight)
        self.uph_weight = float(uph_weight)
        self.unit_weight = float(unit_weight)

    def forward(self, pred, batch):
        depth_pred = pred[:, 0:1]
        phase_pred = pred[:, 1:4]
        depth_target = batch["height_01"].to(pred.device, non_blocking=True)
        phase_target = batch["phase_target"].to(pred.device, non_blocking=True)
        mask = batch["mask"].to(pred.device, non_blocking=True)
        depth_loss = self.depth_loss(depth_pred, depth_target)

        pred_sc = normalize_sincos(phase_pred[:, 0:2])
        phase_err = torch.abs(pred_sc - phase_target[:, 0:2]).sum(dim=1, keepdim=True)
        uph_err = torch.abs(torch.clamp(phase_pred[:, 2:3], 0.0, 1.0) - phase_target[:, 2:3])
        unit = torch.sqrt((phase_pred[:, 0:1] ** 2 + phase_pred[:, 1:2] ** 2).clamp(min=1e-8))
        unit_err = torch.abs(unit - 1.0)
        denom = mask.sum().clamp(min=1.0)
        phase_loss = (phase_err * mask).sum() / denom
        uph_loss = (uph_err * mask).sum() / denom
        unit_loss = (unit_err * mask).sum() / denom
        return depth_loss + self.phase_weight * phase_loss + self.uph_weight * uph_loss + self.unit_weight * unit_loss


def load_matching_state(target_module, checkpoint_path: str, device, source_prefix="", target_prefix=""):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    target = target_module.state_dict()
    loaded = {}
    skipped = []
    partial = {}
    for key, value in state.items():
        src_key = key
        if source_prefix:
            if not src_key.startswith(source_prefix):
                continue
            src_key = src_key[len(source_prefix):]
        dst_key = f"{target_prefix}{src_key}"
        if dst_key in target and tuple(target[dst_key].shape) == tuple(value.shape):
            loaded[dst_key] = value
        elif dst_key in target and dst_key in {"out.weight", "out.bias"}:
            # Reuse the pretrained depth head for channel 0 when the new model
            # predicts depth plus phase channels.
            dst = target[dst_key].clone()
            if value.ndim == dst.ndim and value.shape[0] == 1 and dst.shape[0] > 1 and tuple(value.shape[1:]) == tuple(dst.shape[1:]):
                dst[:1] = value
                partial[dst_key] = dst
            else:
                skipped.append(key)
        else:
            skipped.append(key)
    target.update(loaded)
    target.update(partial)
    target_module.load_state_dict(target)
    return len(loaded) + len(partial), skipped[:20]


def freeze_backbone_features(model):
    for name, param in model.backbone.named_parameters():
        if not name.startswith("out."):
            param.requires_grad_(False)


def save_rows(rows, path, keys):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


@torch.no_grad()
def evaluate(model, loader, device, phase_channels, out_dir=None, save_images=False):
    model.eval()
    rows = []
    if save_images and out_dir is not None:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc="eval"):
        fringe = batch["fringe"].to(device, non_blocking=True)
        cond = select_cond(batch, device, phase_channels)
        pred = model(fringe, cond)
        pred_depth = pred[:, 0:1]
        pred_mm = prediction_to_mm(pred_depth, batch)
        target_raw = batch["height_raw"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        phase_target = batch["phase_target"].to(device, non_blocking=True)
        for j in range(pred.shape[0]):
            sample_idx = len(rows)
            depth_m = compute_metrics(pred_mm[j:j + 1], target_raw[j:j + 1], mask=mask[j:j + 1])
            phase_m = phase_metrics(pred[j:j + 1, 1:4], phase_target[j:j + 1], mask=mask[j:j + 1])
            rows.append({"sample": sample_idx, **depth_m, **phase_m})
            if save_images and out_dir is not None and sample_idx < 8:
                save_comparison(
                    fringe[j:j + 1],
                    target_raw[j:j + 1],
                    pred_mm[j:j + 1],
                    out_dir / "samples" / f"sample_{sample_idx:02d}.png",
                    title=f"phase-route RMSE {depth_m['rmse']:.2f}mm",
                    mask=mask[j:j + 1],
                )
    return rows


def summarize_extra(rows):
    summary = summarize(rows)
    for key in PHASE_METRIC_KEYS:
        vals = np.array([float(r[key]) for r in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
        }
    return summary


def checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best_val_rmse, history):
    return {
        "epoch": ep,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": vars(args),
        "best_val_rmse": best_val_rmse,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_phase_cache_960")
    parser.add_argument("--save_dir", default="/root/autodl-tmp/diffusion_fpp_v5/results/fpp960_e1_phase_route_mt")
    parser.add_argument("--base_checkpoint", default="/root/autodl-tmp/diffusion_fpp_v5/results/fpp960_a_fringe_unet_control/checkpoints/best.pt")
    parser.add_argument("--phase_channels", default="1-12")
    parser.add_argument("--freeze_backbone_features", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--depth_alpha", type=float, default=0.7)
    parser.add_argument("--phase_weight", type=float, default=0.25)
    parser.add_argument("--uph_weight", type=float, default=0.10)
    parser.add_argument("--unit_weight", type=float, default=0.02)
    parser.add_argument("--adapter_hidden", type=int, default=32)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--train_epoch_repeats", type=int, default=1)
    parser.add_argument("--train_crop", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
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
        train_crop_h=args.train_crop,
        train_crop_w=args.train_crop,
        train_epoch_repeats=args.train_epoch_repeats,
        require_cache=True,
    )
    max_channel = loaders["cond_channels"] - 1
    args.phase_channel_indices = parse_channel_spec(args.phase_channels, max_channel=max_channel)
    cond_channels = len(args.phase_channel_indices) if args.phase_channel_indices is not None else loaders["cond_channels"]

    save_dir = Path(args.save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    model = OfficialUNetFPPAdapter(
        cond_channels=cond_channels,
        out_channels=4,
        dropout_rate=0.0,
        adapter_hidden=args.adapter_hidden,
    ).to(device)
    if args.base_checkpoint:
        loaded, skipped = load_matching_state(model.backbone, args.base_checkpoint, device)
        print(f"Loaded {loaded} matching backbone tensors from {args.base_checkpoint}; skipped {len(skipped)} examples={skipped[:5]}")
    if args.freeze_backbone_features:
        freeze_backbone_features(model)

    criterion = PhaseRouteLoss(
        depth_alpha=args.depth_alpha,
        phase_weight=args.phase_weight,
        uph_weight=args.uph_weight,
        unit_weight=args.unit_weight,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.RMSprop(trainable, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10, min_lr=1e-6
    )
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))

    print(f"Device: {device}")
    print(f"Phase channels: {args.phase_channel_indices}")
    print(f"Train {len(loaders['train'].dataset)} | Val {len(loaders['val'].dataset)} | Test {len(loaders['test'].dataset)}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | trainable {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    history = []
    best_val_rmse = float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"phase-route {ep}/{args.epochs}"):
            fringe = batch["fringe"].to(device, non_blocking=True)
            cond = select_cond(batch, device, args.phase_channel_indices)
            batch_gpu = {
                "height_01": batch["height_01"].to(device, non_blocking=True),
                "phase_target": batch["phase_target"].to(device, non_blocking=True),
                "mask": batch["mask"].to(device, non_blocking=True),
            }
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                pred = model(fringe, cond)
                loss = criterion(pred, batch_gpu)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item())
            seen += 1
            if args.max_train_batches and seen >= args.max_train_batches:
                break

        train_loss = total / max(1, seen)
        log = {
            "epoch": ep,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        }
        if ep == 1 or ep % args.eval_every == 0:
            val_rows = evaluate(model, loaders["val"], device, args.phase_channel_indices)
            val_summary = summarize_extra(val_rows)
            val_rmse = val_summary["rmse"]["mean"]
            scheduler.step(val_rmse)
            log.update({f"val_{k}": val_summary[k]["mean"] for k in METRIC_KEYS + PHASE_METRIC_KEYS})
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                torch.save(
                    checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best_val_rmse, history),
                    save_dir / "checkpoints" / "best_rmse.pt",
                )
                first = next(iter(loaders["val"]))
                pred = model(
                    first["fringe"].to(device, non_blocking=True),
                    select_cond(first, device, args.phase_channel_indices),
                )
                save_comparison(
                    first["fringe"].to(device),
                    first["height_raw"].to(device),
                    prediction_to_mm(pred[:, 0:1], first),
                    save_dir / "visualizations" / f"val_ep{ep:03d}.png",
                    title=f"E1 val RMSE {val_rmse:.2f}mm",
                    mask=first["mask"].to(device),
                )
        else:
            scheduler.step(train_loss)

        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        if args.save_every > 0 and (ep == 1 or ep == args.epochs or ep % args.save_every == 0):
            torch.save(
                checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best_val_rmse, history),
                save_dir / "checkpoints" / "latest.pt",
            )

    best_path = save_dir / "checkpoints" / "best_rmse.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_rows = evaluate(model, loaders["test"], device, args.phase_channel_indices, out_dir=save_dir / "evaluation", save_images=True)
        keys = ["sample"] + METRIC_KEYS + PHASE_METRIC_KEYS
        save_rows(test_rows, save_dir / "evaluation" / "per_sample_metrics.csv", keys)
        summary = summarize_extra(test_rows)
        summary["n"] = len(test_rows)
        summary["checkpoint"] = str(best_path)
        summary["args"] = vars(args)
        with open(save_dir / "evaluation" / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("Final test:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
