"""C4/E6 adapter UNet with diffusion phase also injected into the main input.

The earlier adapter-only route showed a hard ceiling: even GT phase could not
help much when the C4 backbone was frozen and phase entered only as feature
biases.  This script keeps the stable adapter checkpoint initialization, but
expands the UNet backbone first convolution to receive restored phase channels
directly beside the raw fringe.  Newly added input weights are initialized to
zero, so the model starts exactly from the checkpoint behavior.
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
from models.official_unet import OfficialUNetFPP, ZeroCondAdapter
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


class OfficialUNetFPPDirectAdapter(torch.nn.Module):
    def __init__(self, cond_channels, direct_channels=3, out_channels=1, dropout_rate=0.0, adapter_hidden=32):
        super().__init__()
        self.direct_channels = int(direct_channels)
        self.backbone = OfficialUNetFPP(
            in_channels=1 + self.direct_channels,
            out_channels=out_channels,
            dropout_rate=dropout_rate,
        )
        self.adapter1 = ZeroCondAdapter(cond_channels, 64, adapter_hidden)
        self.adapter2 = ZeroCondAdapter(cond_channels, 128, adapter_hidden)
        self.adapter3 = ZeroCondAdapter(cond_channels, 256, adapter_hidden)
        self.adapter4 = ZeroCondAdapter(cond_channels, 512, adapter_hidden)
        self.adapter_mid = ZeroCondAdapter(cond_channels, 1024, adapter_hidden)

    def forward(self, fringe, direct_phase, cond):
        x0 = torch.cat([fringe, direct_phase], dim=1)
        skip1 = self.backbone.down1.conv(x0)
        skip1 = skip1 + self.adapter1(cond, skip1.shape[-2:])
        x = self.backbone.down1.pool(skip1)

        skip2 = self.backbone.down2.conv(x)
        skip2 = skip2 + self.adapter2(cond, skip2.shape[-2:])
        x = self.backbone.down2.pool(skip2)

        skip3 = self.backbone.down3.conv(x)
        skip3 = skip3 + self.adapter3(cond, skip3.shape[-2:])
        x = self.backbone.down3.pool(skip3)

        skip4 = self.backbone.down4.conv(x)
        skip4 = skip4 + self.adapter4(cond, skip4.shape[-2:])
        x = self.backbone.down4.pool(skip4)

        x = self.backbone.dropout(self.backbone.bottleneck(x))
        x = x + self.adapter_mid(cond, x.shape[-2:])
        x = self.backbone.up1(x, skip4)
        x = self.backbone.up2(x, skip3)
        x = self.backbone.up3(x, skip2)
        x = self.backbone.up4(x, skip1)
        return self.backbone.out(x)


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
    phase_pred = batch["phase_pred"].to(device, non_blocking=True)
    return torch.cat([cond, phase_pred], dim=1)


def select_direct_phase(batch, device, direct_channels):
    phase_pred = batch["phase_pred"].to(device, non_blocking=True)
    return phase_pred[:, :direct_channels]


def first_backbone_conv_key():
    return "backbone.down1.conv.conv.0.weight"


def load_expanded_state(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    source = ckpt.get("model_state_dict", ckpt)
    target = model.state_dict()
    loaded = {}
    expanded = {}
    skipped = []
    first_key = first_backbone_conv_key()
    for key, value in source.items():
        if key not in target:
            skipped.append(key)
            continue
        dst = target[key]
        if tuple(dst.shape) == tuple(value.shape):
            loaded[key] = value
            continue
        if key == first_key and value.ndim == 4 and dst.ndim == 4:
            if value.shape[0] == dst.shape[0] and value.shape[2:] == dst.shape[2:] and value.shape[1] <= dst.shape[1]:
                new_value = dst.clone()
                new_value.zero_()
                new_value[:, : value.shape[1]] = value
                expanded[key] = new_value
                continue
        if key.endswith("net.0.weight") and value.ndim == 4 and dst.ndim == 4:
            if value.shape[0] == dst.shape[0] and value.shape[2:] == dst.shape[2:] and value.shape[1] <= dst.shape[1]:
                new_value = dst.clone()
                new_value.zero_()
                new_value[:, : value.shape[1]] = value
                expanded[key] = new_value
                continue
        skipped.append(key)
    target.update(loaded)
    target.update(expanded)
    model.load_state_dict(target)
    return {"loaded": len(loaded), "expanded": len(expanded), "skipped_examples": skipped[:20], "source_args": ckpt.get("args", {})}


def freeze_backbone_except_direct_input(model):
    for param in model.backbone.parameters():
        param.requires_grad_(False)
    conv0 = model.backbone.down1.conv.conv[0]
    conv0.weight.requires_grad_(True)
    if conv0.bias is not None:
        conv0.bias.requires_grad_(True)


def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad_(False)


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
def evaluate_loss(model, loader, criterion, device, physics_channels, direct_channels):
    model.eval()
    total = 0.0
    seen = 0
    for batch in tqdm(loader, desc="val loss", leave=False):
        fringe = batch["fringe"].to(device, non_blocking=True)
        cond = select_cond(batch, device, physics_channels)
        direct = select_direct_phase(batch, device, direct_channels)
        target = batch["height_01"].to(device, non_blocking=True)
        total += float(criterion(model(fringe, direct, cond), target).item())
        seen += 1
    return total / max(1, seen)


@torch.no_grad()
def evaluate_metrics(model, loader, device, physics_channels, direct_channels, out_dir=None, save_images=False):
    model.eval()
    rows = []
    if save_images and out_dir is not None:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc="eval metrics"):
        fringe = batch["fringe"].to(device, non_blocking=True)
        cond = select_cond(batch, device, physics_channels)
        direct = select_direct_phase(batch, device, direct_channels)
        pred = model(fringe, direct, cond)
        pred_mm = prediction_to_mm(pred, batch)
        target_raw = batch["height_raw"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
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
                    title=f"direct phase adapter RMSE {metrics['rmse']:.2f}mm",
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
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_phase_cache_960")
    parser.add_argument("--phase_pred_prefix", default="phase_pred_e5_sincos_uph_ddim10")
    parser.add_argument("--save_dir", default="/root/autodl-tmp/diffusion_fpp_v5/results/fpp960_direct_phase_adapter")
    parser.add_argument("--init_checkpoint", required=True)
    parser.add_argument("--physics_channels", default="1,2,3,4,5,6,9,10")
    parser.add_argument("--include_ftp", action="store_true")
    parser.add_argument("--direct_channels", type=int, default=3)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--freeze_backbone_except_direct_input", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--adapter_hidden", type=int, default=32)
    parser.add_argument("--eval_metrics_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--eval_initial", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.physics_channel_indices = parse_channel_spec(args.physics_channels, args.include_ftp)
    args.physics_channel_names = channel_names(args.physics_channel_indices)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    loaders = create_fpp_ml_bench_loaders(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        include_ftp=args.include_ftp,
        image_h=args.image_size,
        image_w=args.image_size,
        require_cache=True,
        phase_cache_dir=args.phase_cache_dir,
        phase_pred_prefix=args.phase_pred_prefix,
    )
    phase_pred_channels = int(loaders.get("phase_pred_channels", 0))
    if phase_pred_channels <= 0:
        raise ValueError("phase_pred_prefix did not load phase prediction channels")
    args.direct_channels = min(int(args.direct_channels), phase_pred_channels)
    cond_channels = len(args.physics_channel_indices) + phase_pred_channels

    save_dir = Path(args.save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    model = OfficialUNetFPPDirectAdapter(
        cond_channels=cond_channels,
        direct_channels=args.direct_channels,
        out_channels=1,
        dropout_rate=args.dropout,
        adapter_hidden=args.adapter_hidden,
    ).to(device)
    load_info = load_expanded_state(model, args.init_checkpoint, device)
    if args.freeze_backbone:
        freeze_backbone(model)
    if args.freeze_backbone_except_direct_input:
        freeze_backbone_except_direct_input(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    criterion = HybridL1Loss(alpha=args.alpha)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))

    print(f"Device: {device}")
    print(f"Physics channels: {args.physics_channel_indices} | {args.physics_channel_names}")
    print(f"Phase pred channels: {phase_pred_channels} | direct={args.direct_channels}")
    print(f"Init load: {load_info}")
    print(f"Train {len(loaders['train'].dataset)} | Val {len(loaders['val'].dataset)} | Test {len(loaders['test'].dataset)}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | trainable {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    history = []
    best_val_loss = float("inf")
    best_val_rmse = float("inf")
    if args.eval_initial:
        val_loss = evaluate_loss(model, loaders["val"], criterion, device, args.physics_channel_indices, args.direct_channels)
        val_rows = evaluate_metrics(model, loaders["val"], device, args.physics_channel_indices, args.direct_channels)
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
        for batch in tqdm(loaders["train"], desc=f"direct phase adapter {ep}/{args.epochs}"):
            fringe = batch["fringe"].to(device, non_blocking=True)
            cond = select_cond(batch, device, args.physics_channel_indices)
            direct = select_direct_phase(batch, device, args.direct_channels)
            target = batch["height_01"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                loss = criterion(model(fringe, direct, cond), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item())
            seen += 1
            if args.max_train_batches and seen >= args.max_train_batches:
                break
        train_loss = total / max(1, seen)
        val_loss = evaluate_loss(model, loaders["val"], criterion, device, args.physics_channel_indices, args.direct_channels)
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
            val_rows = evaluate_metrics(model, loaders["val"], device, args.physics_channel_indices, args.direct_channels)
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
        rows = evaluate_metrics(
            model,
            loaders["test"],
            device,
            args.physics_channel_indices,
            args.direct_channels,
            out_dir=save_dir / "evaluation",
            save_images=True,
        )
        summary = write_eval_outputs(rows, save_dir / "evaluation", best_path, args)
        print("Final test:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
