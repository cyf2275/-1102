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

from data.dataset_fpp_phase import create_fpp_phase_loaders
from models.single_frame_baselines import build_single_frame_baseline
from train_fpp_official_style_unet import METRIC_KEYS, summarize
from utils.metrics import compute_metrics
from utils.visualization import save_comparison


PHASE_XY_TO_DEPTH_COEF = (
    26294.245854143974,
    -702.0035381190783,
    -0.7098495222018522,
    46809.68618991401,
    4.9209790510550935,
    0.016255433663514382,
    -656.3297862483804,
    0.6346697283961983,
    -2.076325866036234,
    21882.703414268024,
)


def save_rows(rows, path):
    keys = ["sample"] + METRIC_KEYS + ["phase_rmse", "phase_mae"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def masked_mean(value, mask=None, eps=1e-6):
    if mask is None:
        return value.mean()
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.to(dtype=value.dtype, device=value.device)
    return (value * mask).sum() / mask.sum().clamp_min(eps)


def masked_hybrid_loss(pred, target, mask=None, alpha=0.7, eps=1e-6):
    err = pred - target
    l1 = masked_mean(torch.abs(err), mask, eps=eps)
    rmse = torch.sqrt(masked_mean(err * err, mask, eps=eps) + eps)
    return float(alpha) * l1 + (1.0 - float(alpha)) * rmse


def wrapped01_from_sincos(sin_map, cos_map):
    wrapped = torch.atan2(sin_map.float(), cos_map.float())
    return ((wrapped + np.pi) / (2.0 * np.pi)).to(dtype=sin_map.dtype)


def phase01_to_abs(phase01, phase_minmax):
    lo = phase_minmax[:, 0].view(-1, 1, 1, 1).to(device=phase01.device, dtype=phase01.dtype)
    hi = phase_minmax[:, 1].view(-1, 1, 1, 1).to(device=phase01.device, dtype=phase01.dtype)
    return phase01 * (hi - lo).clamp_min(1e-6) + lo


def abs_phase_to_global01(phase_abs, global_min, global_max):
    return ((phase_abs - float(global_min)) / max(float(global_max) - float(global_min), 1e-6)).clamp(0.0, 1.0)


def global01_to_abs_phase(phase_global01, global_min, global_max):
    return phase_global01.clamp(0.0, 1.0) * (float(global_max) - float(global_min)) + float(global_min)


def phase_xy_to_depth_mm(phase_abs, cond):
    x = cond[:, 11:12].to(device=phase_abs.device, dtype=phase_abs.dtype)
    y = cond[:, 12:13].to(device=phase_abs.device, dtype=phase_abs.dtype)
    c = torch.as_tensor(PHASE_XY_TO_DEPTH_COEF, device=phase_abs.device, dtype=phase_abs.dtype)
    return (
        c[0]
        + c[1] * phase_abs
        + c[2] * x
        + c[3] * y
        + c[4] * phase_abs * phase_abs
        + c[5] * phase_abs * x
        + c[6] * phase_abs * y
        + c[7] * x * x
        + c[8] * x * y
        + c[9] * y * y
    )


def make_model_input(batch, args, device):
    fringe = batch["fringe"].to(device, non_blocking=True)
    if args.input_mode == "fringe":
        return fringe
    if args.input_mode == "fringe_xy":
        cond = batch["cond"].to(device, non_blocking=True)
        return torch.cat([fringe, cond[:, 11:13]], dim=1)
    raise ValueError(f"unknown input_mode: {args.input_mode}")


def phase_metrics(pred_phase, target_phase, mask, eps=1e-6):
    err = pred_phase - target_phase
    mae = masked_mean(torch.abs(err), mask, eps=eps)
    rmse = torch.sqrt(masked_mean(err * err, mask, eps=eps) + eps)
    return float(rmse.detach().item()), float(mae.detach().item())


def mps_phase_loss(output, batch, args):
    phase = batch["phase_target"]
    mask = batch.get("mask")
    gt_sin = phase[:, 0:1]
    gt_cos = phase[:, 1:2]
    gt_wrapped01 = wrapped01_from_sincos(gt_sin, gt_cos)
    gt_unwrapped_abs = phase01_to_abs(phase[:, 2:3], batch["phase_minmax"])
    gt_unwrapped_global01 = abs_phase_to_global01(gt_unwrapped_abs, args.global_phase_min, args.global_phase_max)

    fenzi_loss = masked_hybrid_loss(output["fenzi"], gt_sin, mask, alpha=args.alpha)
    fenmu_loss = masked_hybrid_loss(output["fenmu"], gt_cos, mask, alpha=args.alpha)
    wrapped_loss = masked_hybrid_loss(output["wrapped"], gt_wrapped01, mask, alpha=args.alpha)
    unwrapped_loss = masked_hybrid_loss(output["unwrapped"], gt_unwrapped_global01, mask, alpha=args.alpha)
    loss = (
        args.w1 * fenzi_loss
        + args.w2 * fenmu_loss
        + args.w3 * wrapped_loss
        + args.w4 * unwrapped_loss
    )
    return loss, {
        "fenzi_loss": float(fenzi_loss.detach().item()),
        "fenmu_loss": float(fenmu_loss.detach().item()),
        "wrapped_loss": float(wrapped_loss.detach().item()),
        "unwrapped_loss": float(unwrapped_loss.detach().item()),
    }


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


def summarize_phase(rows):
    out = {}
    for key in ("phase_rmse", "phase_mae"):
        vals = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            out[key] = {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std(ddof=0)),
            }
    return out


@torch.no_grad()
def evaluate_loss(model, loader, device, args):
    model.eval()
    total = 0.0
    parts_total = {"fenzi_loss": 0.0, "fenmu_loss": 0.0, "wrapped_loss": 0.0, "unwrapped_loss": 0.0}
    seen = 0
    for batch in tqdm(loader, desc="val loss", leave=False):
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        output = model(make_model_input(batch, args, device))
        loss, parts = mps_phase_loss(output, batch, args)
        total += float(loss.item())
        for key in parts_total:
            parts_total[key] += float(parts[key])
        seen += 1
    out = {k: v / max(1, seen) for k, v in parts_total.items()}
    return total / max(1, seen), out


@torch.no_grad()
def evaluate_metrics(model, loader, device, args, out_dir=None, save_images=False):
    model.eval()
    rows = []
    if save_images and out_dir is not None:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc="eval metrics"):
        fringe = batch["fringe"].to(device, non_blocking=True)
        cond = batch["cond"].to(device, non_blocking=True)
        output = model(make_model_input(batch, args, device))
        pred_phase = global01_to_abs_phase(output["unwrapped"], args.global_phase_min, args.global_phase_max)
        pred_mm = phase_xy_to_depth_mm(pred_phase, cond)
        target = batch["height_raw"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        target_phase = phase01_to_abs(batch["phase_target"][:, 2:3].to(device, non_blocking=True), batch["phase_minmax"].to(device, non_blocking=True))
        for j in range(pred_mm.shape[0]):
            sample_idx = len(rows)
            metrics = compute_metrics(pred_mm[j:j + 1], target[j:j + 1], mask=mask[j:j + 1])
            ph_rmse, ph_mae = phase_metrics(pred_phase[j:j + 1], target_phase[j:j + 1], mask[j:j + 1])
            rows.append({"sample": sample_idx, **metrics, "phase_rmse": ph_rmse, "phase_mae": ph_mae})
            if save_images and out_dir is not None and sample_idx < 8:
                save_comparison(
                    fringe[j:j + 1],
                    target[j:j + 1],
                    pred_mm[j:j + 1],
                    out_dir / "samples" / f"sample_{sample_idx:02d}.png",
                    title=f"MPS phase-proxy RMSE {metrics['rmse']:.2f}mm",
                    mask=mask[j:j + 1],
                )
    return rows


def write_eval_outputs(rows, out_dir, checkpoint, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rows(rows, out_dir / "per_sample_metrics.csv")
    summary = summarize(rows)
    summary.update(summarize_phase(rows))
    summary["n"] = len(rows)
    summary["checkpoint"] = str(checkpoint)
    summary["args"] = vars(args)
    summary["method_note"] = (
        "MPS_XNet-style phase-first baseline. The network predicts numerator, "
        "denominator, wrapped phase, and globally-normalized unwrapped phase "
        "from a single fringe. Depth is evaluated through a fixed train-split "
        "phase+xy->depth proxy; no teacher phase/minmax is used as test-time input."
    )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def get_train_global_phase_minmax(loaders):
    ds = loaders["train"].dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    phase_minmax = np.asarray(ds.phase_minmax, dtype=np.float32)
    return float(np.min(phase_minmax[:, 0])), float(np.max(phase_minmax[:, 1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_phase_cache_960")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--base_channels", type=int, default=8)
    parser.add_argument("--input_mode", choices=["fringe", "fringe_xy"], default="fringe")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument("--w3", type=float, default=1.0)
    parser.add_argument("--w4", type=float, default=50.0)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_metrics_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--require_cache", action="store_true")
    parser.add_argument("--preload_ram", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=251)
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

    loaders = create_fpp_phase_loaders(
        base_cache_dir=args.base_cache_dir,
        phase_cache_dir=args.phase_cache_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        image_h=args.image_size,
        image_w=args.image_size,
        require_cache=args.require_cache,
        preload_ram=args.preload_ram,
        train_minimal=True,
        train_extra_keys={"mask", "phase_minmax"},
    )
    args.global_phase_min, args.global_phase_max = get_train_global_phase_minmax(loaders)

    save_dir = Path(args.save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    model = build_single_frame_baseline(
        "mps_xnet_phase",
        in_channels=3 if args.input_mode == "fringe_xy" else 1,
        out_channels=1,
        base_channels=args.base_channels,
        dropout_rate=args.dropout,
    ).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10, min_lr=1e-6
    )
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))

    print(f"Device: {device}")
    print(
        f"Arch: MPS-XNet phase-first proxy | input={args.input_mode} | "
        f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )
    print(f"Global train phase range: [{args.global_phase_min:.6f}, {args.global_phase_max:.6f}] rad")
    print(f"Train {len(loaders['train'].dataset)} | Val {len(loaders['val'].dataset)} | Test {len(loaders['test'].dataset)}")

    history = []
    best_val_loss = float("inf")
    best_val_rmse = float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        totals = {"loss": 0.0, "fenzi_loss": 0.0, "fenmu_loss": 0.0, "wrapped_loss": 0.0, "unwrapped_loss": 0.0}
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"mps_xnet_phase {ep}/{args.epochs}"):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                output = model(make_model_input(batch, args, device))
                loss, parts = mps_phase_loss(output, batch, args)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.item())
            for key in ("fenzi_loss", "fenmu_loss", "wrapped_loss", "unwrapped_loss"):
                totals[key] += parts[key]
            seen += 1
            if args.max_train_batches and seen >= args.max_train_batches:
                break
        train = {k: v / max(1, seen) for k, v in totals.items()}

        val_loss = None
        val_parts = {}
        if ep == 1 or ep == args.epochs or ep % max(1, args.eval_every) == 0:
            val_loss, val_parts = evaluate_loss(model, loaders["val"], device, args)
            scheduler.step(val_loss)

        log = {
            "epoch": ep,
            **{f"train_{k}": v for k, v in train.items()},
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_parts.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        }
        improved_rmse = False
        if ep == 1 or ep == args.epochs or ep % max(1, args.eval_metrics_every) == 0:
            val_rows = evaluate_metrics(model, loaders["val"], device, args)
            val_summary = summarize(val_rows)
            phase_summary = summarize_phase(val_rows)
            log.update({f"val_{k}": val_summary[k]["mean"] for k in METRIC_KEYS})
            log.update({f"val_{k}": v["mean"] for k, v in phase_summary.items()})
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

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
