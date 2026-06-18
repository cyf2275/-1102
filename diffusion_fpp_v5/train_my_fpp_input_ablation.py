"""Real-capture my_fpp input and teacher-phase validation.

This script intentionally separates legal single-frame inputs from teacher-only
diagnostics. Legal configs never feed phase_y/phase_x, bc_y/bc_x, or masks to
the model. Those arrays are used only as labels, weights, metrics, or QC.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset_my_fpp import (
    ALL_INPUT_MODES,
    LEGAL_INPUT_MODES,
    canonical_input_mode,
    create_my_fpp_loaders,
    is_legal_single_frame_mode,
    smoke_summary,
)
from models import ConditionalUNet
from utils.metrics import compute_metrics
from utils.visualization import save_comparison


METRIC_KEYS = ["rmse", "mae", "edge_rmse", "normal_deg", "ssim"]
ROI_PREFIXES = ["object", "valid"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mean(x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    if weight is None:
        return x.mean()
    weight = torch.clamp(weight.to(device=x.device, dtype=x.dtype), min=0.0)
    return (x * weight).sum() / weight.sum().clamp(min=1.0)


def charbonnier(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-3) -> torch.Tensor:
    return masked_mean(torch.sqrt((pred - target) ** 2 + eps * eps), weight=weight)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    return masked_mean((pred - target) ** 2, weight=weight)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    pdx = pred[..., :, 1:] - pred[..., :, :-1]
    tdx = target[..., :, 1:] - target[..., :, :-1]
    pdy = pred[..., 1:, :] - pred[..., :-1, :]
    tdy = target[..., 1:, :] - target[..., :-1, :]
    if weight is None:
        return torch.mean(torch.abs(pdx - tdx)) + torch.mean(torch.abs(pdy - tdy))
    wx = weight[..., :, 1:] * weight[..., :, :-1]
    wy = weight[..., 1:, :] * weight[..., :-1, :]
    return masked_mean(torch.abs(pdx - tdx), wx) + masked_mean(torch.abs(pdy - tdy), wy)


def train_weight(batch: Dict[str, object], device: torch.device, object_weight: float) -> torch.Tensor:
    valid = batch["valid_mask"].to(device, non_blocking=True).float()  # type: ignore[index]
    obj = batch["object_mask"].to(device, non_blocking=True).float()  # type: ignore[index]
    extra = max(float(object_weight) - 1.0, 0.0)
    return valid * (1.0 + extra * obj)


def prediction_to_height_mm(pred_norm: torch.Tensor, batch: Dict[str, object]) -> torch.Tensor:
    scale = batch["scale_mm"].to(pred_norm.device, non_blocking=True).view(-1, 1, 1, 1)  # type: ignore[index]
    return torch.clamp(pred_norm, -1.0, 1.0) * scale


def build_model(cond_channels: int, out_channels: int, args: argparse.Namespace) -> ConditionalUNet:
    return ConditionalUNet(
        in_channels=1,
        cond_channels=cond_channels,
        out_channels=out_channels,
        base_ch=args.base_channels,
        ch_mult=tuple(args.ch_mult),
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        time_emb_dim=args.time_emb_dim,
    )


def forward_model(model: torch.nn.Module, batch: Dict[str, object], device: torch.device) -> torch.Tensor:
    cond = batch["cond"].to(device, non_blocking=True).float()  # type: ignore[index]
    zeros = torch.zeros((cond.shape[0], 1, cond.shape[-2], cond.shape[-1]), device=device)
    t = torch.zeros((cond.shape[0],), dtype=torch.long, device=device)
    return torch.tanh(model(zeros, t, cond))


def teacher_aux_loss(pred: torch.Tensor, batch: Dict[str, object], device: torch.device) -> torch.Tensor:
    if pred.shape[1] < 5:
        return pred.new_tensor(0.0)
    phase_pred = pred[:, 1:5]
    phase_target = batch["phase_target"].to(device, non_blocking=True).float()  # type: ignore[index]
    phase_conf = batch["phase_conf"].to(device, non_blocking=True).float()  # type: ignore[index]
    valid = batch["valid_mask"].to(device, non_blocking=True).float()  # type: ignore[index]
    phase_conf = phase_conf * valid
    return charbonnier(phase_pred, phase_target, weight=phase_conf)


def compute_loss(pred: torch.Tensor, batch: Dict[str, object], device: torch.device, args: argparse.Namespace) -> torch.Tensor:
    height_pred = pred[:, :1]
    target = batch["height"].to(device, non_blocking=True).float()  # type: ignore[index]
    weight = train_weight(batch, device, args.object_mask_weight)
    loss = charbonnier(height_pred, target, weight=weight)
    loss = loss + args.lambda_mse * masked_mse(height_pred, target, weight=weight)
    if args.lambda_grad > 0:
        loss = loss + args.lambda_grad * gradient_loss(height_pred, target, weight=weight)
    if args.config == "teacher_aux" and args.lambda_teacher_phase > 0:
        loss = loss + args.lambda_teacher_phase * teacher_aux_loss(pred, batch, device)
    return loss


def metric_row(pred_mm: torch.Tensor, target_mm: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    return compute_metrics(pred_mm, target_mm, mask=mask)


def mean_std(rows: List[Dict[str, object]], key: str) -> Tuple[float, float]:
    vals = np.array([float(r[key]) for r in rows], dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std(ddof=1) if vals.size > 1 else 0.0)


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {"n": len(rows)}
    for roi in ROI_PREFIXES:
        roi_summary = {}
        for key in METRIC_KEYS:
            mean, std = mean_std(rows, f"{roi}_{key}")
            roi_summary[key] = {"mean": mean, "std": std}
        out[roi] = roi_summary
    per_object: Dict[str, Dict[str, float]] = {}
    for obj in sorted({int(r["object_id"]) for r in rows}):
        subset = [r for r in rows if int(r["object_id"]) == obj]
        per_object[f"obj{obj:04d}"] = {
            "n": len(subset),
            "object_rmse_mean": mean_std(subset, "object_rmse")[0],
            "valid_rmse_mean": mean_std(subset, "valid_rmse")[0],
        }
    out["per_object"] = per_object
    return out


def save_rows(rows: List[Dict[str, object]], path: Path) -> None:
    keys = [
        "sample_id",
        "object_id",
        "pose_id",
        "legal_single_frame",
        "config",
    ]
    for roi in ROI_PREFIXES:
        keys.extend([f"{roi}_{key}" for key in METRIC_KEYS])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Iterable[Dict[str, object]],
    device: torch.device,
    args: argparse.Namespace,
    out_dir: Path | None = None,
    save_images: bool = False,
) -> List[Dict[str, object]]:
    model.eval()
    rows: List[Dict[str, object]] = []
    if save_images and out_dir is not None:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc="eval", leave=False):
        pred = forward_model(model, batch, device)[:, :1]
        pred_mm = prediction_to_height_mm(pred, batch)
        target_mm = batch["height_raw"].to(device, non_blocking=True).float()  # type: ignore[index]
        object_mask = batch["object_mask"].to(device, non_blocking=True).float()  # type: ignore[index]
        valid_mask = batch["valid_mask"].to(device, non_blocking=True).float()  # type: ignore[index]
        fringe = batch["fringe"].to(device, non_blocking=True).float()  # type: ignore[index]
        sample_ids = batch["sample_id"]  # type: ignore[index]
        object_ids = batch["object_id"].detach().cpu().numpy().tolist()  # type: ignore[index]
        pose_ids = batch["pose_id"].detach().cpu().numpy().tolist()  # type: ignore[index]
        for j in range(pred.shape[0]):
            object_metrics = metric_row(pred_mm[j:j + 1], target_mm[j:j + 1], object_mask[j:j + 1])
            valid_metrics = metric_row(pred_mm[j:j + 1], target_mm[j:j + 1], valid_mask[j:j + 1])
            row: Dict[str, object] = {
                "sample_id": sample_ids[j],
                "object_id": int(object_ids[j]),
                "pose_id": int(pose_ids[j]),
                "legal_single_frame": bool(args.legal_single_frame),
                "config": args.config,
            }
            for key, value in object_metrics.items():
                row[f"object_{key}"] = value
            for key, value in valid_metrics.items():
                row[f"valid_{key}"] = value
            rows.append(row)
            if save_images and out_dir is not None and len(rows) <= args.save_eval_images:
                save_comparison(
                    fringe[j:j + 1],
                    target_mm[j:j + 1],
                    pred_mm[j:j + 1],
                    out_dir / "samples" / f"{len(rows):02d}_{sample_ids[j]}.png",
                    title=f"{args.config} object RMSE {object_metrics['rmse']:.3f}mm",
                    mask=object_mask[j:j + 1],
                )
    return rows


def write_eval_outputs(rows: List[Dict[str, object]], out_dir: Path, checkpoint: Path, args: argparse.Namespace) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rows(rows, out_dir / "per_sample_metrics.csv")
    summary = summarize(rows)
    summary.update({
        "checkpoint": str(checkpoint),
        "config": args.config,
        "seed": args.seed,
        "legal_single_frame": args.legal_single_frame,
        "input_mode": args.input_mode,
        "target": "wall_normal_height",
        "metric_scope": "self-built dataset only; do not compare directly to FPP-ML-Bench depth RMSE",
        "args": vars(args),
    })
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def save_dataset_qc(loaders: Dict[str, object], out_dir: Path, count: int = 8) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = loaders["train_eval"]
    saved = 0
    for batch in loader:  # type: ignore[assignment]
        b = batch
        n = b["fringe"].shape[0]  # type: ignore[index]
        for j in range(n):
            sample_id = b["sample_id"][j]  # type: ignore[index]
            raw = b["fringe"][j, 0].numpy()  # type: ignore[index]
            height = b["height_raw"][j, 0].numpy()  # type: ignore[index]
            valid = b["valid_mask"][j, 0].numpy() > 0.5  # type: ignore[index]
            obj = b["object_mask"][j, 0].numpy() > 0.5  # type: ignore[index]
            phase_target = b["phase_target"][j].numpy()  # type: ignore[index]
            bc_y = b["bc_y"][j, 0].numpy()  # type: ignore[index]
            bc_x = b["bc_x"][j, 0].numpy()  # type: ignore[index]
            height_show = np.ma.masked_where(~valid, height)
            obj_overlay = np.zeros((*raw.shape, 3), dtype=np.float32)
            obj_overlay[..., 1] = valid.astype(np.float32)
            obj_overlay[..., 0] = obj.astype(np.float32)
            fig, axes = plt.subplots(2, 4, figsize=(18, 8))
            panels = [
                (raw, "single_input", "gray"),
                (height_show, "wall_normal_height", "viridis"),
                (obj_overlay, "green=valid red=object", None),
                (phase_target[0], "sin phase_y teacher QC", "twilight"),
                (phase_target[2], "sin phase_x teacher QC", "twilight"),
                (bc_y, "bc_y QC only", "magma"),
                (bc_x, "bc_x QC only", "magma"),
                (b["cond"][j, 0].numpy(), "cond ch0 single_input", "gray"),  # type: ignore[index]
            ]
            for ax, (img, title, cmap) in zip(axes.flat, panels):
                ax.imshow(img, cmap=cmap) if cmap else ax.imshow(img)
                ax.set_title(title)
                ax.axis("off")
            fig.suptitle(str(sample_id))
            fig.tight_layout()
            fig.savefig(out_dir / f"{saved + 1:02d}_{sample_id}_qc.png", dpi=130, bbox_inches="tight")
            plt.close(fig)
            saved += 1
            if saved >= count:
                return


def save_checkpoint(path: Path, ep: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: GradScaler, args: argparse.Namespace, best: float, history: List[Dict[str, object]]) -> None:
    torch.save({
        "epoch": ep,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": vars(args),
        "best_val_object_rmse": best,
        "history": history,
    }, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="my_fpp_dataset_v1")
    parser.add_argument("--processed_dir", default="")
    parser.add_argument("--split_dir", default="")
    parser.add_argument("--save_dir", default="cloud_results/A_20260611_my_fpp_physics_validation/runs/debug")
    parser.add_argument("--config", default="raw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_epoch_repeats", type=int, default=1)
    parser.add_argument("--train_subset", type=int, default=0)
    parser.add_argument("--image_h", type=int, default=240)
    parser.add_argument("--image_w", type=int, default=320)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--ch_mult", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--time_emb_dim", type=int, default=128)
    parser.add_argument("--lambda_mse", type=float, default=0.5)
    parser.add_argument("--lambda_grad", type=float, default=0.10)
    parser.add_argument("--lambda_teacher_phase", type=float, default=0.05)
    parser.add_argument("--object_mask_weight", type=float, default=3.0)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--save_eval_images", type=int, default=8)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--cache_features", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--smoke_only", action="store_true")
    parser.add_argument("--visualize_only", action="store_true")
    parser.add_argument("--visualize_count", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.config = canonical_input_mode(args.config)
    args.input_mode = args.config
    args.legal_single_frame = is_legal_single_frame_mode(args.config)
    if args.config == "teacher_aux":
        args.input_mode = "teacher_aux"
    if args.config == "teacher_oracle":
        args.input_mode = "teacher_oracle"
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    loaders = create_my_fpp_loaders(
        data_root=args.data_root,
        processed_dir=args.processed_dir or None,
        split_dir=args.split_dir or None,
        input_mode=args.input_mode,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        image_h=args.image_h,
        image_w=args.image_w,
        train_epoch_repeats=args.train_epoch_repeats,
        train_subset=args.train_subset,
        cache_features=args.cache_features,
    )
    args.channel_names = loaders["channel_names"]
    args.cond_channels = int(loaders["cond_channels"])
    args.height_stats = loaders["stats"]

    smoke = smoke_summary(loaders)
    with (save_dir / "loader_smoke_summary.json").open("w", encoding="utf-8") as f:
        json.dump(smoke, f, indent=2, ensure_ascii=False)
    print(json.dumps(smoke, indent=2, ensure_ascii=False))
    if args.smoke_only:
        return
    save_dataset_qc(loaders, save_dir / "qc_visualizations", count=args.visualize_count)
    if args.visualize_only:
        return

    out_channels = 5 if args.config == "teacher_aux" else 1
    model = build_model(args.cond_channels, out_channels, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))
    best = float("inf")
    history: List[Dict[str, object]] = []
    (save_dir / "checkpoints").mkdir(exist_ok=True)
    (save_dir / "visualizations").mkdir(exist_ok=True)

    print(f"Device: {device}")
    print(f"Config: {args.config} | legal_single_frame={args.legal_single_frame}")
    print(f"Channels ({args.cond_channels}): {args.channel_names}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"{args.config} {ep}/{args.epochs}"):  # type: ignore[index]
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                pred = forward_model(model, batch, device)
                loss = compute_loss(pred, batch, device, args)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item())
            seen += 1
            if args.max_train_batches and seen >= args.max_train_batches:
                break
        scheduler.step()
        log: Dict[str, object] = {
            "epoch": ep,
            "train_loss": total / max(1, seen),
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - t0,
        }
        if ep == 1 or ep % args.eval_every == 0:
            val_rows = evaluate(model, loaders["val"], device, args)  # type: ignore[arg-type,index]
            val_summary = summarize(val_rows)
            log["val_object_rmse"] = val_summary["object"]["rmse"]["mean"]  # type: ignore[index]
            log["val_valid_rmse"] = val_summary["valid"]["rmse"]["mean"]  # type: ignore[index]
            val_rmse = float(log["val_object_rmse"])
            if val_rmse < best:
                best = val_rmse
                save_checkpoint(save_dir / "checkpoints" / "best.pt", ep, model, optimizer, scaler, args, best, history)
                first = next(iter(loaders["val"]))  # type: ignore[index]
                pred = forward_model(model, first, device)[:, :1]
                pred_mm = prediction_to_height_mm(pred, first)
                save_comparison(
                    first["fringe"].to(device),  # type: ignore[index]
                    first["height_raw"].to(device),  # type: ignore[index]
                    pred_mm,
                    save_dir / "visualizations" / f"val_ep{ep:03d}.png",
                    title=f"{args.config} val object RMSE {val_rmse:.3f}mm",
                    mask=first["object_mask"].to(device),  # type: ignore[index]
                )
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        with (save_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        if args.save_every > 0 and (ep == 1 or ep == args.epochs or ep % args.save_every == 0):
            save_checkpoint(save_dir / "checkpoints" / "latest.pt", ep, model, optimizer, scaler, args, best, history)

    best_path = save_dir / "checkpoints" / "best.pt"
    if not best_path.exists():
        best_path = save_dir / "checkpoints" / "latest.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    test_rows = evaluate(model, loaders["test"], device, args, out_dir=save_dir / "evaluation", save_images=True)  # type: ignore[arg-type,index]
    summary = write_eval_outputs(test_rows, save_dir / "evaluation", best_path, args)
    print("Final test:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
