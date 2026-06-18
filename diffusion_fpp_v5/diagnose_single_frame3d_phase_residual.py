"""Phase-direction and residual learnability diagnostics for single-frame 3D data.

The formal inputs remain legal single-frame inputs. Oracle phase branches are
diagnostic only: they answer whether true teacher/PMP phase could help if the
single-frame phase predictor were perfect.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from train_single_frame3d_physics_diffusion import (
    build_model,
    charbonnier,
    forward_direct,
    gradient_loss,
    load_base_model,
    masked_mse,
    pred_to_depth_mm,
    row_from_prediction,
    save_checkpoint,
    set_seed,
    summarize_rows,
    train_weight,
)
from train_single_frame3d_full_pip_rcpc import (
    create_loaders,
    dp_predict,
    load_residual,
    phi_predict,
    zero_time_forward,
)


DIR_INDEX = {
    "y": [0, 1, 4, 6],
    "x": [2, 3, 5, 7],
}


def model_args_from_ckpt(path: Path, fallback: argparse.Namespace) -> argparse.Namespace:
    if not path.exists():
        return fallback
    ckpt = torch.load(str(path), map_location="cpu")
    saved = ckpt.get("args", {})
    if not isinstance(saved, dict):
        saved = {}
    return argparse.Namespace(
        base_channels=int(saved.get("base_channels", fallback.base_channels)),
        ch_mult=list(saved.get("ch_mult", fallback.ch_mult)),
        num_res_blocks=int(saved.get("num_res_blocks", fallback.num_res_blocks)),
        dropout=float(saved.get("dropout", fallback.dropout)),
        time_emb_dim=int(saved.get("time_emb_dim", fallback.time_emb_dim)),
    )


def load_model(path: Path, in_channels: int, out_channels: int, fallback: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    ns = model_args_from_ckpt(path, fallback)
    model = build_model(in_channels, out_channels, ns).to(device)
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def dir_phi_predict(model: torch.nn.Module, batch: Dict[str, object], direction: str, device: torch.device) -> torch.Tensor:
    cond = batch["cond"].to(device, non_blocking=True).float()  # type: ignore[index]
    out = zero_time_forward(model, cond)
    out = out.clone()
    out[:, 2:4] = (out[:, 2:4] + 1.0) * 0.5
    return out


def dir_phi_oracle(batch: Dict[str, object], direction: str, device: torch.device) -> torch.Tensor:
    idx = DIR_INDEX[direction]
    return batch["phi_target"].to(device, non_blocking=True).float()[:, idx]  # type: ignore[index]


def dir_phi_loss(pred: torch.Tensor, batch: Dict[str, object], direction: str, device: torch.device) -> torch.Tensor:
    idx = DIR_INDEX[direction]
    target = batch["phi_target"].to(device, non_blocking=True).float()[:, idx]  # type: ignore[index]
    weight = batch["phi_weight"].to(device, non_blocking=True).float()[:, idx]  # type: ignore[index]
    valid = batch["valid_mask"].to(device, non_blocking=True).float()  # type: ignore[index]
    loss_sc = charbonnier(pred[:, :2], target[:, :2], weight=weight[:, :2] * valid)
    loss_rest = masked_mse(pred[:, 2:4], target[:, 2:4], weight=weight[:, 2:4] * valid)
    return loss_sc + loss_rest


@torch.no_grad()
def eval_dir_phi_loss(model: torch.nn.Module, loader: Iterable[Dict[str, object]], direction: str, device: torch.device) -> float:
    vals: List[float] = []
    model.eval()
    for batch in loader:
        vals.append(float(dir_phi_loss(dir_phi_predict(model, batch, direction, device), batch, direction, device).item()))
    return float(np.mean(vals)) if vals else float("nan")


def train_dir_phi(args: argparse.Namespace, loaders: Dict[str, object], direction: str, device: torch.device) -> torch.nn.Module:
    save_dir = Path(args.save_dir) / f"direction_{direction}" / "phi_predictor"
    ckpt_path = save_dir / "checkpoints" / "best.pt"
    if args.reuse_checkpoints and ckpt_path.exists():
        return load_model(ckpt_path, args.cond_channels, 4, args, device)
    model = build_model(args.cond_channels, 4, args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.direction_epochs))
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))
    best = float("inf")
    history: List[Dict[str, object]] = []
    for ep in range(1, args.direction_epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"P_{direction} {ep}/{args.direction_epochs}"):
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                loss = dir_phi_loss(dir_phi_predict(model, batch, direction, device), batch, direction, device)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.item())
            seen += 1
        sched.step()
        val = eval_dir_phi_loss(model, loaders["val"], direction, device)
        log = {"stage": f"phi_{direction}", "epoch": ep, "train_loss": total / max(1, seen), "val_loss": val, "seconds": time.time() - t0}
        history.append(log)
        print(json.dumps(log, ensure_ascii=False), flush=True)
        if val < best:
            best = val
            save_checkpoint(ckpt_path, ep, model, opt, scaler, args, best, history)
    return load_model(ckpt_path, args.cond_channels, 4, args, device)


def dir_dp_predict(phi_model: torch.nn.Module, dp_model: torch.nn.Module, batch: Dict[str, object], direction: str, device: torch.device) -> torch.Tensor:
    cond = batch["cond"].to(device, non_blocking=True).float()  # type: ignore[index]
    phi = dir_phi_predict(phi_model, batch, direction, device).detach()
    return zero_time_forward(dp_model, torch.cat([cond, phi], dim=1))[:, :1]


def dir_dp_oracle(dp_model: torch.nn.Module, batch: Dict[str, object], direction: str, device: torch.device) -> torch.Tensor:
    cond = batch["cond"].to(device, non_blocking=True).float()  # type: ignore[index]
    phi = dir_phi_oracle(batch, direction, device)
    return zero_time_forward(dp_model, torch.cat([cond, phi], dim=1))[:, :1]


@torch.no_grad()
def eval_dp_fast(pred_fn: Callable[[Dict[str, object]], torch.Tensor], loader: Iterable[Dict[str, object]], device: torch.device) -> float:
    vals: List[float] = []
    for batch in loader:
        pred = pred_to_depth_mm(pred_fn(batch), batch)
        target = batch["depth_raw"].to(device, non_blocking=True).float()  # type: ignore[index]
        mask = batch["object_mask"].to(device, non_blocking=True).float()  # type: ignore[index]
        count = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        rmse = torch.sqrt((((pred - target) ** 2) * mask).sum(dim=(1, 2, 3)) / count)
        vals.extend(float(x) for x in rmse.detach().cpu().tolist())
    return float(np.mean(vals)) if vals else float("nan")


def train_dir_dp(
    args: argparse.Namespace,
    loaders: Dict[str, object],
    phi_model: torch.nn.Module,
    direction: str,
    device: torch.device,
    oracle: bool = False,
) -> torch.nn.Module:
    in_ch = args.cond_channels + 4
    subdir = "phase_depth_oracle_evidence" if oracle else "phase_depth_evidence"
    save_dir = Path(args.save_dir) / f"direction_{direction}" / subdir
    ckpt_path = save_dir / "checkpoints" / "best.pt"
    if args.reuse_checkpoints and ckpt_path.exists():
        return load_model(ckpt_path, in_ch, 1, args, device)
    model = build_model(in_ch, 1, args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.direction_epochs))
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))
    best = float("inf")
    history: List[Dict[str, object]] = []
    for ep in range(1, args.direction_epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        tag = f"D_{direction}_{'oracle' if oracle else 'pred'}"
        for batch in tqdm(loaders["train"], desc=f"{tag} {ep}/{args.direction_epochs}"):
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                pred = dir_dp_oracle(model, batch, direction, device) if oracle else dir_dp_predict(phi_model, model, batch, direction, device)
                target = batch["depth"].to(device, non_blocking=True).float()  # type: ignore[index]
                weight = train_weight(batch, device, args.object_mask_weight)
                loss = charbonnier(pred, target, weight=weight)
                loss = loss + args.lambda_mse * masked_mse(pred, target, weight=weight)
                if args.lambda_grad > 0:
                    loss = loss + args.lambda_grad * gradient_loss(pred, target, weight=weight)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.item())
            seen += 1
        sched.step()
        val_fn = (lambda b: dir_dp_oracle(model, b, direction, device)) if oracle else (lambda b: dir_dp_predict(phi_model, model, b, direction, device))
        val = eval_dp_fast(val_fn, loaders["val"], device)
        log = {"stage": f"dp_{direction}_{'oracle' if oracle else 'pred'}", "epoch": ep, "train_loss": total / max(1, seen), "val_object_rmse": val, "seconds": time.time() - t0}
        history.append(log)
        print(json.dumps(log, ensure_ascii=False), flush=True)
        if val < best:
            best = val
            save_checkpoint(ckpt_path, ep, model, opt, scaler, args, best, history)
    return load_model(ckpt_path, in_ch, 1, args, device)


@torch.no_grad()
def rows_for_loader(loader: Iterable[Dict[str, object]], pred_fn: Callable[[Dict[str, object]], torch.Tensor], mode: str, device: torch.device) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for batch in tqdm(loader, desc=f"eval {mode}", leave=False):
        pred = pred_fn(batch)
        for j in range(pred.shape[0]):
            rows.append(row_from_prediction(pred, batch, j, "phase_residual_diagnosis", mode))
    return rows


def full_oracle_dp(dp_model: torch.nn.Module, batch: Dict[str, object], device: torch.device) -> torch.Tensor:
    cond = batch["cond"].to(device, non_blocking=True).float()  # type: ignore[index]
    phi = batch["phi_target"].to(device, non_blocking=True).float()  # type: ignore[index]
    return zero_time_forward(dp_model, torch.cat([cond, phi], dim=1))[:, :1]


def corrcoef(a: List[float], b: List[float]) -> float:
    if len(a) < 3:
        return float("nan")
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        keep = np.isfinite(x) & np.isfinite(y)
        x = x[keep]
        y = y[keep]
    if x.size < 3 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


@torch.no_grad()
def residual_diagnostics(
    loaders: Dict[str, object],
    base_model: torch.nn.Module,
    device: torch.device,
    max_pixels_per_batch: int = 12000,
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for split in [s for s in ("val", "test", "ood") if s in loaders]:
        abs_res: List[float] = []
        signed_res: List[float] = []
        edge_vals: List[float] = []
        low_conf_vals: List[float] = []
        depth_vals: List[float] = []
        sample_rows: List[Dict[str, object]] = []
        for batch in tqdm(loaders[split], desc=f"residual diag {split}", leave=False):
            base_norm = forward_direct(base_model, batch, device)[:, :1]
            base_mm = pred_to_depth_mm(base_norm, batch)
            target = batch["depth_raw"].to(device, non_blocking=True).float()  # type: ignore[index]
            mask = batch["object_mask"].to(device, non_blocking=True).float() > 0.5  # type: ignore[index]
            fringe = batch["fringe"].to(device, non_blocking=True).float()  # type: ignore[index]
            conf = batch["teacher_conf"].to(device, non_blocking=True).float()  # type: ignore[index]
            dx = F.pad(fringe[..., :, 1:] - fringe[..., :, :-1], (0, 1, 0, 0))
            dy = F.pad(fringe[..., 1:, :] - fringe[..., :-1, :], (0, 0, 0, 1))
            edge = torch.sqrt(dx * dx + dy * dy)
            res = target - base_mm
            for j in range(base_norm.shape[0]):
                sample_rows.append(row_from_prediction(base_norm, batch, j, "phase_residual_diagnosis", "direct_base"))
            flat_mask = mask.flatten()
            if flat_mask.sum().item() <= 0:
                continue
            idx = flat_mask.nonzero(as_tuple=False).flatten()
            if idx.numel() > max_pixels_per_batch:
                perm = torch.randperm(idx.numel(), device=idx.device)[:max_pixels_per_batch]
                idx = idx[perm]
            res_flat = res.flatten()[idx].detach().cpu().numpy()
            edge_flat = edge.flatten()[idx].detach().cpu().numpy()
            conf_flat = conf.flatten()[idx].detach().cpu().numpy()
            depth_flat = target.flatten()[idx].detach().cpu().numpy()
            signed_res.extend(float(x) for x in res_flat)
            abs_res.extend(float(abs(x)) for x in res_flat)
            edge_vals.extend(float(x) for x in edge_flat)
            low_conf_vals.extend(float(1.0 - x) for x in conf_flat)
            depth_vals.extend(float(x) for x in depth_flat)
        signed = np.asarray(signed_res, dtype=np.float64)
        abs_arr = np.asarray(abs_res, dtype=np.float64)
        out[split] = {
            "n_rows": len(sample_rows),
            "base_summary": summarize_rows(sample_rows),
            "residual_mm_mean": float(np.mean(signed)) if signed.size else float("nan"),
            "residual_mm_std": float(np.std(signed)) if signed.size else float("nan"),
            "abs_residual_mm_mean": float(np.mean(abs_arr)) if abs_arr.size else float("nan"),
            "positive_residual_fraction": float(np.mean(signed > 0.0)) if signed.size else float("nan"),
            "corr_absres_edge": corrcoef(abs_res, edge_vals),
            "corr_absres_low_teacher_conf": corrcoef(abs_res, low_conf_vals),
            "corr_absres_depth": corrcoef(abs_res, depth_vals),
        }
    return out


def save_rows_csv(rows: List[Dict[str, object]], path: Path) -> None:
    keys = ["split", "sample_id", "object_id", "pose_id", "config", "mode", "legal_single_frame"]
    for roi in ("object", "valid"):
        for metric in ("rmse", "mae", "edge_rmse", "normal_deg", "ssim"):
            keys.append(f"{roi}_{metric}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def make_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Phase Direction and Residual Diagnosis",
        "",
        "Object ROI RMSE is the primary metric. Oracle branches are diagnostic only and are not legal test-time inputs.",
        "",
        "## Main Metrics",
        "",
        "| split | branch | object RMSE | valid RMSE |",
        "|---|---|---:|---:|",
    ]
    for split, data in summary["splits"].items():  # type: ignore[union-attr]
        for branch, metrics in data.items():  # type: ignore[union-attr]
            lines.append(f"| {split} | {branch} | {metrics['object']['rmse']['mean']:.4f} | {metrics['valid']['rmse']['mean']:.4f} |")
    lines += ["", "## Residual Diagnostics", "", "```json", json.dumps(summary["residual_diagnostics"], indent=2, ensure_ascii=False), "```", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--teacher_extra_root", required=True)
    parser.add_argument("--ood_root", default="")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--residual_ckpt", default="")
    parser.add_argument("--pilot_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--eval_batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--direction_epochs", type=int, default=20)
    parser.add_argument("--directions", nargs="+", choices=["y", "x"], default=["y", "x"])
    parser.add_argument("--include_full_yx", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--ch_mult", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--time_emb_dim", type=int, default=128)
    parser.add_argument("--object_mask_weight", type=float, default=3.0)
    parser.add_argument("--lambda_mse", type=float, default=0.2)
    parser.add_argument("--lambda_grad", type=float, default=0.05)
    parser.add_argument("--cache_features", action="store_true")
    parser.add_argument("--feature_cache_dir", default="")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--reuse_checkpoints", action="store_true")
    parser.add_argument("--smoke_only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    loaders_obj = create_loaders(args)
    args.cond_channels = int(loaders_obj["cond_channels"])
    args.dp_cond_channels = int(loaders_obj["dp_cond_channels"])
    args.normalization = loaders_obj["norm"]
    args.split_counts = loaders_obj["split_counts"]
    (save_dir / "diagnosis_smoke.json").write_text(json.dumps({
        "device": str(device),
        "split_counts": args.split_counts,
        "cond_channels": args.cond_channels,
        "dp_cond_channels": args.dp_cond_channels,
        "normalization": args.normalization,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.smoke_only:
        print((save_dir / "diagnosis_smoke.json").read_text(encoding="utf-8"))
        return

    loaders: Dict[str, object] = loaders_obj["loaders"]  # type: ignore[assignment]
    base_model, base_args = load_base_model(args.base_ckpt, args.cond_channels, device)
    pilot_dir = Path(args.pilot_dir)
    full_phi = load_model(pilot_dir / "phi_predictor" / "checkpoints" / "best.pt", args.cond_channels, 8, args, device)
    full_dp = load_model(pilot_dir / "phase_depth_evidence" / "checkpoints" / "best.pt", args.dp_cond_channels, 1, args, device)

    dir_models: Dict[str, Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]] = {}
    for direction in args.directions:
        phi = train_dir_phi(args, loaders, direction, device)
        dp_pred = train_dir_dp(args, loaders, phi, direction, device, oracle=False)
        dp_oracle = train_dir_dp(args, loaders, phi, direction, device, oracle=True)
        dir_models[direction] = (phi, dp_pred, dp_oracle)

    residual = residual_diagnostics(loaders, base_model, device)
    summary: Dict[str, object] = {
        "stage": "phase_direction_residual_diagnosis",
        "seed": args.seed,
        "legal_single_frame_note": "predicted branches use only input_vertical_0120 plus derived features; oracle branches are diagnostic only",
        "base_ckpt": args.base_ckpt,
        "base_args": base_args,
        "pilot_dir": str(pilot_dir),
        "split_counts": args.split_counts,
        "normalization": args.normalization,
        "splits": {},
        "residual_diagnostics": residual,
    }
    all_rows: List[Dict[str, object]] = []
    for split in [s for s in ("val", "test", "ood") if s in loaders]:
        split_rows: List[Dict[str, object]] = []
        split_rows += rows_for_loader(loaders[split], lambda b: forward_direct(base_model, b, device)[:, :1], "direct_base", device)  # type: ignore[index]
        if args.include_full_yx:
            split_rows += rows_for_loader(loaders[split], lambda b: dp_predict(full_phi, full_dp, b, device), "full_predicted_yx_phase", device)  # type: ignore[index]
            split_rows += rows_for_loader(loaders[split], lambda b: full_oracle_dp(full_dp, b, device), "full_oracle_yx_phase_mismatch", device)  # type: ignore[index]
        for direction, (phi, dp_pred, dp_oracle) in dir_models.items():
            split_rows += rows_for_loader(loaders[split], lambda b, p=phi, d=dp_pred, di=direction: dir_dp_predict(p, d, b, di, device), f"{direction}_predicted_phase", device)  # type: ignore[index]
            split_rows += rows_for_loader(loaders[split], lambda b, d=dp_oracle, di=direction: dir_dp_oracle(d, b, di, device), f"{direction}_oracle_phase", device)  # type: ignore[index]
        if args.residual_ckpt:
            posterior, residual_args = load_residual(args.residual_ckpt, args.cond_channels, device)
            # Keep this branch deterministic enough for diagnostics by using a light mean sample.
            rows_res = []
            for batch in tqdm(loaders[split], desc=f"eval residual {split}", leave=False):  # type: ignore[index]
                base = forward_direct(base_model, batch, device)[:, :1]
                _, dd, _ = posterior.sample(batch, base_model, steps=8, ensemble_size=2)
                for j in range(dd.shape[0]):
                    rows_res.append(row_from_prediction(dd, batch, j, "phase_residual_diagnosis", "residual_posterior_light"))
            split_rows += rows_res
            summary["residual_args"] = residual_args
        save_rows_csv([{**r, "split": split} for r in split_rows], save_dir / f"{split}_phase_direction_metrics.csv")
        all_rows += [{**r, "split": split} for r in split_rows]
        by_mode: Dict[str, List[Dict[str, object]]] = {}
        for row in split_rows:
            by_mode.setdefault(str(row["mode"]), []).append(row)
        summary["splits"][split] = {mode: summarize_rows(rows) for mode, rows in sorted(by_mode.items())}  # type: ignore[index]
    save_rows_csv(all_rows, save_dir / "phase_direction_metrics_all.csv")
    (save_dir / "phase_residual_diagnosis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (save_dir / "phase_residual_diagnosis_report.md").write_text(make_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
