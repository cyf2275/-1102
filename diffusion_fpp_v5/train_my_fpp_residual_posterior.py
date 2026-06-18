"""Constrained residual diffusion posterior for my_fpp real-capture data.

This script is a fairer diffusion-vs-UNet check than full-depth diffusion:
it freezes a trained direct UNet as the posterior mean and trains diffusion
only on a bounded residual. Test-time inputs remain legal single-frame inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset_my_fpp import canonical_input_mode, create_my_fpp_loaders, is_legal_single_frame_mode, smoke_summary
from models import ConditionalUNet
from train_my_fpp_input_ablation import (
    METRIC_KEYS,
    build_model,
    charbonnier,
    gradient_loss,
    masked_mse,
    metric_row,
    prediction_to_height_mm,
    save_comparison,
    set_seed,
    summarize,
    train_weight,
)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    acp = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    acp = acp / acp[0]
    betas = 1 - (acp[1:] / acp[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def namespace_from_dict(data: Dict[str, object]) -> argparse.Namespace:
    defaults = {
        "base_channels": 32,
        "ch_mult": [1, 2, 4, 8],
        "num_res_blocks": 1,
        "dropout": 0.05,
        "time_emb_dim": 128,
    }
    merged = dict(defaults)
    merged.update(data)
    return argparse.Namespace(**merged)


def load_base_model(path: str | Path, cond_channels: int, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, object]]:
    ckpt = torch.load(str(path), map_location=device)
    saved_args = ckpt.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = {}
    base_config = canonical_input_mode(str(saved_args.get("config", "raw")))
    out_channels = 5 if base_config == "teacher_aux" else 1
    model_args = namespace_from_dict(saved_args)
    model = build_model(cond_channels, out_channels, model_args).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, saved_args


@torch.no_grad()
def base_predict_norm(base_model: torch.nn.Module, batch: Dict[str, object], device: torch.device) -> torch.Tensor:
    cond = batch["cond"].to(device, non_blocking=True).float()  # type: ignore[index]
    zeros = torch.zeros((cond.shape[0], 1, cond.shape[-2], cond.shape[-1]), device=device)
    t = torch.zeros((cond.shape[0],), dtype=torch.long, device=device)
    return torch.tanh(base_model(zeros, t, cond))[:, :1]


class ResidualPosterior:
    def __init__(self, model: ConditionalUNet, timesteps: int, residual_scale: float, device: torch.device) -> None:
        self.model = model
        self.timesteps = int(timesteps)
        self.residual_scale = float(residual_scale)
        self.device = device
        betas = cosine_beta_schedule(self.timesteps).to(device)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        self.sqrt_acp = torch.sqrt(acp)
        self.sqrt_om = torch.sqrt(1.0 - acp)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self.sqrt_acp[t].view(-1, 1, 1, 1)
        so = self.sqrt_om[t].view(-1, 1, 1, 1)
        return sa * x0 + so * noise

    def cond_with_base(self, batch: Dict[str, object], base_norm: torch.Tensor) -> torch.Tensor:
        cond = batch["cond"].to(self.device, non_blocking=True).float()  # type: ignore[index]
        return torch.cat([cond, base_norm.detach()], dim=1)

    def target_residual(self, batch: Dict[str, object], base_norm: torch.Tensor) -> torch.Tensor:
        target = batch["height"].to(self.device, non_blocking=True).float()  # type: ignore[index]
        residual = (target - base_norm.detach()) / max(self.residual_scale, 1e-6)
        return torch.clamp(residual, -1.0, 1.0)

    def training_loss(self, batch: Dict[str, object], base_model: torch.nn.Module, args: argparse.Namespace) -> torch.Tensor:
        base_norm = base_predict_norm(base_model, batch, self.device)
        cond = self.cond_with_base(batch, base_norm)
        target_res = self.target_residual(batch, base_norm)
        b = target_res.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.device)
        noisy = self.q_sample(target_res, t)
        pred_res = torch.tanh(self.model(noisy, t, cond))
        weight = train_weight(batch, self.device, args.object_mask_weight)
        loss = charbonnier(pred_res, target_res, weight=weight)
        loss = loss + args.lambda_mse * masked_mse(pred_res, target_res, weight=weight)
        if args.lambda_grad > 0:
            final_pred = torch.clamp(base_norm + self.residual_scale * pred_res, -1.0, 1.0)
            target = batch["height"].to(self.device, non_blocking=True).float()  # type: ignore[index]
            loss = loss + args.lambda_grad * gradient_loss(final_pred, target, weight=weight)
        if args.lambda_final > 0:
            final_pred = torch.clamp(base_norm + self.residual_scale * pred_res, -1.0, 1.0)
            target = batch["height"].to(self.device, non_blocking=True).float()  # type: ignore[index]
            loss = loss + args.lambda_final * charbonnier(final_pred, target, weight=weight)
        return loss

    @torch.no_grad()
    def sample(
        self,
        batch: Dict[str, object],
        base_model: torch.nn.Module,
        steps: int,
        ensemble_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_norm = base_predict_norm(base_model, batch, self.device)
        cond = self.cond_with_base(batch, base_norm)
        b, _, h, w = base_norm.shape
        seq = torch.linspace(self.timesteps - 1, 0, int(steps), device=self.device).long()
        preds = []
        for _ in range(max(1, int(ensemble_size))):
            x = torch.randn((b, 1, h, w), device=self.device)
            for t_val in seq:
                t_int = int(t_val.item())
                t = torch.full((b,), t_int, device=self.device, dtype=torch.long)
                x0 = torch.tanh(self.model(x, t, cond))
                if t_int == 0:
                    x = x0
                    continue
                prev_t = max(t_int - max(1, self.timesteps // max(1, int(steps))), 0)
                eps = (x - self.sqrt_acp[t].view(-1, 1, 1, 1) * x0) / self.sqrt_om[t].view(-1, 1, 1, 1).clamp(min=1e-6)
                x = self.sqrt_acp[prev_t].view(1, 1, 1, 1) * x0 + self.sqrt_om[prev_t].view(1, 1, 1, 1) * eps
            preds.append(torch.clamp(base_norm + self.residual_scale * torch.clamp(x, -1.0, 1.0), -1.0, 1.0))
        stack = torch.stack(preds, dim=0)
        mean = stack.mean(dim=0)
        unc = stack.std(dim=0, unbiased=False) if stack.shape[0] > 1 else torch.zeros_like(mean)
        return base_norm, mean, unc


def compute_row(
    pred_norm: torch.Tensor,
    batch: Dict[str, object],
    j: int,
    mode: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    device = pred_norm.device
    scale = batch["scale_mm"].to(device, non_blocking=True).float()[j].view(1, 1, 1, 1)  # type: ignore[index]
    pred_j = j if pred_norm.shape[0] > j else 0
    pred_mm = torch.clamp(pred_norm[pred_j:pred_j + 1], -1.0, 1.0) * scale
    target_mm = batch["height_raw"].to(device, non_blocking=True).float()[j:j + 1]  # type: ignore[index]
    object_mask = batch["object_mask"].to(device, non_blocking=True).float()[j:j + 1]  # type: ignore[index]
    valid_mask = batch["valid_mask"].to(device, non_blocking=True).float()[j:j + 1]  # type: ignore[index]
    object_metrics = metric_row(pred_mm, target_mm, object_mask)
    valid_metrics = metric_row(pred_mm, target_mm, valid_mask)
    row: Dict[str, object] = {
        "sample_id": batch["sample_id"][j],  # type: ignore[index]
        "object_id": int(batch["object_id"][j].item()),  # type: ignore[index]
        "pose_id": int(batch["pose_id"][j].item()),  # type: ignore[index]
        "legal_single_frame": bool(args.legal_single_frame),
        "config": f"{args.config}_residual_posterior_{mode}",
        "mode": mode,
    }
    for key in METRIC_KEYS:
        row[f"object_{key}"] = object_metrics[key]
        row[f"valid_{key}"] = valid_metrics[key]
    return row


def save_rows(rows: List[Dict[str, object]], path: Path) -> None:
    keys = ["sample_id", "object_id", "pose_id", "legal_single_frame", "config", "mode"]
    for roi in ("object", "valid"):
        keys.extend([f"{roi}_{key}" for key in METRIC_KEYS])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


@torch.no_grad()
def collect_predictions(
    posterior: ResidualPosterior,
    base_model: torch.nn.Module,
    loader: Iterable[Dict[str, object]],
    device: torch.device,
    args: argparse.Namespace,
    out_dir: Path | None = None,
    save_images: bool = False,
) -> Dict[str, object]:
    posterior.model.eval()
    samples: List[Dict[str, object]] = []
    rows_base: List[Dict[str, object]] = []
    rows_mean: List[Dict[str, object]] = []
    if save_images and out_dir is not None:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc="eval residual posterior", leave=False):
        base_norm, mean_norm, unc_norm = posterior.sample(
            batch,
            base_model,
            steps=args.sample_steps,
            ensemble_size=args.ensemble_size,
        )
        for j in range(base_norm.shape[0]):
            rows_base.append(compute_row(base_norm, batch, j, "base_unet", args))
            rows_mean.append(compute_row(mean_norm, batch, j, "posterior_mean", args))
            samples.append({
                "batch": batch,
                "index": j,
                "base_norm": base_norm[j:j + 1].detach().cpu(),
                "mean_norm": mean_norm[j:j + 1].detach().cpu(),
                "unc_norm": unc_norm[j:j + 1].detach().cpu(),
            })
            if save_images and out_dir is not None and len(samples) <= args.save_eval_images:
                scale = batch["scale_mm"].to(device, non_blocking=True).float()[j].view(1, 1, 1, 1)  # type: ignore[index]
                save_comparison(
                    batch["fringe"][j:j + 1].to(device),  # type: ignore[index]
                    batch["height_raw"][j:j + 1].to(device),  # type: ignore[index]
                    torch.clamp(mean_norm[j:j + 1], -1.0, 1.0) * scale,
                    out_dir / "samples" / f"{len(samples):02d}_{batch['sample_id'][j]}_posterior_mean.png",  # type: ignore[index]
                    title="residual posterior mean",
                    mask=batch["object_mask"][j:j + 1].to(device),  # type: ignore[index]
                )
    return {"samples": samples, "base": rows_base, "posterior_mean": rows_mean}


def parse_alpha_grid(text: str) -> List[float]:
    vals = []
    for part in str(text).replace(",", " ").split():
        try:
            vals.append(float(part))
        except Exception:
            pass
    vals = [min(max(v, 0.0), 1.0) for v in vals]
    return sorted(set(vals)) or [0.0, 0.25, 0.5, 0.75, 1.0]


def rows_for_gate(samples: List[Dict[str, object]], tau: float, alpha: float, args: argparse.Namespace, device: torch.device) -> Tuple[List[Dict[str, object]], float]:
    rows: List[Dict[str, object]] = []
    accepted = 0.0
    total = 0.0
    for item in samples:
        batch = item["batch"]  # type: ignore[assignment]
        j = int(item["index"])
        base = item["base_norm"].to(device)  # type: ignore[union-attr]
        mean = item["mean_norm"].to(device)  # type: ignore[union-attr]
        unc = item["unc_norm"].to(device)  # type: ignore[union-attr]
        if tau < 0:
            use = torch.zeros_like(base, dtype=torch.bool)
        else:
            correction = torch.abs(mean - base)
            use = (unc <= float(tau)) & (correction <= float(args.max_gate_correction))
        blended = torch.clamp(base + float(alpha) * (mean - base), -1.0, 1.0)
        pred = torch.where(use, blended, base)
        accepted += float(use.float().mean().item())
        total += 1.0
        rows.append(compute_row(pred, batch, j, "posterior_gate", args))
    return rows, accepted / max(total, 1.0)


def choose_gate(samples: List[Dict[str, object]], args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    unc_vals = []
    for item in samples:
        unc = item["unc_norm"].numpy().reshape(-1)  # type: ignore[union-attr]
        if unc.size:
            unc_vals.append(unc)
    if unc_vals:
        vals = np.concatenate(unc_vals)
        qs = np.percentile(vals, [10, 20, 40, 60, 80, 90, 95, 99]).tolist()
        thresholds = [-1.0, 0.0] + [float(x) for x in qs] + [float(np.max(vals) + 1e-6)]
    else:
        thresholds = [-1.0]
    best = {"threshold": -1.0, "alpha": 0.0, "object_rmse": float("inf"), "valid_rmse": float("inf"), "accepted_fraction": 0.0, "rows": []}
    for tau in thresholds:
        for alpha in parse_alpha_grid(args.alpha_grid):
            rows, accepted = rows_for_gate(samples, float(tau), float(alpha), args, device)
            summary = summarize(rows)
            obj = float(summary["object"]["rmse"]["mean"])  # type: ignore[index]
            valid = float(summary["valid"]["rmse"]["mean"])  # type: ignore[index]
            if obj < float(best["object_rmse"]) - 1e-12 or (
                abs(obj - float(best["object_rmse"])) <= 1e-12
                and (float(alpha), accepted) < (float(best["alpha"]), float(best["accepted_fraction"]))
            ):
                best = {
                    "threshold": float(tau),
                    "alpha": float(alpha),
                    "object_rmse": obj,
                    "valid_rmse": valid,
                    "accepted_fraction": float(accepted),
                    "rows": rows,
                }
    return best


def write_eval_outputs(
    collected: Dict[str, object],
    gate_rows: List[Dict[str, object]],
    gate: Dict[str, object],
    out_dir: Path,
    checkpoint: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_base = collected["base"]  # type: ignore[assignment]
    rows_mean = collected["posterior_mean"]  # type: ignore[assignment]
    save_rows(rows_base, out_dir / "base_unet_per_sample_metrics.csv")
    save_rows(rows_mean, out_dir / "posterior_mean_per_sample_metrics.csv")
    save_rows(gate_rows, out_dir / "per_sample_metrics.csv")
    base_summary = summarize(rows_base)
    mean_summary = summarize(rows_mean)
    gate_summary = summarize(gate_rows)
    summary = dict(gate_summary)
    summary.update({
        "checkpoint": str(checkpoint),
        "base_checkpoint": str(args.base_ckpt),
        "config": f"{args.config}_residual_posterior",
        "seed": args.seed,
        "legal_single_frame": args.legal_single_frame,
        "input_mode": args.input_mode,
        "target": "wall_normal_height",
        "experiment_role": "constrained residual diffusion posterior",
        "metric_scope": "self-built dataset only; direct comparison is against frozen direct UNet checkpoints on the same split",
        "gate": {k: v for k, v in gate.items() if k != "rows"},
        "comparison": {
            "base_unet": base_summary,
            "posterior_mean": mean_summary,
            "posterior_gate": gate_summary,
        },
        "args": vars(args),
    })
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (out_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary["comparison"], f, indent=2, ensure_ascii=False)
    return summary


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
    parser.add_argument("--save_dir", default="cloud_results/A_20260611_my_fpp_diffusion_vs_unet/runs/debug")
    parser.add_argument("--config", default="raw_single_phys")
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_epoch_repeats", type=int, default=4)
    parser.add_argument("--image_h", type=int, default=480)
    parser.add_argument("--image_w", type=int, default=640)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--ch_mult", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--time_emb_dim", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--sample_steps", type=int, default=12)
    parser.add_argument("--ensemble_size", type=int, default=3)
    parser.add_argument("--residual_scale", type=float, default=0.25)
    parser.add_argument("--max_gate_correction", type=float, default=0.25)
    parser.add_argument("--alpha_grid", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--lambda_mse", type=float, default=0.5)
    parser.add_argument("--lambda_grad", type=float, default=0.10)
    parser.add_argument("--lambda_final", type=float, default=0.50)
    parser.add_argument("--object_mask_weight", type=float, default=3.0)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--save_eval_images", type=int, default=8)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--cache_features", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--smoke_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.config = canonical_input_mode(args.config)
    args.input_mode = args.config
    args.legal_single_frame = is_legal_single_frame_mode(args.config)
    if not args.legal_single_frame:
        raise ValueError("residual posterior comparison only supports legal single-frame configs")
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
        cache_features=args.cache_features,
    )
    args.channel_names = loaders["channel_names"]
    args.cond_channels = int(loaders["cond_channels"])
    args.posterior_cond_channels = args.cond_channels + 1
    args.height_stats = loaders["stats"]

    with (save_dir / "loader_smoke_summary.json").open("w", encoding="utf-8") as f:
        json.dump(smoke_summary(loaders), f, indent=2, ensure_ascii=False)
    if args.smoke_only:
        print((save_dir / "loader_smoke_summary.json").read_text(encoding="utf-8"))
        return

    base_model, base_args = load_base_model(args.base_ckpt, args.cond_channels, device)
    args.base_config = canonical_input_mode(str(base_args.get("config", args.config)))
    model = ConditionalUNet(
        in_channels=1,
        cond_channels=args.posterior_cond_channels,
        out_channels=1,
        base_ch=args.base_channels,
        ch_mult=tuple(args.ch_mult),
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        time_emb_dim=args.time_emb_dim,
    ).to(device)
    posterior = ResidualPosterior(model, timesteps=args.timesteps, residual_scale=args.residual_scale, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))
    best = float("inf")
    history: List[Dict[str, object]] = []
    (save_dir / "checkpoints").mkdir(exist_ok=True)

    print(f"Device: {device}")
    print(f"Config: {args.config} | base={args.base_config} | role=constrained residual diffusion posterior")
    print(f"Channels: cond={args.cond_channels}, posterior_cond={args.posterior_cond_channels}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"residual-posterior {args.config} {ep}/{args.epochs}"):  # type: ignore[index]
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                loss = posterior.training_loss(batch, base_model, args)
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
            collected = collect_predictions(posterior, base_model, loaders["val"], device, args)  # type: ignore[index]
            gate = choose_gate(collected["samples"], args, device)  # type: ignore[arg-type]
            base_summary = summarize(collected["base"])  # type: ignore[arg-type]
            mean_summary = summarize(collected["posterior_mean"])  # type: ignore[arg-type]
            log["val_base_object_rmse"] = base_summary["object"]["rmse"]["mean"]  # type: ignore[index]
            log["val_posterior_mean_object_rmse"] = mean_summary["object"]["rmse"]["mean"]  # type: ignore[index]
            log["val_gate_object_rmse"] = gate["object_rmse"]
            log["val_gate_threshold"] = gate["threshold"]
            log["val_gate_accept"] = gate["accepted_fraction"]
            val_rmse = float(gate["object_rmse"])
            if val_rmse < best:
                best = val_rmse
                save_checkpoint(save_dir / "checkpoints" / "best.pt", ep, model, optimizer, scaler, args, best, history)
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        with (save_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        if args.save_every > 0 and (ep == 1 or ep == args.epochs or ep % args.save_every == 0):
            save_checkpoint(save_dir / "checkpoints" / "latest.pt", ep, model, optimizer, scaler, args, best, history)

    best_path = save_dir / "checkpoints" / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    val_collected = collect_predictions(posterior, base_model, loaders["val"], device, args)  # type: ignore[index]
    gate = choose_gate(val_collected["samples"], args, device)  # type: ignore[arg-type]
    test_collected = collect_predictions(posterior, base_model, loaders["test"], device, args, out_dir=save_dir / "evaluation", save_images=True)  # type: ignore[index]
    test_gate_rows, test_accept = rows_for_gate(test_collected["samples"], float(gate["threshold"]), float(gate["alpha"]), args, device)  # type: ignore[arg-type]
    gate["test_accepted_fraction"] = float(test_accept)
    summary = write_eval_outputs(test_collected, test_gate_rows, gate, save_dir / "evaluation", best_path, args)
    print("Final test:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
