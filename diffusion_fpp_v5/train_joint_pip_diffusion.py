"""Joint PIP-DiffFPP training on FPP-ML-Bench.

This script is the network-level PIP-DiffFPP line:
  fringe/physics instruction -> Coarse low-pass branch -> controlled diffusion
  residual branch -> final depth loss.

It intentionally keeps the old post-hoc gate/fusion code out of the loop.
The final prediction is supervised directly:
    D_pred = D_base + residual_scale * gate * R_pred
where D_base is the cached deterministic C4 adapter prediction.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from diffusion_pip import (
    charbonnier,
    confidence_edge_loss,
    cosine_beta_schedule,
    grad_xy_padded,
    masked_mean,
    normal_loss,
    oriented_gradient_loss,
    weighted_charbonnier,
    weighted_mse,
)
from models import CoarseLowpassNet, ConditionalUNetAdapter
from physics_features_pip import FEATURE_ORDER
from train_fpp_official_style_unet import parse_channel_spec
from train_pip_lite import prediction_to_mm, zero_initialize_prediction_head
from utils.metrics import compute_metrics
from utils.visualization import save_comparison


METRIC_KEYS = ["rmse", "mae", "edge_rmse", "normal_deg", "ssim"]


def logit(p):
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return float(np.log(p / (1.0 - p)))


def mean_std(rows, key):
    vals = np.array([r[key] for r in rows], dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)


def summarize_prefixed(rows, prefix):
    out = {}
    for key in METRIC_KEYS:
        mean, std = mean_std(rows, f"{prefix}_{key}")
        out[key] = {"mean": mean, "std": std}
    return out


def save_rows(rows, path):
    keys = ["sample", "object_index"]
    for prefix in ("pred", "base"):
        keys.extend([f"{prefix}_{k}" for k in METRIC_KEYS])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def resolve_residual_scale(cache_dir, base_prefix, explicit_scale):
    if explicit_scale and explicit_scale > 0:
        return float(explicit_scale)
    stats_path = Path(cache_dir) / f"{base_prefix}_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"missing residual stats file: {stats_path}; pass --residual_scale if needed"
        )
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    for key in ("p99_abs_residual", "residual_scale"):
        value = float(stats.get(key, 0.0))
        if value > 0:
            return value
    raise ValueError(f"{stats_path} does not contain p99_abs_residual/residual_scale")


def low_freq_grad_loss(pred, target, mask=None):
    pdx, pdy = grad_xy_padded(pred)
    tdx, tdy = grad_xy_padded(target)
    if mask is None:
        return F.l1_loss(pdx, tdx) + F.l1_loss(pdy, tdy)
    weight = torch.clamp(mask.to(device=pred.device, dtype=pred.dtype), 0.0, 1.0)
    return masked_mean(torch.abs(pdx - tdx), weight) + masked_mean(torch.abs(pdy - tdy), weight)


def masked_heteroscedastic_l1(pred, target, log_var, mask=None):
    loss = torch.exp(-log_var) * torch.abs(pred - target) + log_var
    if mask is None:
        return loss.mean()
    return masked_mean(loss, mask)


class JointPIPDiffFPP(nn.Module):
    def __init__(
        self,
        cond_channels,
        cond_indices=None,
        joint_mode="full",
        base_channels=24,
        adapter_hidden=24,
        coarse_channels=24,
        ch_mult=(1, 2, 4, 8),
        dropout=0.05,
        learned_residual_gate=False,
        gate_init=0.05,
    ):
        super().__init__()
        self.cond_indices = list(cond_indices) if cond_indices is not None else None
        self.joint_mode = str(joint_mode)
        if self.joint_mode not in {"full", "no_unc", "no_coarse"}:
            raise ValueError(f"unknown joint_mode: {self.joint_mode}")
        selected_channels = len(self.cond_indices) if self.cond_indices is not None else int(cond_channels)
        self.use_coarse = self.joint_mode in {"full", "no_unc"}
        if self.use_coarse:
            self.coarse = CoarseLowpassNet(selected_channels, base_ch=coarse_channels)
            extra = 2 if self.joint_mode == "full" else 1
        else:
            self.coarse = None
            extra = 0
        self.diffusion_cond_channels = selected_channels + extra
        self.learned_residual_gate = bool(learned_residual_gate)
        self.residual_net = ConditionalUNetAdapter(
            in_channels=1,
            cond_channels=self.diffusion_cond_channels,
            out_channels=1,
            base_ch=base_channels,
            ch_mult=ch_mult,
            num_res_blocks=2,
            dropout=dropout,
            adapter_hidden=adapter_hidden,
        )
        if self.learned_residual_gate:
            gate_hidden = max(8, min(32, self.diffusion_cond_channels * 2))
            self.gate_head = nn.Sequential(
                nn.Conv2d(self.diffusion_cond_channels, gate_hidden, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(gate_hidden, 1, 3, padding=1),
            )
            nn.init.zeros_(self.gate_head[-1].weight)
            nn.init.constant_(self.gate_head[-1].bias, logit(gate_init))
        else:
            self.gate_head = None

    def select_cond(self, cond):
        if self.cond_indices is None:
            return cond
        return cond[:, self.cond_indices]

    def forward(self, x_t, t, cond):
        cond_sel = self.select_cond(cond)
        d_low = None
        log_var = None
        cond_aug = cond_sel
        if self.use_coarse:
            d_low, log_var = self.coarse(cond_sel)
            if self.joint_mode == "full":
                unc_feat = torch.tanh(log_var / 4.0)
                cond_aug = torch.cat([cond_sel, d_low, unc_feat], dim=1)
            else:
                cond_aug = torch.cat([cond_sel, d_low], dim=1)
        residual = torch.clamp(self.residual_net(x_t, t, cond_aug), -1.0, 1.0)
        gate = torch.sigmoid(self.gate_head(cond_aug)) if self.gate_head is not None else None
        return residual, d_low, log_var, gate


class JointDiffusionRunner:
    def __init__(
        self,
        model,
        timesteps,
        device,
        residual_scale,
        base_residual_gate=1.0,
        train_t_min_ratio=0.0,
        train_t_max_ratio=0.15,
    ):
        self.model = model
        self.timesteps = int(timesteps)
        self.device = device
        self.residual_scale = float(residual_scale)
        self.base_residual_gate = float(base_residual_gate)
        self.train_t_min_ratio = float(train_t_min_ratio)
        self.train_t_max_ratio = float(train_t_max_ratio)
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

    def effective_residual_scale(self):
        return self.residual_scale * self.base_residual_gate

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sa = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        so = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sa * x_start + so * noise

    def residual_target(self, target, base):
        scale = self.effective_residual_scale()
        return torch.clamp((target - base) / scale, -1.0, 1.0)

    def gated_residual(self, residual, gate=None):
        return residual if gate is None else residual * gate

    def residual_to_depth(self, residual, base, gate=None):
        return torch.clamp(base + self.gated_residual(residual, gate) * self.effective_residual_scale(), -1.0, 1.0)

    def sample_timesteps(self, batch_size):
        t_min = max(0, min(self.timesteps - 1, int((self.timesteps - 1) * self.train_t_min_ratio)))
        t_max = max(t_min, min(self.timesteps - 1, int((self.timesteps - 1) * self.train_t_max_ratio)))
        return torch.randint(t_min, t_max + 1, (batch_size,), device=self.device, dtype=torch.long)

    @torch.no_grad()
    def sample_ddim(self, batch, steps=20, start_ratio=0.05, ensemble_size=1):
        if ensemble_size > 1:
            preds = []
            for i in range(int(ensemble_size)):
                preds.append(self._sample_single(batch, steps=steps, start_ratio=start_ratio, seed=i * 17))
            return torch.median(torch.stack(preds, dim=0), dim=0).values
        return self._sample_single(batch, steps=steps, start_ratio=start_ratio, seed=0)

    @torch.no_grad()
    def _sample_single(self, batch, steps=20, start_ratio=0.05, seed=0):
        cond = batch["cond"].to(self.device, non_blocking=True)
        base = torch.clamp(batch["base_height"].to(self.device, non_blocking=True), -1.0, 1.0)
        b = base.shape[0]
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        start_t = max(1, min(self.timesteps - 1, int(self.timesteps * float(start_ratio))))
        t0 = torch.full((b,), start_t, device=self.device, dtype=torch.long)
        x = self.q_sample(
            base,
            t0,
            noise=torch.randn(base.shape, device=self.device, generator=gen),
        )
        n_steps = max(2, min(int(steps), start_t + 1))
        times = torch.linspace(start_t, 0, n_steps, device=self.device).round().long().unique_consecutive()
        if times[-1] != 0:
            times = torch.cat([times, torch.zeros(1, device=self.device, dtype=torch.long)])
        for i in range(len(times) - 1):
            t = times[i]
            t_next = times[i + 1]
            tb = torch.full((b,), int(t.item()), device=self.device, dtype=torch.long)
            pred_res, _, _, gate = self.model(x, tb, cond)
            x0 = self.residual_to_depth(pred_res, base, gate)
            alpha = self.alphas_cumprod[t]
            alpha_next = self.alphas_cumprod[t_next]
            eps = (x - alpha.sqrt() * x0) / (1.0 - alpha).sqrt().clamp(min=1e-8)
            x = alpha_next.sqrt() * x0 + (1.0 - alpha_next).sqrt() * eps
        tb = torch.zeros((b,), device=self.device, dtype=torch.long)
        pred_res, _, _, gate = self.model(x, tb, cond)
        return self.residual_to_depth(pred_res, base, gate)


def base_error_loss_weight(target, base, residual_scale, gamma=1.0, strength=0.0, mask=None):
    weight = torch.ones_like(target)
    if strength > 0:
        err = torch.clamp(torch.abs(target - base) / max(float(residual_scale), 1e-6), 0.0, 1.0)
        if abs(float(gamma) - 1.0) > 1e-6:
            err = torch.pow(err, float(gamma))
        weight = weight * (1.0 + float(strength) * err)
    if mask is not None:
        weight = weight * torch.clamp(mask.to(device=target.device, dtype=target.dtype), 0.0, 1.0)
    return weight


def hard_region_target(batch, target, base, residual_scale, args, valid_mask=None):
    """Continuous hard-region target for masked residual diffusion.

    The score combines two signals:
      1) oracle base residual during training, so the gate learns where C4 fails;
      2) physical edge/confidence score, available at train and test time.

    The returned map is only a training target/loss weight. At inference the model
    still uses its learned gate from condition features, not ground-truth error.
    """

    if str(args.hard_mask_mode) == "none":
        return None
    scale = max(float(residual_scale), 1e-6)
    err_score = torch.clamp(torch.abs(target - base) / scale, 0.0, 1.0)
    edge = torch.clamp(batch["edge_score"].to(target.device, non_blocking=True), 0.0, 1.0)
    conf = torch.clamp(batch["phase_conf"].to(target.device, non_blocking=True), 0.0, 1.0)
    phys_score = edge * torch.pow(conf, float(args.hard_conf_power))

    mode = str(args.hard_mask_mode)
    if mode == "oracle_error":
        score = err_score
    elif mode == "physics":
        score = phys_score
    elif mode == "mixed":
        score = torch.maximum(
            float(args.hard_error_weight) * err_score,
            float(args.hard_physics_weight) * phys_score,
        )
    else:
        raise ValueError(f"unknown hard_mask_mode: {args.hard_mask_mode}")

    if float(args.hard_mask_threshold) > 0:
        score = torch.sigmoid((score - float(args.hard_mask_threshold)) * float(args.hard_mask_sharpness))
    score = torch.clamp(score, 0.0, 1.0)
    if valid_mask is not None:
        score = score * torch.clamp(valid_mask.to(device=score.device, dtype=score.dtype), 0.0, 1.0)
    return score


def training_loss(runner, batch, args):
    device = runner.device
    target = batch["height"].to(device, non_blocking=True)
    base = torch.clamp(batch["base_height"].to(device, non_blocking=True), -1.0, 1.0)
    cond = batch["cond"].to(device, non_blocking=True)
    valid_mask = batch.get("mask")
    if valid_mask is not None:
        valid_mask = torch.clamp(valid_mask.to(device, non_blocking=True), 0.0, 1.0)

    residual_target = runner.residual_target(target, base)
    b = target.shape[0]
    t = runner.sample_timesteps(b)
    x_t = runner.q_sample(base, t)
    pred_res, d_low, log_var, gate = runner.model(x_t, t, cond)
    pred_res_eff = runner.gated_residual(pred_res, gate)
    pred_depth = runner.residual_to_depth(pred_res, base, gate)

    weight = base_error_loss_weight(
        target,
        base,
        runner.effective_residual_scale(),
        gamma=args.base_error_loss_gamma,
        strength=args.base_error_loss_weight,
        mask=valid_mask,
    )
    hard_target = hard_region_target(batch, target, base, runner.effective_residual_scale(), args, valid_mask)
    if hard_target is not None and args.hard_mask_focus_weight > 0:
        weight = weight * (1.0 + float(args.hard_mask_focus_weight) * hard_target)

    loss_res = weighted_charbonnier(pred_res_eff, residual_target, weight=weight)
    loss_res = loss_res + 0.5 * weighted_mse(pred_res_eff, residual_target, weight=weight)
    loss_depth = weighted_charbonnier(pred_depth, target, weight=weight)
    loss_depth = loss_depth + 0.5 * weighted_mse(pred_depth, target, weight=weight)
    loss = args.lambda_depth * loss_depth + args.lambda_residual * loss_res

    losses = {
        "loss_depth": float(loss_depth.detach().item()),
        "loss_residual": float(loss_res.detach().item()),
    }
    if gate is not None:
        losses["gate_mean"] = float(gate.detach().mean().item())
        losses["gate_max"] = float(gate.detach().amax().item())
        if args.lambda_gate_l1 > 0:
            loss_gate = gate.mean()
            loss = loss + args.lambda_gate_l1 * loss_gate
            losses["loss_gate_l1"] = float(loss_gate.detach().item())
        if hard_target is not None and args.lambda_gate_supervision > 0:
            loss_gate_sup = masked_mean(torch.abs(gate - hard_target), valid_mask)
            loss = loss + args.lambda_gate_supervision * loss_gate_sup
            losses["loss_gate_supervision"] = float(loss_gate_sup.detach().item())
            losses["hard_target_mean"] = float(hard_target.detach().mean().item())

    if d_low is not None and args.lambda_coarse > 0:
        target_low = batch["height_low"].to(device, non_blocking=True)
        loss_coarse = charbonnier(d_low, target_low, mask=valid_mask)
        if args.lambda_coarse_grad > 0:
            loss_coarse = loss_coarse + args.lambda_coarse_grad * low_freq_grad_loss(d_low, target_low, valid_mask)
        loss = loss + args.lambda_coarse * loss_coarse
        losses["loss_coarse"] = float(loss_coarse.detach().item())

    if d_low is not None and log_var is not None and args.lambda_uncertainty > 0:
        target_low = batch["height_low"].to(device, non_blocking=True)
        loss_unc = masked_heteroscedastic_l1(d_low, target_low, log_var, mask=valid_mask)
        loss = loss + args.lambda_uncertainty * loss_unc
        losses["loss_uncertainty"] = float(loss_unc.detach().item())

    if args.lambda_oriented > 0:
        loss_oriented = oriented_gradient_loss(
            pred_depth,
            target,
            batch["phase_sin"].to(device, non_blocking=True),
            batch["phase_cos"].to(device, non_blocking=True),
            batch["phase_conf"].to(device, non_blocking=True),
            mask=valid_mask,
        )
        loss = loss + args.lambda_oriented * loss_oriented
        losses["loss_oriented"] = float(loss_oriented.detach().item())

    if args.lambda_edge > 0:
        loss_edge = confidence_edge_loss(
            pred_depth,
            target,
            batch["edge_score"].to(device, non_blocking=True),
            batch["phase_conf"].to(device, non_blocking=True),
            mask=valid_mask,
        )
        loss = loss + args.lambda_edge * loss_edge
        losses["loss_edge"] = float(loss_edge.detach().item())

    if args.lambda_normal > 0:
        loss_norm = normal_loss(pred_depth, target, mask=valid_mask)
        loss = loss + args.lambda_normal * loss_norm
        losses["loss_normal"] = float(loss_norm.detach().item())

    losses["loss_total"] = float(loss.detach().item())
    return loss, losses


@torch.no_grad()
def evaluate_split(runner, loader, device, height_scale, split_name, args, out_dir=None, save_images=False):
    runner.model.eval()
    rows = []
    if save_images and out_dir is not None:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    for idx, batch in enumerate(tqdm(loader, desc=f"eval {split_name}")):
        pred = runner.sample_ddim(
            batch,
            steps=args.ddim_steps,
            start_ratio=args.sample_start_ratio,
            ensemble_size=args.ensemble,
        )
        pred_mm = prediction_to_mm(pred, batch, height_scale)
        base = torch.clamp(batch["base_height"].to(device, non_blocking=True), -1.0, 1.0)
        base_mm = prediction_to_mm(base, batch, height_scale)
        target_raw = batch["height_raw"].to(device, non_blocking=True)
        metric_mask = batch.get("mask")
        if metric_mask is not None:
            metric_mask = metric_mask.to(device, non_blocking=True)
        object_index = batch.get("object_index")
        for j in range(pred_mm.shape[0]):
            mask_j = metric_mask[j:j + 1] if metric_mask is not None else None
            pred_metrics = compute_metrics(pred_mm[j:j + 1], target_raw[j:j + 1], mask=mask_j)
            base_metrics = compute_metrics(base_mm[j:j + 1], target_raw[j:j + 1], mask=mask_j)
            row = {
                "sample": len(rows),
                "object_index": int(object_index[j].item()) if object_index is not None else -1,
            }
            row.update({f"pred_{k}": v for k, v in pred_metrics.items()})
            row.update({f"base_{k}": v for k, v in base_metrics.items()})
            rows.append(row)
            if save_images and out_dir is not None and len(rows) <= 8:
                save_comparison(
                    batch["fringe"][j:j + 1].to(device, non_blocking=True),
                    target_raw[j:j + 1],
                    pred_mm[j:j + 1],
                    out_dir / "samples" / f"{split_name}_{len(rows)-1:03d}.png",
                    title=f"Joint PIP RMSE {pred_metrics['rmse']:.2f}mm",
                    mask=mask_j,
                )
    return rows


def write_eval_outputs(rows, out_dir, args, checkpoint):
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rows(rows, out_dir / "per_sample_metrics.csv")
    pred_summary = summarize_prefixed(rows, "pred")
    base_summary = summarize_prefixed(rows, "base")
    summary = {
        "n": len(rows),
        "checkpoint": str(checkpoint),
        "pred": pred_summary,
        "base": base_summary,
        "delta_rmse_vs_base": pred_summary["rmse"]["mean"] - base_summary["rmse"]["mean"],
        "args": vars(args),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--save_dir", default="/root/autodl-tmp/diffusion_fpp_v5/results/joint_pip_diffusion")
    parser.add_argument("--base_prefix", default="base_c4_adapter")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_pspquad_cache_960")
    parser.add_argument("--phase_pred_prefix", default="")
    parser.add_argument("--append_phase_pred_to_cond", action="store_true")
    parser.add_argument("--include_ftp", action="store_true")
    parser.add_argument("--physics_channels", default="")
    parser.add_argument("--joint_mode", choices=["full", "no_unc", "no_coarse"], default="full")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--adapter_hidden", type=int, default=24)
    parser.add_argument("--coarse_channels", type=int, default=24)
    parser.add_argument("--learned_residual_gate", action="store_true")
    parser.add_argument("--gate_init", type=float, default=0.05)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=4)
    parser.add_argument("--image_h", type=int, default=960)
    parser.add_argument("--image_w", type=int, default=1280)
    parser.add_argument("--lowpass_factor", type=int, default=8)
    parser.add_argument("--train_crop_h", type=int, default=0)
    parser.add_argument("--train_crop_w", type=int, default=0)
    parser.add_argument("--train_epoch_repeats", type=int, default=1)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--require_cache", action="store_true")
    parser.add_argument("--residual_scale", type=float, default=0.0)
    parser.add_argument("--base_residual_gate", type=float, default=1.0)
    parser.add_argument("--train_t_min_ratio", type=float, default=0.0)
    parser.add_argument("--train_t_max_ratio", type=float, default=0.15)
    parser.add_argument("--sample_start_ratio", type=float, default=0.05)
    parser.add_argument("--base_error_loss_weight", type=float, default=1.0)
    parser.add_argument("--base_error_loss_gamma", type=float, default=1.0)
    parser.add_argument("--hard_mask_mode", choices=["none", "oracle_error", "physics", "mixed"], default="none")
    parser.add_argument("--hard_error_weight", type=float, default=1.0)
    parser.add_argument("--hard_physics_weight", type=float, default=1.0)
    parser.add_argument("--hard_conf_power", type=float, default=1.0)
    parser.add_argument("--hard_mask_threshold", type=float, default=0.35)
    parser.add_argument("--hard_mask_sharpness", type=float, default=8.0)
    parser.add_argument("--hard_mask_focus_weight", type=float, default=0.0)
    parser.add_argument("--lambda_depth", type=float, default=1.0)
    parser.add_argument("--lambda_residual", type=float, default=0.2)
    parser.add_argument("--lambda_coarse", type=float, default=0.2)
    parser.add_argument("--lambda_coarse_grad", type=float, default=0.1)
    parser.add_argument("--lambda_uncertainty", type=float, default=0.05)
    parser.add_argument("--lambda_oriented", type=float, default=0.08)
    parser.add_argument("--lambda_edge", type=float, default=0.03)
    parser.add_argument("--lambda_normal", type=float, default=0.01)
    parser.add_argument("--lambda_gate_l1", type=float, default=0.0)
    parser.add_argument("--lambda_gate_supervision", type=float, default=0.0)
    parser.add_argument("--disable_zero_residual_init", action="store_true")
    parser.add_argument("--skip_final_test", action="store_true")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    args.physics_channel_indices = parse_channel_spec(args.physics_channels, args.include_ftp)
    args.physics_channel_names = (
        [FEATURE_ORDER[idx] for idx in args.physics_channel_indices]
        if args.physics_channel_indices is not None else None
    )
    args.resolved_residual_scale = resolve_residual_scale(args.cache_dir, args.base_prefix, args.residual_scale)

    save_dir = Path(args.save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    loaders = create_fpp_ml_bench_loaders(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        include_ftp=args.include_ftp,
        image_h=args.image_h,
        image_w=args.image_w,
        lowpass_factor=args.lowpass_factor,
        require_cache=args.require_cache,
        train_epoch_repeats=args.train_epoch_repeats,
        base_prefix=args.base_prefix,
        phase_cache_dir=args.phase_cache_dir,
        phase_pred_prefix=args.phase_pred_prefix or None,
        append_phase_pred_to_cond=args.append_phase_pred_to_cond,
        train_crop_h=args.train_crop_h,
        train_crop_w=args.train_crop_w,
    )

    model = JointPIPDiffFPP(
        cond_channels=loaders["cond_channels"],
        cond_indices=args.physics_channel_indices,
        joint_mode=args.joint_mode,
        base_channels=args.base_channels,
        adapter_hidden=args.adapter_hidden,
        coarse_channels=args.coarse_channels,
        learned_residual_gate=args.learned_residual_gate,
        gate_init=args.gate_init,
    ).to(device)
    if not args.disable_zero_residual_init and not args.resume:
        zero_initialize_prediction_head(model.residual_net)

    runner = JointDiffusionRunner(
        model=model,
        timesteps=args.timesteps,
        device=device,
        residual_scale=args.resolved_residual_scale,
        base_residual_gate=args.base_residual_gate,
        train_t_min_ratio=args.train_t_min_ratio,
        train_t_max_ratio=args.train_t_max_ratio,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=(device.type == "cuda"))
    best = float("inf")
    history = []
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            sch.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        best = float(ckpt.get("best_val_rmse", best))
        history = ckpt.get("history", history)
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    print(f"Device: {device}")
    print(f"Train {len(loaders['train'].dataset)} | Val {len(loaders['val'].dataset)} | Test {len(loaders['test'].dataset)}")
    print(f"Cond channels: {loaders['cond_channels']} | selected: {args.physics_channel_indices}")
    print(f"Joint mode: {args.joint_mode} | diffusion cond channels: {model.diffusion_cond_channels}")
    print(f"Learned residual gate: {args.learned_residual_gate} | gate_init={args.gate_init:.4f}")
    print(f"Base prefix: {args.base_prefix} | residual_scale={args.resolved_residual_scale:.6f} | gate={args.base_residual_gate:.3f}")
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    with open(save_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    for ep in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        sums = {}
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"joint {ep}/{args.epochs}"):
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                loss, parts = training_loss(runner, batch, args)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            seen += 1
            if args.max_train_batches and seen >= args.max_train_batches:
                break
        sch.step()
        log = {
            "epoch": ep,
            "lr": sch.get_last_lr()[0],
            "seconds": time.time() - t0,
        }
        for key, value in sorted(sums.items()):
            log[key] = value / max(1, seen)

        if ep == 1 or ep % args.eval_every == 0:
            val_rows = evaluate_split(runner, loaders["val"], device, loaders["height_scale"], "val", args)
            pred_summary = summarize_prefixed(val_rows, "pred")
            base_summary = summarize_prefixed(val_rows, "base")
            val_rmse = pred_summary["rmse"]["mean"]
            log.update({
                "val_rmse": val_rmse,
                "val_mae": pred_summary["mae"]["mean"],
                "val_edge_rmse": pred_summary["edge_rmse"]["mean"],
                "val_normal_deg": pred_summary["normal_deg"]["mean"],
                "val_base_rmse": base_summary["rmse"]["mean"],
                "val_delta_rmse_vs_base": val_rmse - base_summary["rmse"]["mean"],
            })
            if val_rmse < best:
                best = val_rmse
                torch.save(
                    {
                        "epoch": ep,
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "height_scale": loaders["height_scale"],
                        "best_val_rmse": best,
                        "cond_channels": loaders["cond_channels"],
                        "diffusion_cond_channels": model.diffusion_cond_channels,
                    },
                    save_dir / "checkpoints" / "best.pt",
                )
                first = next(iter(loaders["val"]))
                pred = runner.sample_ddim(first, steps=args.ddim_steps, start_ratio=args.sample_start_ratio)
                pred_mm = prediction_to_mm(pred, first, loaders["height_scale"])
                first_mask = first.get("mask")
                if first_mask is not None:
                    first_mask = first_mask.to(device, non_blocking=True)
                save_comparison(
                    first["fringe"].to(device, non_blocking=True),
                    first["height_raw"].to(device, non_blocking=True),
                    pred_mm,
                    save_dir / "visualizations" / f"val_ep{ep:03d}.png",
                    title=f"Joint PIP val RMSE {best:.2f}mm",
                    mask=first_mask,
                )

        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        if args.save_every > 0 and (ep == 1 or ep == args.epochs or ep % args.save_every == 0):
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": sch.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "args": vars(args),
                    "height_scale": loaders["height_scale"],
                    "best_val_rmse": best,
                    "history": history,
                },
                save_dir / "checkpoints" / "latest.pt",
            )

    best_path = save_dir / "checkpoints" / "best.pt"
    if best_path.exists() and not args.skip_final_test:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_rows = evaluate_split(
            runner,
            loaders["test"],
            device,
            loaders["height_scale"],
            "test",
            args,
            out_dir=save_dir / "evaluation",
            save_images=True,
        )
        summary = write_eval_outputs(test_rows, save_dir / "evaluation", args, best_path)
        print("Final test:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
