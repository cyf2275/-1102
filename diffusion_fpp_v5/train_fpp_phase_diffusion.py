"""Phase restoration diffusion for FPP-ML-Bench.

E2 in the reset plan: the diffusion model denoises/restores the wrapped phase
representation, instead of acting as a small depth residual refiner.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset_fpp_phase import create_fpp_phase_loaders
from diffusion_pip import cosine_beta_schedule, gaussian_blur_3x3, normalize01_per_sample
from models import ConditionalUNetAdapter


PHASE_METRIC_KEYS = [
    "phase_mae_rad",
    "phase_rmse_rad",
    "phase_aligned_mae_rad",
    "phase_aligned_rmse_rad",
    "uph_mae_01",
    "uph_rmse_01",
    "calib_depth_rmse_mm",
    "calib_depth_mae_mm",
]


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
    out = []
    for idx in selected:
        if idx not in out:
            out.append(idx)
    invalid = [idx for idx in out if idx < 0 or idx > max_channel]
    if invalid:
        raise ValueError(f"invalid phase instruction channels {invalid}; max={max_channel}")
    return out


def parse_ch_mult(text: str):
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def masked_mean(x, mask=None):
    if mask is None:
        return x.mean()
    mask = torch.clamp(mask.to(device=x.device, dtype=x.dtype), 0.0, 1.0)
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def normalize_sincos(x):
    sc = x[:, 0:2]
    norm = torch.sqrt((sc * sc).sum(dim=1, keepdim=True)).clamp(min=1e-6)
    return sc / norm


def project_phase_state(x):
    sc = normalize_sincos(x)
    if x.shape[1] <= 2:
        return sc
    # The third channel is normalized unwrapped phase in [0, 1].  Sigmoid is
    # used instead of clamp during training so gradients remain usable.
    extra = torch.sigmoid(x[:, 2:])
    return torch.cat([sc, extra], dim=1)


def sanitize_phase_state(x):
    sc = normalize_sincos(x)
    if x.shape[1] <= 2:
        return sc
    return torch.cat([sc, torch.clamp(x[:, 2:], 0.0, 1.0)], dim=1)


def angle_from_sincos(sc):
    return torch.atan2(sc[:, 0:1], sc[:, 1:2])


def angular_diff(pred_sc, target_sc):
    pred = angle_from_sincos(pred_sc)
    target = angle_from_sincos(target_sc)
    return torch.atan2(torch.sin(pred - target), torch.cos(pred - target))


def grad_xy_padded(x):
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return dx, dy


def select_cond(batch, device, channels):
    cond = batch["cond"].to(device, non_blocking=True)
    if channels is not None:
        cond = cond[:, channels]
    return cond


def phase_target(
    batch,
    device,
    target_channels=2,
    uph_norm="sample",
    uph_global_min=0.0,
    uph_global_max=1.0,
):
    target_channels = int(target_channels)
    target = batch["phase_target"][:, :target_channels].to(device, non_blocking=True)
    if target_channels >= 3 and str(uph_norm) == "global":
        mm = batch["phase_minmax"].to(device, non_blocking=True)
        lo = mm[:, 0].view(-1, 1, 1, 1)
        hi = mm[:, 1].view(-1, 1, 1, 1)
        raw = torch.clamp(target[:, 2:3], 0.0, 1.0) * (hi - lo) + lo
        gmin = float(uph_global_min)
        gmax = float(uph_global_max)
        uph = torch.clamp((raw - gmin) / max(gmax - gmin, 1e-6), 0.0, 1.0)
        target = torch.cat([target[:, :2], uph], dim=1)
    return target


def source_phase(batch, source: str, device):
    cond = batch["cond"].to(device, non_blocking=True)
    if source == "ftp":
        sc = cond[:, 5:7]
    elif source == "hilbert":
        sc = cond[:, 1:3]
    elif source == "target":
        sc = batch["phase_target"][:, 0:2].to(device, non_blocking=True)
    elif source == "zero":
        sc = torch.zeros_like(batch["phase_target"][:, 0:2].to(device, non_blocking=True))
        sc[:, 1:2] = 1.0
    else:
        raise ValueError(f"unknown phase source: {source}")
    return normalize_sincos(sc)


def source_uph_prior(
    batch,
    mode: str,
    device,
    uph_norm="sample",
    uph_global_min=0.0,
    uph_global_max=1.0,
):
    cond = batch["cond"].to(device, non_blocking=True)
    mode = str(mode)
    if mode == "target":
        return torch.clamp(
            phase_target(
                batch,
                device,
                target_channels=3,
                uph_norm=uph_norm,
                uph_global_min=uph_global_min,
                uph_global_max=uph_global_max,
            )[:, 2:3],
            0.0,
            1.0,
        )
    if mode == "zero":
        return torch.zeros_like(cond[:, 0:1])
    if mode == "half":
        return torch.full_like(cond[:, 0:1], 0.5)
    if mode == "x":
        return torch.clamp(cond[:, 11:12] * 0.5 + 0.5, 0.0, 1.0)
    if mode == "y":
        return torch.clamp(cond[:, 12:13] * 0.5 + 0.5, 0.0, 1.0)
    if mode == "ftp_residual":
        return torch.clamp(cond[:, 7:8] * 0.5 + 0.5, 0.0, 1.0)
    if mode == "coord_auto":
        x01 = torch.clamp(cond[:, 11:12] * 0.5 + 0.5, 0.0, 1.0)
        y01 = torch.clamp(cond[:, 12:13] * 0.5 + 0.5, 0.0, 1.0)
        ftp = normalize_sincos(cond[:, 5:7])
        dx, dy = grad_xy_padded(ftp)
        score_x = torch.mean(torch.abs(dx).sum(dim=1, keepdim=True), dim=(2, 3), keepdim=True)
        score_y = torch.mean(torch.abs(dy).sum(dim=1, keepdim=True), dim=(2, 3), keepdim=True)
        use_x = (score_x >= score_y).to(x01.dtype)
        return use_x * x01 + (1.0 - use_x) * y01
    raise ValueError(f"unknown uph prior mode: {mode}")


def uph_prior_features_torch(cond, basis="xy2_phase"):
    x = cond[:, 11:12]
    y = cond[:, 12:13]
    feats = [torch.ones_like(x), x, y, x * x, x * y, y * y]
    basis = str(basis)
    if basis in {"xy2_phase", "xy2_phase_edge"}:
        hres = cond[:, 3:4]
        fres = cond[:, 7:8]
        feats.extend([hres, fres, hres * x, hres * y, fres * x, fres * y])
    if basis == "xy2_phase_edge":
        dwt = cond[:, 9:10]
        grad = cond[:, 10:11]
        feats.extend([dwt, grad, dwt * x, dwt * y, grad * x, grad * y])
    return torch.cat(feats, dim=1)


def uph_prior_raw_from_cond(cond, coef, basis="xy2_phase"):
    if coef is None:
        raise ValueError("uph prior coefficients are required for prior_residual representation")
    feats = uph_prior_features_torch(cond, basis=basis)
    coef_t = torch.as_tensor(coef, device=cond.device, dtype=cond.dtype).view(1, -1, 1, 1)
    if feats.shape[1] != coef_t.shape[1]:
        raise ValueError(f"UPH prior feature count {feats.shape[1]} != coef count {coef_t.shape[1]}")
    return torch.sum(feats * coef_t, dim=1, keepdim=True)


def phase_target_encoded(
    batch,
    device,
    target_channels=2,
    uph_norm="sample",
    uph_global_min=0.0,
    uph_global_max=1.0,
    uph_representation="absolute",
    uph_prior_coef=None,
    uph_prior_basis="xy2_phase",
    uph_residual_scale=1.0,
):
    target = phase_target(
        batch,
        device,
        target_channels=target_channels,
        uph_norm=uph_norm,
        uph_global_min=uph_global_min,
        uph_global_max=uph_global_max,
    )
    if int(target_channels) < 3 or str(uph_representation) == "absolute":
        return target
    if str(uph_representation) != "prior_residual":
        raise ValueError(f"unknown uph representation: {uph_representation}")
    mm = batch["phase_minmax"].to(device, non_blocking=True)
    lo = mm[:, 0].view(-1, 1, 1, 1)
    hi = mm[:, 1].view(-1, 1, 1, 1)
    raw = torch.clamp(batch["phase_target"][:, 2:3].to(device, non_blocking=True), 0.0, 1.0) * (hi - lo) + lo
    cond = batch["cond"].to(device, non_blocking=True)
    prior = uph_prior_raw_from_cond(cond, uph_prior_coef, basis=uph_prior_basis)
    scale = max(float(uph_residual_scale), 1e-6)
    residual01 = torch.clamp(0.5 + 0.5 * (raw - prior) / scale, 0.0, 1.0)
    return torch.cat([target[:, :2], residual01], dim=1)


def source_state(
    batch,
    source: str,
    device,
    target_channels=2,
    uph_start_from="coord_auto",
    uph_norm="sample",
    uph_global_min=0.0,
    uph_global_max=1.0,
):
    target_channels = int(target_channels)
    if source == "target":
        return phase_target(
            batch,
            device,
            target_channels=target_channels,
            uph_norm=uph_norm,
            uph_global_min=uph_global_min,
            uph_global_max=uph_global_max,
        )
    sc = source_phase(batch, source, device)
    if target_channels <= 2:
        return sc
    extras = [
        source_uph_prior(
            batch,
            uph_start_from,
            device,
            uph_norm=uph_norm,
            uph_global_min=uph_global_min,
            uph_global_max=uph_global_max,
        )
    ]
    while 2 + len(extras) < target_channels:
        extras.append(torch.zeros_like(extras[0]))
    return torch.cat([sc] + extras[: target_channels - 2], dim=1)


def source_confidence(batch, source: str, device):
    cond = batch["cond"].to(device, non_blocking=True)
    if source == "ftp":
        return torch.clamp(cond[:, 8:9], 0.0, 1.0)
    if source == "hilbert":
        return torch.clamp(cond[:, 4:5], 0.0, 1.0)
    return torch.ones_like(cond[:, 0:1])


def edge_score(batch, device):
    cond = batch["cond"].to(device, non_blocking=True)
    return normalize01_per_sample(torch.clamp(0.5 * cond[:, 9:10] + 0.5 * cond[:, 10:11], 0.0, 1.0))


def phase_loss(pred_raw, target_sc, mask, grad_weight=0.05, unit_weight=0.02):
    pred_sc = normalize_sincos(pred_raw)
    l1 = masked_mean(torch.abs(pred_sc - target_sc).sum(dim=1, keepdim=True), mask)
    mse = masked_mean(((pred_sc - target_sc) ** 2).sum(dim=1, keepdim=True), mask)
    cos = torch.clamp((pred_sc * target_sc).sum(dim=1, keepdim=True), -1.0, 1.0)
    angle = masked_mean(1.0 - cos, mask)
    loss = l1 + 0.5 * mse + 0.25 * angle

    if grad_weight > 0:
        pdx, pdy = grad_xy_padded(pred_sc)
        tdx, tdy = grad_xy_padded(target_sc)
        grad_err = torch.abs(pdx - tdx).sum(dim=1, keepdim=True) + torch.abs(pdy - tdy).sum(dim=1, keepdim=True)
        loss = loss + float(grad_weight) * masked_mean(grad_err, mask)

    if unit_weight > 0:
        norm = torch.sqrt((pred_raw[:, 0:2] ** 2).sum(dim=1, keepdim=True).clamp(min=1e-8))
        loss = loss + float(unit_weight) * masked_mean(torch.abs(norm - 1.0), mask)
    return loss


def uph_loss(pred_raw, target, mask, uph_weight=1.0, uph_grad_weight=0.05):
    if pred_raw.shape[1] < 3 or target.shape[1] < 3 or float(uph_weight) <= 0:
        return pred_raw.new_tensor(0.0)
    pred_uph = torch.sigmoid(pred_raw[:, 2:3])
    target_uph = torch.clamp(target[:, 2:3], 0.0, 1.0)
    err = pred_uph - target_uph
    loss = masked_mean(torch.abs(err), mask) + 0.5 * masked_mean(err * err, mask)
    if uph_grad_weight > 0:
        pdx, pdy = grad_xy_padded(pred_uph)
        tdx, tdy = grad_xy_padded(target_uph)
        grad_err = torch.abs(pdx - tdx) + torch.abs(pdy - tdy)
        loss = loss + float(uph_grad_weight) * masked_mean(grad_err, mask)
    return float(uph_weight) * loss


def calibrated_depth_from_uph(raw_uph, cond_full, coef):
    x = cond_full[:, 11:12].to(device=raw_uph.device, dtype=raw_uph.dtype)
    y = cond_full[:, 12:13].to(device=raw_uph.device, dtype=raw_uph.dtype)
    terms = [
        torch.ones_like(raw_uph),
        raw_uph,
        x,
        y,
        raw_uph * raw_uph,
        raw_uph * x,
        raw_uph * y,
        x * x,
        x * y,
        y * y,
    ]
    if coef.numel() > 10:
        terms.extend([raw_uph ** 3, raw_uph * raw_uph * x, raw_uph * raw_uph * y, raw_uph * x * y, x ** 3, y ** 3])
    pred = raw_uph.new_zeros(raw_uph.shape)
    for c, term in zip(coef.to(device=raw_uph.device, dtype=raw_uph.dtype), terms):
        pred = pred + c * term
    return pred


def phase_metrics_tensor(pred_raw, target_sc, mask=None, allow_offset=True):
    pred_raw = torch.nan_to_num(pred_raw, nan=0.0, posinf=1.0, neginf=-1.0)
    target_sc = torch.nan_to_num(target_sc, nan=0.0, posinf=1.0, neginf=-1.0)
    pred_full = pred_raw
    target_full = target_sc
    pred_sc = normalize_sincos(pred_raw)
    target_sc = normalize_sincos(target_sc)
    diff = angular_diff(pred_sc, target_sc)
    if mask is not None:
        valid = mask.to(device=diff.device, dtype=torch.bool)
        if valid.any():
            diff_v = diff[valid]
        else:
            diff_v = diff.flatten()
    else:
        valid = None
        diff_v = diff.flatten()
    out = {
        "phase_mae_rad": float(torch.mean(torch.abs(diff_v)).item()),
        "phase_rmse_rad": float(torch.sqrt(torch.mean(diff_v * diff_v)).item()),
    }
    if allow_offset:
        pred_ang = angle_from_sincos(pred_sc)
        target_ang = angle_from_sincos(target_sc)
        if valid is not None and valid.any():
            delta = target_ang[valid] - pred_ang[valid]
            offset = torch.atan2(torch.sin(delta).mean(), torch.cos(delta).mean())
            aligned = torch.atan2(torch.sin(pred_ang[valid] + offset - target_ang[valid]),
                                  torch.cos(pred_ang[valid] + offset - target_ang[valid]))
        else:
            delta = target_ang.flatten() - pred_ang.flatten()
            offset = torch.atan2(torch.sin(delta).mean(), torch.cos(delta).mean())
            aligned = torch.atan2(torch.sin(pred_ang.flatten() + offset - target_ang.flatten()),
                                  torch.cos(pred_ang.flatten() + offset - target_ang.flatten()))
        out["phase_aligned_mae_rad"] = float(torch.mean(torch.abs(aligned)).item())
        out["phase_aligned_rmse_rad"] = float(torch.sqrt(torch.mean(aligned * aligned)).item())
    else:
        out["phase_aligned_mae_rad"] = out["phase_mae_rad"]
        out["phase_aligned_rmse_rad"] = out["phase_rmse_rad"]
    if pred_full.shape[1] >= 3 and target_full.shape[1] >= 3:
        pred_uph = torch.clamp(pred_full[:, 2:3], 0.0, 1.0)
        target_uph = torch.clamp(target_full[:, 2:3], 0.0, 1.0)
        uph_err = pred_uph - target_uph
        if mask is not None:
            valid_uph = mask.to(device=uph_err.device, dtype=torch.bool)
            uph_v = uph_err[valid_uph] if valid_uph.any() else uph_err.flatten()
        else:
            uph_v = uph_err.flatten()
        out["uph_mae_01"] = float(torch.mean(torch.abs(uph_v)).item())
        out["uph_rmse_01"] = float(torch.sqrt(torch.mean(uph_v * uph_v)).item())
    return out


def depth_metrics_tensor(pred_depth, target_depth, mask=None):
    err = pred_depth - target_depth
    if mask is not None:
        valid = mask.to(device=err.device, dtype=torch.bool)
        err_v = err[valid] if valid.any() else err.flatten()
    else:
        err_v = err.flatten()
    err_v = torch.nan_to_num(err_v, nan=1e4, posinf=1e4, neginf=-1e4)
    return {
        "calib_depth_rmse_mm": float(torch.sqrt(torch.mean(err_v * err_v)).item()),
        "calib_depth_mae_mm": float(torch.mean(torch.abs(err_v)).item()),
    }


class PhaseRestorationDiffusion:
    def __init__(
        self,
        model,
        timesteps=200,
        image_h=960,
        image_w=960,
        device="cuda",
        cond_indices=None,
        phase_weight=1.0,
        grad_weight=0.05,
        unit_weight=0.02,
        target_channels=2,
        uph_weight=0.0,
        uph_grad_weight=0.05,
        uph_start_from="coord_auto",
        uph_norm="sample",
        uph_global_min=0.0,
        uph_global_max=1.0,
        calib_depth_weight=0.0,
        calib_depth_coef=None,
        calib_depth_scale=100.0,
        calib_depth_clip=300.0,
        calib_depth_start_epoch=0,
        uph_representation="absolute",
        uph_prior_coef=None,
        uph_prior_basis="xy2_phase",
        uph_residual_scale=1.0,
    ):
        self.model = model
        self.timesteps = int(timesteps)
        self.image_h = int(image_h)
        self.image_w = int(image_w)
        self.device = device
        self.cond_indices = list(cond_indices) if cond_indices is not None else None
        self.phase_weight = float(phase_weight)
        self.grad_weight = float(grad_weight)
        self.unit_weight = float(unit_weight)
        self.target_channels = int(target_channels)
        self.uph_weight = float(uph_weight)
        self.uph_grad_weight = float(uph_grad_weight)
        self.uph_start_from = str(uph_start_from)
        self.uph_norm = str(uph_norm)
        self.uph_global_min = float(uph_global_min)
        self.uph_global_max = float(uph_global_max)
        self.calib_depth_weight = float(calib_depth_weight)
        self.calib_depth_scale = float(calib_depth_scale)
        self.calib_depth_clip = float(calib_depth_clip)
        self.calib_depth_start_epoch = int(calib_depth_start_epoch)
        self.uph_representation = str(uph_representation)
        self.uph_prior_coef = [float(v) for v in uph_prior_coef] if uph_prior_coef is not None else None
        self.uph_prior_basis = str(uph_prior_basis)
        self.uph_residual_scale = float(uph_residual_scale)
        self.calib_depth_coef = None
        if calib_depth_coef is not None:
            self.calib_depth_coef = torch.as_tensor(calib_depth_coef, dtype=torch.float32, device=device)
        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

    def condition(self, batch):
        return select_cond(batch, self.device, self.cond_indices)

    def target(self, batch):
        return phase_target_encoded(
            batch,
            self.device,
            target_channels=self.target_channels,
            uph_norm=self.uph_norm,
            uph_global_min=self.uph_global_min,
            uph_global_max=self.uph_global_max,
            uph_representation=self.uph_representation,
            uph_prior_coef=self.uph_prior_coef,
            uph_prior_basis=self.uph_prior_basis,
            uph_residual_scale=self.uph_residual_scale,
        )

    def decode_uph_raw(self, batch, encoded_uph):
        encoded_uph = torch.clamp(encoded_uph, 0.0, 1.0)
        if self.uph_representation == "prior_residual":
            cond_full = batch["cond"].to(self.device, non_blocking=True)
            prior = uph_prior_raw_from_cond(cond_full, self.uph_prior_coef, basis=self.uph_prior_basis)
            return prior + (encoded_uph - 0.5) * (2.0 * max(self.uph_residual_scale, 1e-6))
        return encoded_uph * (self.uph_global_max - self.uph_global_min) + self.uph_global_min

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sa = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        so = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sa * x_start + so * noise

    def p_loss(self, batch, t_min_ratio=0.0, t_max_ratio=1.0, train_start_from="target"):
        target = self.target(batch)
        mask = torch.clamp(batch["mask"].to(self.device, non_blocking=True), 0.0, 1.0)
        b = target.shape[0]
        t_min = max(0, min(self.timesteps - 1, int((self.timesteps - 1) * float(t_min_ratio))))
        t_max = max(t_min, min(self.timesteps - 1, int((self.timesteps - 1) * float(t_max_ratio))))
        t = torch.randint(t_min, t_max + 1, (b,), device=self.device, dtype=torch.long)
        train_start_from = str(train_start_from)
        if train_start_from == "mix":
            ftp_sc = source_phase(batch, "ftp", self.device)
            target_sc = target[:, 0:2]
            use_ftp = (torch.rand((b, 1, 1, 1), device=self.device) < 0.5).to(target.dtype)
            start_sc = torch.where(use_ftp > 0.5, ftp_sc, target_sc)
            if self.target_channels > 2:
                start = torch.cat([start_sc, target[:, 2:self.target_channels]], dim=1)
            else:
                start = start_sc
        else:
            if train_start_from == "target":
                start = target
            else:
                start = source_state(
                    batch,
                    train_start_from,
                    self.device,
                    target_channels=self.target_channels,
                    uph_start_from=self.uph_start_from,
                    uph_norm=self.uph_norm,
                    uph_global_min=self.uph_global_min,
                    uph_global_max=self.uph_global_max,
                )
        x_t = self.q_sample(start, t)
        pred = self.model(x_t, t, self.condition(batch))
        loss = self.phase_weight * phase_loss(pred, target[:, 0:2], mask, grad_weight=self.grad_weight, unit_weight=self.unit_weight) + uph_loss(
            pred,
            target,
            mask,
            uph_weight=self.uph_weight,
            uph_grad_weight=self.uph_grad_weight,
        )
        if (
            self.calib_depth_coef is not None
            and self.calib_depth_weight > 0
            and self.target_channels >= 3
            and (self.uph_norm == "global" or self.uph_representation == "prior_residual")
            and int(getattr(self, "current_epoch", 0)) >= self.calib_depth_start_epoch
        ):
            pred_uph = torch.sigmoid(pred[:, 2:3])
            raw_uph = self.decode_uph_raw(batch, pred_uph)
            cond_full = batch["cond"].to(self.device, non_blocking=True)
            pred_depth = calibrated_depth_from_uph(raw_uph.float(), cond_full.float(), self.calib_depth_coef.float())
            target_depth = batch["height_raw"].to(self.device, non_blocking=True).float()
            depth_delta = pred_depth - target_depth
            if self.calib_depth_clip > 0:
                clip = float(self.calib_depth_clip)
                depth_delta = torch.nan_to_num(depth_delta, nan=0.0, posinf=clip, neginf=-clip)
                depth_delta = torch.clamp(depth_delta, -clip, clip)
            depth_err = (torch.sqrt(depth_delta * depth_delta + 1.0) - 1.0) / max(self.calib_depth_scale, 1e-6)
            loss = loss + self.calib_depth_weight * masked_mean(depth_err, mask)
        return loss

    def _apply_observation_guidance(self, x0, batch, guidance):
        if not guidance or float(guidance.get("weight", 0.0)) <= 0:
            return sanitize_phase_state(x0)
        source = str(guidance.get("source", "ftp"))
        lam = float(guidance.get("weight", 0.0))
        grad_clip = float(guidance.get("grad_clip", 0.05))
        eta = float(guidance.get("eta", 1.0))
        k = float(guidance.get("k", 8.0))
        tau = float(guidance.get("tau", 0.4))
        obs = source_phase(batch, source, self.device)
        conf = source_confidence(batch, source, self.device)
        edge = edge_score(batch, self.device)
        with torch.enable_grad():
            z = x0.detach().requires_grad_(True)
            z_sc = normalize_sincos(z)
            err = 1.0 - torch.clamp((z_sc * obs).sum(dim=1, keepdim=True), -1.0, 1.0)
            energy = (err * conf).mean()
            grad = torch.autograd.grad(energy, z, retain_graph=False, create_graph=False)[0]
        grad_blur = gaussian_blur_3x3(grad)
        w = torch.pow(conf, eta) * torch.sigmoid(k * (edge - tau))
        grad_final = (1.0 - w) * grad_blur + w * grad
        norm = grad_final.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp(min=1e-8)
        grad_final = grad_final * torch.clamp(grad_clip / norm, max=1.0)
        return sanitize_phase_state(x0 - lam * conf * grad_final)

    @torch.no_grad()
    def sample_ddim(
        self,
        batch,
        steps=20,
        ensemble_size=1,
        start_from="noise",
        start_ratio=1.0,
        guidance=None,
        progress=False,
    ):
        if ensemble_size > 1:
            preds = []
            for i in range(int(ensemble_size)):
                preds.append(self._sample_single(
                    batch,
                    steps=steps,
                    seed=17 * i,
                    start_from=start_from,
                    start_ratio=start_ratio,
                    guidance=guidance,
                    progress=False,
                ))
            stack = torch.stack(preds, dim=0)
            return sanitize_phase_state(torch.median(stack, dim=0).values)
        return self._sample_single(
            batch,
            steps=steps,
            seed=0,
            start_from=start_from,
            start_ratio=start_ratio,
            guidance=guidance,
            progress=progress,
        )

    def _sample_single(self, batch, steps=20, seed=0, start_from="noise", start_ratio=1.0, guidance=None, progress=False):
        cond = self.condition(batch)
        b = cond.shape[0]
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        start_from = str(start_from)
        if start_from == "noise":
            start_t = self.timesteps - 1
            x = torch.randn((b, self.target_channels, self.image_h, self.image_w), device=self.device, generator=gen)
        else:
            start_t = max(1, min(self.timesteps - 1, int(self.timesteps * float(start_ratio))))
            t0 = torch.full((b,), start_t, device=self.device, dtype=torch.long)
            start_phase = source_state(
                batch,
                start_from,
                self.device,
                target_channels=self.target_channels,
                uph_start_from=self.uph_start_from,
                uph_norm=self.uph_norm,
                uph_global_min=self.uph_global_min,
                uph_global_max=self.uph_global_max,
            )
            x = self.q_sample(
                start_phase,
                t0,
                noise=torch.randn(start_phase.shape, device=self.device, generator=gen),
            )

        n_steps = max(2, min(int(steps), start_t + 1))
        times = torch.linspace(start_t, 0, n_steps, device=self.device).round().long().unique_consecutive()
        if times[-1] != 0:
            times = torch.cat([times, torch.zeros(1, device=self.device, dtype=torch.long)])
        iterator = range(len(times) - 1)
        if progress:
            iterator = tqdm(iterator, desc=f"phase DDIM {len(times) - 1} steps")
        for i in iterator:
            t = times[i]
            t_next = times[i + 1]
            tb = torch.full((b,), int(t.item()), device=self.device, dtype=torch.long)
            pred_x0 = project_phase_state(self.model(x, tb, cond))
            denoise_progress = float(i + 1) / float(max(1, len(times) - 1))
            if guidance and denoise_progress >= float(guidance.get("apply_start_ratio", 0.7)):
                pred_x0 = self._apply_observation_guidance(pred_x0, batch, guidance)
            alpha = self.alphas_cumprod[t]
            alpha_next = self.alphas_cumprod[t_next]
            eps = (x - alpha.sqrt() * pred_x0) / (1.0 - alpha).sqrt().clamp(min=1e-8)
            x = alpha_next.sqrt() * pred_x0 + (1.0 - alpha_next).sqrt() * eps
        tb = torch.zeros((b,), device=self.device, dtype=torch.long)
        return project_phase_state(self.model(x, tb, cond))


def summarize(rows, prefix=""):
    summary = {}
    for key in PHASE_METRIC_KEYS:
        row_key = f"{prefix}{key}"
        if row_key not in rows[0]:
            continue
        vals = np.array([float(r[f"{prefix}{key}"]) for r in rows], dtype=np.float64)
        summary[f"{prefix}{key}"] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
        }
    return summary


def save_rows(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_phase_visual(batch, pred_sc, path, title="phase diffusion"):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fringe = batch["fringe"][0, 0].detach().cpu().numpy()
    target_sc = batch["phase_target"][0:1, :pred_sc.shape[1]].to(pred_sc.device)
    mask = batch["mask"][0, 0].detach().cpu().numpy() > 0.5
    pred_ang = angle_from_sincos(pred_sc[0:1]).detach().cpu().numpy()[0, 0]
    target_ang = angle_from_sincos(target_sc).detach().cpu().numpy()[0, 0]
    err = np.abs(np.angle(np.exp(1j * (pred_ang - target_ang))))
    err = np.where(mask, err, np.nan)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    axes[0].imshow(fringe, cmap="gray")
    axes[0].set_title("A0")
    axes[1].imshow(target_ang, cmap="twilight", vmin=-math.pi, vmax=math.pi)
    axes[1].set_title("GT wrapped")
    axes[2].imshow(pred_ang, cmap="twilight", vmin=-math.pi, vmax=math.pi)
    axes[2].set_title("pred wrapped")
    im = axes[3].imshow(err, cmap="magma", vmin=0.0, vmax=math.pi)
    axes[3].set_title("abs error")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def validation_score(summary, args):
    if args.selection_metric == "phase_uph_score":
        phase = summary["phase_aligned_mae_rad"]["mean"]
        uph = summary.get("uph_mae_01", {"mean": 0.0})["mean"]
        return float(phase) + float(args.uph_select_weight) * float(uph)
    if args.selection_metric == "calib_depth_rmse_mm":
        return summary["calib_depth_rmse_mm"]["mean"]
    return summary[args.selection_metric]["mean"]


def resolve_uph_global_range(args):
    if str(args.uph_norm) != "global":
        return float(args.uph_global_min), float(args.uph_global_max)
    if args.uph_global_max > args.uph_global_min:
        return float(args.uph_global_min), float(args.uph_global_max)
    mm_path = Path(args.phase_cache_dir) / "phase_minmax_train_float32.npy"
    mm = np.load(mm_path, mmap_mode="r")
    return float(np.min(mm[:, 0])), float(np.max(mm[:, 1]))


def load_calib_depth_coef(args):
    if float(args.calib_depth_weight) <= 0:
        return None
    path = str(args.calib_depth_summary or "").strip()
    if not path:
        raise ValueError("--calib_depth_summary is required when --calib_depth_weight > 0")
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    source = str(args.calib_depth_source)
    try:
        coef = summary["sources"][source]["coef"]
    except KeyError as exc:
        raise KeyError(f"calibration source {source!r} not found in {path}") from exc
    return [float(v) for v in coef]


def resolve_uph_prior_config(args):
    if str(args.uph_representation) != "prior_residual":
        args.uph_prior_coef = None
        return
    path = str(args.uph_prior_summary or "").strip()
    if not path:
        raise ValueError("--uph_prior_summary is required when --uph_representation prior_residual")
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    args.uph_prior_coef = [float(v) for v in summary["prior_coef"]]
    if not str(args.uph_prior_basis or "").strip():
        args.uph_prior_basis = str(summary.get("args", {}).get("basis", "xy2_phase"))
    if float(args.uph_residual_scale) <= 0:
        args.uph_residual_scale = float(summary.get("recommended_residual_scale", 1.0))


@torch.no_grad()
def evaluate(diffusion, loader, device, args, split_name, out_dir=None, save_images=False):
    diffusion.model.eval()
    rows = []
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    guidance = None
    if args.guidance_weight > 0:
        guidance = {
            "weight": args.guidance_weight,
            "source": args.guidance_source,
            "apply_start_ratio": args.guidance_apply_start_ratio,
            "grad_clip": args.guidance_grad_clip,
            "eta": args.guidance_eta,
            "k": args.guidance_k,
            "tau": args.guidance_tau,
        }
    for batch in tqdm(loader, desc=f"eval phase {split_name}"):
        pred = diffusion.sample_ddim(
            batch,
            steps=args.ddim_steps,
            ensemble_size=args.ensemble,
            start_from=args.sample_start_from,
            start_ratio=args.sample_start_ratio,
            guidance=guidance,
            progress=False,
        )
        target = diffusion.target(batch)
        mask = batch["mask"].to(device, non_blocking=True)
        ftp = source_phase(batch, "ftp", device)
        hilbert = source_phase(batch, "hilbert", device)
        pred_depth = None
        if diffusion.calib_depth_coef is not None and diffusion.target_channels >= 3:
            pred_raw_uph = diffusion.decode_uph_raw(batch, torch.clamp(pred[:, 2:3], 0.0, 1.0))
            cond_full = batch["cond"].to(device, non_blocking=True)
            pred_depth = calibrated_depth_from_uph(pred_raw_uph.float(), cond_full.float(), diffusion.calib_depth_coef.float())
            target_depth = batch["height_raw"].to(device, non_blocking=True).float()
        for j in range(pred.shape[0]):
            row = {"sample": len(rows)}
            row.update(phase_metrics_tensor(pred[j:j + 1], target[j:j + 1], mask=mask[j:j + 1]))
            if pred_depth is not None:
                row.update(depth_metrics_tensor(pred_depth[j:j + 1], target_depth[j:j + 1], mask=mask[j:j + 1]))
            row.update({f"ftp_{k}": v for k, v in phase_metrics_tensor(ftp[j:j + 1], target[j:j + 1], mask=mask[j:j + 1]).items()})
            row.update({f"hilbert_{k}": v for k, v in phase_metrics_tensor(hilbert[j:j + 1], target[j:j + 1], mask=mask[j:j + 1]).items()})
            rows.append(row)
            if save_images and out_dir is not None and row["sample"] < 8:
                one = {k: v[j:j + 1].detach().cpu() if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == pred.shape[0] else v for k, v in batch.items()}
                save_phase_visual(
                    one,
                    pred[j:j + 1],
                    out_dir / "visualizations" / f"{split_name}_{row['sample']:02d}.png",
                    title=f"E2 {split_name} MAE {row['phase_mae_rad']:.3f} rad",
                )
    return rows


def checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best_val, history):
    return {
        "epoch": ep,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": vars(args),
        "best_val_phase_mae_rad": best_val,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_phase_cache_960")
    parser.add_argument("--save_dir", default="/root/autodl-tmp/diffusion_fpp_v5/results/fpp960_e2_phase_diffusion")
    parser.add_argument("--phase_channels", default="0-12")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--train_crop", type=int, default=0)
    parser.add_argument("--train_epoch_repeats", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--ch_mult", default="1,2,4,8,8")
    parser.add_argument("--adapter_hidden", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target_channels", type=int, choices=[2, 3], default=2)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--sample_start_from", choices=["noise", "ftp", "hilbert"], default="ftp")
    parser.add_argument("--sample_start_ratio", type=float, default=0.7)
    parser.add_argument("--train_start_from", choices=["target", "ftp", "hilbert", "mix"], default="target")
    parser.add_argument("--train_t_min_ratio", type=float, default=0.0)
    parser.add_argument("--train_t_max_ratio", type=float, default=1.0)
    parser.add_argument("--phase_weight", type=float, default=1.0)
    parser.add_argument("--grad_weight", type=float, default=0.05)
    parser.add_argument("--unit_weight", type=float, default=0.02)
    parser.add_argument("--uph_weight", type=float, default=0.0)
    parser.add_argument("--uph_grad_weight", type=float, default=0.05)
    parser.add_argument(
        "--uph_start_from",
        choices=["coord_auto", "x", "y", "ftp_residual", "zero", "half", "target"],
        default="coord_auto",
    )
    parser.add_argument("--uph_norm", choices=["sample", "global"], default="sample")
    parser.add_argument("--uph_global_min", type=float, default=0.0)
    parser.add_argument("--uph_global_max", type=float, default=0.0)
    parser.add_argument("--uph_representation", choices=["absolute", "prior_residual"], default="absolute")
    parser.add_argument("--uph_prior_summary", default="")
    parser.add_argument("--uph_prior_basis", default="")
    parser.add_argument("--uph_residual_scale", type=float, default=0.0)
    parser.add_argument("--calib_depth_weight", type=float, default=0.0)
    parser.add_argument("--calib_depth_summary", default="")
    parser.add_argument("--calib_depth_source", default="gt_raw")
    parser.add_argument("--calib_depth_scale", type=float, default=100.0)
    parser.add_argument("--calib_depth_clip", type=float, default=300.0)
    parser.add_argument("--calib_depth_start_epoch", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument(
        "--selection_metric",
        choices=["phase_mae_rad", "phase_aligned_mae_rad", "phase_uph_score", "calib_depth_rmse_mm"],
        default="phase_mae_rad",
        help="Validation metric used for best_phase.pt. Raw phase is stricter; aligned phase measures restoration shape.",
    )
    parser.add_argument("--uph_select_weight", type=float, default=2.0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--guidance_weight", type=float, default=0.0)
    parser.add_argument("--guidance_source", choices=["ftp", "hilbert"], default="ftp")
    parser.add_argument("--guidance_apply_start_ratio", type=float, default=0.7)
    parser.add_argument("--guidance_grad_clip", type=float, default=0.05)
    parser.add_argument("--guidance_eta", type=float, default=1.0)
    parser.add_argument("--guidance_k", type=float, default=8.0)
    parser.add_argument("--guidance_tau", type=float, default=0.4)
    args = parser.parse_args()
    args.ch_mult_tuple = parse_ch_mult(args.ch_mult)
    args.uph_global_min, args.uph_global_max = resolve_uph_global_range(args)
    resolve_uph_prior_config(args)
    args.calib_depth_coef = load_calib_depth_coef(args)

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

    model = ConditionalUNetAdapter(
        in_channels=args.target_channels,
        cond_channels=cond_channels,
        out_channels=args.target_channels,
        base_ch=args.base_channels,
        ch_mult=args.ch_mult_tuple,
        dropout=args.dropout,
        adapter_hidden=args.adapter_hidden,
    ).to(device)
    diffusion = PhaseRestorationDiffusion(
        model,
        timesteps=args.timesteps,
        image_h=args.image_size,
        image_w=args.image_size,
        device=device,
        cond_indices=args.phase_channel_indices,
        phase_weight=args.phase_weight,
        grad_weight=args.grad_weight,
        unit_weight=args.unit_weight,
        target_channels=args.target_channels,
        uph_weight=args.uph_weight,
        uph_grad_weight=args.uph_grad_weight,
        uph_start_from=args.uph_start_from,
        uph_norm=args.uph_norm,
        uph_global_min=args.uph_global_min,
        uph_global_max=args.uph_global_max,
        calib_depth_weight=args.calib_depth_weight,
        calib_depth_coef=args.calib_depth_coef,
        calib_depth_scale=args.calib_depth_scale,
        calib_depth_clip=args.calib_depth_clip,
        calib_depth_start_epoch=args.calib_depth_start_epoch,
        uph_representation=args.uph_representation,
        uph_prior_coef=args.uph_prior_coef,
        uph_prior_basis=args.uph_prior_basis,
        uph_residual_scale=args.uph_residual_scale,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))

    print(f"Device: {device}")
    print(f"Phase channels: {args.phase_channel_indices}")
    print(f"Train {len(loaders['train'].dataset)} | Val {len(loaders['val'].dataset)} | Test {len(loaders['test'].dataset)}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Sampling: start={args.sample_start_from} ratio={args.sample_start_ratio} steps={args.ddim_steps}")
    print(
        f"Target channels: {args.target_channels} | phase_weight={args.phase_weight} | uph_weight={args.uph_weight} | "
        f"uph_start={args.uph_start_from} | uph_norm={args.uph_norm} "
        f"[{args.uph_global_min:.3f}, {args.uph_global_max:.3f}] | "
        f"uph_repr={args.uph_representation} scale={args.uph_residual_scale:.3f} | "
        f"calib_depth_weight={args.calib_depth_weight} "
        f"calib_depth_clip={args.calib_depth_clip} "
        f"calib_depth_start_epoch={args.calib_depth_start_epoch}"
    )

    history = []
    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        diffusion.current_epoch = ep
        total = 0.0
        seen = 0
        skipped = 0
        for batch in tqdm(loaders["train"], desc=f"phase diffusion {ep}/{args.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                loss = diffusion.p_loss(
                    batch,
                    t_min_ratio=args.train_t_min_ratio,
                    t_max_ratio=args.train_t_max_ratio,
                    train_start_from=args.train_start_from,
                )
            if not torch.isfinite(loss):
                skipped += 1
                continue
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item())
            seen += 1
            if args.max_train_batches and seen >= args.max_train_batches:
                break
        scheduler.step()
        log = {
            "epoch": ep,
            "train_loss": total / max(1, seen),
            "seen_batches": seen,
            "skipped_batches": skipped,
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - t0,
        }
        if ep == 1 or ep % args.eval_every == 0:
            val_rows = evaluate(diffusion, loaders["val"], device, args, "val")
            val_summary = summarize(val_rows)
            ftp_summary = summarize(val_rows, prefix="ftp_")
            hilbert_summary = summarize(val_rows, prefix="hilbert_")
            val_mae = validation_score(val_summary, args)
            log["val_selection_score"] = val_mae
            log.update({f"val_{k}": v["mean"] for k, v in val_summary.items()})
            log.update({f"ftp_val_{k[len('ftp_'):]}": v["mean"] for k, v in ftp_summary.items() if k.startswith("ftp_")})
            log.update({f"hilbert_val_{k[len('hilbert_'):]}": v["mean"] for k, v in hilbert_summary.items() if k.startswith("hilbert_")})
            if val_mae < best_val:
                best_val = val_mae
                torch.save(
                    checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best_val, history),
                    save_dir / "checkpoints" / "best_phase.pt",
                )
                first = next(iter(loaders["val"]))
                first_pred = diffusion.sample_ddim(
                    first,
                    steps=args.ddim_steps,
                    ensemble_size=args.ensemble,
                    start_from=args.sample_start_from,
                    start_ratio=args.sample_start_ratio,
                    progress=False,
                )
                save_phase_visual(
                    first,
                    first_pred,
                    save_dir / "visualizations" / f"val_ep{ep:03d}.png",
                    title=f"E2 val {args.selection_metric} {val_mae:.3f} rad",
                )
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        if args.save_every > 0 and (ep == 1 or ep == args.epochs or ep % args.save_every == 0):
            torch.save(
                checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best_val, history),
                save_dir / "checkpoints" / "latest.pt",
            )

    best_path = save_dir / "checkpoints" / "best_phase.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        for split in ("val", "test"):
            rows = evaluate(
                diffusion,
                loaders[split],
                device,
                args,
                split,
                out_dir=save_dir / "evaluation",
                save_images=True,
            )
            split_dir = save_dir / "evaluation" / split
            split_dir.mkdir(parents=True, exist_ok=True)
            save_rows(rows, split_dir / "per_sample_phase_metrics.csv")
            summary = summarize(rows)
            summary.update(summarize(rows, prefix="ftp_"))
            summary.update(summarize(rows, prefix="hilbert_"))
            summary["n"] = len(rows)
            summary["checkpoint"] = str(best_path)
            summary["args"] = vars(args)
            with open(split_dir / "phase_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"Final {split}:")
            print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
