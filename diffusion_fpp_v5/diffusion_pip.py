from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def masked_mean(x, mask=None):
    if mask is None:
        return x.mean()
    mask = torch.clamp(mask.to(device=x.device, dtype=x.dtype), 0.0, 1.0)
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def weighted_mean(x, weight=None):
    if weight is None:
        return x.mean()
    weight = weight.to(device=x.device, dtype=x.dtype)
    return (x * weight).sum() / weight.sum().clamp(min=1e-6)


def charbonnier(pred, target, eps=1e-3, mask=None):
    return masked_mean(torch.sqrt((pred - target) ** 2 + eps * eps), mask=mask)


def masked_mse(pred, target, mask=None):
    return masked_mean((pred - target) ** 2, mask=mask)


def weighted_charbonnier(pred, target, weight=None, eps=1e-3):
    return weighted_mean(torch.sqrt((pred - target) ** 2 + eps * eps), weight=weight)


def weighted_mse(pred, target, weight=None):
    return weighted_mean((pred - target) ** 2, weight=weight)


def grad_xy_padded(x):
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return dx, dy


def normal_loss(pred, target, mask=None):
    pdx, pdy = grad_xy_padded(pred)
    tdx, tdy = grad_xy_padded(target)
    pn = torch.cat([-pdx, -pdy, torch.ones_like(pred)], dim=1)
    tn = torch.cat([-tdx, -tdy, torch.ones_like(target)], dim=1)
    pn = F.normalize(pn, dim=1)
    tn = F.normalize(tn, dim=1)
    err = 1.0 - (pn * tn).sum(dim=1, keepdim=True)
    return masked_mean(err, mask=mask)


def oriented_gradient_loss(pred, target, phase_sin, phase_cos, phase_conf, conf_thresh=0.05, mask=None):
    dsx, dsy = grad_xy_padded(phase_sin)
    dcx, dcy = grad_xy_padded(phase_cos)
    # dphi = cos(phi) d sin(phi) - sin(phi) d cos(phi), avoiding wrap jumps.
    gx = phase_cos * dsx - phase_sin * dcx
    gy = phase_cos * dsy - phase_sin * dcy
    norm = torch.sqrt(gx * gx + gy * gy).clamp(min=1e-6)
    nx, ny = gx / norm, gy / norm
    tx, ty = -ny, nx
    pdx, pdy = grad_xy_padded(pred)
    tdx, tdy = grad_xy_padded(target)
    p_n = pdx * nx + pdy * ny
    t_n = tdx * nx + tdy * ny
    p_t = pdx * tx + pdy * ty
    t_t = tdx * tx + tdy * ty
    weight = (phase_conf > conf_thresh).float()
    if mask is not None:
        weight = weight * torch.clamp(mask.to(device=weight.device, dtype=weight.dtype), 0.0, 1.0)
    denom = weight.sum().clamp(min=1.0)
    loss_n = (torch.abs(p_n - t_n) * weight).sum() / denom
    loss_t = (torch.abs(p_t - t_t) * weight).sum() / denom
    return loss_n + 0.3 * loss_t


def confidence_edge_loss(pred, target, edge_score, phase_conf, mask=None):
    weight = torch.clamp(edge_score, 0.0, 1.0) * torch.clamp(phase_conf, 0.0, 1.0)
    if mask is not None:
        weight = weight * torch.clamp(mask.to(device=weight.device, dtype=weight.dtype), 0.0, 1.0)
    denom = weight.sum().clamp(min=1.0)
    return (torch.abs(pred - target) * weight).sum() / denom


def gaussian_blur_3x3(x):
    kernel = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                          dtype=x.dtype, device=x.device)
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=x.shape[1])


def normalize01_per_sample(x):
    flat = x.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return torch.clamp((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)


class PIPDiffusion:
    def __init__(
        self,
        model,
        timesteps=200,
        image_h=480,
        image_w=640,
        device="cuda",
        phase_head=None,
        lambda_oriented=0.08,
        lambda_edge=0.03,
        lambda_normal=0.01,
        lambda_phase=0.0,
        cond_indices=None,
        target_mode="full_x0",
        residual_scale=1.0,
        base_residual_gate=1.0,
        train_start_from_base=False,
        train_t_min_ratio=0.0,
        train_t_max_ratio=1.0,
        base_error_loss_weight=0.0,
        base_error_loss_gamma=1.0,
        low_edge_loss_weight=0.0,
        low_edge_threshold=0.467,
        blend_loss_alpha=0.0,
    ):
        self.model = model
        self.timesteps = int(timesteps)
        self.image_h = int(image_h)
        self.image_w = int(image_w)
        self.device = device
        self.phase_head = phase_head
        self.lambda_oriented = float(lambda_oriented)
        self.lambda_edge = float(lambda_edge)
        self.lambda_normal = float(lambda_normal)
        self.lambda_phase = float(lambda_phase)
        self.cond_indices = list(cond_indices) if cond_indices is not None else None
        self.target_mode = str(target_mode)
        self.residual_scale = float(residual_scale)
        self.base_residual_gate = float(base_residual_gate)
        self.train_start_from_base = bool(train_start_from_base)
        self.train_t_min_ratio = float(train_t_min_ratio)
        self.train_t_max_ratio = float(train_t_max_ratio)
        self.base_error_loss_weight = float(base_error_loss_weight)
        self.base_error_loss_gamma = float(base_error_loss_gamma)
        self.low_edge_loss_weight = float(low_edge_loss_weight)
        self.low_edge_threshold = float(low_edge_threshold)
        self.blend_loss_alpha = float(blend_loss_alpha)
        if self.target_mode not in {"full_x0", "residual", "base_residual"}:
            raise ValueError(f"unknown target_mode: {self.target_mode}")
        if self.target_mode in {"residual", "base_residual"} and self.residual_scale <= 0:
            raise ValueError("residual_scale must be > 0 for residual target modes")
        if self.base_residual_gate <= 0:
            raise ValueError("base_residual_gate must be > 0")
        if self.train_start_from_base and self.target_mode not in {"full_x0", "base_residual"}:
            raise ValueError("train_start_from_base is only supported for full_x0/base_residual target modes")
        if self.base_error_loss_weight < 0 or self.low_edge_loss_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.base_error_loss_gamma <= 0:
            raise ValueError("base_error_loss_gamma must be > 0")
        if self.blend_loss_alpha < 0:
            raise ValueError("blend_loss_alpha must be non-negative")
        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def register(self, name, tensor):
        setattr(self, name, tensor.to(self.device))

    def condition(self, batch):
        cond = batch["cond"].to(self.device, non_blocking=True)
        if self.cond_indices is not None:
            cond = cond[:, self.cond_indices]
        return cond

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sa = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        so = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sa * x_start + so * noise

    def base_height(self, batch):
        if "base_height" not in batch:
            raise KeyError("base-conditioned target mode requires batch['base_height']")
        return torch.clamp(batch["base_height"].to(self.device, non_blocking=True), -1.0, 1.0)

    def effective_residual_scale(self):
        if self.target_mode == "base_residual":
            return self.residual_scale * self.base_residual_gate
        return self.residual_scale

    def residual_target(self, target, base, scale=None):
        scale = self.residual_scale if scale is None else float(scale)
        return torch.clamp((target - base) / scale, -1.0, 1.0)

    def residual_to_depth(self, residual, base, scale=None):
        scale = self.residual_scale if scale is None else float(scale)
        return torch.clamp(base + residual * scale, -1.0, 1.0)

    def base_residual_loss_weight(self, x_start, base, batch, valid_mask=None):
        if base is None or (self.base_error_loss_weight <= 0 and self.low_edge_loss_weight <= 0):
            return valid_mask
        weight = torch.ones_like(x_start)
        if self.base_error_loss_weight > 0:
            # Weight the training residual where the deterministic base is already wrong.
            scale = max(float(self.residual_scale), 1e-6)
            err = torch.clamp(torch.abs(x_start - base) / scale, 0.0, 1.0)
            if abs(self.base_error_loss_gamma - 1.0) > 1e-6:
                err = torch.pow(err, self.base_error_loss_gamma)
            weight = weight * (1.0 + self.base_error_loss_weight * err)
        if self.low_edge_loss_weight > 0:
            edge = torch.clamp(batch["edge_score"].to(self.device, non_blocking=True), 0.0, 1.0)
            threshold = max(self.low_edge_threshold, 1e-6)
            if self.low_edge_threshold < 1.0:
                low_edge = torch.clamp((threshold - edge) / threshold, 0.0, 1.0)
            else:
                low_edge = 1.0 - edge
            weight = weight * (1.0 + self.low_edge_loss_weight * low_edge)
        if valid_mask is not None:
            weight = weight * torch.clamp(valid_mask.to(device=weight.device, dtype=weight.dtype), 0.0, 1.0)
        return weight

    def phase_consistency_loss(self, pred, batch):
        if self.phase_head is None or self.lambda_phase <= 0:
            return pred.new_tensor(0.0)
        xy = batch["xy"].to(pred.device, non_blocking=True)
        phase_target = torch.cat([
            batch["phase_sin"].to(pred.device, non_blocking=True),
            batch["phase_cos"].to(pred.device, non_blocking=True),
        ], dim=1)
        phase_pred = self.phase_head(self.phase_head_depth_input(pred, batch), xy)
        conf = batch["phase_conf"].to(pred.device, non_blocking=True)
        if "mask" in batch:
            conf = conf * torch.clamp(batch["mask"].to(pred.device, non_blocking=True), 0.0, 1.0)
        err = torch.abs(phase_pred - phase_target).sum(dim=1, keepdim=True)
        return (err * conf).sum() / conf.sum().clamp(min=1.0)

    def phase_head_depth_input(self, depth_norm, batch):
        mode = str(getattr(self.phase_head, "phase_depth_input", "height_norm"))
        if mode == "height_norm":
            return depth_norm
        if mode == "depth01":
            return torch.clamp((depth_norm + 1.0) * 0.5, 0.0, 1.0) * 2.0 - 1.0
        if mode == "raw_mm":
            if "depth_minmax" not in batch:
                return depth_norm
            minmax = batch["depth_minmax"].to(depth_norm.device, non_blocking=True)
            dmin = minmax[:, 0].view(-1, 1, 1, 1)
            dmax = minmax[:, 1].view(-1, 1, 1, 1)
            depth01 = torch.clamp((depth_norm + 1.0) * 0.5, 0.0, 1.0)
            depth_mm = depth01 * (dmax - dmin).clamp(min=1e-6) + dmin
            center = float(getattr(self.phase_head, "raw_depth_center", 0.0))
            scale = max(float(getattr(self.phase_head, "raw_depth_scale", 1.0)), 1e-6)
            return torch.clamp((depth_mm - center) / scale, -2.0, 2.0)
        return depth_norm

    def p_loss(self, batch):
        x_start = batch["height"].to(self.device, non_blocking=True)
        cond = self.condition(batch)
        base = None
        diffusion_target = x_start
        if self.target_mode in {"residual", "base_residual"}:
            base = self.base_height(batch)
            diffusion_target = self.residual_target(x_start, base, self.effective_residual_scale())
        valid_mask = batch.get("mask")
        if valid_mask is not None:
            valid_mask = torch.clamp(valid_mask.to(self.device, non_blocking=True), 0.0, 1.0)
        b = x_start.shape[0]
        t_min = max(0, min(self.timesteps - 1, int((self.timesteps - 1) * self.train_t_min_ratio)))
        t_max = max(t_min, min(self.timesteps - 1, int((self.timesteps - 1) * self.train_t_max_ratio)))
        t = torch.randint(t_min, t_max + 1, (b,), device=self.device, dtype=torch.long)
        if self.target_mode == "base_residual":
            x_t = self.q_sample(base, t)
        elif self.train_start_from_base:
            train_base = torch.clamp(batch["base_height"].to(self.device, non_blocking=True), -1.0, 1.0)
            x_t = self.q_sample(train_base, t)
        else:
            x_t = self.q_sample(diffusion_target, t)
        pred_target = torch.clamp(self.model(x_t, t, cond), -1.0, 1.0)
        if self.target_mode in {"residual", "base_residual"}:
            x0 = self.residual_to_depth(pred_target, base, self.effective_residual_scale())
            supervised_depth = x0
            if self.blend_loss_alpha > 0:
                alpha = min(float(self.blend_loss_alpha), 1.0)
                supervised_depth = torch.clamp(base + alpha * (x0 - base), -1.0, 1.0)
            loss_weight = self.base_residual_loss_weight(x_start, base, batch, valid_mask=valid_mask)
            loss = weighted_charbonnier(pred_target, diffusion_target, weight=loss_weight)
            loss = loss + 0.5 * weighted_mse(pred_target, diffusion_target, weight=loss_weight)
            loss = loss + 0.5 * weighted_charbonnier(supervised_depth, x_start, weight=loss_weight)
        else:
            x0 = pred_target
            supervised_depth = x0
            loss_weight = valid_mask
            if (
                self.train_start_from_base
                and "base_height" in batch
                and (self.base_error_loss_weight > 0 or self.low_edge_loss_weight > 0)
            ):
                train_base = torch.clamp(batch["base_height"].to(self.device, non_blocking=True), -1.0, 1.0)
                loss_weight = self.base_residual_loss_weight(x_start, train_base, batch, valid_mask=valid_mask)
            loss = weighted_charbonnier(x0, x_start, weight=loss_weight)
            loss = loss + 0.5 * weighted_mse(x0, x_start, weight=loss_weight)
        if self.lambda_oriented > 0:
            loss = loss + self.lambda_oriented * oriented_gradient_loss(
                supervised_depth, x_start,
                batch["phase_sin"].to(self.device, non_blocking=True),
                batch["phase_cos"].to(self.device, non_blocking=True),
                batch["phase_conf"].to(self.device, non_blocking=True),
                mask=valid_mask,
            )
        if self.lambda_edge > 0:
            loss = loss + self.lambda_edge * confidence_edge_loss(
                supervised_depth, x_start,
                batch["edge_score"].to(self.device, non_blocking=True),
                batch["phase_conf"].to(self.device, non_blocking=True),
                mask=valid_mask,
            )
        if self.lambda_normal > 0:
            loss = loss + self.lambda_normal * normal_loss(supervised_depth, x_start, mask=valid_mask)
        if self.lambda_phase > 0 and self.phase_head is not None:
            loss = loss + self.lambda_phase * self.phase_consistency_loss(supervised_depth, batch)
        return loss

    @torch.no_grad()
    def sample_ddim(
        self,
        batch,
        steps=50,
        ensemble_size=1,
        guidance=None,
        progress=False,
        start_from_base=False,
        start_ratio=1.0,
    ):
        if ensemble_size > 1:
            preds = []
            for i in range(ensemble_size):
                preds.append(self._sample_single(
                    batch,
                    steps=steps,
                    seed=i * 13,
                    guidance=guidance,
                    progress=False,
                    start_from_base=start_from_base,
                    start_ratio=start_ratio,
                ))
            return torch.median(torch.stack(preds, dim=0), dim=0).values
        return self._sample_single(
            batch,
            steps=steps,
            seed=0,
            guidance=guidance,
            progress=progress,
            start_from_base=start_from_base,
            start_ratio=start_ratio,
        )

    def _apply_posterior_guidance(self, x0, batch, progress, guidance):
        if not guidance or self.phase_head is None:
            return x0
        if progress < float(guidance.get("apply_start_ratio", 0.7)):
            return x0
        lam = float(guidance.get("weight", 0.0))
        if lam <= 0:
            return x0
        grad_clip = float(guidance.get("grad_clip", 0.05))
        eta = float(guidance.get("eta", 1.0))
        k = float(guidance.get("k", 8.0))
        tau = float(guidance.get("tau", 0.4))

        xy = batch["xy"].to(self.device, non_blocking=True)
        phase_target = torch.cat([
            batch["phase_sin"].to(self.device, non_blocking=True),
            batch["phase_cos"].to(self.device, non_blocking=True),
        ], dim=1)
        phase_conf = torch.clamp(batch["phase_conf"].to(self.device, non_blocking=True), 0.0, 1.0)
        edge_score = normalize01_per_sample(batch["edge_score"].to(self.device, non_blocking=True))
        with torch.enable_grad():
            z = x0.detach().requires_grad_(True)
            phase_pred = self.phase_head(self.phase_head_depth_input(z, batch), xy)
            err = torch.abs(phase_pred - phase_target).sum(dim=1, keepdim=True)
            energy = (err * phase_conf).mean()
            grad = torch.autograd.grad(energy, z, retain_graph=False, create_graph=False)[0]
        grad_blur = gaussian_blur_3x3(grad)
        w = torch.pow(phase_conf, eta) * torch.sigmoid(k * (edge_score - tau))
        grad_final = (1.0 - w) * grad_blur + w * grad
        norm = grad_final.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp(min=1e-8)
        scale = torch.clamp(grad_clip / norm, max=1.0)
        grad_final = grad_final * scale
        return torch.clamp(x0 - lam * phase_conf * grad_final, -1.0, 1.0)

    def _sample_single(
        self,
        batch,
        steps=50,
        seed=0,
        guidance=None,
        progress=False,
        start_from_base=False,
        start_ratio=1.0,
    ):
        cond = self.condition(batch)
        b = cond.shape[0]
        base = self.base_height(batch) if self.target_mode in {"residual", "base_residual"} else None
        gen = torch.Generator(device=self.device).manual_seed(seed)
        start_t = self.timesteps - 1
        if start_from_base or self.target_mode == "base_residual":
            if "base_height" not in batch:
                raise KeyError("start_from_base=True requires batch['base_height']")
            start_base = base if base is not None else torch.clamp(batch["base_height"].to(self.device, non_blocking=True), -1.0, 1.0)
            start_t = max(1, min(self.timesteps - 1, int(self.timesteps * float(start_ratio))))
            t0 = torch.full((b,), start_t, device=self.device, dtype=torch.long)
            x = self.q_sample(
                start_base,
                t0,
                noise=torch.randn(start_base.shape, device=self.device, generator=gen),
            )
        else:
            x = torch.randn((b, 1, self.image_h, self.image_w), device=self.device, generator=gen)
        n_steps = max(2, min(int(steps), start_t + 1))
        times = torch.linspace(start_t, 0, n_steps, device=self.device).round().long().unique_consecutive()
        if times[-1] != 0:
            times = torch.cat([times, torch.zeros(1, device=self.device, dtype=torch.long)])
        iterator = range(len(times) - 1)
        if progress:
            iterator = tqdm(iterator, desc=f"PIP DDIM {len(times)-1} steps")
        for i in iterator:
            t = times[i]
            t_next = times[i + 1]
            tb = torch.full((b,), int(t.item()), device=self.device, dtype=torch.long)
            with torch.no_grad():
                pred_target = torch.clamp(self.model(x, tb, cond), -1.0, 1.0)
                x0 = self.residual_to_depth(pred_target, base, self.effective_residual_scale()) if self.target_mode in {"residual", "base_residual"} else pred_target
            denoise_progress = float(i + 1) / float(max(1, len(times) - 1))
            x0 = self._apply_posterior_guidance(x0, batch, denoise_progress, guidance)
            model_target = self.residual_target(x0, base, self.effective_residual_scale()) if self.target_mode == "residual" else x0
            alpha = self.alphas_cumprod[t]
            alpha_next = self.alphas_cumprod[t_next]
            eps = (x - alpha.sqrt() * model_target) / (1 - alpha).sqrt().clamp(min=1e-8)
            x = alpha_next.sqrt() * model_target + (1 - alpha_next).sqrt() * eps
        tb = torch.zeros((b,), device=self.device, dtype=torch.long)
        with torch.no_grad():
            pred_target = torch.clamp(self.model(x, tb, cond), -1.0, 1.0)
            x0 = self.residual_to_depth(pred_target, base, self.effective_residual_scale()) if self.target_mode in {"residual", "base_residual"} else pred_target
        return self._apply_posterior_guidance(x0, batch, 1.0, guidance)
