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


def gradient_xy(x):
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx, dy


def gradient_loss(pred, target):
    pdx, pdy = gradient_xy(pred)
    tdx, tdy = gradient_xy(target)
    return F.l1_loss(pdx, tdx) + F.l1_loss(pdy, tdy)


def edge_weighted_loss(pred, target, quantile=0.80):
    with torch.no_grad():
        dx, dy = gradient_xy(target)
        mag = F.pad(torch.sqrt(dx[..., :-1, :] * dx[..., :-1, :] + dy[..., :, :-1] * dy[..., :, :-1]), (0, 1, 0, 1))
        flat = mag.flatten(1)
        thresh = torch.quantile(flat, quantile, dim=1).view(-1, 1, 1, 1)
        mask = mag >= thresh
    if mask.any():
        return F.l1_loss(pred[mask], target[mask])
    return F.l1_loss(pred, target)


def normal_loss(pred, target):
    pdx, pdy = gradient_xy(pred)
    tdx, tdy = gradient_xy(target)
    pdx = F.pad(pdx, (0, 1, 0, 0))
    tdx = F.pad(tdx, (0, 1, 0, 0))
    pdy = F.pad(pdy, (0, 0, 0, 1))
    tdy = F.pad(tdy, (0, 0, 0, 1))
    pn = torch.cat([-pdx, -pdy, torch.ones_like(pred)], dim=1)
    tn = torch.cat([-tdx, -tdy, torch.ones_like(target)], dim=1)
    pn = F.normalize(pn, dim=1)
    tn = F.normalize(tn, dim=1)
    return 1.0 - (pn * tn).sum(dim=1).mean()


class PhaseEdgeDiffusion:
    """v3-style full-height x0 diffusion with phase/edge conditions."""

    def __init__(self, model, timesteps=200, image_h=480, image_w=640, device="cuda",
                 lambda_grad=0.2, lambda_edge=0.15, lambda_normal=0.05):
        self.model = model
        self.timesteps = int(timesteps)
        self.image_h = int(image_h)
        self.image_w = int(image_w)
        self.device = device
        self.lambda_grad = float(lambda_grad)
        self.lambda_edge = float(lambda_edge)
        self.lambda_normal = float(lambda_normal)
        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def register(self, name, tensor):
        setattr(self, name, tensor.to(self.device))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sa = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        so = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sa * x_start + so * noise

    def p_loss(self, x_start, cond):
        b = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.device, dtype=torch.long)
        x_t = self.q_sample(x_start, t)
        x0 = self.model(x_t, t, cond)
        loss = F.l1_loss(x0, x_start) + 0.5 * F.mse_loss(x0, x_start)
        if self.lambda_grad > 0:
            loss = loss + self.lambda_grad * gradient_loss(x0, x_start)
        if self.lambda_edge > 0:
            loss = loss + self.lambda_edge * edge_weighted_loss(x0, x_start)
        if self.lambda_normal > 0:
            loss = loss + self.lambda_normal * normal_loss(x0, x_start)
        return loss

    @torch.no_grad()
    def sample_ddim(self, cond, steps=50, ensemble_size=1, progress=False):
        if ensemble_size > 1:
            preds = []
            for i in range(ensemble_size):
                preds.append(self._sample_single(cond, steps=steps, seed=i * 13, progress=False))
            return torch.median(torch.stack(preds, dim=0), dim=0).values
        return self._sample_single(cond, steps=steps, seed=0, progress=progress)

    def _sample_single(self, cond, steps=50, seed=0, progress=False):
        b = cond.shape[0]
        gen = torch.Generator(device=self.device).manual_seed(seed)
        x = torch.randn((b, 1, self.image_h, self.image_w), device=self.device, generator=gen)
        stride = max(1, self.timesteps // int(steps))
        times = torch.arange(self.timesteps - 1, -1, -stride, device=self.device).long()
        if times[-1] != 0:
            times = torch.cat([times, torch.zeros(1, device=self.device, dtype=torch.long)])
        iterator = range(len(times) - 1)
        if progress:
            iterator = tqdm(iterator, desc=f"DDIM {len(times)-1} steps")
        for i in iterator:
            t = times[i]
            t_next = times[i + 1]
            tb = torch.full((b,), int(t.item()), device=self.device, dtype=torch.long)
            x0 = torch.clamp(self.model(x, tb, cond), -1.0, 1.0)
            alpha = self.alphas_cumprod[t]
            alpha_next = self.alphas_cumprod[t_next]
            eps = (x - alpha.sqrt() * x0) / (1 - alpha).sqrt().clamp(min=1e-8)
            x = alpha_next.sqrt() * x0 + (1 - alpha_next).sqrt() * eps
        tb = torch.zeros((b,), device=self.device, dtype=torch.long)
        return torch.clamp(self.model(x, tb, cond), -1.0, 1.0)
