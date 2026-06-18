import math

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


def gradient_loss(pred, target):
    pdx = pred[..., :, 1:] - pred[..., :, :-1]
    tdx = target[..., :, 1:] - target[..., :, :-1]
    pdy = pred[..., 1:, :] - pred[..., :-1, :]
    tdy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pdx, tdx) + F.l1_loss(pdy, tdy)


def normalized_fft_loss(pred, target):
    pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
    target_fft = torch.fft.rfft2(target.float(), norm="ortho")
    pred_mag = torch.log1p(torch.abs(pred_fft))
    target_mag = torch.log1p(torch.abs(target_fft))
    return F.l1_loss(pred_mag, target_mag)


class PhysicsConditionedDiffusion:
    """Full-height x0-pred diffusion with fringe-only physics conditions."""

    def __init__(self, model, timesteps=200, image_h=480, image_w=640, device="cuda",
                 lambda_grad=0.2, lambda_fft=0.05):
        self.model = model
        self.timesteps = int(timesteps)
        self.image_h = int(image_h)
        self.image_w = int(image_w)
        self.device = device
        self.lambda_grad = float(lambda_grad)
        self.lambda_fft = float(lambda_fft)

        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register("betas", betas)
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
        if self.lambda_fft > 0:
            fft = normalized_fft_loss(x0, x_start)
            if torch.isfinite(fft):
                loss = loss + self.lambda_fft * fft
        return loss

    @torch.no_grad()
    def sample_ddim(self, cond, steps=50, ensemble_size=1, coarse=None,
                    start_ratio=0.55, progress=False):
        if ensemble_size > 1:
            preds = []
            for i in range(ensemble_size):
                preds.append(self._sample_single(cond, steps, coarse, start_ratio, seed=i * 13, progress=False))
            return torch.median(torch.stack(preds, dim=0), dim=0).values
        return self._sample_single(cond, steps, coarse, start_ratio, seed=0, progress=progress)

    def _sample_single(self, cond, steps, coarse, start_ratio, seed=0, progress=False):
        b = cond.shape[0]
        gen = torch.Generator(device=self.device).manual_seed(seed)
        if coarse is not None:
            start_t = max(1, min(self.timesteps - 1, int(self.timesteps * float(start_ratio))))
            t0 = torch.full((b,), start_t, device=self.device, dtype=torch.long)
            x = self.q_sample(torch.clamp(coarse, -1.0, 1.0), t0,
                              noise=torch.randn(coarse.shape, device=self.device, generator=gen))
        else:
            start_t = self.timesteps - 1
            x = torch.randn((b, 1, self.image_h, self.image_w), device=self.device, generator=gen)

        n_steps = max(2, min(int(steps), start_t + 1))
        times = torch.linspace(start_t, 0, n_steps, device=self.device).round().long().unique_consecutive()
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

