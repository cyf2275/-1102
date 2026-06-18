from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointwisePhaseProjectionHead(nn.Module):
    """Depth-to-phase projection head with no spatial receptive field.

    Inputs must be limited to D, x, y. Do not feed fringe-derived phase
    instructions here, otherwise the head can learn an identity shortcut.
    """

    def __init__(self, hidden_dim=64, num_layers=4):
        super().__init__()
        hidden_dim = min(int(hidden_dim), 64)
        num_layers = max(3, min(int(num_layers), 4))
        layers = []
        in_ch = 3
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(in_ch, hidden_dim, 1))
            layers.append(nn.SiLU())
            in_ch = hidden_dim
        layers.append(nn.Conv2d(in_ch, 2, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, depth_norm, xy):
        """Return normalized sin/cos phase predictions.

        depth_norm: Bx1xHxW, typically normalized to [-1, 1]
        xy: Bx2xHxW containing x/y coordinate maps
        """
        out = self.net(torch.cat([depth_norm, xy], dim=1))
        return F.normalize(out, dim=1, eps=1e-6)


class CoarseLowpassNet(nn.Module):
    """Lightweight low-pass depth and uncertainty predictor.

    This branch predicts low-frequency structure only. Its uncertainty output
    is trained against low-pass error, not full-resolution edge error.
    """

    def __init__(self, in_channels, base_ch=32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
        )
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            nn.GroupNorm(8, base_ch * 2),
            nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.SiLU(),
        )
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)
        self.mid = nn.Sequential(
            nn.GroupNorm(8, base_ch * 4),
            nn.SiLU(),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.SiLU(),
        )
        self.up2 = nn.Conv2d(base_ch * 4 + base_ch * 2, base_ch * 2, 3, padding=1)
        self.up1 = nn.Conv2d(base_ch * 2 + base_ch, base_ch, 3, padding=1)
        self.out = nn.Conv2d(base_ch, 2, 3, padding=1)

    def forward(self, cond):
        e1 = self.enc1(cond)
        e2 = self.enc2(self.down1(e1))
        mid = self.mid(self.down2(e2))
        h = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        h = F.silu(self.up2(torch.cat([h, e2], dim=1)))
        h = F.interpolate(h, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        h = F.silu(self.up1(torch.cat([h, e1], dim=1)))
        out = self.out(h)
        depth_low = torch.tanh(out[:, :1])
        log_var = torch.clamp(out[:, 1:2], -6.0, 4.0)
        return depth_low, log_var


def heteroscedastic_l1(pred, target, log_var):
    abs_err = torch.abs(pred - target)
    return (torch.exp(-log_var) * abs_err + log_var).mean()
