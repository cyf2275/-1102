import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=timesteps.device, dtype=torch.float32) *
                      -(math.log(10000.0) / max(half - 1, 1)))
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _groups(ch):
    g = 32
    while g > 1 and ch % g != 0:
        g -= 1
    return g


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(t_emb)[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = torch.chunk(self.qkv(self.norm(x)), 3, dim=1)
        q = q.reshape(b, c, h * w).transpose(1, 2)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).transpose(1, 2)
        attn = torch.softmax((q @ k) * (c ** -0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class ConditionalUNet(nn.Module):
    """x0-pred full-height diffusion U-Net with configurable condition channels."""

    def __init__(self, in_channels=1, cond_channels=8, out_channels=1,
                 base_ch=48, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
                 dropout=0.05, time_emb_dim=256):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        time_dim = time_emb_dim * 4
        self.conv_in = nn.Conv2d(in_channels + cond_channels, base_ch, 3, padding=1)
        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_ch
        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                blocks.append(ResBlock(ch if j == 0 else out_ch, out_ch, time_dim, dropout))
            self.down_blocks.append(blocks)
            self.downsamples.append(
                nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
                if i < len(ch_mult) - 1 else None
            )
            ch = out_ch

        self.mid1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid2 = ResBlock(ch, ch, time_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in reversed(range(len(ch_mult))):
            out_ch = base_ch * ch_mult[i]
            blocks = nn.ModuleList()
            blocks.append(ResBlock(ch + out_ch, out_ch, time_dim, dropout))
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock(out_ch, out_ch, time_dim, dropout))
            self.up_blocks.append(blocks)
            self.upsamples.append(
                nn.ConvTranspose2d(ch, ch, 3, stride=2, padding=1, output_padding=1)
                if i > 0 else None
            )
            ch = out_ch

        self.norm_out = nn.GroupNorm(_groups(ch), ch)
        self.out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x, t, cond):
        t_emb = self.time_embed(get_timestep_embedding(t, self.time_emb_dim))
        h = self.conv_in(torch.cat([x, cond], dim=1))

        skips = []
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)
            if self.downsamples[i] is not None:
                h = self.downsamples[i](h)

        h = self.mid2(self.mid_attn(self.mid1(h, t_emb)), t_emb)

        for i, blocks in enumerate(self.up_blocks):
            if self.upsamples[i] is not None:
                h = self.upsamples[i](h)
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, t_emb)

        return self.out(F.silu(self.norm_out(h)))


class ZeroCondAdapter(nn.Module):
    """Zero-initialized 1x1 condition adapter for diffusion U-Nets."""

    def __init__(self, cond_channels, out_channels, hidden_channels=32):
        super().__init__()
        hidden_channels = max(8, min(int(hidden_channels), max(8, out_channels // 4)))
        self.net = nn.Sequential(
            nn.Conv2d(cond_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond, size):
        if cond.shape[-2:] != size:
            cond = F.interpolate(cond, size=size, mode="bilinear", align_corners=False)
        return self.net(cond)


class ConditionalUNetAdapter(nn.Module):
    """x0-pred diffusion U-Net with zero adapters instead of input concat."""

    def __init__(self, in_channels=1, cond_channels=8, out_channels=1,
                 base_ch=48, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
                 dropout=0.05, time_emb_dim=256, adapter_hidden=32):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        time_dim = time_emb_dim * 4
        self.conv_in = nn.Conv2d(in_channels, base_ch, 3, padding=1)
        self.adapter_in = ZeroCondAdapter(cond_channels, base_ch, adapter_hidden)
        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.down_adapters = nn.ModuleList()
        ch = base_ch
        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                blocks.append(ResBlock(ch if j == 0 else out_ch, out_ch, time_dim, dropout))
            self.down_blocks.append(blocks)
            self.down_adapters.append(ZeroCondAdapter(cond_channels, out_ch, adapter_hidden))
            self.downsamples.append(
                nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
                if i < len(ch_mult) - 1 else None
            )
            ch = out_ch

        self.mid1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid2 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_adapter = ZeroCondAdapter(cond_channels, ch, adapter_hidden)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.up_adapters = nn.ModuleList()
        for i in reversed(range(len(ch_mult))):
            out_ch = base_ch * ch_mult[i]
            blocks = nn.ModuleList()
            blocks.append(ResBlock(ch + out_ch, out_ch, time_dim, dropout))
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock(out_ch, out_ch, time_dim, dropout))
            self.up_blocks.append(blocks)
            self.up_adapters.append(ZeroCondAdapter(cond_channels, out_ch, adapter_hidden))
            self.upsamples.append(
                nn.ConvTranspose2d(ch, ch, 3, stride=2, padding=1, output_padding=1)
                if i > 0 else None
            )
            ch = out_ch

        self.norm_out = nn.GroupNorm(_groups(ch), ch)
        self.out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x, t, cond):
        t_emb = self.time_embed(get_timestep_embedding(t, self.time_emb_dim))
        h = self.conv_in(x)
        h = h + self.adapter_in(cond, h.shape[-2:])

        skips = []
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, t_emb)
            h = h + self.down_adapters[i](cond, h.shape[-2:])
            skips.append(h)
            if self.downsamples[i] is not None:
                h = self.downsamples[i](h)

        h = self.mid2(self.mid_attn(self.mid1(h, t_emb)), t_emb)
        h = h + self.mid_adapter(cond, h.shape[-2:])

        for i, blocks in enumerate(self.up_blocks):
            if self.upsamples[i] is not None:
                h = self.upsamples[i](h)
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, t_emb)
            h = h + self.up_adapters[i](cond, h.shape[-2:])

        return self.out(F.silu(self.norm_out(h)))
