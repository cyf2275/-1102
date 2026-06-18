"""Single-frame physics features for v3.5.

Feature order:
0 raw fringe
1 Hilbert wrapped phase sin
2 Hilbert wrapped phase cos
3 row-wise detrended unwrapped Hilbert phase residual
4 Haar-DWT high-frequency energy
5 fringe gradient magnitude
6 x coordinate
7 y coordinate
"""
from __future__ import annotations

import numpy as np


EPS = 1e-6


def robust_unit(x: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    lo, hi = np.percentile(x, [lo_q, hi_q])
    return np.clip((x - lo) / (hi - lo + EPS), 0.0, 1.0).astype(np.float32)


def zscore_clip(x: np.ndarray, clip: float = 3.0) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    z = (x - float(x.mean())) / (float(x.std()) + EPS)
    z = np.clip(z, -clip, clip) / clip
    return z.astype(np.float32)


def analytic_signal_x(fringe_hw: np.ndarray) -> np.ndarray:
    n = fringe_hw.shape[-1]
    spectrum = np.fft.fft(fringe_hw, axis=-1)
    h = np.zeros(n, dtype=np.float32)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * h[None, :], axis=-1)


def detrend_phase_x(phase_hw: np.ndarray) -> np.ndarray:
    """Remove a row-wise linear carrier trend from unwrapped phase."""
    h, w = phase_hw.shape
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    xc = x - float(x.mean())
    denom = float(np.sum(xc * xc)) + EPS
    row_mean = phase_hw.mean(axis=1, keepdims=True)
    slope = ((phase_hw - row_mean) * xc[None, :]).sum(axis=1, keepdims=True) / denom
    trend = row_mean + slope * xc[None, :]
    return (phase_hw - trend).astype(np.float32)


def gradient_magnitude(fringe_hw: np.ndarray) -> np.ndarray:
    dx = np.zeros_like(fringe_hw, dtype=np.float32)
    dy = np.zeros_like(fringe_hw, dtype=np.float32)
    dx[:, 1:] = fringe_hw[:, 1:] - fringe_hw[:, :-1]
    dy[1:, :] = fringe_hw[1:, :] - fringe_hw[:-1, :]
    return np.sqrt(dx * dx + dy * dy).astype(np.float32)


def haar_dwt_energy(fringe_hw: np.ndarray) -> np.ndarray:
    """One-level Haar high-frequency energy, upsampled back to HxW."""
    h, w = fringe_hw.shape
    hc, wc = h - (h % 2), w - (w % 2)
    f = fringe_hw[:hc, :wc].astype(np.float32, copy=False)
    a = f[0::2, 0::2]
    b = f[0::2, 1::2]
    c = f[1::2, 0::2]
    d = f[1::2, 1::2]
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    energy = np.sqrt(lh * lh + hl * hl + hh * hh).astype(np.float32)
    up = np.repeat(np.repeat(energy, 2, axis=0), 2, axis=1)
    out = np.zeros((h, w), dtype=np.float32)
    out[:hc, :wc] = up[:hc, :wc]
    if hc < h:
        out[hc:, :wc] = out[hc - 1:hc, :wc]
    if wc < w:
        out[:, wc:] = out[:, wc - 1:wc]
    return out


def build_v35_features(fringe_chw: np.ndarray) -> np.ndarray:
    fringe = fringe_chw[0].astype(np.float32, copy=False)
    analytic = analytic_signal_x(fringe)
    phase = np.angle(analytic).astype(np.float32)
    unwrapped = np.unwrap(phase, axis=1).astype(np.float32)
    residual = detrend_phase_x(unwrapped)

    dwt_energy = haar_dwt_energy(fringe)
    grad = gradient_magnitude(fringe)

    h, w = fringe.shape
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None].repeat(w, axis=1)

    return np.stack([
        fringe,
        np.sin(phase).astype(np.float32),
        np.cos(phase).astype(np.float32),
        zscore_clip(residual),
        robust_unit(dwt_energy) * 2.0 - 1.0,
        robust_unit(grad) * 2.0 - 1.0,
        x,
        y,
    ], axis=0).astype(np.float32)
