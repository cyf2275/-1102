"""Frequency-normalized single-frame FPP physics instructions for PIP-DiffFPP.

Feature order:
0 raw fringe                           [0, 1] or original normalized range
1 hilbert_sin                          [-1, 1]
2 hilbert_cos                          [-1, 1]
3 hilbert_detrended_phase_residual     [-1, 1]
4 hilbert_amplitude_confidence         [0, 1]
5 dwt_high_frequency_energy            [0, 1]
6 fringe_gradient_magnitude            [0, 1]
7 x coordinate                         [-1, 1]
8 y coordinate                         [-1, 1]
9 ftp_detrended_phase_residual         [-1, 1]
10 ftp_spectral_confidence             [0, 1]

The default PIP-lite model uses channels 0..8. FTP channels are cached for
ablation but are not used by default.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


EPS = 1e-6


def robust_unit(x: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    lo, hi = np.percentile(x, [lo_q, hi_q])
    return np.clip((x - lo) / (hi - lo + EPS), 0.0, 1.0).astype(np.float32)


def zscore_clip(x: np.ndarray, clip: float = 3.0) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    z = (x - float(x.mean())) / (float(x.std()) + EPS)
    return (np.clip(z, -clip, clip) / clip).astype(np.float32)


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


def estimate_carrier_fft(fringe_hw: np.ndarray, dc_radius_ratio: float = 0.06) -> Dict[str, float]:
    """Estimate dominant carrier peak from a 2-D FFT magnitude map.

    Returns normalized cycles-per-pixel offsets and a simple spectral
    confidence. The peak search masks the DC area and chooses one sideband.
    """
    h, w = fringe_hw.shape
    centered = fringe_hw.astype(np.float32, copy=False) - float(fringe_hw.mean())
    spec = np.fft.fftshift(np.fft.fft2(centered))
    mag = np.abs(spec).astype(np.float32)
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dc_radius = max(4, int(min(h, w) * dc_radius_ratio))
    mag[(yy - cy) ** 2 + (xx - cx) ** 2 <= dc_radius ** 2] = 0.0

    peak_y, peak_x = np.unravel_index(int(np.argmax(mag)), mag.shape)
    dy = int(peak_y - cy)
    dx = int(peak_x - cx)
    # Use a deterministic sideband orientation to stabilize demodulation.
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
        peak_x = cx + dx
        peak_y = cy + dy

    peak = float(mag[peak_y, peak_x])
    total = float(mag.sum()) + EPS
    confidence = peak / total
    return {
        "dx": float(dx),
        "dy": float(dy),
        "fx": float(dx) / float(w),
        "fy": float(dy) / float(h),
        "peak_y": float(peak_y),
        "peak_x": float(peak_x),
        "spectral_confidence": float(confidence),
    }


def ftp_demodulation(fringe_hw: np.ndarray, carrier: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate single-sideband FTP demodulation using the estimated carrier."""
    h, w = fringe_hw.shape
    centered = fringe_hw.astype(np.float32, copy=False) - float(fringe_hw.mean())
    spec = np.fft.fftshift(np.fft.fft2(centered))
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    peak_y = float(carrier["peak_y"])
    peak_x = float(carrier["peak_x"])
    sigma = max(3.0, 0.035 * min(h, w))
    window = np.exp(-((yy - peak_y) ** 2 + (xx - peak_x) ** 2) / (2.0 * sigma * sigma))
    sideband = np.fft.ifft2(np.fft.ifftshift(spec * window))

    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]
    carrier_phase = 2.0 * math.pi * (carrier["fx"] * x + carrier["fy"] * y)
    demodulated = sideband * np.exp(-1j * carrier_phase)
    phase = np.angle(demodulated).astype(np.float32)
    amplitude = np.abs(demodulated).astype(np.float32)
    unwrapped = np.unwrap(np.unwrap(phase, axis=1), axis=0).astype(np.float32)
    residual = detrend_phase_x(unwrapped)
    return phase, residual, amplitude


def build_pip_features(fringe_chw: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    fringe = fringe_chw[0].astype(np.float32, copy=False)
    carrier = estimate_carrier_fft(fringe)

    analytic = analytic_signal_x(fringe)
    hilbert_phase = np.angle(analytic).astype(np.float32)
    hilbert_amp = np.abs(analytic).astype(np.float32)
    hilbert_unwrapped = np.unwrap(hilbert_phase, axis=1).astype(np.float32)
    hilbert_residual = detrend_phase_x(hilbert_unwrapped)

    _, ftp_residual, ftp_amp = ftp_demodulation(fringe, carrier)
    dwt_energy = haar_dwt_energy(fringe)
    grad = gradient_magnitude(fringe)

    h, w = fringe.shape
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None].repeat(w, axis=1)

    features = np.stack([
        fringe.astype(np.float32),
        np.sin(hilbert_phase).astype(np.float32),
        np.cos(hilbert_phase).astype(np.float32),
        zscore_clip(hilbert_residual),
        robust_unit(hilbert_amp),
        robust_unit(dwt_energy),
        robust_unit(grad),
        x,
        y,
        zscore_clip(ftp_residual),
        robust_unit(ftp_amp) * float(np.clip(carrier["spectral_confidence"] * 100.0, 0.0, 1.0)),
    ], axis=0).astype(np.float32)
    return features, carrier


FEATURE_ORDER = [
    "raw_fringe",
    "hilbert_sin",
    "hilbert_cos",
    "hilbert_detrended_phase_residual",
    "hilbert_amplitude_confidence",
    "dwt_high_frequency_energy",
    "fringe_gradient_magnitude",
    "x",
    "y",
    "ftp_detrended_phase_residual",
    "ftp_spectral_confidence",
]
