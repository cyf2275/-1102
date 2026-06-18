"""Shearlet-lite single-frame fringe physics features.

This is not a full ShearLab implementation.  It follows the useful part of
one-shot shearlet FPP for our pipeline: estimate the fringe carrier in the
2-D Fourier domain, apply a small directional filter bank around the carrier,
and expose wrapped phase plus a coefficient-quality map as network
conditions.

Feature order:
0 raw fringe
1 Hilbert wrapped phase sin
2 Hilbert wrapped phase cos
3 row-wise detrended Hilbert residual
4 directional carrier phase sin
5 directional carrier phase cos
6 directional coefficient amplitude
7 directional coefficient quality
8 x coordinate
9 y coordinate
"""
from __future__ import annotations

import numpy as np

from physics_features_v35 import (
    EPS,
    analytic_signal_x,
    detrend_phase_x,
    robust_unit,
    zscore_clip,
)


DEFAULT_SHEARS = (-0.45, -0.22, 0.0, 0.22, 0.45)


def _fft_frequency_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    fy = np.fft.fftshift(np.fft.fftfreq(h)).astype(np.float32)
    fx = np.fft.fftshift(np.fft.fftfreq(w)).astype(np.float32)
    return np.meshgrid(fx, fy)


def estimate_carrier(fringe_hw: np.ndarray, dc_radius: float = 0.035) -> tuple[float, float, float]:
    """Return the dominant positive carrier vector in cycles/pixel.

    The sign is made deterministic so equivalent +/- Fourier peaks do not
    flip the phase channel between samples.
    """
    h, w = fringe_hw.shape
    f = fringe_hw.astype(np.float32, copy=False)
    spectrum = np.fft.fftshift(np.fft.fft2(f - float(f.mean())))
    mag = np.abs(spectrum).astype(np.float32)
    kx, ky = _fft_frequency_grid(h, w)
    radius = np.sqrt(kx * kx + ky * ky)
    mag[radius < dc_radius] = 0.0
    mag[:, : w // 2] *= 0.85
    py, px = np.unravel_index(int(np.argmax(mag)), mag.shape)
    fx = float(kx[py, px])
    fy = float(ky[py, px])
    if fx < 0 or (abs(fx) < EPS and fy < 0):
        fx, fy = -fx, -fy
    f0 = max(float(np.sqrt(fx * fx + fy * fy)), 1.0 / max(h, w))
    return fx, fy, f0


def directional_carrier_responses(
    fringe_hw: np.ndarray,
    shears: tuple[float, ...] = DEFAULT_SHEARS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directional complex responses, max amplitude, and quality map."""
    h, w = fringe_hw.shape
    fx0, fy0, f0 = estimate_carrier(fringe_hw)
    ux, uy = fx0 / f0, fy0 / f0

    kx, ky = _fft_frequency_grid(h, w)
    u = kx * ux + ky * uy
    v = -kx * uy + ky * ux

    # Slightly broad filters keep the phase stable on small Nguyen/Wang data.
    radial_sigma = max(0.012, f0 * 0.42)
    perp_sigma = max(0.008, f0 * 0.23)

    spectrum = np.fft.fftshift(np.fft.fft2(fringe_hw.astype(np.float32) - float(fringe_hw.mean())))
    responses = []
    for shear in shears:
        # Shear term lets the passband rotate locally around the dominant
        # carrier, approximating the direction selectivity used by shearlets.
        vv = v - shear * (u - f0)
        filt = np.exp(-0.5 * ((u - f0) / radial_sigma) ** 2 - 0.5 * (vv / perp_sigma) ** 2)
        filt = filt.astype(np.float32)
        filt[u <= 0.0] = 0.0
        response = np.fft.ifft2(np.fft.ifftshift(spectrum * filt)).astype(np.complex64)
        responses.append(response)

    resp = np.stack(responses, axis=0)
    amps = np.abs(resp).astype(np.float32)
    best_idx = np.argmax(amps, axis=0)
    best = np.take_along_axis(resp, best_idx[None, ...], axis=0)[0]
    max_amp = np.take_along_axis(amps, best_idx[None, ...], axis=0)[0]

    if len(shears) > 1:
        sorted_amp = np.sort(amps, axis=0)
        second = sorted_amp[-2]
        dominance = (max_amp - second) / (max_amp + EPS)
    else:
        dominance = np.ones_like(max_amp, dtype=np.float32)
    amp_norm = robust_unit(max_amp)
    quality = np.clip(amp_norm * dominance, 0.0, 1.0).astype(np.float32)
    return best, max_amp, quality


def build_shearlet_lite_features(fringe_chw: np.ndarray) -> np.ndarray:
    fringe = fringe_chw[0].astype(np.float32, copy=False)

    analytic = analytic_signal_x(fringe)
    hilbert_phase = np.angle(analytic).astype(np.float32)
    unwrapped = np.unwrap(hilbert_phase, axis=1).astype(np.float32)
    residual = detrend_phase_x(unwrapped)

    directional, amp, quality = directional_carrier_responses(fringe)
    directional_phase = np.angle(directional).astype(np.float32)

    h, w = fringe.shape
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None].repeat(w, axis=1)

    return np.stack(
        [
            fringe,
            np.sin(hilbert_phase).astype(np.float32),
            np.cos(hilbert_phase).astype(np.float32),
            zscore_clip(residual),
            np.sin(directional_phase).astype(np.float32),
            np.cos(directional_phase).astype(np.float32),
            robust_unit(amp) * 2.0 - 1.0,
            quality * 2.0 - 1.0,
            x,
            y,
        ],
        axis=0,
    ).astype(np.float32)
