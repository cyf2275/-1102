from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image
from tqdm import tqdm

from physics_features_pip import (
    EPS,
    estimate_carrier_fft,
    gradient_magnitude,
    haar_dwt_energy,
    robust_unit,
    zscore_clip,
)


PHASE_INSTR_ORDER = [
    "raw_fringe",
    "hilbert_aligned_sin",
    "hilbert_aligned_cos",
    "hilbert_aligned_residual",
    "hilbert_aligned_confidence",
    "ftp_sin",
    "ftp_cos",
    "ftp_residual",
    "ftp_confidence",
    "dwt_high_frequency_energy",
    "fringe_gradient_magnitude",
    "x",
    "y",
]

PHASE_TARGET_ORDER = [
    "gt_wrapped_sin",
    "gt_wrapped_cos",
    "gt_unwrapped_01",
]


def analytic_signal_axis(fringe_hw: np.ndarray, axis: int) -> np.ndarray:
    n = fringe_hw.shape[axis]
    spectrum = np.fft.fft(fringe_hw, axis=axis)
    h = np.zeros(n, dtype=np.float32)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    shape = [1] * fringe_hw.ndim
    shape[axis] = n
    return np.fft.ifft(spectrum * h.reshape(shape), axis=axis)


def detrend_phase_axis(phase_hw: np.ndarray, axis: int) -> np.ndarray:
    if axis == 1:
        data = phase_hw
    else:
        data = phase_hw.T
    h, w = data.shape
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    xc = x - float(x.mean())
    denom = float(np.sum(xc * xc)) + EPS
    row_mean = data.mean(axis=1, keepdims=True)
    slope = ((data - row_mean) * xc[None, :]).sum(axis=1, keepdims=True) / denom
    trend = row_mean + slope * xc[None, :]
    residual = (data - trend).astype(np.float32)
    return residual if axis == 1 else residual.T


def ftp_full_phase(fringe_hw: np.ndarray, carrier: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = fringe_hw.shape
    centered = fringe_hw.astype(np.float32, copy=False) - float(fringe_hw.mean())
    spec = np.fft.fftshift(np.fft.fft2(centered))
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    peak_y = float(carrier["peak_y"])
    peak_x = float(carrier["peak_x"])
    sigma = max(3.0, 0.035 * min(h, w))
    window = np.exp(-((yy - peak_y) ** 2 + (xx - peak_x) ** 2) / (2.0 * sigma * sigma))
    sideband = np.fft.ifft2(np.fft.ifftshift(spec * window))
    phase = np.angle(sideband).astype(np.float32)
    amp = np.abs(sideband).astype(np.float32)
    axis = 1 if abs(float(carrier.get("dx", 0.0))) >= abs(float(carrier.get("dy", 0.0))) else 0
    unwrapped = np.unwrap(phase, axis=axis).astype(np.float32)
    residual = detrend_phase_axis(unwrapped, axis)
    return phase, residual, amp


def make_phase_instr(fringe_hw: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    carrier = estimate_carrier_fft(fringe_hw)
    axis = 1 if abs(carrier["dx"]) >= abs(carrier["dy"]) else 0

    analytic = analytic_signal_axis(fringe_hw, axis=axis)
    h_phase = np.angle(analytic).astype(np.float32)
    h_amp = np.abs(analytic).astype(np.float32)
    h_unwrapped = np.unwrap(h_phase, axis=axis).astype(np.float32)
    h_residual = detrend_phase_axis(h_unwrapped, axis=axis)

    ftp_phase, ftp_residual, ftp_amp = ftp_full_phase(fringe_hw, carrier)
    dwt = haar_dwt_energy(fringe_hw)
    grad = gradient_magnitude(fringe_hw)

    h, w = fringe_hw.shape
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None].repeat(w, axis=1)

    features = np.stack(
        [
            fringe_hw.astype(np.float32),
            np.sin(h_phase).astype(np.float32),
            np.cos(h_phase).astype(np.float32),
            zscore_clip(h_residual),
            robust_unit(h_amp),
            np.sin(ftp_phase).astype(np.float32),
            np.cos(ftp_phase).astype(np.float32),
            zscore_clip(ftp_residual),
            robust_unit(ftp_amp) * float(np.clip(carrier["spectral_confidence"] * 100.0, 0.0, 1.0)),
            robust_unit(dwt),
            robust_unit(grad),
            x,
            y,
        ],
        axis=0,
    ).astype(np.float32)
    carrier = dict(carrier)
    carrier["hilbert_axis"] = "x" if axis == 1 else "y"
    return features, carrier


def phase_error_from_complex(pred_sin, pred_cos, gt_sin, gt_cos, mask=None, allow_sign=True):
    pred = np.arctan2(pred_sin, pred_cos)
    gt = np.arctan2(gt_sin, gt_cos)
    valid = np.ones_like(gt, dtype=bool) if mask is None else mask > 0.5
    if not np.any(valid):
        return {"mae_rad": float("nan"), "rmse_rad": float("nan"), "offset": 0.0, "sign": 1}

    best = None
    for sign in ([1, -1] if allow_sign else [1]):
        signed = sign * pred
        diff0 = gt[valid] - signed[valid]
        offset = float(np.angle(np.mean(np.exp(1j * diff0))))
        diff = np.angle(np.exp(1j * (signed[valid] + offset - gt[valid])))
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff * diff)))
        cand = {"mae_rad": mae, "rmse_rad": rmse, "offset": offset, "sign": sign}
        if best is None or cand["mae_rad"] < best["mae_rad"]:
            best = cand
    return best


def load_mat_key(path: Path, preferred: tuple[str, ...]) -> np.ndarray:
    mat = sio.loadmat(path)
    for key in preferred:
        if key in mat:
            return np.asarray(mat[key])
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if not keys:
        raise KeyError(f"no arrays found in {path}")
    return np.asarray(mat[keys[0]])


def open_memmaps(out_dir: Path, split: str, n: int, h: int, w: int):
    return {
        "instr": np.lib.format.open_memmap(
            out_dir / f"phase_instr_{split}_float16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(n, len(PHASE_INSTR_ORDER), h, w),
        ),
        "target": np.lib.format.open_memmap(
            out_dir / f"phase_target_{split}_float16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(n, len(PHASE_TARGET_ORDER), h, w),
        ),
        "minmax": np.lib.format.open_memmap(
            out_dir / f"phase_minmax_{split}_float32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n, 2),
        ),
    }


def sample_raw_dir(raw_root: Path, sample: dict) -> Path:
    return raw_root / sample["object"] / sample["angle"]


def build_split(args, manifest, split: str):
    cache_dir = Path(args.base_cache_dir)
    out_dir = Path(args.phase_cache_dir)
    raw_root = Path(args.raw_root)
    split_meta = manifest["splits"][split]
    samples = split_meta["samples"]
    n = len(samples)
    fringe_cache = np.load(cache_dir / f"fringe_{split}_uint8.npy", mmap_mode="r")
    mask_cache = np.load(cache_dir / f"mask_{split}_uint8.npy", mmap_mode="r")
    old_physics = np.load(cache_dir / f"physics_pip_{split}_float16.npy", mmap_mode="r")
    _, _, h, w = fringe_cache.shape
    maps = open_memmaps(out_dir, split, n, h, w)

    rows = []
    carriers = []
    for i, sample in enumerate(tqdm(samples, desc=f"phase cache {split}")):
        raw_dir = sample_raw_dir(raw_root, sample)
        wph = load_mat_key(raw_dir / "wrapped_phase.mat", ("wph", "wrapped_phase")).astype(np.float32)
        uph = load_mat_key(raw_dir / "unwrapped_phase.mat", ("uph", "unwrapped_phase")).astype(np.float32)
        if wph.shape != (h, w):
            wph = np.asarray(Image.fromarray(wph).resize((w, h), Image.BILINEAR), dtype=np.float32)
        if uph.shape != (h, w):
            uph = np.asarray(Image.fromarray(uph).resize((w, h), Image.BILINEAR), dtype=np.float32)

        fringe = np.asarray(fringe_cache[i, 0]).astype(np.float32) / 255.0
        mask = np.asarray(mask_cache[i, 0]).astype(np.float32)
        valid = mask > 0.5
        instr, carrier = make_phase_instr(fringe)
        carriers.append({"sample": sample["sample"], **carrier})

        gt_sin = np.sin(wph).astype(np.float32)
        gt_cos = np.cos(wph).astype(np.float32)
        if np.any(valid):
            lo = float(np.percentile(uph[valid], 0.5))
            hi = float(np.percentile(uph[valid], 99.5))
        else:
            lo = float(np.percentile(uph, 0.5))
            hi = float(np.percentile(uph, 99.5))
        uph01 = np.clip((uph - lo) / (hi - lo + EPS), 0.0, 1.0).astype(np.float32)

        maps["instr"][i] = instr.astype(np.float16)
        maps["target"][i, 0] = gt_sin.astype(np.float16)
        maps["target"][i, 1] = gt_cos.astype(np.float16)
        maps["target"][i, 2] = uph01.astype(np.float16)
        maps["minmax"][i] = np.array([lo, hi], dtype=np.float32)

        old_h = phase_error_from_complex(
            np.asarray(old_physics[i, 1]).astype(np.float32),
            np.asarray(old_physics[i, 2]).astype(np.float32),
            gt_sin,
            gt_cos,
            mask=valid,
        )
        new_h = phase_error_from_complex(instr[1], instr[2], gt_sin, gt_cos, mask=valid)
        ftp = phase_error_from_complex(instr[5], instr[6], gt_sin, gt_cos, mask=valid)
        rows.append(
            {
                "sample": sample["sample"],
                "object": sample["object"],
                "angle": sample["angle"],
                "carrier_dx": carrier["dx"],
                "carrier_dy": carrier["dy"],
                "hilbert_axis": carrier["hilbert_axis"],
                "old_hilbert_mae_rad": old_h["mae_rad"],
                "aligned_hilbert_mae_rad": new_h["mae_rad"],
                "ftp_mae_rad": ftp["mae_rad"],
                "old_hilbert_rmse_rad": old_h["rmse_rad"],
                "aligned_hilbert_rmse_rad": new_h["rmse_rad"],
                "ftp_rmse_rad": ftp["rmse_rad"],
                "aligned_hilbert_sign": new_h["sign"],
                "ftp_sign": ftp["sign"],
                "valid_fraction": float(valid.mean()),
                "uph_p005": lo,
                "uph_p995": hi,
            }
        )

    for v in maps.values():
        v.flush()

    csv_path = out_dir / f"phase_diagnostic_{split}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / f"carrier_{split}.json", "w", encoding="utf-8") as f:
        json.dump(carriers, f, indent=2)

    def mean(key):
        vals = [float(r[key]) for r in rows if math.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "n": n,
        "shape": [n, len(PHASE_INSTR_ORDER), h, w],
        "target_shape": [n, len(PHASE_TARGET_ORDER), h, w],
        "diagnostic": {
            "old_hilbert_mae_rad": mean("old_hilbert_mae_rad"),
            "aligned_hilbert_mae_rad": mean("aligned_hilbert_mae_rad"),
            "ftp_mae_rad": mean("ftp_mae_rad"),
            "old_hilbert_rmse_rad": mean("old_hilbert_rmse_rad"),
            "aligned_hilbert_rmse_rad": mean("aligned_hilbert_rmse_rad"),
            "ftp_rmse_rad": mean("ftp_rmse_rad"),
        },
        "objects": split_meta["objects"],
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_phase_cache_960")
    parser.add_argument("--raw_root", default="/root/autodl-tmp/datasets/fpp-ml-bench/fpp_synthetic_dataset")
    parser.add_argument("--splits", default="train,val,test")
    args = parser.parse_args()

    base_cache_dir = Path(args.base_cache_dir)
    phase_cache_dir = Path(args.phase_cache_dir)
    phase_cache_dir.mkdir(parents=True, exist_ok=True)
    with open(base_cache_dir / "manifest.json", "r", encoding="utf-8") as f:
        base_manifest = json.load(f)

    manifest = {
        "base_cache_dir": str(base_cache_dir),
        "raw_root": str(Path(args.raw_root)),
        "phase_instr_order": PHASE_INSTR_ORDER,
        "phase_target_order": PHASE_TARGET_ORDER,
        "splits": {},
    }
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        manifest["splits"][split] = build_split(args, base_manifest, split)

    with open(phase_cache_dir / "phase_cache_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    stats = {
        split: data["diagnostic"]
        for split, data in manifest["splits"].items()
    }
    with open(phase_cache_dir / "phase_target_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

