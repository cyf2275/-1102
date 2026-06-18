"""Build PSP quadrature targets for single-frame physical diffusion.

The input remains the official single A0 fringe through the existing phase
instruction cache.  The target is computed from the first 18 phase-shifted
frames:

    A = 2/N * sum I_k cos(2*pi*k/N)
    B = -2/N * sum I_k sin(2*pi*k/N)

The diffusion model can then restore the missing PSP quadrature field from one
frame, and the wrapped phase is obtained by atan2(B, A).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm


EPS = 1e-6
SPLITS = ("train", "val", "test")


def load_mat_key(path: Path, names):
    mat = loadmat(path)
    for name in names:
        if name in mat:
            return np.asarray(mat[name])
    keys = [k for k in mat.keys() if not k.startswith("__")]
    raise KeyError(f"{path} missing keys {names}; available={keys}")


def resize_float(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape == (h, w):
        return arr
    return np.asarray(Image.fromarray(arr, mode="F").resize((w, h), Image.BILINEAR), dtype=np.float32)


def sample_raw_dir(raw_root: Path, sample: dict) -> Path:
    return raw_root / sample["object"] / sample["angle"]


def circular_aligned_mae(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    if not np.any(valid):
        valid = np.ones_like(gt, dtype=bool)
    vals = []
    for sign in (1.0, -1.0):
        delta = sign * pred[valid] - gt[valid]
        offset = math.atan2(float(np.sin(delta).mean()), float(np.cos(delta).mean()))
        err = np.arctan2(np.sin(sign * pred - offset - gt), np.cos(sign * pred - offset - gt))
        vals.append(float(np.mean(np.abs(err[valid]))))
    return min(vals)


def link_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def open_targets(out_dir: Path, split: str, n: int, h: int, w: int):
    return {
        "target": np.lib.format.open_memmap(
            out_dir / f"phase_target_{split}_float16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(n, 3, h, w),
        ),
        "minmax": np.lib.format.open_memmap(
            out_dir / f"phase_minmax_{split}_float32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n, 2),
        ),
    }


def build_split(args, base_manifest: dict, split: str):
    src_phase_dir = Path(args.source_phase_cache_dir)
    out_dir = Path(args.output_phase_cache_dir)
    raw_root = Path(args.raw_root)
    samples = base_manifest["splits"][split]["samples"]
    n = len(samples)

    instr_src = src_phase_dir / f"phase_instr_{split}_float16.npy"
    instr_dst = out_dir / f"phase_instr_{split}_float16.npy"
    link_or_copy(instr_src, instr_dst)

    instr = np.load(instr_src, mmap_mode="r")
    _, _, h, w = instr.shape
    maps = open_targets(out_dir, split, n, h, w)
    theta = (2.0 * np.pi * np.arange(int(args.steps), dtype=np.float32) / float(args.steps)).astype(np.float32)
    cos_t = np.cos(theta)[:, None, None]
    sin_t = np.sin(theta)[:, None, None]

    rows = []
    for i, sample in enumerate(tqdm(samples, desc=f"psp quad {split}")):
        raw_dir = sample_raw_dir(raw_root, sample)
        frames = []
        for k in range(int(args.steps)):
            img = Image.open(raw_dir / f"A_{k}.png").convert("L")
            arr = np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
            frames.append(arr)
        stack = np.stack(frames, axis=0)
        a = (2.0 / float(args.steps)) * np.sum(stack * cos_t, axis=0)
        b = (-2.0 / float(args.steps)) * np.sum(stack * sin_t, axis=0)
        amp = np.sqrt(a * a + b * b).astype(np.float32)
        valid = amp > float(args.amp_floor)
        if (raw_dir / "wrapped_phase.mat").exists():
            wph = load_mat_key(raw_dir / "wrapped_phase.mat", ("wph", "wrapped_phase")).astype(np.float32)
            wph = resize_float(wph, h, w)
        else:
            wph = np.arctan2(b, a).astype(np.float32)
        qsin = b / (amp + EPS)
        qcos = a / (amp + EPS)
        if np.any(valid):
            amp_hi = float(np.percentile(amp[valid], float(args.amp_percentile)))
        else:
            amp_hi = float(np.percentile(amp, float(args.amp_percentile)))
        amp01 = np.clip(amp / max(amp_hi, EPS), 0.0, 1.0)

        maps["target"][i, 0] = qsin.astype(np.float16)
        maps["target"][i, 1] = qcos.astype(np.float16)
        maps["target"][i, 2] = amp01.astype(np.float16)
        maps["minmax"][i] = np.array([0.0, amp_hi], dtype=np.float32)

        pred_phase = np.arctan2(qsin, qcos).astype(np.float32)
        rows.append(
            {
                "sample": sample["sample"],
                "object": sample["object"],
                "angle": sample["angle"],
                "amp_p99": amp_hi,
                "valid_fraction": float(valid.mean()),
                "psp_quad_to_wph_aligned_mae_rad": circular_aligned_mae(pred_phase, wph, valid),
            }
        )

    for mm in maps.values():
        mm.flush()
    with open(out_dir / f"psp_quad_diagnostic_{split}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    vals = [r["psp_quad_to_wph_aligned_mae_rad"] for r in rows]
    return {
        "n": n,
        "shape": [n, 3, h, w],
        "samples": samples,
        "psp_quad_to_wph_aligned_mae_rad": float(np.mean(vals)),
        "target_order": ["psp_quad_sin", "psp_quad_cos", "psp_modulation_01"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--source_phase_cache_dir", default="/root/autodl-tmp/fpp_ml_phase_cache_960")
    parser.add_argument("--output_phase_cache_dir", default="/root/autodl-tmp/fpp_ml_pspquad_cache_960")
    parser.add_argument("--raw_root", default="/root/autodl-tmp/datasets/fpp-ml-bench/fpp_synthetic_dataset")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--amp_percentile", type=float, default=99.0)
    parser.add_argument("--amp_floor", type=float, default=1e-4)
    args = parser.parse_args()

    out_dir = Path(args.output_phase_cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(Path(args.base_cache_dir) / "manifest.json", "r", encoding="utf-8") as f:
        base_manifest = json.load(f)
    with open(Path(args.source_phase_cache_dir) / "phase_cache_manifest.json", "r", encoding="utf-8") as f:
        src_manifest = json.load(f)

    manifest = {
        "dataset": "fpp_ml_bench_psp_quadrature",
        "source_phase_cache_dir": str(Path(args.source_phase_cache_dir)),
        "raw_root": str(Path(args.raw_root)),
        "steps": int(args.steps),
        "feature_order": src_manifest.get("feature_order", []),
        "target_order": ["psp_quad_sin", "psp_quad_cos", "psp_modulation_01"],
        "splits": {},
    }
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split}")
        manifest["splits"][split] = build_split(args, base_manifest, split)
    with open(out_dir / "phase_cache_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: {"n": v["n"], "mae": v["psp_quad_to_wph_aligned_mae_rad"]} for k, v in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
