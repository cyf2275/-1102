"""Precompute cached shearlet-lite physics features for Nguyen/Wang."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from physics_features_shearlet_lite import build_shearlet_lite_features


FEATURE_ORDER = [
    "raw_fringe",
    "hilbert_sin",
    "hilbert_cos",
    "hilbert_detrended_phase_residual",
    "shearlet_lite_directional_phase_sin",
    "shearlet_lite_directional_phase_cos",
    "shearlet_lite_coefficient_amplitude",
    "shearlet_lite_coefficient_quality",
    "x",
    "y",
]


def precompute_split(data_dir: Path, cache_dir: Path, split: str, dtype=np.float16) -> dict:
    src_name = "X_train_fringe.npy" if split == "train" else "X_test_fringe.npy"
    out_name = f"physics_shearlet_lite_{split}_{np.dtype(dtype).name}.npy"
    src = np.load(data_dir / src_name, mmap_mode="r")
    n, h, w, _ = src.shape
    out_path = cache_dir / out_name
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=dtype, shape=(n, len(FEATURE_ORDER), h, w))
    t0 = time.time()
    for i in tqdm(range(n), desc=f"precompute shearlet-lite {split}"):
        fringe_chw = np.transpose(np.asarray(src[i]), (2, 0, 1)).astype(np.float32)
        arr[i] = build_shearlet_lite_features(fringe_chw).astype(dtype)
        if (i + 1) % 16 == 0:
            arr.flush()
    arr.flush()
    return {
        "split": split,
        "source": str(data_dir / src_name),
        "output": str(out_path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "seconds": time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/root/diffusion_fpp_v5/data")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/diffusion_fpp_shearlet_lite_cache")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if args.dtype == "float16" else np.float32

    results = [
        precompute_split(data_dir, cache_dir, "train", dtype=dtype),
        precompute_split(data_dir, cache_dir, "test", dtype=dtype),
    ]
    manifest = {
        "name": "shearlet_lite_nguyen_wang",
        "note": "Directional Fourier/shear filter bank around the estimated fringe carrier; not full ShearLab.",
        "feature_order": FEATURE_ORDER,
        "results": results,
    }
    with open(cache_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
