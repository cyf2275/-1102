"""Precompute cached v3.5 physics features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from physics_features_v35 import build_v35_features


def precompute_split(data_dir: Path, cache_dir: Path, split: str, dtype=np.float16) -> Path:
    src_name = "X_train_fringe.npy" if split == "train" else "X_test_fringe.npy"
    out_name = f"physics_v35_{split}_float16.npy"
    src = np.load(data_dir / src_name, mmap_mode="r")
    n, h, w, _ = src.shape
    out_path = cache_dir / out_name
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=dtype, shape=(n, 8, h, w))
    for i in tqdm(range(n), desc=f"precompute {split}"):
        fringe_chw = np.transpose(np.asarray(src[i]), (2, 0, 1)).astype(np.float32)
        arr[i] = build_v35_features(fringe_chw).astype(dtype)
    arr.flush()
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/root/diffusion_fpp_v5/data")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/diffusion_fpp_v35_cache")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = precompute_split(data_dir, cache_dir, "train")
    test_path = precompute_split(data_dir, cache_dir, "test")
    manifest = {
        "feature_order": [
            "raw_fringe",
            "hilbert_sin",
            "hilbert_cos",
            "hilbert_detrended_phase_residual",
            "dwt_energy",
            "fringe_gradient",
            "x",
            "y",
        ],
        "train": str(train_path),
        "test": str(test_path),
    }
    with open(cache_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
