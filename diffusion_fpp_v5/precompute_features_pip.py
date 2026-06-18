"""Precompute PIP-DiffFPP single-frame physics instructions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from physics_features_pip import FEATURE_ORDER, build_pip_features


def precompute_split(fringe_path: Path, out_path: Path):
    data = np.load(fringe_path, mmap_mode="r")
    n = int(data.shape[0])
    sample_features, sample_meta = build_pip_features(
        np.transpose(np.asarray(data[0]), (2, 0, 1)).astype(np.float32)
    )
    channels, h, w = sample_features.shape
    out = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float16,
        shape=(n, channels, h, w),
    )
    carriers = []
    out[0] = sample_features.astype(np.float16)
    carriers.append(sample_meta)
    for i in tqdm(range(1, n), desc=f"precompute {fringe_path.name}"):
        fringe = np.transpose(np.asarray(data[i]), (2, 0, 1)).astype(np.float32)
        features, meta = build_pip_features(fringe)
        out[i] = features.astype(np.float16)
        carriers.append(meta)
    out.flush()
    return {"n": n, "shape": list(out.shape), "carrier": carriers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/root/diffusion_fpp_v5/data")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/diffusion_fpp_pip_cache")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "feature_order": FEATURE_ORDER,
        "feature_dtype": "float16",
        "default_condition_channels": list(range(9)),
        "ftp_ablation_channels": [9, 10],
        "splits": {},
    }
    manifest["splits"]["train"] = precompute_split(
        data_dir / "X_train_fringe.npy",
        cache_dir / "physics_pip_train_float16.npy",
    )
    manifest["splits"]["test"] = precompute_split(
        data_dir / "X_test_fringe.npy",
        cache_dir / "physics_pip_test_float16.npy",
    )
    with open(cache_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
