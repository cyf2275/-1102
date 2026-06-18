"""Precompute fringe-only physics condition channels.

This removes the CPU bottleneck caused by doing Hilbert/FFT feature extraction
inside DataLoader workers every epoch.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.dataset import build_physics_features


def precompute(fringe_path, out_path, dtype=np.float16):
    fringe = np.load(fringe_path, mmap_mode="r")
    n, h, w, _ = fringe.shape
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=dtype,
        shape=(n, 7, h, w),
    )
    t0 = time.time()
    for i in tqdm(range(n), desc=f"precompute {fringe_path.name}"):
        f = np.transpose(np.asarray(fringe[i]), (2, 0, 1)).astype(np.float32)
        arr[i] = build_physics_features(f).astype(dtype)
        if (i + 1) % 32 == 0:
            arr.flush()
    arr.flush()
    return {
        "source": str(fringe_path),
        "output": str(out_path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "seconds": time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/root/diffusion_fpp_v5/data")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/diffusion_fpp_v5_cache")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    dtype = np.float16 if args.dtype == "float16" else np.float32

    results = [
        precompute(data_dir / "X_train_fringe.npy", cache_dir / f"physics_train_{args.dtype}.npy", dtype=dtype),
        precompute(data_dir / "X_test_fringe.npy", cache_dir / f"physics_test_{args.dtype}.npy", dtype=dtype),
    ]

    # The Dataset expects these stable names.
    for split in ("train", "test"):
        src = cache_dir / f"physics_{split}_{args.dtype}.npy"
        dst = cache_dir / f"physics_{split}_float16.npy"
        if args.dtype == "float16" and src != dst:
            src.replace(dst)

    with open(cache_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
