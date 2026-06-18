"""Build GPU-friendly memmap caches for FPP-ML-Bench.

The raw FPP-ML-Bench training split stores fringe images as PNG files and depth
targets as MATLAB .mat files. This script converts the official object-level
train/val/test split into fixed-resolution NumPy memmaps and precomputes the
PIP physics instruction channels, so training does not decode PNG/MAT files in
the DataLoader hot path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm

from physics_features_pip import FEATURE_ORDER, build_pip_features


SPLITS = ("train", "val", "test")


def _resize_float(arr: np.ndarray, size: tuple[int, int], resample=Image.BILINEAR) -> np.ndarray:
    h, w = size
    if arr.shape == (h, w):
        return arr.astype(np.float32, copy=False)
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    return np.asarray(img.resize((w, h), resample=resample), dtype=np.float32)


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    if mask.shape == (h, w):
        return mask.astype(np.uint8, copy=False)
    img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    out = np.asarray(img.resize((w, h), resample=Image.NEAREST), dtype=np.uint8)
    return (out > 127).astype(np.uint8)


def _resize_depth_foreground(depth: np.ndarray, valid: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Resize sparse depth without mixing zero background into object edges."""
    h, w = size
    valid_f = valid.astype(np.float32)
    if depth.shape == (h, w):
        out_mask = valid.astype(np.uint8, copy=False)
        return np.where(out_mask > 0, depth, 0.0).astype(np.float32), out_mask
    num = _resize_float(depth * valid_f, size, Image.BILINEAR)
    den = _resize_float(valid_f, size, Image.BILINEAR)
    out = np.zeros((h, w), dtype=np.float32)
    keep = den > 1e-4
    out[keep] = num[keep] / den[keep]
    out_mask = _resize_mask(valid, size)
    out = np.where(out_mask > 0, out, 0.0).astype(np.float32)
    return out, out_mask


def _load_depth(path: Path) -> np.ndarray:
    mat = loadmat(path)
    for key in ("depthMapNormalized", "depthMap"):
        if key in mat:
            return np.asarray(mat[key], dtype=np.float32)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    raise KeyError(f"no supported depth key in {path}; keys={keys}")


def _sample_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    obj, angle = stem.rsplit("_", 1)
    return obj, angle


def _open_memmaps(cache_dir: Path, split: str, n: int, h: int, w: int):
    return {
        "fringe": np.lib.format.open_memmap(
            cache_dir / f"fringe_{split}_uint8.npy", mode="w+", dtype=np.uint8, shape=(n, 1, h, w)
        ),
        "depth01": np.lib.format.open_memmap(
            cache_dir / f"depth01_{split}_float16.npy", mode="w+", dtype=np.float16, shape=(n, 1, h, w)
        ),
        "depth_mm": np.lib.format.open_memmap(
            cache_dir / f"depth_mm_{split}_float32.npy", mode="w+", dtype=np.float32, shape=(n, 1, h, w)
        ),
        "mask": np.lib.format.open_memmap(
            cache_dir / f"mask_{split}_uint8.npy", mode="w+", dtype=np.uint8, shape=(n, 1, h, w)
        ),
        "physics": np.lib.format.open_memmap(
            cache_dir / f"physics_pip_{split}_float16.npy", mode="w+", dtype=np.float16, shape=(n, len(FEATURE_ORDER), h, w)
        ),
        "minmax": np.lib.format.open_memmap(
            cache_dir / f"depth_minmax_{split}_float32.npy", mode="w+", dtype=np.float32, shape=(n, 2)
        ),
    }


def build_split(args, split: str):
    data_root = Path(args.data_root)
    train_root = data_root / "training_datasets"
    variant_root = train_root / args.variant / split
    raw_root = train_root / args.raw_variant / split
    params_root = train_root / "info_depth_params" / split
    fringe_paths = sorted((variant_root / "fringe").glob("*.png"))
    if args.max_samples:
        fringe_paths = fringe_paths[: int(args.max_samples)]
    if not fringe_paths:
        raise FileNotFoundError(f"no fringe PNG files found under {variant_root / 'fringe'}")

    h = w = int(args.image_size)
    maps = _open_memmaps(Path(args.cache_dir), split, len(fringe_paths), h, w)
    samples = []
    object_to_idx: dict[str, int] = {}

    for i, fringe_path in enumerate(tqdm(fringe_paths, desc=f"cache {split}")):
        obj, angle = _sample_name(fringe_path)
        object_to_idx.setdefault(obj, len(object_to_idx))
        depth_path = variant_root / "depth" / f"{fringe_path.stem}.mat"
        raw_path = raw_root / "depth" / f"{fringe_path.stem}.mat"
        params_path = params_root / "depth" / f"{fringe_path.stem}.mat"

        fringe_u8 = np.asarray(Image.open(fringe_path).convert("L").resize((w, h), Image.BILINEAR), dtype=np.uint8)
        raw_depth = _load_depth(raw_path)
        raw_valid = raw_depth > 0.0
        depth_norm = _load_depth(depth_path)
        depth01, mask = _resize_depth_foreground(np.clip(depth_norm, 0.0, 1.0), raw_valid, (h, w))
        depth_mm, _ = _resize_depth_foreground(raw_depth, raw_valid, (h, w))

        params = loadmat(params_path)
        depth_min = float(np.asarray(params["depth_min"]).reshape(-1)[0])
        depth_max = float(np.asarray(params["depth_max"]).reshape(-1)[0])

        fringe_float = (fringe_u8.astype(np.float32) / 255.0)[None, ...]
        physics, carrier = build_pip_features(fringe_float)

        maps["fringe"][i, 0] = fringe_u8
        maps["depth01"][i, 0] = depth01.astype(np.float16)
        maps["depth_mm"][i, 0] = depth_mm
        maps["mask"][i, 0] = mask
        maps["physics"][i] = physics.astype(np.float16)
        maps["minmax"][i] = np.array([depth_min, depth_max], dtype=np.float32)
        samples.append({
            "index": i,
            "sample": fringe_path.stem,
            "object": obj,
            "object_index": object_to_idx[obj],
            "angle": angle,
            "fringe_path": str(fringe_path),
            "depth_path": str(depth_path),
            "raw_depth_path": str(raw_path),
            "params_path": str(params_path),
            "carrier": carrier,
        })

    for mm in maps.values():
        mm.flush()
    return {
        "n": len(fringe_paths),
        "shape": [len(fringe_paths), 1, h, w],
        "physics_shape": [len(fringe_paths), len(FEATURE_ORDER), h, w],
        "objects": sorted(object_to_idx, key=object_to_idx.get),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/root/autodl-tmp/datasets/fpp-ml-bench")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_480")
    parser.add_argument("--image_size", type=int, default=480)
    parser.add_argument("--variant", default="training_data_depth_individual_normalized")
    parser.add_argument("--raw_variant", default="training_data_depth_raw")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "fpp_ml_bench",
        "data_root": str(Path(args.data_root)),
        "variant": args.variant,
        "raw_variant": args.raw_variant,
        "image_size": int(args.image_size),
        "feature_order": FEATURE_ORDER,
        "feature_dtype": "float16",
        "default_condition_channels": list(range(9)),
        "ftp_ablation_channels": [9, 10],
        "splits": {},
    }
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split}")
        manifest["splits"][split] = build_split(args, split)

    with open(cache_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "cache_dir": str(cache_dir),
        "image_size": int(args.image_size),
        "splits": {k: {"n": v["n"], "objects": v["objects"]} for k, v in manifest["splits"].items()},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
