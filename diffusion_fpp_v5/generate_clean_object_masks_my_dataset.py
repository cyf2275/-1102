from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def keep_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=bool)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            out |= labels == i
    return out


def choose_seed_component(seed: np.ndarray, height: np.ndarray) -> np.ndarray:
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(seed.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(seed, dtype=bool)

    h, w = seed.shape
    best_i = 0
    best_score = -1.0
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < 500:
            continue
        cy = float(centroids[i][1])
        comp = labels == i
        vals = height[comp & np.isfinite(height)]
        if vals.size == 0:
            continue
        p90 = float(np.percentile(vals, 90))
        # Prefer large components away from the bottom support region.
        vertical_bonus = 1.0 + max(0.0, (0.80 * h - cy) / h)
        central_bonus = 1.0 + max(0.0, 1.0 - abs(float(centroids[i][0]) - 0.5 * w) / (0.5 * w)) * 0.15
        score = area * vertical_bonus * central_bonus * (1.0 + min(p90, 10.0) / 20.0)
        if score > best_score:
            best_score = score
            best_i = i
    if best_i == 0:
        return np.zeros_like(seed, dtype=bool)
    return labels == best_i


def clean_object_mask(
    height: np.ndarray,
    valid_mask: np.ndarray,
    h_min: float = 1.0,
    seed_height: float = 2.0,
    bbox_pad: int = 24,
    bbox_down_pad: int = 45,
    min_area: int = 800,
) -> tuple[np.ndarray, dict[str, float | int]]:
    valid = valid_mask.astype(bool) & np.isfinite(height)
    coarse = valid & (height > h_min)

    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    coarse = cv2.morphologyEx(coarse.astype(np.uint8), cv2.MORPH_OPEN, kernel3).astype(bool)
    coarse = keep_components(coarse, min_area=min_area)

    seed = valid & (height > seed_height)
    seed = cv2.morphologyEx(seed.astype(np.uint8), cv2.MORPH_OPEN, kernel3).astype(bool)
    seed = keep_components(seed, min_area=max(200, min_area // 4))
    seed_main = choose_seed_component(seed, height)

    if not seed_main.any():
        return coarse, {
            "used_fallback": 1,
            "coarse_pixels": int(coarse.sum()),
            "seed_pixels": int(seed.sum()),
            "clean_pixels": int(coarse.sum()),
        }

    ys, xs = np.where(seed_main)
    h, w = height.shape
    y0 = max(0, int(ys.min()) - bbox_pad)
    y1 = min(h, int(ys.max()) + bbox_down_pad)
    x0 = max(0, int(xs.min()) - bbox_pad)
    x1 = min(w, int(xs.max()) + bbox_pad)
    roi = np.zeros_like(coarse, dtype=bool)
    roi[y0:y1, x0:x1] = True

    candidate = coarse & roi

    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    clean = np.zeros_like(candidate, dtype=bool)
    best_i = 0
    best_overlap = -1
    best_area = -1
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == i
        overlap = int((comp & seed_main).sum())
        if overlap > best_overlap or (overlap == best_overlap and area > best_area):
            best_overlap = overlap
            best_area = area
            best_i = i
    if best_i > 0:
        clean = labels == best_i
    else:
        clean = candidate

    clean = cv2.morphologyEx(clean.astype(np.uint8), cv2.MORPH_CLOSE, kernel5).astype(bool)
    clean = cv2.morphologyEx(clean.astype(np.uint8), cv2.MORPH_OPEN, kernel3).astype(bool)
    clean = keep_components(clean, min_area=min_area)

    return clean, {
        "used_fallback": 0,
        "coarse_pixels": int(coarse.sum()),
        "seed_pixels": int(seed.sum()),
        "seed_main_pixels": int(seed_main.sum()),
        "clean_pixels": int(clean.sum()),
        "bbox_x0": int(x0),
        "bbox_y0": int(y0),
        "bbox_x1": int(x1),
        "bbox_y1": int(y1),
    }


def save_overlay(path: Path, single: np.ndarray, old_mask: np.ndarray, clean_mask: np.ndarray, height: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    axes[0].imshow(single, cmap="gray")
    axes[0].set_title("single input")

    axes[1].imshow(single, cmap="gray")
    old_rgba = np.zeros((*old_mask.shape, 4), dtype=float)
    old_rgba[old_mask] = [1.0, 0.0, 0.0, 0.35]
    axes[1].imshow(old_rgba)
    axes[1].set_title("old object_mask")

    axes[2].imshow(single, cmap="gray")
    new_rgba = np.zeros((*clean_mask.shape, 4), dtype=float)
    new_rgba[clean_mask] = [0.0, 0.8, 1.0, 0.45]
    axes[2].imshow(new_rgba)
    axes[2].set_title("clean object_mask v1")

    vals = height[clean_mask & np.isfinite(height)]
    if vals.size:
        vmin, vmax = np.percentile(vals, [2, 98])
    else:
        vals = height[old_mask & np.isfinite(height)]
        vmin, vmax = np.percentile(vals, [2, 98]) if vals.size else (0.0, 1.0)
    im = axes[3].imshow(np.where(clean_mask, height, np.nan), cmap="turbo", vmin=vmin, vmax=vmax)
    axes[3].set_title("clean height")
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="my_fpp_dataset_v1/processed/orderfix_0610")
    parser.add_argument("--out-dir", default="cloud_results/my_dataset_clean_object_masks_v1_20260611")
    parser.add_argument("--h-min", type=float, default=1.0)
    parser.add_argument("--seed-height", type=float, default=2.0)
    parser.add_argument("--bbox-pad", type=int, default=24)
    parser.add_argument("--bbox-down-pad", type=int, default=45)
    parser.add_argument("--min-area", type=int, default=800)
    parser.add_argument("--preview-limit", type=int, default=20)
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    out = Path(args.out_dir)
    mask_dir = out / "masks"
    fig_dir = out / "figures"
    mask_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for idx, npz_path in enumerate(sorted(processed.glob("*.npz"))):
        sid = npz_path.stem
        with np.load(npz_path) as z:
            single = z["single_input"]
            height = z["wall_normal_height"]
            valid = z["valid_mask"].astype(bool)
            old = z["object_mask"].astype(bool)
            clean, meta = clean_object_mask(
                height,
                valid,
                h_min=args.h_min,
                seed_height=args.seed_height,
                bbox_pad=args.bbox_pad,
                bbox_down_pad=args.bbox_down_pad,
                min_area=args.min_area,
            )

        cv2.imwrite(str(mask_dir / f"{sid}_clean_object_mask.png"), (clean.astype(np.uint8) * 255))
        if idx < args.preview_limit or sid in {"obj0003_pose0005", "obj0011_pose0001", "obj0012_pose0001"}:
            save_overlay(fig_dir / f"{sid}_clean_mask_compare.png", single, old, clean, height)

        old_pixels = int(old.sum())
        clean_pixels = int(clean.sum())
        rows.append(
            {
                "sample_id": sid,
                "old_pixels": old_pixels,
                "clean_pixels": clean_pixels,
                "clean_over_old": float(clean_pixels / old_pixels) if old_pixels else 0.0,
                "intersection_pixels": int((old & clean).sum()),
                **meta,
            }
        )

    with (out / "clean_mask_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    readme = (
        "# Clean Object Mask v1\n\n"
        "This directory contains a non-destructive clean-mask proposal. Original NPZ files are not modified.\n\n"
        "Pipeline:\n"
        "1. Start from coarse foreground: valid_mask and wall_normal_height > h_min.\n"
        "2. Build high-confidence seed: valid_mask and wall_normal_height > seed_height.\n"
        "3. Select the dominant seed connected component, preferring large components away from the bottom support region.\n"
        "4. Grow back inside the coarse foreground within the seed bounding box.\n"
        "5. Keep the component overlapping the seed and apply small morphology cleanup.\n\n"
        "Default parameters: h_min=1.0 mm, seed_height=2.0 mm, bbox_pad=24 px, bbox_down_pad=45 px, min_area=800 px.\n\n"
        "Use this as a review artifact first. If acceptable, a later dataset version can add `object_mask_clean_v1` to each NPZ.\n"
    )
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(out.resolve())


if __name__ == "__main__":
    main()
