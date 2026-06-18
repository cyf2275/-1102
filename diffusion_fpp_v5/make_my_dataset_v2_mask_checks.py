from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from generate_clean_object_masks_my_dataset import save_overlay


def sample_id(object_id: int, pose_id: int) -> str:
    return f"obj{object_id:04d}_pose{pose_id:04d}"


def parse_sample(text: str) -> str:
    if ":" in text:
        obj, pose = text.split(":", 1)
        pose = pose.lower().replace("pose", "")
        return sample_id(int(obj), int(pose))
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path(r"I:\cyf\my_fpp_dataset_v2_0612_new\processed\orderfix_0612_cleanmask_v1"))
    parser.add_argument("--out-dir", type=Path, default=Path(r"I:\cyf\my_dataset_v2_0612_checks\mask_checks"))
    parser.add_argument(
        "--samples",
        nargs="+",
        default=["22:pose2", "50:pose2", "60:pose2", "61:pose2", "62:pose1", "63:pose1", "64:pose1", "64:pose2"],
        help="Sample ids like obj0022_pose0002 or shorthand like 22:pose2.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int | float]] = []
    for item in args.samples:
        sid = parse_sample(item)
        npz_path = args.processed_dir / f"{sid}.npz"
        if not npz_path.exists():
            rows.append({"sample_id": sid, "status": "missing"})
            print(f"[missing] {npz_path}")
            continue
        with np.load(npz_path) as z:
            single = z["single_input"]
            height = z["wall_normal_height"]
            valid = z["valid_mask"].astype(bool)
            old = z["object_mask"].astype(bool)
            clean = z["object_mask_clean_v1"].astype(bool) if "object_mask_clean_v1" in z.files else old

        save_overlay(args.out_dir / f"{sid}_clean_mask_compare.png", single, old, clean, height)
        vals = height[clean & np.isfinite(height)]
        rows.append(
            {
                "sample_id": sid,
                "status": "ok",
                "valid_pixels": int(valid.sum()),
                "object_mask_pixels": int(old.sum()),
                "object_mask_clean_v1_pixels": int(clean.sum()),
                "clean_over_object": float(clean.sum() / old.sum()) if old.sum() else 0.0,
                "clean_height_p02": float(np.percentile(vals, 2)) if vals.size else float("nan"),
                "clean_height_p50": float(np.percentile(vals, 50)) if vals.size else float("nan"),
                "clean_height_p98": float(np.percentile(vals, 98)) if vals.size else float("nan"),
            }
        )
        print(f"[ok] {sid}")

    with (args.out_dir / "mask_check_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(args.out_dir.resolve())


if __name__ == "__main__":
    main()
