from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from generate_clean_object_masks_my_dataset import clean_object_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", default="my_fpp_dataset_v1/processed/orderfix_0610")
    parser.add_argument("--dst-dir", default="my_fpp_dataset_v1/processed/orderfix_0610_cleanmask_v1")
    parser.add_argument("--h-min", type=float, default=1.0)
    parser.add_argument("--seed-height", type=float, default=2.0)
    parser.add_argument("--bbox-pad", type=int, default=24)
    parser.add_argument("--bbox-down-pad", type=int, default=45)
    parser.add_argument("--min-area", type=int, default=800)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for src_npz in sorted(src_dir.glob("*.npz")):
        sid = src_npz.stem
        dst_npz = dst_dir / src_npz.name
        src_json = src_npz.with_suffix(".json")
        dst_json = dst_dir / src_json.name

        with np.load(src_npz) as z:
            arrays = {k: z[k] for k in z.files}

        clean, meta = clean_object_mask(
            arrays["wall_normal_height"],
            arrays["valid_mask"].astype(bool),
            h_min=args.h_min,
            seed_height=args.seed_height,
            bbox_pad=args.bbox_pad,
            bbox_down_pad=args.bbox_down_pad,
            min_area=args.min_area,
        )
        arrays["object_mask_clean_v1"] = clean.astype(np.uint8)
        np.savez_compressed(dst_npz, **arrays)

        if src_json.exists():
            info = json.loads(src_json.read_text(encoding="utf-8"))
        else:
            info = {"sample_id": sid}
        info["npz"] = str(dst_npz.as_posix())
        info["object_mask_clean_v1_pixels"] = int(clean.sum())
        info["object_mask_clean_v1_rule"] = {
            "h_min_mm": args.h_min,
            "seed_height_mm": args.seed_height,
            "bbox_pad_px": args.bbox_pad,
            "bbox_down_pad_px": args.bbox_down_pad,
            "min_area_px": args.min_area,
        }
        info["object_mask_clean_v1_note"] = (
            "Derived from valid_mask and wall_normal_height for object-only evaluation. "
            "It should not replace valid_mask for full-image training/evaluation."
        )
        dst_json.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

        old_pixels = int(arrays["object_mask"].astype(bool).sum())
        clean_pixels = int(clean.sum())
        rows.append(
            {
                "sample_id": sid,
                "src_npz": str(src_npz.as_posix()),
                "dst_npz": str(dst_npz.as_posix()),
                "old_object_pixels": old_pixels,
                "clean_object_pixels": clean_pixels,
                "clean_over_old": float(clean_pixels / old_pixels) if old_pixels else 0.0,
                **meta,
            }
        )

    with (dst_dir / "cleanmask_v1_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    readme = """# orderfix_0610_cleanmask_v1

This directory mirrors `processed/orderfix_0610` and adds one extra NPZ field:

```text
object_mask_clean_v1
```

The original fields are unchanged. The original `object_mask` is retained.

Recommended use:

- `valid_mask`: full valid reconstruction region for training/evaluation.
- `object_mask`: coarse foreground region, includes supports/base fragments.
- `object_mask_clean_v1`: cleaner object-body region for object-only metrics, visualization, and optional auxiliary object-focused loss.

Do not use `object_mask_clean_v1` as the only training mask unless the task is explicitly object-body-only reconstruction.
"""
    (dst_dir / "README.md").write_text(readme, encoding="utf-8")
    print(dst_dir.resolve())


if __name__ == "__main__":
    main()
