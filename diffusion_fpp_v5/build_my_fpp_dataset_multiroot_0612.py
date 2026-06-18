from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from build_my_fpp_dataset import (
    OBJECT_MASK_HEIGHT_THRESHOLD,
    PMP_FRAME_COUNT,
    SINGLE_INPUT_INDEX,
    compute_pmp_orderfix,
    copy_pose_raw,
    fit_reference_wall,
    image_stats,
    numeric_key,
    sample_id,
)
from export_cloudcompare_wall_aligned_pointclouds import transform_to_wall
from generate_clean_object_masks_my_dataset import clean_object_mask
from reconstruct_my_dataset_pmp_ftp import parse_pmp_calibration, read_gray


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def source_root_for_object(object_id: str, roots: list[Path]) -> Path:
    for root in roots:
        if (root / object_id).is_dir():
            return root
    raise FileNotFoundError(f"Object {object_id} was not found in roots: {roots}")


def discover_multiroot_samples(roots: list[Path], min_object: int, max_object: int) -> list[tuple[str, str, Path, Path]]:
    samples: list[tuple[str, str, Path, Path]] = []
    for obj_num in range(min_object, max_object + 1):
        obj = str(obj_num)
        try:
            root = source_root_for_object(obj, roots)
        except FileNotFoundError:
            continue
        obj_dir = root / obj
        for pose_dir in sorted(obj_dir.glob("pose*"), key=lambda p: numeric_key(p.name.replace("pose", ""))):
            pmp_dir = pose_dir / "pmp"
            if not pmp_dir.is_dir():
                continue
            if len(list(pmp_dir.glob("*.bmp"))) != PMP_FRAME_COUNT:
                continue
            samples.append((obj, pose_dir.name, pose_dir, root))
    return samples


def parse_sample_key(text: str) -> tuple[str, str]:
    if ":" in text:
        obj, pose = text.split(":", 1)
        return str(int(obj)), f"pose{int(pose.lower().replace('pose', ''))}"
    if "_pose" in text:
        obj_part, pose_part = text.split("_pose", 1)
        return str(int(obj_part.replace("obj", ""))), f"pose{int(pose_part)}"
    raise ValueError(f"Unsupported priority sample notation: {text}")


def sort_with_priority(samples: list[tuple[str, str, Path, Path]], priority_items: list[str]) -> list[tuple[str, str, Path, Path]]:
    priority = {parse_sample_key(item): idx for idx, item in enumerate(priority_items)}

    def key(item: tuple[str, str, Path, Path]) -> tuple[int, int, int]:
        obj, pose, _, _ = item
        if (obj, pose) in priority:
            return 0, priority[(obj, pose)], 0
        return 1, int(obj), int(pose.replace("pose", ""))

    return sorted(samples, key=key)


def infer_phase_order(pmat: np.ndarray, requested: str) -> str:
    if requested != "auto":
        return requested
    if np.linalg.norm(pmat[2]) > 1e-8:
        return "xy"
    return "yx"


def split_for_object_0612(object_id: str) -> str:
    n = int(object_id)
    if n >= 61:
        return "test"
    if n >= 55:
        return "val"
    return "train"


def write_npz_sample_with_clean(
    out_npz: Path,
    data_root: Path,
    object_id: str,
    pose_name: str,
    rec: dict[str, np.ndarray],
    wall: dict[str, np.ndarray],
    wall_model: dict[str, np.ndarray | float | list[float]],
    args: argparse.Namespace,
) -> tuple[dict, dict[str, float | int]]:
    single_input = read_gray(data_root / object_id / pose_name / "pmp" / f"{SINGLE_INPUT_INDEX:04d}.bmp").astype(np.uint8)
    obj_points = np.stack([rec["X"], rec["Y"], rec["Z"]], axis=2)
    obj_coords_ref = transform_to_wall(
        obj_points,
        wall_model["center"],  # type: ignore[arg-type]
        wall_model["normal"],  # type: ignore[arg-type]
        wall_model["e1"],  # type: ignore[arg-type]
        wall_model["e2"],  # type: ignore[arg-type]
    ).astype(np.float32)
    wall_coords = wall_model["wall_coords"]  # type: ignore[assignment]
    reference_height = (obj_coords_ref[:, :, 2] - wall_coords[:, :, 2]).astype(np.float32)  # type: ignore[index]
    reference_valid = (
        rec["valid_mask"]
        & wall["valid_mask"]
        & np.isfinite(reference_height)
        & np.isfinite(rec["X"])
        & np.isfinite(rec["Y"])
        & np.isfinite(rec["Z"])
    )

    if args.height_mode == "reference_wall":
        height = reference_height
        valid_mask = reference_valid
        sample_plane_fit_mask = np.zeros_like(valid_mask, dtype=bool)
        sample_plane_meta = {}
    elif args.height_mode == "sample_plane":
        h_img, w_img = rec["Z"].shape
        yy, xx = np.indices((h_img, w_img))
        sample_plane_fit_mask = (
            rec["valid_mask"]
            & np.isfinite(rec["X"])
            & np.isfinite(rec["Y"])
            & np.isfinite(rec["Z"])
            & (yy < int(0.58 * h_img))
            & ((xx < int(0.28 * w_img)) | (xx > int(0.72 * w_img)) | (yy < int(0.22 * h_img)))
        )
        if int(sample_plane_fit_mask.sum()) < 5000:
            sample_plane_fit_mask = (
                rec["valid_mask"]
                & np.isfinite(rec["X"])
                & np.isfinite(rec["Y"])
                & np.isfinite(rec["Z"])
                & (yy < int(0.50 * h_img))
            )
        from export_cloudcompare_wall_aligned_pointclouds import fit_plane, wall_basis

        points = np.stack(
            [rec["X"][sample_plane_fit_mask], rec["Y"][sample_plane_fit_mask], rec["Z"][sample_plane_fit_mask]],
            axis=1,
        )
        center, normal, rms = fit_plane(points)
        e1, e2 = wall_basis(normal)
        sample_coords = transform_to_wall(obj_points, center, normal, e1, e2).astype(np.float32)
        height = sample_coords[:, :, 2].astype(np.float32)
        valid_mask = (
            rec["valid_mask"]
            & np.isfinite(height)
            & np.isfinite(rec["X"])
            & np.isfinite(rec["Y"])
            & np.isfinite(rec["Z"])
        )
        vals = height[valid_mask]
        if vals.size and abs(float(np.percentile(vals, 1))) > abs(float(np.percentile(vals, 99))):
            height = -height
            normal = -normal
        sample_plane_meta = {
            "sample_plane_center": center.astype(float).tolist(),
            "sample_plane_normal": normal.astype(float).tolist(),
            "sample_plane_rms": float(rms),
            "sample_plane_fit_pixels": int(sample_plane_fit_mask.sum()),
        }
    else:
        raise ValueError(f"Unsupported height_mode: {args.height_mode}")

    candidate = valid_mask & (np.abs(height) > args.object_height_threshold)
    if np.any(candidate) and float(np.nanmedian(height[candidate])) < 0.0:
        height = -height
    object_mask = valid_mask & (height > args.object_height_threshold)
    clean, meta = clean_object_mask(
        height,
        valid_mask,
        h_min=args.h_min,
        seed_height=args.seed_height,
        bbox_pad=args.bbox_pad,
        bbox_down_pad=args.bbox_down_pad,
        min_area=args.min_area,
    )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_npz = np.savez_compressed if args.npz_compression == "compressed" else np.savez
    save_npz(
        out_npz,
        single_input=single_input,
        phase_y_capture=rec["phase_y_capture"],
        phase_x_capture=rec["phase_x_capture"],
        ac_y=rec["ac_y"],
        ac_x=rec["ac_x"],
        bc_y=rec["bc_y"],
        bc_x=rec["bc_x"],
        X=rec["X"],
        Y=rec["Y"],
        Z=rec["Z"],
        wall_normal_height=height,
        reference_wall_normal_height=reference_height,
        valid_mask=valid_mask.astype(np.uint8),
        object_mask=object_mask.astype(np.uint8),
        object_mask_clean_v1=clean.astype(np.uint8),
        wall_plane_fit_mask=wall_model["fit_mask"].astype(np.uint8),  # type: ignore[union-attr]
        sample_plane_fit_mask=sample_plane_fit_mask.astype(np.uint8),
    )

    vals = height[object_mask & np.isfinite(height)]
    meta = dict(meta)
    meta["object_mask_clean_v1_pixels"] = int(clean.sum())
    row = {
        "sample_id": sample_id(object_id, pose_name),
        "object_id": object_id,
        "pose": pose_name,
        "npz": str(out_npz.as_posix()),
        "single_input_raw": str(data_root / object_id / pose_name / "pmp" / f"{SINGLE_INPUT_INDEX:04d}.bmp"),
        "valid_pixels": int(valid_mask.sum()),
        "object_pixels": int(object_mask.sum()),
        "height_p02": float(np.percentile(vals, 2)) if vals.size else None,
        "height_p50": float(np.percentile(vals, 50)) if vals.size else None,
        "height_p98": float(np.percentile(vals, 98)) if vals.size else None,
        "single_input_stats": image_stats(single_input),
        **sample_plane_meta,
    }
    return row, meta


def write_readme(out_root: Path, summary: dict) -> None:
    text = f"""# My FPP Dataset v2 0612

This dataset was generated from two local raw-data roots:

- `{summary["source_roots"][0]}`
- `{summary["source_roots"][1]}`

It excludes the old object range used in v1 and processes objects `{summary["object_range"][0]}` to `{summary["object_range"][1]}` with the latest 0612 calibration.

## Processed Directory

```text
processed/orderfix_0612_cleanmask_v1/
```

Each sample is a compressed NPZ with:

```text
single_input
phase_y_capture, phase_x_capture
ac_y, ac_x, bc_y, bc_x
X, Y, Z
wall_normal_height
valid_mask
object_mask
object_mask_clean_v1
wall_plane_fit_mask
```

## Split Rule

```text
objects 13-54 -> train
objects 55-60 -> val
objects 61-64 -> test
```

This follows the collection note that objects after 60 are held for final testing. Object 60 is kept in validation for now so it can be inspected without touching the final test group.

## Important Boundaries

- `single_input` is `pmp/0048.bmp`, the Y-direction f=32 phase-step-0 fringe.
- PMP frames are used only to generate teacher depth/phase/XYZ.
- `valid_mask` is the main valid reconstruction region.
- `object_mask_clean_v1` is for object-only visualization, metrics, and optional auxiliary losses. It should not replace `valid_mask` for full-image training unless the task is explicitly object-body-only.
- Raw BMP frames are not copied by default. Source raw paths are recorded in the manifest and sidecar JSON files.

## Calibration

Calibration file: `{summary["calibration"]}`

Phase order convention:

```text
{summary["phase_order"]}
```

Height target mode:

```text
{summary["height_mode"]}
```
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data_my"))
    parser.add_argument("--extra-root", type=Path, default=Path(r"I:\cyf"))
    parser.add_argument("--out-root", type=Path, default=Path(r"I:\cyf\my_fpp_dataset_v2_0612_new"))
    parser.add_argument("--calibration", type=Path, default=Path("data_my") / "calibrate_0612.cp")
    parser.add_argument("--reference-root", type=Path, default=Path("data_my"))
    parser.add_argument("--reference-object", default="0")
    parser.add_argument("--reference-pose", default="pose1")
    parser.add_argument("--wall-check-object", default="00")
    parser.add_argument("--wall-check-pose", default="pose1")
    parser.add_argument("--min-object", type=int, default=13)
    parser.add_argument("--max-object", type=int, default=64)
    parser.add_argument("--raw-mode", choices=["none", "hardlink", "copy"], default="none")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--h-min", type=float, default=1.0)
    parser.add_argument("--seed-height", type=float, default=2.0)
    parser.add_argument("--bbox-pad", type=int, default=24)
    parser.add_argument("--bbox-down-pad", type=int, default=45)
    parser.add_argument("--min-area", type=int, default=800)
    parser.add_argument("--npz-compression", choices=["stored", "compressed"], default="stored")
    parser.add_argument("--phase-order", choices=["auto", "yx", "xy"], default="auto")
    parser.add_argument("--height-mode", choices=["reference_wall", "sample_plane"], default="sample_plane")
    parser.add_argument("--object-height-threshold", type=float, default=3.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument(
        "--priority-samples",
        nargs="+",
        default=["22:pose2", "50:pose2", "60:pose2", "61:pose2", "62:pose1", "63:pose1", "64:pose1", "64:pose2"],
    )
    args = parser.parse_args()

    roots = [resolve_path(args.data_root), resolve_path(args.extra_root)]
    out_root = resolve_path(args.out_root)
    processed_dir = out_root / "processed" / "orderfix_0612_cleanmask_v1"
    processed_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "splits").mkdir(parents=True, exist_ok=True)
    (out_root / "calibration").mkdir(parents=True, exist_ok=True)

    cmat, pmat, cal_meta = parse_pmp_calibration(resolve_path(args.calibration))
    phase_order = infer_phase_order(pmat, args.phase_order)
    shutil.copy2(args.calibration, out_root / "calibration" / args.calibration.name)
    (out_root / "calibration" / "calibration_meta.json").write_text(
        json.dumps(
            {
                "source": str(resolve_path(args.calibration)),
                "cm": cmat.tolist(),
                "pm": pmat.tolist(),
                "meta": cal_meta,
                "phase_order": phase_order,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    reference_root = resolve_path(args.reference_root)
    wall = compute_pmp_orderfix(reference_root, args.reference_object, args.reference_pose, cmat, pmat, phase_order=phase_order)
    wall_model = fit_reference_wall(wall)

    raw_counts: dict[str, int] = {}
    if args.raw_mode != "none":
        ref_src = reference_root / args.reference_object / args.reference_pose
        ref_dst = out_root / "raw" / "reference_wall" / "pose0001"
        for k, v in copy_pose_raw(ref_src, ref_dst, args.raw_mode, args.overwrite).items():
            raw_counts[k] = raw_counts.get(k, 0) + v
        wall_check_src = reference_root / args.wall_check_object / args.wall_check_pose
        if wall_check_src.exists():
            wall_check_dst = out_root / "raw" / "wall_checks" / f"wall{args.wall_check_object}_pose0001"
            for k, v in copy_pose_raw(wall_check_src, wall_check_dst, args.raw_mode, args.overwrite).items():
                raw_counts[k] = raw_counts.get(k, 0) + v

    samples = sort_with_priority(discover_multiroot_samples(roots, args.min_object, args.max_object), args.priority_samples)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    rows: list[dict] = []
    clean_rows: list[dict] = []
    splits = {"train": [], "val": [], "test": []}

    for idx, (obj, pose, pose_dir, root) in enumerate(samples, start=1):
        sid = sample_id(obj, pose)
        print(f"[{idx:03d}/{len(samples):03d}] building {sid} from {root}")
        npz_path = processed_dir / f"{sid}.npz"
        sidecar_path = npz_path.with_suffix(".json")
        if args.resume and npz_path.exists() and npz_path.stat().st_size > 0 and sidecar_path.exists():
            row = json.loads(sidecar_path.read_text(encoding="utf-8"))
            split = row.get("split", split_for_object_0612(obj))
            rows.append(row)
            splits[split].append(sid)
            clean_rows.append({"sample_id": sid, "status": "resumed"})
            print(f"[{idx:03d}/{len(samples):03d}] skipped existing {sid}")
            continue
        if args.raw_mode != "none":
            pose_num = int(pose.replace("pose", ""))
            raw_dst = out_root / "raw" / "objects" / f"obj{int(obj):04d}" / f"pose{pose_num:04d}"
            for k, v in copy_pose_raw(pose_dir, raw_dst, args.raw_mode, args.overwrite).items():
                raw_counts[k] = raw_counts.get(k, 0) + v

        rec = compute_pmp_orderfix(root, obj, pose, cmat, pmat, phase_order=phase_order)
        row, clean_meta = write_npz_sample_with_clean(npz_path, root, obj, pose, rec, wall, wall_model, args)

        split = split_for_object_0612(obj)
        row["split"] = split
        row["source_root"] = str(root)
        row["source_pose_dir"] = str(pose_dir)
        row["single_input_raw"] = str((pose_dir / "pmp" / f"{SINGLE_INPUT_INDEX:04d}.bmp"))
        row["object_mask_clean_v1_pixels"] = clean_meta["object_mask_clean_v1_pixels"]
        row["object_mask_clean_v1_rule"] = {
            "h_min_mm": args.h_min,
            "seed_height_mm": args.seed_height,
            "object_height_threshold": args.object_height_threshold,
            "bbox_pad_px": args.bbox_pad,
            "bbox_down_pad_px": args.bbox_down_pad,
            "min_area_px": args.min_area,
        }
        rows.append(row)
        splits[split].append(sid)
        clean_rows.append({"sample_id": sid, **clean_meta})
        sidecar_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

    for split, ids in splits.items():
        (out_root / "splits" / f"{split}.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "sample_id",
            "split",
            "object_id",
            "pose",
            "npz",
            "source_root",
            "source_pose_dir",
            "single_input_raw",
            "valid_pixels",
            "object_pixels",
            "object_mask_clean_v1_pixels",
            "height_p02",
            "height_p50",
            "height_p98",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    clean_manifest_path = processed_dir / "cleanmask_v1_manifest.csv"
    with clean_manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in clean_rows for k in row.keys()}) if clean_rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(clean_rows)

    split_counts = {k: len(v) for k, v in splits.items()}
    summary = {
        "dataset": "my_fpp_dataset_v2_0612_new",
        "source_roots": [str(p) for p in roots],
        "output_root": str(out_root),
        "calibration": str(resolve_path(args.calibration)),
        "processed_dir": str(processed_dir),
        "object_range": [args.min_object, args.max_object],
        "reference_wall": f"{args.reference_object}/{args.reference_pose}",
        "reference_root": str(reference_root),
        "object_count": len(sorted({obj for obj, _, _, _ in samples}, key=numeric_key)),
        "sample_count": len(samples),
        "split_counts": split_counts,
        "split_rule": "13-54 train, 55-60 val, 61-64 test",
        "priority_samples": args.priority_samples,
        "resume": bool(args.resume),
        "raw_mode": args.raw_mode,
        "raw_counts": raw_counts,
        "npz_compression": args.npz_compression,
        "manifest": str(manifest_path),
        "cleanmask_manifest": str(clean_manifest_path),
        "wall_plane": {
            "center": np.asarray(wall_model["center"]).tolist(),
            "normal": np.asarray(wall_model["normal"]).tolist(),
            "rms": wall_model["rms"],
            "fit_rule": "0/pose1 valid upper wall, row < 0.72H, 20 < col < W-20",
        },
        "single_input_index": SINGLE_INPUT_INDEX,
        "single_input_file": f"pmp/{SINGLE_INPUT_INDEX:04d}.bmp",
        "phase_order": phase_order,
        "height_mode": args.height_mode,
        "object_height_threshold": args.object_height_threshold,
    }
    (out_root / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(out_root, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
