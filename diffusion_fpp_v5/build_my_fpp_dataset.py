from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from export_cloudcompare_wall_aligned_pointclouds import fit_plane, transform_to_wall, wall_basis
from reconstruct_my_dataset_pmp_ftp import (
    compute_multifrequency_phase,
    load_pmp_stack,
    parse_pmp_calibration,
    read_gray,
    solve_projective,
)


FREQUENCIES = [1, 4, 8, 16, 32, 64]
PHASE_STEPS = 12
PMP_FRAME_COUNT = 144
SINGLE_INPUT_INDEX = 48
OBJECT_MASK_HEIGHT_THRESHOLD = 1.0


def numeric_key(text: str) -> tuple[int, str]:
    try:
        return int(text), text
    except ValueError:
        return 10**9, text


def sample_id(object_id: str, pose_name: str) -> str:
    pose_num = int(pose_name.replace("pose", ""))
    return f"obj{int(object_id):04d}_pose{pose_num:04d}"


def link_or_copy_file(src: Path, dst: Path, mode: str, overwrite: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if overwrite:
            dst.unlink()
        else:
            return "exists"
    if mode == "none":
        return "skipped"
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copied"
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlinked"
        except OSError:
            shutil.copy2(src, dst)
            return "copied_fallback"
    raise ValueError(f"Unsupported raw mode: {mode}")


def copy_pose_raw(src_pose: Path, dst_pose: Path, mode: str, overwrite: bool) -> dict[str, int]:
    counts: dict[str, int] = {}
    pmp_src = src_pose / "pmp"
    pmp_dst = dst_pose / "pmp"
    pmp_dst.mkdir(parents=True, exist_ok=True)
    for i in range(PMP_FRAME_COUNT):
        status = link_or_copy_file(pmp_src / f"{i:04d}.bmp", pmp_dst / f"{i:04d}.bmp", mode, overwrite)
        counts[status] = counts.get(status, 0) + 1
    for name in ["metadata.json", "capture_complete.flag"]:
        src = src_pose / name
        if src.exists():
            status = link_or_copy_file(src, dst_pose / name, "copy", overwrite)
            counts[status] = counts.get(status, 0) + 1
    return counts


def compute_pmp_orderfix(
    data_root: Path,
    obj: str,
    pose: str,
    cmat: np.ndarray,
    pmat: np.ndarray,
    phase_order: str = "yx",
) -> dict[str, np.ndarray]:
    stack = load_pmp_stack(data_root / obj / pose)
    phase_y, ac_y, bc_y = compute_multifrequency_phase(stack, 0)
    phase_x, ac_x, bc_x = compute_multifrequency_phase(stack, 72)
    valid_mask = (bc_y > 5.0) & (bc_x > 5.0)
    if phase_order == "yx":
        # Legacy 0609/0610 affine projector format:
        # captured Y phase maps to projector row 0, captured X phase maps to row 1.
        X, Y, Z = solve_projective(cmat, pmat, phase_y, phase_x, valid_mask)
    elif phase_order == "xy":
        # Full 0612 projector-coordinate format:
        # captured X phase maps to projector row 0, captured Y phase maps to row 1.
        X, Y, Z = solve_projective(cmat, pmat, phase_x, phase_y, valid_mask)
    else:
        raise ValueError(f"Unsupported phase_order: {phase_order}")
    valid_mask = valid_mask & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    return {
        "phase_y_capture": phase_y.astype(np.float32),
        "phase_x_capture": phase_x.astype(np.float32),
        "ac_y": ac_y.astype(np.float32),
        "ac_x": ac_x.astype(np.float32),
        "bc_y": bc_y.astype(np.float32),
        "bc_x": bc_x.astype(np.float32),
        "valid_mask": valid_mask,
        "X": X.astype(np.float32),
        "Y": Y.astype(np.float32),
        "Z": Z.astype(np.float32),
    }


def fit_reference_wall(wall: dict[str, np.ndarray]) -> dict[str, np.ndarray | float | list[float]]:
    h, w = wall["Z"].shape
    yy, xx = np.indices((h, w))
    valid = wall["valid_mask"] & np.isfinite(wall["X"]) & np.isfinite(wall["Y"]) & np.isfinite(wall["Z"])
    # Use upper wall to avoid the support table and low-reflectance bottom edge.
    fit_mask = valid & (yy < int(h * 0.72)) & (xx > 20) & (xx < w - 20)
    points = np.stack([wall["X"][fit_mask], wall["Y"][fit_mask], wall["Z"][fit_mask]], axis=1)
    center, normal, rms = fit_plane(points)
    e1, e2 = wall_basis(normal)
    wall_coords = transform_to_wall(np.stack([wall["X"], wall["Y"], wall["Z"]], axis=2), center, normal, e1, e2)
    return {
        "center": center.astype(np.float32),
        "normal": normal.astype(np.float32),
        "e1": e1.astype(np.float32),
        "e2": e2.astype(np.float32),
        "rms": float(rms),
        "fit_mask": fit_mask,
        "wall_coords": wall_coords.astype(np.float32),
    }


def split_for_object(object_id: str) -> str:
    n = int(object_id)
    if n <= 8:
        return "train"
    if n <= 10:
        return "val"
    return "test"


def image_stats(img: np.ndarray) -> dict[str, float | int]:
    return {
        "min": int(np.min(img)),
        "max": int(np.max(img)),
        "mean": float(np.mean(img)),
        "std": float(np.std(img)),
    }


def write_npz_sample(
    out_npz: Path,
    data_root: Path,
    object_id: str,
    pose_name: str,
    rec: dict[str, np.ndarray],
    wall: dict[str, np.ndarray],
    wall_model: dict[str, np.ndarray | float | list[float]],
    compressed: bool = True,
) -> dict:
    single_input = read_gray(data_root / object_id / pose_name / "pmp" / f"{SINGLE_INPUT_INDEX:04d}.bmp").astype(np.uint8)
    obj_points = np.stack([rec["X"], rec["Y"], rec["Z"]], axis=2)
    obj_coords = transform_to_wall(
        obj_points,
        wall_model["center"],  # type: ignore[arg-type]
        wall_model["normal"],  # type: ignore[arg-type]
        wall_model["e1"],  # type: ignore[arg-type]
        wall_model["e2"],  # type: ignore[arg-type]
    ).astype(np.float32)
    wall_coords = wall_model["wall_coords"]  # type: ignore[assignment]
    height = (obj_coords[:, :, 2] - wall_coords[:, :, 2]).astype(np.float32)  # type: ignore[index]
    valid_mask = (
        rec["valid_mask"]
        & wall["valid_mask"]
        & np.isfinite(height)
        & np.isfinite(rec["X"])
        & np.isfinite(rec["Y"])
        & np.isfinite(rec["Z"])
    )
    candidate = valid_mask & (np.abs(height) > OBJECT_MASK_HEIGHT_THRESHOLD)
    if np.any(candidate) and float(np.nanmedian(height[candidate])) < 0.0:
        height = -height
    object_mask = valid_mask & (height > OBJECT_MASK_HEIGHT_THRESHOLD)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_npz = np.savez_compressed if compressed else np.savez
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
        valid_mask=valid_mask.astype(np.uint8),
        object_mask=object_mask.astype(np.uint8),
        wall_plane_fit_mask=wall_model["fit_mask"].astype(np.uint8),  # type: ignore[union-attr]
    )
    vals = height[object_mask & np.isfinite(height)]
    return {
        "sample_id": sample_id(object_id, pose_name),
        "object_id": object_id,
        "pose": pose_name,
        "npz": str(out_npz.as_posix()),
        "single_input_raw": f"raw/objects/obj{int(object_id):04d}/pose{int(pose_name.replace('pose', '')):04d}/pmp/{SINGLE_INPUT_INDEX:04d}.bmp",
        "valid_pixels": int(valid_mask.sum()),
        "object_pixels": int(object_mask.sum()),
        "height_p02": float(np.percentile(vals, 2)) if vals.size else None,
        "height_p50": float(np.percentile(vals, 50)) if vals.size else None,
        "height_p98": float(np.percentile(vals, 98)) if vals.size else None,
        "single_input_stats": image_stats(single_input),
    }


def discover_samples(data_root: Path) -> list[tuple[str, str, Path]]:
    samples: list[tuple[str, str, Path]] = []
    for obj_dir in sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda p: numeric_key(p.name)):
        if obj_dir.name in {"0", "00"}:
            continue
        try:
            int(obj_dir.name)
        except ValueError:
            continue
        for pose_dir in sorted(obj_dir.glob("pose*"), key=lambda p: numeric_key(p.name.replace("pose", ""))):
            pmp = pose_dir / "pmp"
            if not pmp.exists():
                continue
            frames = list(pmp.glob("*.bmp"))
            if len(frames) == PMP_FRAME_COUNT:
                samples.append((obj_dir.name, pose_dir.name, pose_dir))
    return samples


def write_dataset_readme(path: Path, summary: dict) -> None:
    text = f"""# My FPP Dataset v1

This dataset was generated from `data_my` for single-shot FPP experiments.

## Structure

```text
my_fpp_dataset_v1/
  calibration/
  raw/
    reference_wall/
    wall_checks/
    objects/
  processed/orderfix_0610/
  splits/
```

## Important Conventions

- Reference wall: `raw/reference_wall/pose0001`, originally `data_my/0/pose1`.
- Secondary wall check: `raw/wall_checks/wall00_pose0001`, originally `data_my/00/pose1`.
- Single-frame input: `pmp/0048.bmp`.
- Teacher/GT: 6-frequency x 12-step x 2-direction PMP reconstruction.
- Phase order fix for `calibrate_0610.pmp`: captured Y phase maps to projector row 0, captured X phase maps to projector row 1.
- Main target for training: `wall_normal_height` in each processed `.npz`.
- FTP is not used as GT. FTP is only a traditional single-frame baseline/diagnostic.

## Counts

- Objects: {summary["object_count"]}
- Samples: {summary["sample_count"]}
- Train/Val/Test: {summary["split_counts"]}

## Processed NPZ Fields

```text
single_input
phase_y_capture
phase_x_capture
ac_y, ac_x
bc_y, bc_x
X, Y, Z
wall_normal_height
valid_mask
object_mask
wall_plane_fit_mask
```

## Notes

Raw BMP files are organized for traceability. They may be hardlinks to the original `data_my` files if the builder was run with `--raw-mode hardlink`.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data_my"))
    parser.add_argument("--out-root", type=Path, default=Path("my_fpp_dataset_v1"))
    parser.add_argument("--calibration", type=Path, default=Path("data_my") / "calibrate_0610.pmp")
    parser.add_argument("--reference-object", default="0")
    parser.add_argument("--reference-pose", default="pose1")
    parser.add_argument("--wall-check-object", default="00")
    parser.add_argument("--wall-check-pose", default="pose1")
    parser.add_argument("--raw-mode", choices=["hardlink", "copy", "none"], default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = args.out_root
    (out / "calibration").mkdir(parents=True, exist_ok=True)
    (out / "processed" / "orderfix_0610").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)

    cmat, pmat, cal_meta = parse_pmp_calibration(args.calibration)
    shutil.copy2(args.calibration, out / "calibration" / args.calibration.name)
    (out / "calibration" / "calibration_meta.json").write_text(
        json.dumps(
            {
                "source": str(args.calibration),
                "cm": cmat.tolist(),
                "pm": pmat.tolist(),
                "meta": cal_meta,
                "phase_order_fix": "captured Y phase -> projector row 0; captured X phase -> projector row 1",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    raw_counts: dict[str, int] = {}
    ref_src = args.data_root / args.reference_object / args.reference_pose
    ref_dst = out / "raw" / "reference_wall" / "pose0001"
    for k, v in copy_pose_raw(ref_src, ref_dst, args.raw_mode, args.overwrite).items():
        raw_counts[k] = raw_counts.get(k, 0) + v
    wall_check_src = args.data_root / args.wall_check_object / args.wall_check_pose
    if wall_check_src.exists():
        wall_check_dst = out / "raw" / "wall_checks" / f"wall{args.wall_check_object}_pose0001"
        for k, v in copy_pose_raw(wall_check_src, wall_check_dst, args.raw_mode, args.overwrite).items():
            raw_counts[k] = raw_counts.get(k, 0) + v

    wall = compute_pmp_orderfix(args.data_root, args.reference_object, args.reference_pose, cmat, pmat)
    wall_model = fit_reference_wall(wall)

    samples = discover_samples(args.data_root)
    rows: list[dict] = []
    splits = {"train": [], "val": [], "test": []}
    for idx, (obj, pose, pose_dir) in enumerate(samples, start=1):
        sid = sample_id(obj, pose)
        print(f"[{idx:03d}/{len(samples):03d}] building {sid}")
        pose_num = int(pose.replace("pose", ""))
        raw_dst = out / "raw" / "objects" / f"obj{int(obj):04d}" / f"pose{pose_num:04d}"
        for k, v in copy_pose_raw(pose_dir, raw_dst, args.raw_mode, args.overwrite).items():
            raw_counts[k] = raw_counts.get(k, 0) + v
        rec = compute_pmp_orderfix(args.data_root, obj, pose, cmat, pmat)
        npz_path = out / "processed" / "orderfix_0610" / f"{sid}.npz"
        row = write_npz_sample(npz_path, args.data_root, obj, pose, rec, wall, wall_model)
        split = split_for_object(obj)
        row["split"] = split
        rows.append(row)
        splits[split].append(sid)
        sidecar = npz_path.with_suffix(".json")
        sidecar.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

    for split, ids in splits.items():
        (out / "splits" / f"{split}.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")

    manifest_path = out / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id",
            "split",
            "object_id",
            "pose",
            "npz",
            "valid_pixels",
            "object_pixels",
            "height_p02",
            "height_p50",
            "height_p98",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    split_counts = {k: len(v) for k, v in splits.items()}
    summary = {
        "dataset": "my_fpp_dataset_v1",
        "source_data_root": str(args.data_root),
        "output_root": str(out),
        "calibration": str(args.calibration),
        "reference_wall": f"{args.reference_object}/{args.reference_pose}",
        "wall_check": f"{args.wall_check_object}/{args.wall_check_pose}",
        "object_count": len(sorted({obj for obj, _, _ in samples}, key=numeric_key)),
        "sample_count": len(samples),
        "split_counts": split_counts,
        "raw_mode": args.raw_mode,
        "raw_counts": raw_counts,
        "processed_dir": str((out / "processed" / "orderfix_0610").as_posix()),
        "manifest": str(manifest_path.as_posix()),
        "wall_plane": {
            "center": np.asarray(wall_model["center"]).tolist(),
            "normal": np.asarray(wall_model["normal"]).tolist(),
            "rms": wall_model["rms"],
            "fit_rule": "0/pose1 valid upper wall, row < 0.72H, 20 < col < W-20",
        },
        "single_input_index": SINGLE_INPUT_INDEX,
        "single_input_file": f"pmp/{SINGLE_INPUT_INDEX:04d}.bmp",
        "phase_order_fix": "captured Y phase -> projector row 0; captured X phase -> projector row 1",
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_dataset_readme(out / "README.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
