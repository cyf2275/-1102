from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from build_my_fpp_dataset import compute_pmp_orderfix, fit_reference_wall
from export_cloudcompare_wall_aligned_pointclouds import fit_plane, transform_to_wall, wall_basis
from reconstruct_my_dataset_pmp_ftp import parse_pmp_calibration, robust_range


def parse_sample(text: str) -> tuple[str, str]:
    if ":" in text:
        obj, pose = text.split(":", 1)
        return str(int(obj)), f"pose{int(pose.lower().replace('pose', ''))}"
    if "_pose" in text:
        obj_part, pose_part = text.split("_pose", 1)
        return str(int(obj_part.replace("obj", ""))), f"pose{int(pose_part)}"
    raise ValueError(f"Unsupported sample notation: {text}")


def source_root_for_object(object_id: str, roots: list[Path]) -> Path:
    for root in roots:
        if (root / object_id).is_dir():
            return root
    raise FileNotFoundError(f"Object {object_id} not found in roots: {roots}")


def sample_id(object_id: str, pose: str) -> str:
    return f"obj{int(object_id):04d}_pose{int(pose.replace('pose', '')):04d}"


def sample_plane_height(rec: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict]:
    h_img, w_img = rec["Z"].shape
    yy, xx = np.indices((h_img, w_img))
    fit_mask = (
        rec["valid_mask"]
        & np.isfinite(rec["X"])
        & np.isfinite(rec["Y"])
        & np.isfinite(rec["Z"])
        & (yy < int(0.58 * h_img))
        & ((xx < int(0.28 * w_img)) | (xx > int(0.72 * w_img)) | (yy < int(0.22 * h_img)))
    )
    if int(fit_mask.sum()) < 5000:
        fit_mask = (
            rec["valid_mask"]
            & np.isfinite(rec["X"])
            & np.isfinite(rec["Y"])
            & np.isfinite(rec["Z"])
            & (yy < int(0.50 * h_img))
        )
    points = np.stack([rec["X"][fit_mask], rec["Y"][fit_mask], rec["Z"][fit_mask]], axis=1)
    center, normal, rms = fit_plane(points)
    e1, e2 = wall_basis(normal)
    coords = transform_to_wall(np.stack([rec["X"], rec["Y"], rec["Z"]], axis=2), center, normal, e1, e2)
    height = coords[:, :, 2].astype(np.float32)
    valid = rec["valid_mask"] & np.isfinite(height)
    vals = height[valid]
    if vals.size and abs(float(np.percentile(vals, 1))) > abs(float(np.percentile(vals, 99))):
        height = -height
        normal = -normal
    meta = {
        "sample_plane_rms": float(rms),
        "sample_plane_fit_pixels": int(fit_mask.sum()),
        "sample_plane_normal": normal.astype(float).tolist(),
    }
    return height, valid, meta


def reference_wall_height(rec: dict[str, np.ndarray], wall: dict[str, np.ndarray], wall_model: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    coords = transform_to_wall(
        np.stack([rec["X"], rec["Y"], rec["Z"]], axis=2),
        wall_model["center"],
        wall_model["normal"],
        wall_model["e1"],
        wall_model["e2"],
    )
    wall_coords = wall_model["wall_coords"]
    height = (coords[:, :, 2] - wall_coords[:, :, 2]).astype(np.float32)
    valid = rec["valid_mask"] & wall["valid_mask"] & np.isfinite(height)
    return height, valid, {
        "reference_wall_rms": float(wall_model["rms"]),
        "reference_wall_normal": np.asarray(wall_model["normal"]).astype(float).tolist(),
    }


def orient_and_mask(height: np.ndarray, valid: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    h = height.copy()
    candidate = valid & np.isfinite(h) & (np.abs(h) > threshold)
    if np.any(candidate) and float(np.nanmedian(h[candidate])) < 0.0:
        h = -h
    mask = valid & np.isfinite(h) & (h > threshold)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    return h, mask


def write_ply(path: Path, rec: dict[str, np.ndarray], mask: np.ndarray, stride: int = 2) -> int:
    sel = mask & np.isfinite(rec["X"]) & np.isfinite(rec["Y"]) & np.isfinite(rec["Z"])
    sparse = np.zeros_like(sel, dtype=bool)
    sparse[::stride, ::stride] = sel[::stride, ::stride]
    pts = np.stack([rec["X"][sparse], rec["Y"][sparse], rec["Z"][sparse]], axis=1)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    return int(pts.shape[0])


def save_panel(path: Path, single: np.ndarray, panels: list[dict]) -> None:
    fig, axes = plt.subplots(len(panels), 3, figsize=(12.5, 3.7 * len(panels)), constrained_layout=True)
    if len(panels) == 1:
        axes = axes[None, :]
    for row, panel in enumerate(panels):
        height = panel["height"]
        valid = panel["valid"]
        mask = panel["mask"]
        vmin, vmax = robust_range(height, valid)
        axes[row, 0].imshow(single, cmap="gray")
        axes[row, 0].set_title("single input")
        im = axes[row, 1].imshow(np.where(valid, height, np.nan), cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"{panel['name']} height")
        fig.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.03)
        axes[row, 2].imshow(single, cmap="gray")
        rgba = np.zeros((*mask.shape, 4), dtype=float)
        rgba[mask] = [0.0, 0.8, 1.0, 0.45]
        axes[row, 2].imshow(rgba)
        axes[row, 2].set_title(f"mask>{panel['threshold']} px={int(mask.sum())}")
        for ax in axes[row]:
            ax.axis("off")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data_my"))
    parser.add_argument("--extra-root", type=Path, default=Path(r"I:\cyf"))
    parser.add_argument("--out-root", type=Path, default=Path(r"I:\cyf\my_dataset_v2_0612_checks\calibration_mode_pilot"))
    parser.add_argument("--samples", nargs="+", default=["13:pose1", "22:pose2", "50:pose2", "61:pose2", "64:pose2"])
    args = parser.parse_args()

    roots = [args.data_root.resolve(), args.extra_root.resolve()]
    args.out_root.mkdir(parents=True, exist_ok=True)

    mode_specs = [
        {
            "name": "cal0610_yx_reference_wall",
            "calibration": args.data_root / "calibrate_0610.pmp",
            "phase_order": "yx",
            "height_mode": "reference_wall",
            "threshold": 1.0,
        },
        {
            "name": "cal0610_yx_sample_plane",
            "calibration": args.data_root / "calibrate_0610.pmp",
            "phase_order": "yx",
            "height_mode": "sample_plane",
            "threshold": 3.0,
        },
        {
            "name": "cal0612_xy_sample_plane",
            "calibration": args.data_root / "calibrate_0612.cp",
            "phase_order": "xy",
            "height_mode": "sample_plane",
            "threshold": 3.0,
        },
        {
            "name": "cal0612_yx_reference_wall_bad_control",
            "calibration": args.data_root / "calibrate_0612.cp",
            "phase_order": "yx",
            "height_mode": "reference_wall",
            "threshold": 1.0,
        },
    ]

    prepared = {}
    for spec in mode_specs:
        cm, pm, meta = parse_pmp_calibration(spec["calibration"])
        wall = compute_pmp_orderfix(args.data_root, "0", "pose1", cm, pm, phase_order=spec["phase_order"])
        wall_model = fit_reference_wall(wall)
        prepared[spec["name"]] = {"cm": cm, "pm": pm, "meta": meta, "wall": wall, "wall_model": wall_model}

    rows = []
    for sample in args.samples:
        obj, pose = parse_sample(sample)
        sid = sample_id(obj, pose)
        source_root = source_root_for_object(obj, roots)
        sample_out = args.out_root / sid
        sample_out.mkdir(parents=True, exist_ok=True)
        single = cv2.imdecode(np.fromfile(str(source_root / obj / pose / "pmp" / "0048.bmp"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        panels = []
        for spec in mode_specs:
            prep = prepared[spec["name"]]
            rec = compute_pmp_orderfix(source_root, obj, pose, prep["cm"], prep["pm"], phase_order=spec["phase_order"])
            if spec["height_mode"] == "reference_wall":
                height, valid, meta_h = reference_wall_height(rec, prep["wall"], prep["wall_model"])
            else:
                height, valid, meta_h = sample_plane_height(rec)
            height, mask = orient_and_mask(height, valid, spec["threshold"])
            vals = height[valid & np.isfinite(height)]
            object_vals = height[mask & np.isfinite(height)]
            row = {
                "sample_id": sid,
                "mode": spec["name"],
                "source_root": str(source_root),
                "calibration": str(spec["calibration"]),
                "phase_order": spec["phase_order"],
                "height_mode": spec["height_mode"],
                "threshold": spec["threshold"],
                "valid_pixels": int(valid.sum()),
                "object_pixels": int(mask.sum()),
                "height_p02": float(np.percentile(vals, 2)) if vals.size else None,
                "height_p50": float(np.percentile(vals, 50)) if vals.size else None,
                "height_p98": float(np.percentile(vals, 98)) if vals.size else None,
                "object_height_p50": float(np.percentile(object_vals, 50)) if object_vals.size else None,
                "object_height_p98": float(np.percentile(object_vals, 98)) if object_vals.size else None,
                **meta_h,
            }
            rows.append(row)
            panels.append({"name": spec["name"], "height": height, "valid": valid, "mask": mask, "threshold": spec["threshold"]})
            write_ply(sample_out / f"{spec['name']}_object_stride2.ply", rec, mask, stride=2)
        save_panel(sample_out / "height_mask_compare.png", single, panels)
        print(sid)

    with (args.out_root / "calibration_mode_pilot.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.out_root / "calibration_mode_pilot.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out_root.resolve())


if __name__ == "__main__":
    main()
