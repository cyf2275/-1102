from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from build_my_fpp_dataset import compute_pmp_orderfix, fit_reference_wall
from export_cloudcompare_wall_aligned_pointclouds import transform_to_wall
from reconstruct_my_dataset_pmp_ftp import (
    compute_ftp_phase_axis,
    parse_pmp_calibration,
    read_gray,
    robust_range,
    solve_projective,
)


def parse_sample(text: str) -> tuple[str, str]:
    if ":" in text:
        obj, pose = text.split(":", 1)
        return str(int(obj)), f"pose{int(pose.lower().replace('pose', ''))}"
    if "_pose" in text:
        obj_part, pose_part = text.split("_pose", 1)
        obj = str(int(obj_part.replace("obj", "")))
        return obj, f"pose{int(pose_part)}"
    raise ValueError(f"Unsupported sample notation: {text}")


def source_root_for_object(object_id: str, roots: list[Path]) -> Path:
    for root in roots:
        if (root / object_id).is_dir():
            return root
    raise FileNotFoundError(f"Object {object_id} not found in roots: {roots}")


def solve_single_projector_row(cm: np.ndarray, pm: np.ndarray, phase: np.ndarray, mask: np.ndarray, projector_row: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = phase.shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    xc = xx + 1.0
    yc = yy + 1.0
    valid = mask & np.isfinite(phase)
    x = xc[valid]
    y = yc[valid]
    pp = phase[valid].astype(np.float64)
    rows = np.stack(
        [
            np.stack([cm[0, 0] - cm[2, 0] * x, cm[0, 1] - cm[2, 1] * x, cm[0, 2] - cm[2, 2] * x], axis=1),
            np.stack([cm[1, 0] - cm[2, 0] * y, cm[1, 1] - cm[2, 1] * y, cm[1, 2] - cm[2, 2] * y], axis=1),
            np.stack(
                [
                    pm[projector_row, 0] - pm[2, 0] * pp,
                    pm[projector_row, 1] - pm[2, 1] * pp,
                    pm[projector_row, 2] - pm[2, 2] * pp,
                ],
                axis=1,
            ),
        ],
        axis=1,
    )
    rhs = np.stack([x - cm[0, 3], y - cm[1, 3], pp - pm[projector_row, 3]], axis=1)
    coords = np.linalg.solve(rows, rhs)
    X = np.full((h, w), np.nan, dtype=np.float32)
    Y = np.full((h, w), np.nan, dtype=np.float32)
    Z = np.full((h, w), np.nan, dtype=np.float32)
    flat = np.flatnonzero(valid)
    X.flat[flat] = coords[:, 0].astype(np.float32)
    Y.flat[flat] = coords[:, 1].astype(np.float32)
    Z.flat[flat] = coords[:, 2].astype(np.float32)
    return X, Y, Z


def sparse_points(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, mask: np.ndarray, stride: int) -> np.ndarray:
    sel = mask & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    sparse = np.zeros_like(sel, dtype=bool)
    sparse[::stride, ::stride] = sel[::stride, ::stride]
    return np.stack([X[sparse], Y[sparse], Z[sparse]], axis=1)


def write_color_ply(path: Path, clouds: list[tuple[str, np.ndarray, tuple[int, int, int]]]) -> int:
    total = int(sum(points.shape[0] for _, points, _ in clouds))
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {total}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uchar label\n")
        f.write("end_header\n")
        for label_id, (_, points, color) in enumerate(clouds, start=1):
            r, g, b = color
            for p in points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r} {g} {b} {label_id}\n")
    return total


def metrics_against_pmp(pmp: dict[str, np.ndarray], other: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float | int | list[float] | None]:
    common = (
        mask
        & np.isfinite(pmp["X"])
        & np.isfinite(pmp["Y"])
        & np.isfinite(pmp["Z"])
        & np.isfinite(other["X"])
        & np.isfinite(other["Y"])
        & np.isfinite(other["Z"])
    )
    if common.sum() == 0:
        return {"common_pixels": 0}
    dp = np.stack([other["X"][common] - pmp["X"][common], other["Y"][common] - pmp["Y"][common], other["Z"][common] - pmp["Z"][common]], axis=1)
    pmp_pts = np.stack([pmp["X"][common], pmp["Y"][common], pmp["Z"][common]], axis=1)
    other_pts = np.stack([other["X"][common], other["Y"][common], other["Z"][common]], axis=1)
    return {
        "common_pixels": int(common.sum()),
        "xyz_rmse": float(np.sqrt(np.mean(np.sum(dp * dp, axis=1)))),
        "z_rmse": float(np.sqrt(np.mean(dp[:, 2] * dp[:, 2]))),
        "z_mae": float(np.mean(np.abs(dp[:, 2]))),
        "median_abs_z": float(np.median(np.abs(dp[:, 2]))),
        "pmp_centroid": np.mean(pmp_pts, axis=0).tolist(),
        "other_centroid": np.mean(other_pts, axis=0).tolist(),
        "centroid_delta": (np.mean(other_pts, axis=0) - np.mean(pmp_pts, axis=0)).tolist(),
    }


def save_map_compare(path: Path, pmp_z: np.ndarray, ftp_xy_z: np.ndarray, mask: np.ndarray, title: str) -> None:
    vmin, vmax = robust_range(pmp_z, mask)
    err = np.abs(ftp_xy_z - pmp_z)
    emax = float(np.nanpercentile(err[mask & np.isfinite(err)], 98)) if np.any(mask & np.isfinite(err)) else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), constrained_layout=True)
    panels = [
        (pmp_z, "PMP Z", vmin, vmax, "viridis"),
        (ftp_xy_z, "FTP-XY Z", vmin, vmax, "viridis"),
        (err, "|FTP-XY - PMP| Z", 0.0, emax, "magma"),
    ]
    for ax, (arr, name, a, b, cmap) in zip(axes, panels):
        im = ax.imshow(np.where(mask, arr, np.nan), cmap=cmap, vmin=a, vmax=b)
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def process_sample(
    object_id: str,
    pose: str,
    roots: list[Path],
    reference_root: Path,
    out_root: Path,
    cm: np.ndarray,
    pm: np.ndarray,
    wall: dict[str, np.ndarray],
    wall_model: dict,
    stride: int,
) -> dict:
    source_root = source_root_for_object(object_id, roots)
    sid = f"obj{int(object_id):04d}_pose{int(pose.replace('pose', '')):04d}"
    out_dir = out_root / sid
    out_dir.mkdir(parents=True, exist_ok=True)

    pmp = compute_pmp_orderfix(source_root, object_id, pose, cm, pm)
    wall_y_single = read_gray(reference_root / "0" / "pose1" / "pmp" / "0048.bmp")
    wall_x_single = read_gray(reference_root / "0" / "pose1" / "pmp" / "0120.bmp")
    obj_y_single = read_gray(source_root / object_id / pose / "pmp" / "0048.bmp")
    obj_x_single = read_gray(source_root / object_id / pose / "pmp" / "0120.bmp")

    ftp_y_phase, ftp_y_mask, info_y = compute_ftp_phase_axis(
        wall_y_single,
        obj_y_single,
        wall["phase_y_capture"],
        wall["valid_mask"],
        pmp["valid_mask"],
        pmp["phase_y_capture"],
        frequency=32,
        axis="y",
    )
    ftp_x_phase, ftp_x_mask, info_x = compute_ftp_phase_axis(
        wall_x_single,
        obj_x_single,
        wall["phase_x_capture"],
        wall["valid_mask"],
        pmp["valid_mask"],
        pmp["phase_x_capture"],
        frequency=32,
        axis="x",
    )

    # 0612/orderfix convention: captured Y phase -> projector row 0; captured X phase -> projector row 1.
    ftp_y_X, ftp_y_Y, ftp_y_Z = solve_single_projector_row(cm, pm, ftp_y_phase, ftp_y_mask, projector_row=0)
    ftp_x_X, ftp_x_Y, ftp_x_Z = solve_single_projector_row(cm, pm, ftp_x_phase, ftp_x_mask, projector_row=1)
    ftp_xy_mask = ftp_y_mask & ftp_x_mask & wall["valid_mask"] & pmp["valid_mask"]
    ftp_xy_X, ftp_xy_Y, ftp_xy_Z = solve_projective(cm, pm, ftp_y_phase, ftp_x_phase, ftp_xy_mask)

    obj_points = np.stack([pmp["X"], pmp["Y"], pmp["Z"]], axis=2)
    obj_coords = transform_to_wall(obj_points, wall_model["center"], wall_model["normal"], wall_model["e1"], wall_model["e2"])
    wall_coords = wall_model["wall_coords"]
    pmp_height = obj_coords[:, :, 2] - wall_coords[:, :, 2]
    object_mask = pmp["valid_mask"] & wall["valid_mask"] & np.isfinite(pmp_height) & (pmp_height > 1.0)

    pmp_dict = {"X": pmp["X"], "Y": pmp["Y"], "Z": pmp["Z"]}
    ftp_y_dict = {"X": ftp_y_X, "Y": ftp_y_Y, "Z": ftp_y_Z}
    ftp_x_dict = {"X": ftp_x_X, "Y": ftp_x_Y, "Z": ftp_x_Z}
    ftp_xy_dict = {"X": ftp_xy_X, "Y": ftp_xy_Y, "Z": ftp_xy_Z}

    masks = {
        "full": pmp["valid_mask"] & wall["valid_mask"],
        "object": object_mask,
    }
    metrics = {
        "ftp_y_single": {name: metrics_against_pmp(pmp_dict, ftp_y_dict, mask & ftp_y_mask) for name, mask in masks.items()},
        "ftp_x_single": {name: metrics_against_pmp(pmp_dict, ftp_x_dict, mask & ftp_x_mask) for name, mask in masks.items()},
        "ftp_xy": {name: metrics_against_pmp(pmp_dict, ftp_xy_dict, mask & ftp_xy_mask) for name, mask in masks.items()},
    }

    np.savez_compressed(
        out_dir / "pmp_ftp_coordinate_check.npz",
        pmp_X=pmp["X"],
        pmp_Y=pmp["Y"],
        pmp_Z=pmp["Z"],
        pmp_mask=pmp["valid_mask"].astype(np.uint8),
        pmp_object_mask=object_mask.astype(np.uint8),
        ftp_y_phase=ftp_y_phase,
        ftp_y_X=ftp_y_X,
        ftp_y_Y=ftp_y_Y,
        ftp_y_Z=ftp_y_Z,
        ftp_y_mask=ftp_y_mask.astype(np.uint8),
        ftp_x_phase=ftp_x_phase,
        ftp_x_X=ftp_x_X,
        ftp_x_Y=ftp_x_Y,
        ftp_x_Z=ftp_x_Z,
        ftp_x_mask=ftp_x_mask.astype(np.uint8),
        ftp_xy_X=ftp_xy_X,
        ftp_xy_Y=ftp_xy_Y,
        ftp_xy_Z=ftp_xy_Z,
        ftp_xy_mask=ftp_xy_mask.astype(np.uint8),
        pmp_wall_normal_height=pmp_height.astype(np.float32),
    )

    full_mask = pmp["valid_mask"] & ftp_xy_mask
    obj_mask = object_mask & ftp_xy_mask
    save_map_compare(out_dir / "ftp_xy_vs_pmp_Z_full.png", pmp["Z"], ftp_xy_Z, full_mask, f"{sid} full valid")
    save_map_compare(out_dir / "ftp_xy_vs_pmp_Z_object.png", pmp["Z"], ftp_xy_Z, obj_mask, f"{sid} object mask")

    pmp_pts = sparse_points(pmp["X"], pmp["Y"], pmp["Z"], object_mask, stride)
    ftp_y_pts = sparse_points(ftp_y_X, ftp_y_Y, ftp_y_Z, object_mask & ftp_y_mask, stride)
    ftp_x_pts = sparse_points(ftp_x_X, ftp_x_Y, ftp_x_Z, object_mask & ftp_x_mask, stride)
    ftp_xy_pts = sparse_points(ftp_xy_X, ftp_xy_Y, ftp_xy_Z, object_mask & ftp_xy_mask, stride)
    pmp_full_pts = sparse_points(pmp["X"], pmp["Y"], pmp["Z"], pmp["valid_mask"], stride)
    ftp_xy_full_pts = sparse_points(ftp_xy_X, ftp_xy_Y, ftp_xy_Z, ftp_xy_mask, stride)

    ply_counts = {
        "combined_object_pmp_blue_ftpy_green_ftpx_orange_ftpxy_red": write_color_ply(
            out_dir / "combined_object_pmp_blue_ftpy_green_ftpx_orange_ftpxy_red.ply",
            [
                ("pmp", pmp_pts, (40, 110, 255)),
                ("ftp_y_single", ftp_y_pts, (0, 210, 80)),
                ("ftp_x_single", ftp_x_pts, (255, 170, 0)),
                ("ftp_xy", ftp_xy_pts, (255, 35, 35)),
            ],
        ),
        "combined_full_pmp_blue_ftpxy_red": write_color_ply(
            out_dir / "combined_full_pmp_blue_ftpxy_red.ply",
            [
                ("pmp", pmp_full_pts, (40, 110, 255)),
                ("ftp_xy", ftp_xy_full_pts, (255, 35, 35)),
            ],
        ),
        "pmp_object_blue": write_color_ply(out_dir / "pmp_object_blue.ply", [("pmp", pmp_pts, (40, 110, 255))]),
        "ftp_xy_object_red": write_color_ply(out_dir / "ftp_xy_object_red.ply", [("ftp_xy", ftp_xy_pts, (255, 35, 35))]),
    }

    summary = {
        "sample_id": sid,
        "object_id": object_id,
        "pose": pose,
        "source_root": str(source_root),
        "coordinate_convention": "captured Y phase -> projector row 0; captured X phase -> projector row 1",
        "ftp_boundary": "FTP-Y and FTP-X use one f=32 step0 frame each plus wall reference. FTP-XY is a coordinate diagnostic using both single frames.",
        "phase_info_y": info_y,
        "phase_info_x": info_x,
        "pixels": {
            "pmp_valid": int(pmp["valid_mask"].sum()),
            "object_mask": int(object_mask.sum()),
            "ftp_y_valid": int(ftp_y_mask.sum()),
            "ftp_x_valid": int(ftp_x_mask.sum()),
            "ftp_xy_valid": int(ftp_xy_mask.sum()),
        },
        "metrics_against_pmp": metrics,
        "ply_vertices": ply_counts,
        "outputs": {
            "combined_object": str(out_dir / "combined_object_pmp_blue_ftpy_green_ftpx_orange_ftpxy_red.ply"),
            "combined_full": str(out_dir / "combined_full_pmp_blue_ftpxy_red.ply"),
            "npz": str(out_dir / "pmp_ftp_coordinate_check.npz"),
        },
    }
    (out_dir / "coordinate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data_my"))
    parser.add_argument("--extra-root", type=Path, default=Path(r"I:\cyf"))
    parser.add_argument("--reference-root", type=Path, default=Path("data_my"))
    parser.add_argument("--calibration", type=Path, default=Path("data_my") / "calibrate_0612.cp")
    parser.add_argument("--out-root", type=Path, default=Path(r"I:\cyf\my_dataset_v2_0612_checks\ftp_pmp_coordinate_0612"))
    parser.add_argument("--samples", nargs="+", default=["22:pose2", "50:pose2", "61:pose2", "62:pose1", "64:pose2"])
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    roots = [args.data_root.resolve(), args.extra_root.resolve()]
    args.out_root.mkdir(parents=True, exist_ok=True)
    cm, pm, cal_meta = parse_pmp_calibration(args.calibration.resolve())
    wall = compute_pmp_orderfix(args.reference_root.resolve(), "0", "pose1", cm, pm)
    wall_model = fit_reference_wall(wall)

    summaries = []
    for item in args.samples:
        obj, pose = parse_sample(item)
        print(f"[check] obj{int(obj):04d}_{pose}")
        summaries.append(process_sample(obj, pose, roots, args.reference_root.resolve(), args.out_root, cm, pm, wall, wall_model, args.stride))

    with (args.out_root / "coordinate_check_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "sample_id",
            "source_root",
            "pmp_valid",
            "object_mask",
            "ftp_xy_valid",
            "ftp_xy_object_z_rmse",
            "ftp_xy_object_z_mae",
            "ftp_xy_full_z_rmse",
            "combined_object_ply",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow(
                {
                    "sample_id": s["sample_id"],
                    "source_root": s["source_root"],
                    "pmp_valid": s["pixels"]["pmp_valid"],
                    "object_mask": s["pixels"]["object_mask"],
                    "ftp_xy_valid": s["pixels"]["ftp_xy_valid"],
                    "ftp_xy_object_z_rmse": s["metrics_against_pmp"]["ftp_xy"]["object"].get("z_rmse"),
                    "ftp_xy_object_z_mae": s["metrics_against_pmp"]["ftp_xy"]["object"].get("z_mae"),
                    "ftp_xy_full_z_rmse": s["metrics_against_pmp"]["ftp_xy"]["full"].get("z_rmse"),
                    "combined_object_ply": s["outputs"]["combined_object"],
                }
            )
    (args.out_root / "coordinate_check_summary.json").write_text(
        json.dumps({"calibration": str(args.calibration.resolve()), "meta": cal_meta, "samples": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(args.out_root.resolve())


if __name__ == "__main__":
    main()
