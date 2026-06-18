from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from reconstruct_my_dataset_pmp_ftp import (
    compute_ftp_phase_axis,
    compute_multifrequency_phase,
    load_pmp_stack,
    parse_pmp_calibration,
    read_gray,
    robust_range,
    save_depth_png,
    solve_projective,
    write_ply,
)


def pixel_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices((h, w), dtype=np.float64)
    return xx + 1.0, yy + 1.0


def solve_single_phase_axis(cm: np.ndarray, pm: np.ndarray, phase: np.ndarray, mask: np.ndarray, axis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = phase.shape
    xc, yc = pixel_grid(h, w)
    valid = mask & np.isfinite(phase)
    x = xc[valid]
    y = yc[valid]
    pp = phase[valid].astype(np.float64)
    rows = [
        np.stack([cm[0, 0] - cm[2, 0] * x, cm[0, 1] - cm[2, 1] * x, cm[0, 2] - cm[2, 2] * x], axis=1),
        np.stack([cm[1, 0] - cm[2, 0] * y, cm[1, 1] - cm[2, 1] * y, cm[1, 2] - cm[2, 2] * y], axis=1),
    ]
    rhs = [x - cm[0, 3], y - cm[1, 3]]
    row = 0 if axis == "x" else 1
    rows.append(np.stack([pm[row, 0] - pm[2, 0] * pp, pm[row, 1] - pm[2, 1] * pp, pm[row, 2] - pm[2, 2] * pp], axis=1))
    rhs.append(pp - pm[row, 3])
    A = np.stack(rows, axis=1)
    b = np.stack(rhs, axis=1)
    coords = np.linalg.solve(A, b)
    X = np.full((h, w), np.nan, dtype=np.float32)
    Y = np.full((h, w), np.nan, dtype=np.float32)
    Z = np.full((h, w), np.nan, dtype=np.float32)
    idx = np.flatnonzero(valid)
    X.flat[idx] = coords[:, 0].astype(np.float32)
    Y.flat[idx] = coords[:, 1].astype(np.float32)
    Z.flat[idx] = coords[:, 2].astype(np.float32)
    return X, Y, Z


def compute_pmp(data_root: Path, obj: str, pose: str, cm: np.ndarray, pm: np.ndarray) -> dict[str, np.ndarray]:
    stack = load_pmp_stack(data_root / obj / pose)
    phase_y, _, bc_y = compute_multifrequency_phase(stack, 0)
    phase_x, _, bc_x = compute_multifrequency_phase(stack, 72)
    mask = (bc_y > 5.0) & (bc_x > 5.0)
    X, Y, Z = solve_projective(cm, pm, phase_x, phase_y, mask)
    return {"phase_x": phase_x, "phase_y": phase_y, "mask": mask, "X": X, "Y": Y, "Z": Z}


def summarize(arr: np.ndarray, mask: np.ndarray) -> dict:
    vals = arr[mask & np.isfinite(arr)]
    if vals.size == 0:
        return {"valid_pixels": 0}
    return {
        "valid_pixels": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "p02": float(np.percentile(vals, 2)),
        "p98": float(np.percentile(vals, 98)),
    }


def save_phase_png(path: Path, phase: np.ndarray, mask: np.ndarray, title: str) -> None:
    vmin, vmax = robust_range(phase, mask)
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    im = ax.imshow(np.where(mask, phase, np.nan), cmap="twilight", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="unwrapped phase")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_compact(path: Path, single_img: np.ndarray, phase: np.ndarray, ftp_z: np.ndarray, pmp_z: np.ndarray, mask: np.ndarray, title: str) -> None:
    err = np.abs(ftp_z - pmp_z)
    zmin, zmax = robust_range(pmp_z, mask)
    emin, emax = 0.0, float(np.nanpercentile(err[mask & np.isfinite(err)], 98)) if np.any(mask & np.isfinite(err)) else 1.0
    panels = [
        (single_img, "single FTP input", None, None, "gray"),
        (phase, "unwrapped FTP phase", *robust_range(phase, mask), "twilight"),
        (ftp_z, "single FTP-X Z", zmin, zmax, "viridis"),
        (pmp_z, "PMP reference Z", zmin, zmax, "viridis"),
        (err, "|FTP-X - PMP|", emin, emax, "magma"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.8))
    for ax, (arr, name, vmin, vmax, cmap) in zip(axes, panels):
        if vmin is None:
            im = ax.imshow(arr, cmap=cmap)
        else:
            im = ax.imshow(np.where(mask, arr, np.nan), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data_my"))
    parser.add_argument("--calibration", type=Path, default=Path("data_my") / "calibrate_0610.pmp")
    parser.add_argument("--reference-object", default="0")
    parser.add_argument("--reference-pose", default="pose1")
    parser.add_argument("--object", default="12")
    parser.add_argument("--pose", default="pose5")
    parser.add_argument("--axis", choices=["x", "y"], default="x")
    parser.add_argument("--frame-index", type=int, default=120)
    parser.add_argument("--out-root", type=Path, default=Path("cloud_results") / "A_20260610_obj12_pose5_single_ftp_x_0610")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    cm, pm, meta = parse_pmp_calibration(args.calibration)
    ref = compute_pmp(args.data_root, args.reference_object, args.reference_pose, cm, pm)
    target_pmp = compute_pmp(args.data_root, args.object, args.pose, cm, pm)
    ref_img = read_gray(args.data_root / args.reference_object / args.reference_pose / "pmp" / f"{args.frame_index:04d}.bmp")
    obj_img = read_gray(args.data_root / args.object / args.pose / "pmp" / f"{args.frame_index:04d}.bmp")

    ref_phase = ref["phase_x"] if args.axis == "x" else ref["phase_y"]
    target_phase = target_pmp["phase_x"] if args.axis == "x" else target_pmp["phase_y"]
    ftp_phase, ftp_mask, ftp_info = compute_ftp_phase_axis(
        ref_img,
        obj_img,
        ref_phase,
        ref["mask"],
        target_pmp["mask"],
        None,
        frequency=32,
        axis=args.axis,
    )
    # The x-axis convention for the 0610 capture is plus; y-axis is kept only for diagnostics.
    X, Y, Z = solve_single_phase_axis(cm, pm, ftp_phase, ftp_mask, args.axis)
    mask = ftp_mask & target_pmp["mask"] & np.isfinite(Z)
    if np.any(mask):
        lo, hi = np.nanpercentile(Z[mask], [0.5, 99.5])
        mask = mask & (Z >= lo) & (Z <= hi)
    err_mask = mask & np.isfinite(target_pmp["Z"])
    if np.any(err_mask):
        err = Z[err_mask] - target_pmp["Z"][err_mask]
        rmse_to_pmp = float(np.sqrt(np.mean(err * err)))
        mae_to_pmp = float(np.mean(np.abs(err)))
        corr_to_pmp = float(np.corrcoef(Z[err_mask].reshape(-1), target_pmp["Z"][err_mask].reshape(-1))[0, 1])
    else:
        rmse_to_pmp = None
        mae_to_pmp = None
        corr_to_pmp = None

    np.savez_compressed(
        args.out_root / "single_ftp_reconstruction.npz",
        phase=ftp_phase,
        mask=mask,
        X=X,
        Y=Y,
        Z=Z,
        pmp_X=target_pmp["X"],
        pmp_Y=target_pmp["Y"],
        pmp_Z=target_pmp["Z"],
        pmp_phase=target_phase,
    )
    save_phase_png(args.out_root / "single_ftp_unwrapped_phase.png", ftp_phase, mask, f"{args.object}/{args.pose} single FTP-{args.axis.upper()} unwrapped phase")
    save_depth_png(args.out_root / "single_ftp_depth_Z.png", Z, mask, f"{args.object}/{args.pose} single FTP-{args.axis.upper()} Z")
    save_depth_png(args.out_root / "pmp_reference_depth_Z.png", target_pmp["Z"], target_pmp["mask"], f"{args.object}/{args.pose} PMP reference Z")
    save_compact(args.out_root / "single_ftp_vs_pmp_compact.png", obj_img, ftp_phase, Z, target_pmp["Z"], err_mask, f"{args.object}/{args.pose} strict single-frame FTP-{args.axis.upper()} with 0610 calibration")
    vertices = write_ply(args.out_root / "single_ftp_pointcloud_stride2.ply", X, Y, Z, mask, stride=2)
    pmp_vertices = write_ply(args.out_root / "pmp_reference_pointcloud_stride2.ply", target_pmp["X"], target_pmp["Y"], target_pmp["Z"], target_pmp["mask"], stride=2)

    summary = {
        "method": "Strict single-frame Fourier transform profilometry using one object image and a pre-captured wall reference",
        "object": args.object,
        "pose": args.pose,
        "reference_object": args.reference_object,
        "reference_pose": args.reference_pose,
        "calibration": str(args.calibration),
        "axis": args.axis,
        "single_frame": str(args.data_root / args.object / args.pose / "pmp" / f"{args.frame_index:04d}.bmp"),
        "reference_frame": str(args.data_root / args.reference_object / args.reference_pose / "pmp" / f"{args.frame_index:04d}.bmp"),
        "boundary": "Object reconstruction uses one deformed fringe image. The wall reference phase is precomputed from PMP as calibration/reference-plane phase; PMP target is used only for diagnostic comparison.",
        "ftp_info": ftp_info,
        "single_ftp_Z": summarize(Z, mask),
        "pmp_reference_Z": summarize(target_pmp["Z"], target_pmp["mask"]),
        "rmse_to_pmp_Z": rmse_to_pmp,
        "mae_to_pmp_Z": mae_to_pmp,
        "corr_to_pmp_Z": corr_to_pmp,
        "single_ftp_pointcloud_vertices_stride2": vertices,
        "pmp_reference_pointcloud_vertices_stride2": pmp_vertices,
        "calibration_meta": meta,
    }
    (args.out_root / "single_ftp_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
