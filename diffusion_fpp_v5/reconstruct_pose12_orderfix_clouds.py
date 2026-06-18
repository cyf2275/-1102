from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from export_cloudcompare_wall_aligned_pointclouds import (
    colorize,
    fit_plane,
    robust_range,
    transform_to_wall,
    wall_basis,
    write_ply,
)
from reconstruct_my_dataset_pmp_ftp import (
    compute_ftp_phase_axis,
    compute_multifrequency_phase,
    load_pmp_stack,
    parse_pmp_calibration,
    read_gray,
    solve_projective,
)
from reconstruct_single_ftp_pose import solve_single_phase_axis


DATA = Path("data_my")
CAL = DATA / "calibrate_0610.pmp"


def compute_pmp_orderfix(obj: str, pose: str, cmat: np.ndarray, pmat: np.ndarray) -> dict[str, np.ndarray]:
    stack = load_pmp_stack(DATA / obj / pose)
    # Captured order:
    #   0000-0071: horizontal stripes, phase varies along image y.
    #   0072-0143: vertical stripes, phase varies along image x.
    # The 0610 calibration matrix rows empirically match the opposite naming:
    # projector row 0 <- captured Y phase, projector row 1 <- captured X phase.
    phase_y, _, bc_y = compute_multifrequency_phase(stack, 0)
    phase_x, _, bc_x = compute_multifrequency_phase(stack, 72)
    mask = (bc_y > 5.0) & (bc_x > 5.0)
    X, Y, Z = solve_projective(cmat, pmat, phase_y, phase_x, mask)
    return {
        "phase_y_capture": phase_y,
        "phase_x_capture": phase_x,
        "mask": mask,
        "X": X,
        "Y": Y,
        "Z": Z,
    }


def compute_single_ftp(
    wall: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    cmat: np.ndarray,
    pmat: np.ndarray,
    reference_object: str,
    reference_pose: str,
    object_id: str,
    pose: str,
    frame_index: int,
    fft_axis: str,
    projector_row: str,
) -> dict[str, np.ndarray | dict]:
    ref_img = read_gray(DATA / reference_object / reference_pose / "pmp" / f"{frame_index:04d}.bmp")
    obj_img = read_gray(DATA / object_id / pose / "pmp" / f"{frame_index:04d}.bmp")
    ref_phase = wall["phase_y_capture"] if fft_axis == "y" else wall["phase_x_capture"]
    target_phase = target["phase_y_capture"] if fft_axis == "y" else target["phase_x_capture"]
    phase, mask, info = compute_ftp_phase_axis(
        ref_img,
        obj_img,
        ref_phase,
        wall["mask"],
        target["mask"],
        target_phase,
        frequency=32,
        axis=fft_axis,
    )
    X, Y, Z = solve_single_phase_axis(cmat, pmat, phase, mask, projector_row)
    good = mask & target["mask"] & np.isfinite(Z)
    return {"phase": phase, "mask": good, "X": X, "Y": Y, "Z": Z, "info": info}


def wall_align(
    wall: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    ftp_h: dict[str, np.ndarray | dict],
    ftp_v: dict[str, np.ndarray | dict],
) -> dict:
    h, w = wall["Z"].shape
    yy, xx = np.indices((h, w))
    wall_valid = wall["mask"] & np.isfinite(wall["X"]) & np.isfinite(wall["Y"]) & np.isfinite(wall["Z"])
    wall_fit = wall_valid & (yy < int(h * 0.72)) & (xx > 20) & (xx < w - 20)
    wall_pts = np.stack([wall["X"][wall_fit], wall["Y"][wall_fit], wall["Z"][wall_fit]], axis=1)
    center, normal, plane_rms = fit_plane(wall_pts)
    e1, e2 = wall_basis(normal)

    def coords(rec: dict[str, np.ndarray]) -> np.ndarray:
        return transform_to_wall(np.stack([rec["X"], rec["Y"], rec["Z"]], axis=2), center, normal, e1, e2)

    wall_c = coords(wall)
    target_c = coords(target)
    ftp_h_c = coords(ftp_h)  # type: ignore[arg-type]
    ftp_v_c = coords(ftp_v)  # type: ignore[arg-type]

    pmp_height = target_c[:, :, 2] - wall_c[:, :, 2]
    ftp_h_height = ftp_h_c[:, :, 2] - wall_c[:, :, 2]
    ftp_v_height = ftp_v_c[:, :, 2] - wall_c[:, :, 2]
    common = wall_valid & target["mask"] & np.isfinite(pmp_height)
    fg0 = common & (np.abs(pmp_height) > 0.8)
    if np.any(fg0) and np.nanmedian(pmp_height[fg0]) < 0:
        pmp_height = -pmp_height
        ftp_h_height = -ftp_h_height
        ftp_v_height = -ftp_v_height

    foreground = common & (pmp_height > 1.0)
    core_thr = float(np.nanpercentile(pmp_height[foreground], 75)) if np.any(foreground) else 2.0
    object_core = foreground & (pmp_height >= core_thr)
    return {
        "center": center,
        "normal": normal,
        "plane_rms": plane_rms,
        "wall_coords": wall_c,
        "target_coords": target_c,
        "ftp_h_coords": ftp_h_c,
        "ftp_v_coords": ftp_v_c,
        "pmp_height": pmp_height,
        "ftp_h_height": ftp_h_height,
        "ftp_v_height": ftp_v_height,
        "foreground": foreground,
        "object_core": object_core,
        "core_threshold": core_thr,
    }


def write_raw_ply(path: Path, rec: dict[str, np.ndarray], mask: np.ndarray, scalar: np.ndarray, scalar_name: str) -> int:
    sel = mask & np.isfinite(rec["X"]) & np.isfinite(rec["Y"]) & np.isfinite(rec["Z"]) & np.isfinite(scalar)
    idx = np.flatnonzero(sel)
    pts = np.stack([rec["X"].flat[idx], rec["Y"].flat[idx], rec["Z"].flat[idx]], axis=1)
    vals = scalar.flat[idx]
    vmin, vmax = robust_range(vals)
    colors = colorize(vals, vmin, vmax, "turbo")
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"property float {scalar_name}\n")
        f.write("end_header\n")
        for p, c, s in zip(pts, colors, vals):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])} {float(s):.6f}\n")
    return int(pts.shape[0])


def save_preview(path: Path, img: np.ndarray, pmp_h: np.ndarray, ftp_h: np.ndarray, ftp_v: np.ndarray, mask: np.ndarray) -> None:
    err_h = np.abs(ftp_h - pmp_h)
    err_v = np.abs(ftp_v - pmp_h)
    panels = [
        (img, np.ones_like(mask, dtype=bool), "single input 0048", "gray", None),
        (pmp_h, mask, "PMP orderfix height", "coolwarm", "signed"),
        (ftp_h, mask, "single FTP-H/0048 height", "coolwarm", "signed"),
        (err_h, mask, "|FTP-H - PMP|", "magma", "error"),
        (ftp_v, mask, "single FTP-V/0120 height", "coolwarm", "signed"),
        (err_v, mask, "|FTP-V - PMP|", "magma", "error"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (arr, m, title, cmap, mode) in zip(axes.ravel(), panels):
        if mode is None:
            im = ax.imshow(arr, cmap=cmap)
        else:
            vals = arr[m & np.isfinite(arr)]
            if vals.size:
                if mode == "error":
                    vmin, vmax = 0.0, float(np.nanpercentile(vals, 98))
                else:
                    vmax = float(np.nanpercentile(np.abs(vals), 98))
                    vmin = -vmax
            else:
                vmin, vmax = 0.0, 1.0
            im = ax.imshow(np.where(m, arr, np.nan), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def metric(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> dict:
    good = mask & np.isfinite(pred) & np.isfinite(ref)
    if not np.any(good):
        return {"valid": 0}
    e = pred[good] - ref[good]
    return {
        "valid": int(good.sum()),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "mae": float(np.mean(np.abs(e))),
        "median_abs": float(np.median(np.abs(e))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", default="12")
    parser.add_argument("--pose", default="pose5")
    parser.add_argument("--reference-object", default="0")
    parser.add_argument("--reference-pose", default="pose1")
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args()

    out = args.out_root
    if out is None:
        out = Path("cloud_results") / f"A_20260610_obj{args.object}_{args.pose}_phase_orderfix_0610_ref{args.reference_object}_{args.reference_pose}"
    out.mkdir(parents=True, exist_ok=True)
    cmat, pmat, meta = parse_pmp_calibration(CAL)
    wall = compute_pmp_orderfix(args.reference_object, args.reference_pose, cmat, pmat)
    target = compute_pmp_orderfix(args.object, args.pose, cmat, pmat)
    ftp_h = compute_single_ftp(
        wall,
        target,
        cmat,
        pmat,
        args.reference_object,
        args.reference_pose,
        args.object,
        args.pose,
        frame_index=48,
        fft_axis="y",
        projector_row="x",
    )
    ftp_v = compute_single_ftp(
        wall,
        target,
        cmat,
        pmat,
        args.reference_object,
        args.reference_pose,
        args.object,
        args.pose,
        frame_index=120,
        fft_axis="x",
        projector_row="y",
    )
    aligned = wall_align(wall, target, ftp_h, ftp_v)
    fg = aligned["foreground"]
    core = aligned["object_core"]
    pmp_h = aligned["pmp_height"]
    ftp_h_height = aligned["ftp_h_height"]
    ftp_v_height = aligned["ftp_v_height"]

    counts = {
        "pmp_raw_orderfix": write_raw_ply(out / "pmp_raw_orderfix_height_color.ply", target, fg, pmp_h, "wall_normal_height"),
        "pmp_wall_height": write_ply(out / "pmp_wall_aligned_height.ply", aligned["target_coords"], pmp_h, fg, "wall_normal_height", 1.0),
        "pmp_wall_height_x10_visual": write_ply(out / "pmp_wall_aligned_height_x10_visual.ply", aligned["target_coords"], pmp_h, fg, "wall_normal_height", 10.0),
        "ftp_h_wall_height": write_ply(out / "single_ftp_h0048_wall_aligned_height.ply", aligned["ftp_h_coords"], ftp_h_height, fg & ftp_h["mask"], "wall_normal_height", 1.0),
        "ftp_h_wall_height_x10_visual": write_ply(out / "single_ftp_h0048_wall_aligned_height_x10_visual.ply", aligned["ftp_h_coords"], ftp_h_height, fg & ftp_h["mask"], "wall_normal_height", 10.0),
        "ftp_v_wall_height_x10_visual": write_ply(out / "single_ftp_v0120_wall_aligned_height_x10_visual.ply", aligned["ftp_v_coords"], ftp_v_height, fg & ftp_v["mask"], "wall_normal_height", 10.0),
        "pmp_core_x10_visual": write_ply(out / "pmp_object_core_height_x10_visual.ply", aligned["target_coords"], pmp_h, core, "wall_normal_height", 10.0),
        "ftp_h_core_x10_visual": write_ply(out / "single_ftp_h0048_object_core_height_x10_visual.ply", aligned["ftp_h_coords"], ftp_h_height, core & ftp_h["mask"], "wall_normal_height", 10.0),
    }
    img = read_gray(DATA / args.object / args.pose / "pmp" / "0048.bmp")
    save_preview(out / "orderfix_wall_height_preview.png", img, pmp_h, ftp_h_height, ftp_v_height, fg)
    np.savez_compressed(
        out / "orderfix_reconstruction_arrays.npz",
        pmp_height=pmp_h,
        ftp_h_height=ftp_h_height,
        ftp_v_height=ftp_v_height,
        foreground=fg,
        object_core=core,
        pmp_X=target["X"],
        pmp_Y=target["Y"],
        pmp_Z=target["Z"],
    )
    summary = {
        "status": "phase-order corrected diagnostic export",
        "calibration": str(CAL),
        "object": f"data_my/{args.object}/{args.pose}",
        "wall_reference": f"data_my/{args.reference_object}/{args.reference_pose}",
        "phase_order_fix": "solve_projective uses captured Y phase as projector row 0 and captured X phase as projector row 1",
        "single_ftp_h0048": "strict one-frame FTP from horizontal f32 step0 frame 0048, solved with projector row 0",
        "single_ftp_v0120": "strict one-frame FTP from vertical f32 step0 frame 0120, solved with projector row 1",
        "wall_plane_fit_rms": aligned["plane_rms"],
        "wall_plane_normal": aligned["normal"].tolist(),
        "foreground_rule": "PMP wall-normal height > 1.0 after orderfix",
        "object_core_rule": f"PMP wall-normal height >= foreground p75 = {aligned['core_threshold']:.4f}",
        "height_stats": {
            "pmp_foreground_p02_p50_p98": [float(x) for x in np.nanpercentile(pmp_h[fg], [2, 50, 98])] if np.any(fg) else None,
            "ftp_h_foreground_p02_p50_p98": [float(x) for x in np.nanpercentile(ftp_h_height[fg & ftp_h["mask"]], [2, 50, 98])] if np.any(fg & ftp_h["mask"]) else None,
            "ftp_v_foreground_p02_p50_p98": [float(x) for x in np.nanpercentile(ftp_v_height[fg & ftp_v["mask"]], [2, 50, 98])] if np.any(fg & ftp_v["mask"]) else None,
        },
        "height_metrics_to_pmp": {
            "single_ftp_h0048": metric(ftp_h_height, pmp_h, fg & ftp_h["mask"]),
            "single_ftp_v0120": metric(ftp_v_height, pmp_h, fg & ftp_v["mask"]),
        },
        "counts": counts,
        "recommended_cloudcompare_files": [
            "pmp_wall_aligned_height_x10_visual.ply",
            "single_ftp_h0048_wall_aligned_height_x10_visual.ply",
            "single_ftp_v0120_wall_aligned_height_x10_visual.ply",
            "pmp_raw_orderfix_height_color.ply",
        ],
        "calibration_meta": meta,
    }
    (out / "orderfix_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README_ORDERFIX_ASCII.txt").write_text(
        "\n".join(
            [
                "Phase-order-fixed exports for data_my/12/pose5",
                "================================================",
                "",
                "The previous point clouds used captured X phase as projector row 0 and captured Y phase as row 1.",
                "For calibrate_0610.pmp this made the shape too flat.",
                "This folder swaps the phase order: captured Y -> projector row 0, captured X -> projector row 1.",
                "",
                "Open in CloudCompare:",
                "1. pmp_wall_aligned_height_x10_visual.ply",
                "2. single_ftp_h0048_wall_aligned_height_x10_visual.ply",
                "3. single_ftp_v0120_wall_aligned_height_x10_visual.ply",
                "",
                "x10 files exaggerate wall-normal height for inspection only.",
                "For metric use pmp_wall_aligned_height.ply and single_ftp_h0048_wall_aligned_height.ply.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
