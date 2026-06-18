from __future__ import annotations

import json
from pathlib import Path

import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import numpy as np

from reconstruct_my_dataset_pmp_ftp import parse_pmp_calibration
from reconstruct_single_ftp_pose import compute_pmp


DATA = Path("data_my")
CAL = DATA / "calibrate_0610.pmp"
X_DIR = Path("cloud_results") / "A_20260610_obj12_pose5_single_ftp_x_0610"
OUT = Path("cloud_results") / "A_20260610_obj12_pose5_cloudcompare_wall_aligned"


def robust_range(vals: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    a, b = np.percentile(vals, [lo, hi])
    if abs(b - a) < 1e-9:
        b = a + 1.0
    return float(a), float(b)


def colorize(vals: np.ndarray, vmin: float, vmax: float, cmap_name: str = "turbo") -> np.ndarray:
    t = np.clip((vals - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    rgb = mpl_cm.get_cmap(cmap_name)(t)[:, :3]
    return (rgb * 255.0).astype(np.uint8)


def fit_plane(points: np.ndarray, trim_iters: int = 4, keep_percentile: float = 85.0) -> tuple[np.ndarray, np.ndarray, float]:
    pts = points[np.all(np.isfinite(points), axis=1)]
    if pts.shape[0] > 120_000:
        pts = pts[np.linspace(0, pts.shape[0] - 1, 120_000).astype(int)]
    keep = np.ones(pts.shape[0], dtype=bool)
    for _ in range(trim_iters):
        cur = pts[keep]
        center = cur.mean(axis=0)
        _, _, vt = np.linalg.svd(cur - center, full_matrices=False)
        normal = vt[-1]
        residual = np.abs((pts - center) @ normal)
        keep = residual <= np.percentile(residual, keep_percentile)
    cur = pts[keep]
    center = cur.mean(axis=0)
    _, _, vt = np.linalg.svd(cur - center, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    residual = (cur - center) @ normal
    rms = float(np.sqrt(np.mean(residual * residual)))
    return center, normal, rms


def wall_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Pick a stable in-wall x axis close to the global X direction.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(ref, normal))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - np.dot(ref, normal) * normal
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(normal, e1)
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)
    return e1, e2


def transform_to_wall(points: np.ndarray, center: np.ndarray, normal: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    q = points - center
    return np.stack([q @ e1, q @ e2, q @ normal], axis=-1)


def write_ply(path: Path, coords: np.ndarray, height: np.ndarray, mask: np.ndarray, scalar_name: str, height_scale: float = 1.0) -> int:
    sel = mask & np.all(np.isfinite(coords), axis=2) & np.isfinite(height)
    idx = np.flatnonzero(sel)
    pts = coords.reshape(-1, 3)[idx].copy()
    vals = height.reshape(-1)[idx]
    pts[:, 2] = vals * height_scale
    vmin, vmax = robust_range(vals)
    colors = colorize(vals, vmin, vmax, "coolwarm")
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


def save_preview(path: Path, pmp_h: np.ndarray, ftp_h: np.ndarray, mask: np.ndarray, core: np.ndarray) -> None:
    err = np.abs(ftp_h - pmp_h)
    panels = [
        (pmp_h, mask, "PMP wall-normal height", "coolwarm"),
        (ftp_h, mask, "single FTP-X wall-normal height", "coolwarm"),
        (err, mask, "|FTP-X - PMP| height", "magma"),
        (pmp_h, core, "PMP object-core height", "coolwarm"),
        (ftp_h, core, "FTP-X object-core height", "coolwarm"),
        (err, core, "object-core height error", "magma"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, (arr, m, title, cmap) in zip(axes.ravel(), panels):
        vals = arr[m & np.isfinite(arr)]
        if vals.size:
            if "error" in title.lower():
                vmin, vmax = 0.0, float(np.percentile(vals, 98))
            else:
                vmax = float(np.percentile(np.abs(vals), 98))
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cmat, pmat, _ = parse_pmp_calibration(CAL)
    wall = compute_pmp(DATA, "0", "pose1", cmat, pmat)
    obj = compute_pmp(DATA, "12", "pose5", cmat, pmat)
    ftp = np.load(X_DIR / "single_ftp_reconstruction.npz")

    h, w = wall["Z"].shape
    yy, xx = np.indices((h, w))
    wall_valid = wall["mask"] & np.isfinite(wall["X"]) & np.isfinite(wall["Y"]) & np.isfinite(wall["Z"])
    # Use the upper wall as the reference plane. The lower region contains the support table.
    wall_fit = wall_valid & (yy < int(h * 0.72)) & (xx > 20) & (xx < w - 20)
    wall_pts = np.stack([wall["X"][wall_fit], wall["Y"][wall_fit], wall["Z"][wall_fit]], axis=1)
    center, normal, plane_rms = fit_plane(wall_pts)
    e1, e2 = wall_basis(normal)

    wall_coords = transform_to_wall(np.stack([wall["X"], wall["Y"], wall["Z"]], axis=2), center, normal, e1, e2)
    obj_coords = transform_to_wall(np.stack([obj["X"], obj["Y"], obj["Z"]], axis=2), center, normal, e1, e2)
    ftp_coords = transform_to_wall(np.stack([ftp["X"], ftp["Y"], ftp["Z"]], axis=2), center, normal, e1, e2)

    common = wall_valid & obj["mask"] & np.isfinite(obj_coords[:, :, 2])
    pmp_height = obj_coords[:, :, 2] - wall_coords[:, :, 2]
    ftp_height = ftp_coords[:, :, 2] - wall_coords[:, :, 2]
    # The sign depends on normal orientation. Use the sign with positive foreground median for readability.
    fg0 = common & (np.abs(pmp_height) > 0.8)
    if np.any(fg0) and np.nanmedian(pmp_height[fg0]) < 0:
        pmp_height = -pmp_height
        ftp_height = -ftp_height
        obj_coords[:, :, 2] = -obj_coords[:, :, 2]
        ftp_coords[:, :, 2] = -ftp_coords[:, :, 2]

    foreground = common & (pmp_height > 0.8)
    object_core = common & (pmp_height > 1.6)
    ftp_foreground = foreground & ftp["mask"].astype(bool) & np.isfinite(ftp_height)
    ftp_core = object_core & ftp["mask"].astype(bool) & np.isfinite(ftp_height)

    counts = {
        "pmp_wall_aligned_height": write_ply(OUT / "pmp_wall_aligned_height.ply", obj_coords, pmp_height, foreground, "wall_normal_height", 1.0),
        "pmp_wall_aligned_height_x5": write_ply(OUT / "pmp_wall_aligned_height_x5.ply", obj_coords, pmp_height, foreground, "wall_normal_height", 5.0),
        "pmp_wall_aligned_height_x20_visual": write_ply(OUT / "pmp_wall_aligned_height_x20_visual.ply", obj_coords, pmp_height, foreground, "wall_normal_height", 20.0),
        "ftp_x_wall_aligned_height": write_ply(OUT / "single_ftp_x_wall_aligned_height.ply", ftp_coords, ftp_height, ftp_foreground, "wall_normal_height", 1.0),
        "ftp_x_wall_aligned_height_x5": write_ply(OUT / "single_ftp_x_wall_aligned_height_x5.ply", ftp_coords, ftp_height, ftp_foreground, "wall_normal_height", 5.0),
        "ftp_x_wall_aligned_height_x20_visual": write_ply(OUT / "single_ftp_x_wall_aligned_height_x20_visual.ply", ftp_coords, ftp_height, ftp_foreground, "wall_normal_height", 20.0),
        "pmp_object_core_height_x5": write_ply(OUT / "pmp_object_core_height_x5.ply", obj_coords, pmp_height, object_core, "wall_normal_height", 5.0),
        "ftp_x_object_core_height_x5": write_ply(OUT / "single_ftp_x_object_core_height_x5.ply", ftp_coords, ftp_height, ftp_core, "wall_normal_height", 5.0),
    }

    save_preview(OUT / "wall_aligned_height_preview.png", pmp_height, ftp_height, foreground & ftp["mask"].astype(bool), object_core & ftp["mask"].astype(bool))
    summary = {
        "purpose": "Wall-aligned CloudCompare exports. Coordinates are rotated into the fitted wall coordinate system: x/y lie on the wall plane and z is wall-normal height.",
        "why": "The raw calibrated coordinates are dominated by the background wall; viewing them directly in CloudCompare makes the object look coplanar.",
        "calibration": str(CAL),
        "wall_reference": "data_my/0/pose1",
        "object": "data_my/12/pose5",
        "wall_fit_region": "valid pixels with row < 0.72*height and 20 < col < width-20",
        "wall_plane_center": center.tolist(),
        "wall_plane_normal": normal.tolist(),
        "wall_plane_fit_rms": plane_rms,
        "foreground_rule": "PMP wall-normal height > 0.8",
        "object_core_rule": "PMP wall-normal height > 1.6",
        "height_stats": {
            "pmp_foreground": {
                "median": float(np.nanmedian(pmp_height[foreground])) if np.any(foreground) else None,
                "p02_p98": [float(x) for x in np.nanpercentile(pmp_height[foreground], [2, 98])] if np.any(foreground) else None,
            },
            "ftp_foreground": {
                "median": float(np.nanmedian(ftp_height[ftp_foreground])) if np.any(ftp_foreground) else None,
                "p02_p98": [float(x) for x in np.nanpercentile(ftp_height[ftp_foreground], [2, 98])] if np.any(ftp_foreground) else None,
            },
        },
        "counts": counts,
        "recommended_cloudcompare_files": [
            "pmp_wall_aligned_height_x20_visual.ply",
            "single_ftp_x_wall_aligned_height_x20_visual.ply",
            "pmp_wall_aligned_height_x5.ply",
            "single_ftp_x_wall_aligned_height_x5.ply",
            "pmp_object_core_height_x5.ply",
            "single_ftp_x_object_core_height_x5.ply",
        ],
    }
    (OUT / "wall_aligned_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README_WALL_ALIGNED_ASCII.txt").write_text(
        "\n".join(
            [
                "Wall-aligned CloudCompare package",
                "=================================",
                "",
                "Open the *_height_x20_visual.ply files first for visual debugging.",
                "Open *_height_x5.ply next for a less exaggerated view.",
                "These files are not the raw camera/world coordinates. They are rotated into the fitted wall coordinate system.",
                "PLY z is wall-normal height. x5/x20 versions exaggerate height for visual inspection only.",
                "",
                "Recommended:",
                "1. pmp_wall_aligned_height_x20_visual.ply",
                "2. single_ftp_x_wall_aligned_height_x20_visual.ply",
                "3. pmp_wall_aligned_height_x5.ply",
                "4. single_ftp_x_wall_aligned_height_x5.ply",
                "",
                "For metric use, use the non-x5/non-x20 files.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
