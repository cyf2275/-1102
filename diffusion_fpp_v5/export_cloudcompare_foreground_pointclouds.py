from __future__ import annotations

import json
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from reconstruct_single_ftp_pose import compute_pmp
from reconstruct_my_dataset_pmp_ftp import parse_pmp_calibration


DATA = Path("data_my")
CAL = DATA / "calibrate_0610.pmp"
X_DIR = Path("cloud_results") / "A_20260610_obj12_pose5_single_ftp_x_0610"
Y_DIR = Path("cloud_results") / "A_20260610_obj12_pose5_single_ftp_y_0610"
OUT = Path("cloud_results") / "A_20260610_obj12_pose5_cloudcompare_foreground"


def robust_range(vals: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    a, b = np.percentile(vals, [lo, hi])
    if abs(b - a) < 1e-9:
        b = a + 1.0
    return float(a), float(b)


def colorize(vals: np.ndarray, vmin: float, vmax: float, cmap_name: str) -> np.ndarray:
    t = np.clip((vals - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    rgba = cm.get_cmap(cmap_name)(t)
    return (rgba[:, :3] * 255.0).astype(np.uint8)


def write_ply(path: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray, mask: np.ndarray, scalar: np.ndarray, scalar_name: str, cmap: str = "turbo", stride: int = 1) -> int:
    sparse = np.zeros_like(mask, dtype=bool)
    sparse[::stride, ::stride] = mask[::stride, ::stride]
    idx = np.flatnonzero(sparse)
    pts = np.stack([x.flat[idx], y.flat[idx], z.flat[idx]], axis=1)
    vals = scalar.flat[idx]
    vmin, vmax = robust_range(vals)
    colors = colorize(vals, vmin, vmax, cmap)
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


def save_preview(path: Path, obj_z: np.ndarray, ftp_x_z: np.ndarray, ftp_y_z: np.ndarray, rel: np.ndarray, masks: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    panels = [
        (obj_z, masks["obj_full"], "PMP full Z", "viridis"),
        (rel, masks["foreground"], "PMP foreground relZ", "coolwarm"),
        (obj_z, masks["foreground"], "PMP foreground Z", "viridis"),
        (ftp_x_z, masks["ftp_x_fg"], "FTP-X foreground Z", "viridis"),
        (np.abs(ftp_x_z - obj_z), masks["ftp_x_fg"], "|FTP-X-PMP| fg", "magma"),
        (ftp_y_z, masks["ftp_y_fg"], "FTP-Y foreground Z", "viridis"),
    ]
    for ax, (arr, mask, title, cmap) in zip(axes.ravel(), panels):
        vmin, vmax = robust_range(arr[mask])
        if "error" in title.lower() or title.startswith("|"):
            vmin = 0.0
            vmax = float(np.nanpercentile(arr[mask & np.isfinite(arr)], 98)) if np.any(mask & np.isfinite(arr)) else 1.0
        im = ax.imshow(np.where(mask, arr, np.nan), cmap=cmap, vmin=vmin, vmax=vmax)
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
    ftp_x = np.load(X_DIR / "single_ftp_reconstruction.npz")
    ftp_y = np.load(Y_DIR / "single_ftp_reconstruction.npz")

    common_wall = wall["mask"] & obj["mask"] & np.isfinite(wall["Z"]) & np.isfinite(obj["Z"])
    rel = obj["Z"] - wall["Z"]
    # Keep points that differ from the wall. The threshold is deliberately low:
    # it removes the large smooth wall while keeping the object and platform.
    foreground = common_wall & (np.abs(rel) > 1.0)
    # Also keep a tighter object-like mask for quick inspection.
    object_core = common_wall & (np.abs(rel) > 2.5)
    ftp_x_fg = foreground & ftp_x["mask"].astype(bool) & np.isfinite(ftp_x["Z"])
    ftp_y_fg = foreground & ftp_y["mask"].astype(bool) & np.isfinite(ftp_y["Z"])
    ftp_x_core = object_core & ftp_x["mask"].astype(bool) & np.isfinite(ftp_x["Z"])
    ftp_y_core = object_core & ftp_y["mask"].astype(bool) & np.isfinite(ftp_y["Z"])

    counts = {}
    counts["pmp_foreground_relZ"] = write_ply(OUT / "pmp_foreground_relZ_color.ply", obj["X"], obj["Y"], obj["Z"], foreground, rel, "relative_Z_to_wall", "coolwarm")
    counts["pmp_object_core_relZ"] = write_ply(OUT / "pmp_object_core_relZ_color.ply", obj["X"], obj["Y"], obj["Z"], object_core, rel, "relative_Z_to_wall", "coolwarm")
    counts["ftp_x_foreground_relZ"] = write_ply(OUT / "single_ftp_x_foreground_relZ_color.ply", ftp_x["X"], ftp_x["Y"], ftp_x["Z"], ftp_x_fg, rel, "pmp_relative_Z_to_wall", "coolwarm")
    counts["ftp_x_object_core_relZ"] = write_ply(OUT / "single_ftp_x_object_core_relZ_color.ply", ftp_x["X"], ftp_x["Y"], ftp_x["Z"], ftp_x_core, rel, "pmp_relative_Z_to_wall", "coolwarm")
    err_x = np.abs(ftp_x["Z"] - obj["Z"])
    err_y = np.abs(ftp_y["Z"] - obj["Z"])
    counts["ftp_x_foreground_error"] = write_ply(OUT / "single_ftp_x_foreground_error_to_pmp.ply", ftp_x["X"], ftp_x["Y"], ftp_x["Z"], ftp_x_fg, err_x, "abs_z_error_to_pmp", "magma")
    counts["ftp_y_foreground_error"] = write_ply(OUT / "single_ftp_y_foreground_error_to_pmp.ply", ftp_y["X"], ftp_y["Y"], ftp_y["Z"], ftp_y_fg, err_y, "abs_z_error_to_pmp", "magma")

    save_preview(
        OUT / "foreground_export_preview.png",
        obj["Z"],
        ftp_x["Z"],
        ftp_y["Z"],
        rel,
        {
            "obj_full": obj["mask"] & np.isfinite(obj["Z"]),
            "foreground": foreground,
            "ftp_x_fg": ftp_x_fg,
            "ftp_y_fg": ftp_y_fg,
        },
    )

    summary = {
        "purpose": "Foreground-only CloudCompare exports. Full PMP includes the wall/background; these files remove the large wall by thresholding relative Z to wall reference 0/pose1.",
        "reference_wall": "data_my/0/pose1",
        "object": "data_my/12/pose5",
        "foreground_rule": "abs(PMP_Z_object - PMP_Z_wall) > 1.0",
        "object_core_rule": "abs(PMP_Z_object - PMP_Z_wall) > 2.5",
        "counts": counts,
        "recommended": [
            "pmp_foreground_relZ_color.ply",
            "single_ftp_x_foreground_relZ_color.ply",
            "single_ftp_x_foreground_error_to_pmp.ply",
            "pmp_object_core_relZ_color.ply",
            "single_ftp_x_object_core_relZ_color.ply",
        ],
    }
    (OUT / "foreground_export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README_FOREGROUND_ASCII.txt").write_text(
        "\n".join(
            [
                "Foreground CloudCompare package",
                "==============================",
                "",
                "The full PMP cloud contains a large wall/background surface. In CloudCompare it can look like a curved sheet and hide the object.",
                "This package removes most background points by comparing object 12/pose5 against wall reference 0/pose1.",
                "",
                "Recommended files:",
                "1. pmp_foreground_relZ_color.ply",
                "2. single_ftp_x_foreground_relZ_color.ply",
                "3. single_ftp_x_foreground_error_to_pmp.ply",
                "",
                "If you want a tighter crop, use:",
                "1. pmp_object_core_relZ_color.ply",
                "2. single_ftp_x_object_core_relZ_color.ply",
                "",
                "Foreground rule: abs(PMP_Z_object - PMP_Z_wall) > 1.0",
                "Object-core rule: abs(PMP_Z_object - PMP_Z_wall) > 2.5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
