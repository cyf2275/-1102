from __future__ import annotations

import json
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path.cwd()
X_DIR = ROOT / "cloud_results" / "A_20260610_obj12_pose5_single_ftp_x_0610"
Y_DIR = ROOT / "cloud_results" / "A_20260610_obj12_pose5_single_ftp_y_0610"
OUT = ROOT / "cloud_results" / "A_20260610_obj12_pose5_cloudcompare_pointclouds"


def valid_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if mask is not None:
        valid &= mask.astype(bool)
    return valid


def robust_range(vals: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    a, b = np.percentile(vals, [lo, hi])
    if not np.isfinite(a) or not np.isfinite(b) or abs(b - a) < 1e-9:
        a, b = float(np.nanmin(vals)), float(np.nanmax(vals))
    if abs(b - a) < 1e-9:
        b = a + 1.0
    return float(a), float(b)


def colorize(vals: np.ndarray, vmin: float, vmax: float, cmap_name: str = "viridis") -> np.ndarray:
    t = np.clip((vals - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    rgba = cm.get_cmap(cmap_name)(t)
    return np.clip(rgba[:, :3] * 255.0, 0, 255).astype(np.uint8)


def write_ply(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    mask: np.ndarray,
    color_values: np.ndarray | None = None,
    color_range: tuple[float, float] | None = None,
    cmap: str = "viridis",
    scalar_name: str | None = None,
    scalar_values: np.ndarray | None = None,
    stride: int = 1,
) -> int:
    sparse = np.zeros_like(mask, dtype=bool)
    sparse[::stride, ::stride] = mask[::stride, ::stride]
    idx = np.flatnonzero(sparse)
    pts = np.stack([x.flat[idx], y.flat[idx], z.flat[idx]], axis=1)
    if color_values is None:
        color_values = z
    cvals = color_values.flat[idx]
    if color_range is None:
        color_range = robust_range(cvals)
    colors = colorize(cvals, color_range[0], color_range[1], cmap)
    scalars = scalar_values.flat[idx] if scalar_values is not None else None

    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if scalar_name and scalars is not None:
            safe = scalar_name.replace(" ", "_")
            f.write(f"property float {safe}\n")
        f.write("end_header\n")
        if scalars is None:
            for p, c in zip(pts, colors):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        else:
            for p, c, s in zip(pts, colors, scalars):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])} {float(s):.6f}\n")
    return int(pts.shape[0])


def save_preview(path: Path, pmp: dict, ftp_x: dict, ftp_y: dict) -> None:
    common_x = ftp_x["mask"] & pmp["mask"] & np.isfinite(ftp_x["Z"]) & np.isfinite(pmp["Z"])
    common_y = ftp_y["mask"] & pmp["mask"] & np.isfinite(ftp_y["Z"]) & np.isfinite(pmp["Z"])
    err_x = np.abs(ftp_x["Z"] - pmp["Z"])
    err_y = np.abs(ftp_y["Z"] - pmp["Z"])
    zmin, zmax = robust_range(pmp["Z"][pmp["mask"]])
    panels = [
        (pmp["Z"], pmp["mask"], "PMP reference Z", zmin, zmax, "viridis"),
        (ftp_x["Z"], ftp_x["mask"], "single FTP-X Z", zmin, zmax, "viridis"),
        (err_x, common_x, "|FTP-X - PMP|", 0.0, float(np.nanpercentile(err_x[common_x], 98)), "magma"),
        (ftp_y["Z"], ftp_y["mask"], "single FTP-Y Z", zmin, zmax, "viridis"),
        (err_y, common_y, "|FTP-Y - PMP|", 0.0, float(np.nanpercentile(err_y[common_y], 98)), "magma"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
    for ax, (arr, mask, title, vmin, vmax, cmap_name) in zip(axes, panels):
        im = ax.imshow(np.where(mask, arr, np.nan), cmap=cmap_name, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_clouds() -> tuple[dict, dict, dict]:
    x_npz = np.load(X_DIR / "single_ftp_reconstruction.npz")
    y_npz = np.load(Y_DIR / "single_ftp_reconstruction.npz")
    pmp = {
        "X": x_npz["pmp_X"],
        "Y": x_npz["pmp_Y"],
        "Z": x_npz["pmp_Z"],
    }
    pmp["mask"] = valid_xyz(pmp["X"], pmp["Y"], pmp["Z"])
    ftp_x = {"X": x_npz["X"], "Y": x_npz["Y"], "Z": x_npz["Z"], "mask": x_npz["mask"].astype(bool)}
    ftp_y = {"X": y_npz["X"], "Y": y_npz["Y"], "Z": y_npz["Z"], "mask": y_npz["mask"].astype(bool)}
    ftp_x["mask"] &= valid_xyz(ftp_x["X"], ftp_x["Y"], ftp_x["Z"])
    ftp_y["mask"] &= valid_xyz(ftp_y["X"], ftp_y["Y"], ftp_y["Z"])
    return pmp, ftp_x, ftp_y


def metrics(name: str, ftp: dict, pmp: dict) -> dict:
    common = ftp["mask"] & pmp["mask"] & np.isfinite(ftp["Z"]) & np.isfinite(pmp["Z"])
    dz = ftp["Z"][common] - pmp["Z"][common]
    return {
        "name": name,
        "common_pixels": int(common.sum()),
        "rmse_z_to_pmp": float(np.sqrt(np.mean(dz * dz))),
        "mae_z_to_pmp": float(np.mean(np.abs(dz))),
        "median_abs_z_to_pmp": float(np.median(np.abs(dz))),
        "p95_abs_z_to_pmp": float(np.percentile(np.abs(dz), 95)),
        "corr_z_to_pmp": float(np.corrcoef(ftp["Z"][common].reshape(-1), pmp["Z"][common].reshape(-1))[0, 1]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pmp, ftp_x, ftp_y = load_clouds()
    z_range = robust_range(pmp["Z"][pmp["mask"]])
    common_x = ftp_x["mask"] & pmp["mask"]
    common_y = ftp_y["mask"] & pmp["mask"]
    err_x = np.abs(ftp_x["Z"] - pmp["Z"])
    err_y = np.abs(ftp_y["Z"] - pmp["Z"])
    err_x_range = (0.0, float(np.nanpercentile(err_x[common_x & np.isfinite(err_x)], 98)))
    err_y_range = (0.0, float(np.nanpercentile(err_y[common_y & np.isfinite(err_y)], 98)))

    counts = {}
    counts["pmp_reference_full_colorZ"] = write_ply(OUT / "pmp_reference_full_colorZ.ply", pmp["X"], pmp["Y"], pmp["Z"], pmp["mask"], color_range=z_range, cmap="viridis")
    counts["single_ftp_x_full_colorZ"] = write_ply(OUT / "single_ftp_x_full_colorZ.ply", ftp_x["X"], ftp_x["Y"], ftp_x["Z"], ftp_x["mask"], color_range=z_range, cmap="viridis")
    counts["single_ftp_y_full_colorZ"] = write_ply(OUT / "single_ftp_y_full_colorZ.ply", ftp_y["X"], ftp_y["Y"], ftp_y["Z"], ftp_y["mask"], color_range=z_range, cmap="viridis")
    counts["single_ftp_x_error_to_pmp"] = write_ply(
        OUT / "single_ftp_x_error_to_pmp_scalar_color.ply",
        ftp_x["X"],
        ftp_x["Y"],
        ftp_x["Z"],
        common_x,
        color_values=err_x,
        color_range=err_x_range,
        cmap="magma",
        scalar_name="abs_z_error_to_pmp",
        scalar_values=err_x,
    )
    counts["single_ftp_y_error_to_pmp"] = write_ply(
        OUT / "single_ftp_y_error_to_pmp_scalar_color.ply",
        ftp_y["X"],
        ftp_y["Y"],
        ftp_y["Z"],
        common_y,
        color_values=err_y,
        color_range=err_y_range,
        cmap="magma",
        scalar_name="abs_z_error_to_pmp",
        scalar_values=err_y,
    )

    # Lighter versions for quick CloudCompare loading.
    counts["pmp_reference_stride2_colorZ"] = write_ply(OUT / "pmp_reference_stride2_colorZ.ply", pmp["X"], pmp["Y"], pmp["Z"], pmp["mask"], color_range=z_range, cmap="viridis", stride=2)
    counts["single_ftp_x_stride2_colorZ"] = write_ply(OUT / "single_ftp_x_stride2_colorZ.ply", ftp_x["X"], ftp_x["Y"], ftp_x["Z"], ftp_x["mask"], color_range=z_range, cmap="viridis", stride=2)
    counts["single_ftp_y_stride2_colorZ"] = write_ply(OUT / "single_ftp_y_stride2_colorZ.ply", ftp_y["X"], ftp_y["Y"], ftp_y["Z"], ftp_y["mask"], color_range=z_range, cmap="viridis", stride=2)

    save_preview(OUT / "cloudcompare_export_preview.png", pmp, ftp_x, ftp_y)
    summary = {
        "source": {
            "pmp_and_ftp_x": str(X_DIR),
            "ftp_y": str(Y_DIR),
        },
        "z_color_range_from_pmp": z_range,
        "error_color_range_x": err_x_range,
        "error_color_range_y": err_y_range,
        "vertex_counts": counts,
        "metrics": {
            "single_ftp_x_vs_pmp": metrics("single_ftp_x_vs_pmp", ftp_x, pmp),
            "single_ftp_y_vs_pmp": metrics("single_ftp_y_vs_pmp", ftp_y, pmp),
        },
        "recommended_cloudcompare_load_order": [
            "pmp_reference_full_colorZ.ply",
            "single_ftp_x_full_colorZ.ply",
            "single_ftp_x_error_to_pmp_scalar_color.ply",
            "single_ftp_y_full_colorZ.ply",
            "single_ftp_y_error_to_pmp_scalar_color.ply",
        ],
    }
    (OUT / "cloudcompare_export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README_CLOUDCOMPARE.md").write_text(
        "\n".join(
            [
                "# CloudCompare 对比包",
                "",
                "直接在 CloudCompare 中打开以下文件：",
                "",
                "1. `pmp_reference_full_colorZ.ply`：多步 PMP 参考点云，按 Z 着色。",
                "2. `single_ftp_x_full_colorZ.ply`：竖条纹单张 FTP-X 点云，按 PMP 相同 Z 范围着色。",
                "3. `single_ftp_x_error_to_pmp_scalar_color.ply`：FTP-X 点云，按 `abs_z_error_to_pmp` 着色并保存 scalar field。",
                "4. `single_ftp_y_full_colorZ.ply`：横条纹单张 FTP-Y 点云，明显失败，用于对照。",
                "5. `single_ftp_y_error_to_pmp_scalar_color.ply`：FTP-Y 误差点云。",
                "",
                "如果 CloudCompare 打开全分辨率较慢，可以先打开 `*_stride2_colorZ.ply`。",
                "",
                "推荐操作：",
                "",
                "- 先加载 `pmp_reference_full_colorZ.ply` 和 `single_ftp_x_full_colorZ.ply`，切换显示查看形状是否重合。",
                "- 再加载 `single_ftp_x_error_to_pmp_scalar_color.ply`，在 scalar field 里选择 `abs_z_error_to_pmp` 看误差区域。",
                "- 如果想用 CloudCompare 自带距离，选择 FTP 点云和 PMP 点云，执行 `Tools -> Distances -> Cloud/Cloud Dist.`。",
                "",
                "本包内统计：",
                "",
                f"- FTP-X vs PMP: RMSE Z = {summary['metrics']['single_ftp_x_vs_pmp']['rmse_z_to_pmp']:.4f}, corr = {summary['metrics']['single_ftp_x_vs_pmp']['corr_z_to_pmp']:.6f}",
                f"- FTP-Y vs PMP: RMSE Z = {summary['metrics']['single_ftp_y_vs_pmp']['rmse_z_to_pmp']:.4f}, corr = {summary['metrics']['single_ftp_y_vs_pmp']['corr_z_to_pmp']:.6f}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
