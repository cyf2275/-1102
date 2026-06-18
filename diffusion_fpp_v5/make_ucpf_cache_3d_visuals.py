from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in str(text).replace(",", " ").split() if x.strip()]


def load_array(cache_dir: Path, name: str, split: str, dtype: str):
    path = cache_dir / f"{name}_{split}_{dtype}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r")


def norm_to_mm(depth_norm: np.ndarray, depth_minmax: np.ndarray) -> np.ndarray:
    depth01 = np.clip((np.asarray(depth_norm, dtype=np.float32) + 1.0) * 0.5, 0.0, 1.0)
    lo = float(depth_minmax[0])
    hi = float(depth_minmax[1])
    return depth01 * max(hi - lo, 1e-6) + lo


def masked_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool)
    if not np.any(valid):
        return float("nan")
    return float(np.asarray(arr, dtype=np.float64)[valid].mean())


def rmse_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    valid = mask.astype(bool)
    diff = np.asarray(pred, dtype=np.float64)[valid] - np.asarray(target, dtype=np.float64)[valid]
    if diff.size == 0:
        return float("nan"), float("nan")
    return float(np.sqrt(np.mean(diff * diff))), float(np.mean(np.abs(diff)))


def rcpc_fuse(
    d_d_norm: np.ndarray,
    d_p_norm: np.ndarray,
    edge: np.ndarray,
    phase_conf: np.ndarray,
    mask: np.ndarray,
    edge_tau: float,
    delta_max: float,
    phase_conf_max: float,
    high_weight: float,
    low_weight: float,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    delta = np.abs(np.asarray(d_p_norm, dtype=np.float32) - np.asarray(d_d_norm, dtype=np.float32))
    edge_mean = masked_mean(edge, mask)
    delta_mean = masked_mean(delta, mask)
    conf_mean = masked_mean(phase_conf, mask)
    selected = (
        edge_mean >= float(edge_tau)
        and delta_mean <= float(delta_max)
        and conf_mean <= float(phase_conf_max)
    )
    weight = float(high_weight if selected else low_weight)
    fused = np.clip((1.0 - weight) * d_d_norm + weight * d_p_norm, -1.0, 1.0)
    return fused, {
        "selected": bool(selected),
        "weight": weight,
        "edge_mean": edge_mean,
        "delta_mean_norm": delta_mean,
        "phase_conf_mean": conf_mean,
    }


def masked_image(arr: np.ndarray, mask: np.ndarray):
    valid = mask.astype(bool)
    return np.ma.masked_where(~valid, arr)


def surface_arrays(depth: np.ndarray, mask: np.ndarray, step: int):
    z = np.asarray(depth, dtype=np.float32)[::step, ::step].copy()
    m = mask.astype(bool)[::step, ::step]
    z[~m] = np.nan
    h, w = z.shape
    yy, xx = np.mgrid[0:h, 0:w]
    return xx * step, yy * step, z


def set_3d_axes(ax, zmin: float, zmax: float, title: str):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("depth mm")
    ax.set_zlim(zmin, zmax)
    ax.view_init(elev=32, azim=-58)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.45))
    except Exception:
        pass


def write_ply(path: Path, depth: np.ndarray, mask: np.ndarray, step: int, max_points: int):
    valid = mask.astype(bool)
    ys, xs = np.where(valid)
    if step > 1:
        keep = (ys % step == 0) & (xs % step == 0)
        ys, xs = ys[keep], xs[keep]
    if max_points > 0 and len(xs) > max_points:
        rng = np.random.default_rng(20260608)
        idx = rng.choice(len(xs), size=max_points, replace=False)
        ys, xs = ys[idx], xs[idx]
    z = depth[ys, xs]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xs)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, zz in zip(xs, ys, z):
            f.write(f"{float(x):.6f} {float(y):.6f} {float(zz):.6f}\n")


def save_sample_visual(
    save_path: Path,
    sample_id: int,
    fringe: np.ndarray,
    target: np.ndarray,
    d_b: np.ndarray,
    d_p: np.ndarray,
    d_d: np.ndarray,
    final: np.ndarray,
    mask: np.ndarray,
    rcpc_meta: dict[str, float | bool],
    surface_step: int,
):
    valid = mask.astype(bool)
    target_valid = target[valid]
    if target_valid.size:
        zmin = float(np.nanpercentile(target_valid, 1))
        zmax = float(np.nanpercentile(target_valid, 99))
    else:
        zmin, zmax = float(np.nanmin(target)), float(np.nanmax(target))
    err_d = np.abs(d_d - target)
    err_final = np.abs(final - target)
    vmax_err = float(np.nanpercentile(err_d[valid], 95)) if np.any(valid) else float(np.nanmax(err_d))
    vmax_err = max(vmax_err, 1e-3)

    panels_2d = [
        (fringe, "single fringe", "gray", None, None),
        (masked_image(target, mask), "GT depth", "viridis", zmin, zmax),
        (masked_image(d_b, mask), "D_b prior", "viridis", zmin, zmax),
        (masked_image(d_p, mask), "D_p phase", "viridis", zmin, zmax),
        (masked_image(d_d, mask), "D_d diffusion", "viridis", zmin, zmax),
        (masked_image(final, mask), "RCPC final", "viridis", zmin, zmax),
        (masked_image(err_final, mask), "final abs error", "magma", 0.0, vmax_err),
    ]
    panels_3d = [
        (target, "GT 3D"),
        (d_b, "D_b 3D"),
        (d_p, "D_p 3D"),
        (d_d, "D_d 3D"),
        (final, "RCPC 3D"),
    ]

    fig = plt.figure(figsize=(24, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 7, height_ratios=[1.0, 1.15])

    for i, (img, title, cmap, vmin, vmax) in enumerate(panels_2d):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        if i > 0:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for i, (depth, title) in enumerate(panels_3d):
        ax = fig.add_subplot(gs[1, i], projection="3d")
        xx, yy, zz = surface_arrays(depth, mask, surface_step)
        ax.plot_surface(xx, yy, zz, cmap="viridis", linewidth=0, antialiased=False, vmin=zmin, vmax=zmax)
        set_3d_axes(ax, zmin, zmax, title)

    ax = fig.add_subplot(gs[1, 5:])
    im = ax.imshow(masked_image(err_final, mask), cmap="magma", vmin=0.0, vmax=vmax_err)
    ax.set_title("RCPC abs error map", fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    rmse_d, mae_d = rmse_mae(d_d, target, mask)
    rmse_f, mae_f = rmse_mae(final, target, mask)
    fig.suptitle(
        (
            f"test sample {sample_id} | RCPC selected={rcpc_meta['selected']} "
            f"w={rcpc_meta['weight']:.2f} | "
            f"D_d RMSE={rmse_d:.2f} MAE={mae_d:.2f} mm | "
            f"final RMSE={rmse_f:.2f} MAE={mae_f:.2f} mm"
        ),
        fontsize=12,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Make 2D/3D reconstruction visuals from frozen candidate cache.")
    parser.add_argument("--candidate_cache_dir", default="/root/autodl-tmp/fpp_ml_ucpf_hier_orderfix_cache_960_seed180")
    parser.add_argument("--save_dir", default="results/e247_rcpc_3d_visuals")
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", default="1 7 19 21 29 11")
    parser.add_argument("--surface_step", type=int, default=8)
    parser.add_argument("--edge_tau", type=float, default=0.42)
    parser.add_argument("--delta_max", type=float, default=0.11)
    parser.add_argument("--phase_conf_max", type=float, default=0.74)
    parser.add_argument("--high_weight", type=float, default=0.6)
    parser.add_argument("--low_weight", type=float, default=0.0)
    parser.add_argument("--export_ply", action="store_true")
    parser.add_argument("--ply_step", type=int, default=4)
    parser.add_argument("--ply_max_points", type=int, default=160000)
    args = parser.parse_args()

    cache_dir = Path(args.candidate_cache_dir)
    save_dir = Path(args.save_dir)
    split = str(args.split)
    sample_ids = parse_int_list(args.samples)
    if not sample_ids:
        raise ValueError("--samples must contain at least one sample id")

    d_b = load_array(cache_dir, "d_b", split, "float16")
    d_p = load_array(cache_dir, "d_p", split, "float16")
    d_d = load_array(cache_dir, "d_d", split, "float16")
    target_mm = load_array(cache_dir, "target_mm", split, "float32")
    mask = load_array(cache_dir, "mask", split, "uint8")
    fringe = load_array(cache_dir, "fringe", split, "float16")
    edge = load_array(cache_dir, "edge", split, "float16")
    phase_conf = load_array(cache_dir, "phase_conf", split, "float16")
    depth_minmax = load_array(cache_dir, "depth_minmax", split, "float32")
    sample_index = load_array(cache_dir, "sample_index", split, "int32")

    id_to_pos = {int(sample_index[i]): i for i in range(len(sample_index))}
    rows = []
    for sample_id in sample_ids:
        if sample_id not in id_to_pos:
            raise KeyError(f"sample {sample_id} not found in {split} sample_index")
        i = id_to_pos[sample_id]
        m = np.asarray(mask[i, 0], dtype=bool)
        db_mm = norm_to_mm(d_b[i, 0], depth_minmax[i])
        dp_mm = norm_to_mm(d_p[i, 0], depth_minmax[i])
        dd_mm = norm_to_mm(d_d[i, 0], depth_minmax[i])
        final_norm, rcpc_meta = rcpc_fuse(
            np.asarray(d_d[i, 0], dtype=np.float32),
            np.asarray(d_p[i, 0], dtype=np.float32),
            np.asarray(edge[i, 0], dtype=np.float32),
            np.asarray(phase_conf[i, 0], dtype=np.float32),
            m,
            args.edge_tau,
            args.delta_max,
            args.phase_conf_max,
            args.high_weight,
            args.low_weight,
        )
        final_mm = norm_to_mm(final_norm, depth_minmax[i])
        target = np.asarray(target_mm[i, 0], dtype=np.float32)
        save_sample_visual(
            save_dir / f"{split}_{sample_id:03d}_rcpc_3d.png",
            sample_id,
            np.asarray(fringe[i, 0], dtype=np.float32),
            target,
            db_mm,
            dp_mm,
            dd_mm,
            final_mm,
            m,
            rcpc_meta,
            max(1, int(args.surface_step)),
        )
        if args.export_ply:
            write_ply(save_dir / "ply" / f"{split}_{sample_id:03d}_gt_pixel_depth.ply", target, m, args.ply_step, args.ply_max_points)
            write_ply(save_dir / "ply" / f"{split}_{sample_id:03d}_rcpc_pixel_depth.ply", final_mm, m, args.ply_step, args.ply_max_points)
            write_ply(save_dir / "ply" / f"{split}_{sample_id:03d}_d_d_pixel_depth.ply", dd_mm, m, args.ply_step, args.ply_max_points)
        rmse_d, mae_d = rmse_mae(dd_mm, target, m)
        rmse_f, mae_f = rmse_mae(final_mm, target, m)
        rows.append({
            "sample": sample_id,
            **rcpc_meta,
            "d_d_rmse_mm": rmse_d,
            "d_d_mae_mm": mae_d,
            "rcpc_rmse_mm": rmse_f,
            "rcpc_mae_mm": mae_f,
        })

    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / f"{split}_rcpc_3d_visual_summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "candidate_cache_dir": str(cache_dir),
            "split": split,
            "samples": sample_ids,
            "surface_coordinate_note": "3D surfaces use pixel x/y coordinates and depth in mm; they are visualization surfaces, not calibrated camera point clouds.",
            "rcpc_rule": {
                "edge_tau": args.edge_tau,
                "delta_max": args.delta_max,
                "phase_conf_max": args.phase_conf_max,
                "high_weight": args.high_weight,
                "low_weight": args.low_weight,
            },
            "rows": rows,
        }, f, indent=2, ensure_ascii=False)
    print(json.dumps({"save_dir": str(save_dir), "num_samples": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
