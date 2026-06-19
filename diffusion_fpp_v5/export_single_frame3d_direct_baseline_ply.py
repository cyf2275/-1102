"""Export strict-camera PLY for a direct single-frame3d baseline checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch

from train_single_frame3d_backbone_baselines import build_arch, make_loaders, model_depth_norm


def valid_vertices(mask: np.ndarray, stride: int, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask.astype(bool))
    if stride > 1:
        keep = (ys % stride == 0) & (xs % stride == 0)
        ys, xs = ys[keep], xs[keep]
    if max_points > 0 and len(xs) > max_points:
        rng = np.random.default_rng(20260619)
        idx = rng.choice(len(xs), size=max_points, replace=False)
        ys, xs = ys[idx], xs[idx]
    return ys.astype(np.int32), xs.astype(np.int32)


def colors_from_values(values: np.ndarray) -> np.ndarray:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return np.zeros((values.size, 3), dtype=np.uint8)
    lo = float(np.percentile(vals, 1))
    hi = float(np.percentile(vals, 99))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    rgba = cm.get_cmap("viridis")(norm)
    return np.clip(rgba[:, :3] * 255.0, 0, 255).astype(np.uint8)


def backproject_camera_projective(depth_z: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    p = np.asarray(camera_matrix, dtype=np.float64)
    z = np.asarray(depth_z, dtype=np.float64)
    h, w = z.shape
    yy, xx = np.mgrid[0:h, 0:w]
    u = xx.astype(np.float64)
    v = yy.astype(np.float64)

    a00 = u * p[2, 0] - p[0, 0]
    a01 = u * p[2, 1] - p[0, 1]
    b0 = (p[0, 2] - u * p[2, 2]) * z + (p[0, 3] - u * p[2, 3])
    a10 = v * p[2, 0] - p[1, 0]
    a11 = v * p[2, 1] - p[1, 1]
    b1 = (p[1, 2] - v * p[2, 2]) * z + (p[1, 3] - v * p[2, 3])

    det = a00 * a11 - a01 * a10
    good = np.abs(det) > 1e-12
    x = np.full_like(z, np.nan, dtype=np.float64)
    y = np.full_like(z, np.nan, dtype=np.float64)
    x[good] = (b0[good] * a11[good] - a01[good] * b1[good]) / det[good]
    y[good] = (a00[good] * b1[good] - b0[good] * a10[good]) / det[good]
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def load_camera_matrix(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    mat = np.asarray(data["camera_matrix_3x4"], dtype=np.float64)
    if mat.shape != (3, 4):
        raise ValueError(f"camera_matrix_3x4 shape is {mat.shape}, expected (3, 4)")
    return mat


def default_calibration_path(run_args: SimpleNamespace, sample_id: str) -> Path:
    domain = sample_id.split("_obj", 1)[0]
    return Path(run_args.teacher_extra_root) / "calibration" / f"{domain}_calibration.json"


def labels_path_for_sample(run_args: SimpleNamespace, split: str, sample_id: str) -> Path:
    domain, rest = sample_id.split("_obj", 1)
    obj_text, pose_text = rest.split("_pose", 1)
    root = Path(getattr(run_args, "ood_root", "")) if split == "ood" else Path(getattr(run_args, "data_root", ""))
    return root / "samples" / domain / f"obj{int(obj_text):03d}" / f"pose{int(pose_text):02d}" / "labels.npz"


def write_camera_projective_ply(
    path: Path,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    gt_xyz: np.ndarray | None,
    stride: int,
    max_points: int,
    comment: str,
) -> int:
    pred_xyz = backproject_camera_projective(pred, camera_matrix)
    valid = mask.astype(bool) & np.isfinite(pred) & np.isfinite(target) & np.all(np.isfinite(pred_xyz), axis=2)
    ys, xs = valid_vertices(valid, stride=stride, max_points=max_points)
    pts = pred_xyz[ys, xs].astype(np.float32)
    pred_v = pred[ys, xs].astype(np.float32)
    target_v = target[ys, xs].astype(np.float32)
    err_v = np.abs(pred_v - target_v).astype(np.float32)
    if gt_xyz is None:
        gt_v = np.full((len(xs), 3), np.nan, dtype=np.float32)
    else:
        gt_v = gt_xyz[ys, xs].astype(np.float32)
    rgb = colors_from_values(pred_v)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"comment {comment}\n")
        f.write("comment coordinates_are_backprojected_with_camera_matrix_3x4_and_0_based_pixels\n")
        f.write("comment x_y_z_are_in_dataset_calibration_world_unit\n")
        f.write(f"element vertex {len(xs)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property float pixel_u\nproperty float pixel_v\n")
        f.write("property float pred_depth_z\nproperty float gt_depth_z\nproperty float abs_error_z\n")
        f.write("property float gt_camera_x\nproperty float gt_camera_y\nproperty float gt_camera_z\n")
        f.write("end_header\n")
        for i in range(len(xs)):
            r, g, b = rgb[i]
            f.write(
                f"{pts[i, 0]:.6f} {pts[i, 1]:.6f} {pts[i, 2]:.6f} "
                f"{int(r)} {int(g)} {int(b)} {float(xs[i]):.6f} {float(ys[i]):.6f} "
                f"{pred_v[i]:.6f} {target_v[i]:.6f} {err_v[i]:.6f} "
                f"{gt_v[i, 0]:.6f} {gt_v[i, 1]:.6f} {gt_v[i, 2]:.6f}\n"
            )
    return int(len(xs))


def write_pixel_depth_ply(
    path: Path,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    z_scale: float,
    stride: int,
    max_points: int,
    comment: str,
) -> int:
    ys, xs = valid_vertices(mask.astype(bool) & np.isfinite(pred) & np.isfinite(target), stride, max_points)
    pred_v = pred[ys, xs].astype(np.float32)
    target_v = target[ys, xs].astype(np.float32)
    err_v = np.abs(pred_v - target_v).astype(np.float32)
    rgb = colors_from_values(pred_v)
    h, w = pred.shape
    x_coord = xs.astype(np.float32) - (w - 1) * 0.5
    y_coord = (h - 1) * 0.5 - ys.astype(np.float32)
    z_coord = pred_v * float(z_scale)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"comment {comment}\n")
        f.write("comment x_y_are_centered_pixel_coordinates\n")
        f.write(f"comment z_coordinate_is_pred_depth_mm_times_{float(z_scale):.6g}\n")
        f.write(f"element vertex {len(xs)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property float pred_depth_mm\nproperty float gt_depth_mm\nproperty float abs_error_mm\n")
        f.write("end_header\n")
        for i in range(len(xs)):
            r, g, b = rgb[i]
            f.write(
                f"{x_coord[i]:.6f} {y_coord[i]:.6f} {z_coord[i]:.6f} "
                f"{int(r)} {int(g)} {int(b)} "
                f"{pred_v[i]:.6f} {target_v[i]:.6f} {err_v[i]:.6f}\n"
            )
    return int(len(xs))


def export_direct(args: argparse.Namespace) -> Dict[str, object]:
    ckpt = torch.load(args.ckpt, map_location="cpu")
    run_args = SimpleNamespace(**ckpt["args"])
    run_args.eval_batch_size = 1
    run_args.batch_size = 1
    run_args.num_workers = int(args.num_workers)
    if args.data_root:
        run_args.data_root = args.data_root
    if args.teacher_extra_root:
        run_args.teacher_extra_root = args.teacher_extra_root
    if args.ood_root:
        run_args.ood_root = args.ood_root

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loaders_obj = make_loaders(run_args)
    model, disc = build_arch(run_args, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sample_id = args.sample_id or f"new0612_obj{int(args.object_id):03d}_pose{int(args.pose_id):02d}"
    calib = Path(args.calibration_json) if args.calibration_json else default_calibration_path(run_args, sample_id)
    camera_matrix = load_camera_matrix(calib)
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    found = None
    loader = loaders_obj["loaders"][args.split]  # type: ignore[index]
    with torch.no_grad():
        for batch in loader:
            ids = [str(x) for x in batch["sample_id"]]  # type: ignore[index]
            if sample_id not in ids:
                continue
            j = ids.index(sample_id)
            pred_norm = model_depth_norm(model, batch, device, run_args.arch)
            pred_norm_np = pred_norm.detach().cpu().numpy()[:, 0].astype(np.float32)
            target = batch["depth_raw"].detach().cpu().numpy()[:, 0].astype(np.float32)  # type: ignore[index]
            mask = batch["object_mask"].detach().cpu().numpy()[:, 0].astype(bool)  # type: ignore[index]
            scale = batch["scale_mm"].detach().cpu().numpy().astype(np.float32)  # type: ignore[index]
            center = batch["center_mm"].detach().cpu().numpy().astype(np.float32)  # type: ignore[index]
            pred = pred_norm_np[j] * float(scale[j]) + float(center[j])

            label_path = labels_path_for_sample(run_args, args.split, sample_id)
            gt_xyz = None
            if label_path.exists():
                labels = np.load(label_path, allow_pickle=True)
                if "xyz_camera" in labels:
                    gt_xyz = labels["xyz_camera"].astype(np.float32)

            rmse = float(np.sqrt(np.mean((pred[mask[j]] - target[j][mask[j]]) ** 2)))
            base_name = f"{sample_id}_{run_args.arch}_direct_seed{run_args.seed}"
            files: Dict[str, object] = {}
            if args.coordinate_mode in {"camera_projective", "both"}:
                p = out_dir / f"{base_name}_camera_projective.ply"
                n = write_camera_projective_ply(
                    p,
                    pred,
                    target[j],
                    mask[j],
                    camera_matrix,
                    gt_xyz,
                    stride=args.stride,
                    max_points=args.max_points,
                    comment=f"{sample_id} {run_args.arch} direct seed={run_args.seed} split={args.split}",
                )
                files["camera_projective"] = {"path": str(p), "vertices": n}
            if args.coordinate_mode in {"pixel_depth", "both"}:
                for suffix, z_scale in [("true_z", 1.0), ("z10_visual", 10.0)]:
                    p = out_dir / f"{base_name}_{suffix}.ply"
                    n = write_pixel_depth_ply(
                        p,
                        pred,
                        target[j],
                        mask[j],
                        z_scale=z_scale,
                        stride=args.stride,
                        max_points=args.max_points,
                        comment=f"{sample_id} {run_args.arch} direct seed={run_args.seed} split={args.split}",
                    )
                    files[suffix] = {"path": str(p), "vertices": n}
            found = {
                "sample_id": sample_id,
                "split": args.split,
                "arch": run_args.arch,
                "seed": run_args.seed,
                "checkpoint": str(args.ckpt),
                "rmse_mm": rmse,
                "object_vertices_full": int(mask[j].sum()),
                "stride": int(args.stride),
                "max_points": int(args.max_points),
                "coordinate_mode": args.coordinate_mode,
                "calibration_json": str(calib),
                "files": files,
            }
            break
    if found is None:
        raise RuntimeError(f"sample {sample_id} not found in split {args.split}")
    summary_path = out_dir / f"{sample_id}_{run_args.arch}_direct_seed{run_args.seed}_ply_summary.json"
    summary_path.write_text(json.dumps(found, indent=2, ensure_ascii=False), encoding="utf-8")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--split", default="ood")
    parser.add_argument("--sample_id", default="")
    parser.add_argument("--object_id", type=int, default=61)
    parser.add_argument("--pose_id", type=int, default=2)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--teacher_extra_root", default="")
    parser.add_argument("--ood_root", default="")
    parser.add_argument("--calibration_json", default="")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_points", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--coordinate_mode", choices=["camera_projective", "pixel_depth", "both"], default="both")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_direct(args), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
