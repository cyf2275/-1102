from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch

from make_best_anchor_reconstruction_visuals import final_norm, to_mm
from train_refined_xphase_reliability_selector import (
    ReliabilityMLP,
    forward_pack,
    load_all_models,
    rmse_np,
)


def valid_vertices(mask: np.ndarray, stride: int, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask.astype(bool))
    if stride > 1:
        keep = (ys % stride == 0) & (xs % stride == 0)
        ys, xs = ys[keep], xs[keep]
    if max_points > 0 and len(xs) > max_points:
        rng = np.random.default_rng(20260618)
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
    """Back-project known camera/projective Z with the dataset's 3x4 camera matrix.

    The new0612 dataset stores a full 3x4 camera projection matrix, not separated
    intrinsics/extrinsics. For each 0-based pixel (u, v) and known Z, solve the
    two projective camera equations for X and Y.
    """
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
    if "camera_matrix_3x4" not in data:
        raise KeyError(f"{path} does not contain camera_matrix_3x4")
    mat = np.asarray(data["camera_matrix_3x4"], dtype=np.float64)
    if mat.shape != (3, 4):
        raise ValueError(f"camera_matrix_3x4 in {path} has shape {mat.shape}, expected (3, 4)")
    return mat


def labels_path_for_sample(run_args: SimpleNamespace, split: str, sample_id: str) -> Path | None:
    domain, rest = sample_id.split("_obj", 1)
    obj_text, pose_text = rest.split("_pose", 1)
    root = Path(getattr(run_args, "ood_root", "")) if split == "ood" else Path(getattr(run_args, "data_root", ""))
    if not str(root):
        return None
    return root / "samples" / domain / f"obj{int(obj_text):03d}" / f"pose{int(pose_text):02d}" / "labels.npz"


def default_calibration_path(run_args: SimpleNamespace, sample_id: str) -> Path:
    domain = sample_id.split("_obj", 1)[0]
    return Path(run_args.teacher_extra_root) / "calibration" / f"{domain}_calibration.json"


def write_ply(
    path: Path,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    prob: np.ndarray,
    unc: np.ndarray,
    anchor: np.ndarray,
    refined: np.ndarray,
    z_scale: float,
    stride: int,
    max_points: int,
    comment: str,
) -> int:
    ys, xs = valid_vertices(mask, stride=stride, max_points=max_points)
    pred_v = pred[ys, xs].astype(np.float32)
    target_v = target[ys, xs].astype(np.float32)
    err_v = np.abs(pred_v - target_v).astype(np.float32)
    prob_v = prob[ys, xs].astype(np.float32)
    unc_v = unc[ys, xs].astype(np.float32)
    anchor_v = anchor[ys, xs].astype(np.float32)
    refined_v = refined[ys, xs].astype(np.float32)
    rgb = colors_from_values(pred_v)

    h, w = pred.shape
    x_coord = xs.astype(np.float32) - (w - 1) * 0.5
    y_coord = (h - 1) * 0.5 - ys.astype(np.float32)
    z_coord = pred_v * float(z_scale)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment {comment}\n")
        f.write("comment x_y_are_centered_pixel_coordinates\n")
        f.write(f"comment z_coordinate_is_pred_depth_mm_times_{float(z_scale):.6g}\n")
        f.write(f"element vertex {len(xs)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float pred_depth_mm\n")
        f.write("property float gt_depth_mm\n")
        f.write("property float abs_error_mm\n")
        f.write("property float mlp_probability\n")
        f.write("property float posterior_uncertainty\n")
        f.write("property float anchor_depth_mm\n")
        f.write("property float diffusion_candidate_depth_mm\n")
        f.write("end_header\n")
        for i in range(len(xs)):
            r, g, b = rgb[i]
            f.write(
                f"{x_coord[i]:.6f} {y_coord[i]:.6f} {z_coord[i]:.6f} "
                f"{int(r)} {int(g)} {int(b)} "
                f"{pred_v[i]:.6f} {target_v[i]:.6f} {err_v[i]:.6f} "
                f"{prob_v[i]:.6f} {unc_v[i]:.6f} {anchor_v[i]:.6f} {refined_v[i]:.6f}\n"
            )
    return int(len(xs))


def write_camera_projective_ply(
    path: Path,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    prob: np.ndarray,
    unc: np.ndarray,
    anchor: np.ndarray,
    refined: np.ndarray,
    camera_matrix: np.ndarray,
    gt_xyz: np.ndarray | None,
    stride: int,
    max_points: int,
    comment: str,
) -> int:
    pred_xyz = backproject_camera_projective(pred, camera_matrix)
    valid = (
        mask.astype(bool)
        & np.isfinite(pred)
        & np.all(np.isfinite(pred_xyz), axis=2)
        & np.isfinite(target)
        & np.isfinite(prob)
        & np.isfinite(unc)
    )
    ys, xs = valid_vertices(valid, stride=stride, max_points=max_points)
    pts = pred_xyz[ys, xs].astype(np.float32)
    pred_v = pred[ys, xs].astype(np.float32)
    target_v = target[ys, xs].astype(np.float32)
    err_v = np.abs(pred_v - target_v).astype(np.float32)
    prob_v = prob[ys, xs].astype(np.float32)
    unc_v = unc[ys, xs].astype(np.float32)
    anchor_v = anchor[ys, xs].astype(np.float32)
    refined_v = refined[ys, xs].astype(np.float32)
    if gt_xyz is None:
        gt_v = np.full((len(xs), 3), np.nan, dtype=np.float32)
    else:
        gt_v = gt_xyz[ys, xs].astype(np.float32)
    rgb = colors_from_values(pred_v)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment {comment}\n")
        f.write("comment coordinates_are_backprojected_with_camera_matrix_3x4_and_0_based_pixels\n")
        f.write("comment x_y_z_are_in_dataset_calibration_world_unit\n")
        f.write(f"element vertex {len(xs)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float pixel_u\n")
        f.write("property float pixel_v\n")
        f.write("property float pred_depth_z\n")
        f.write("property float gt_depth_z\n")
        f.write("property float abs_error_z\n")
        f.write("property float mlp_probability\n")
        f.write("property float posterior_uncertainty\n")
        f.write("property float anchor_depth_z\n")
        f.write("property float diffusion_candidate_depth_z\n")
        f.write("property float gt_camera_x\n")
        f.write("property float gt_camera_y\n")
        f.write("property float gt_camera_z\n")
        f.write("end_header\n")
        for i in range(len(xs)):
            r, g, b = rgb[i]
            f.write(
                f"{pts[i, 0]:.6f} {pts[i, 1]:.6f} {pts[i, 2]:.6f} "
                f"{int(r)} {int(g)} {int(b)} "
                f"{float(xs[i]):.6f} {float(ys[i]):.6f} "
                f"{pred_v[i]:.6f} {target_v[i]:.6f} {err_v[i]:.6f} "
                f"{prob_v[i]:.6f} {unc_v[i]:.6f} {anchor_v[i]:.6f} {refined_v[i]:.6f} "
                f"{gt_v[i, 0]:.6f} {gt_v[i, 1]:.6f} {gt_v[i, 2]:.6f}\n"
            )
    return int(len(xs))


def export_sample(args: argparse.Namespace) -> Dict[str, object]:
    ckpt = torch.load(args.selector_ckpt, map_location="cpu")
    run_args = SimpleNamespace(**ckpt["args"])
    run_args.num_workers = int(args.num_workers)
    run_args.eval_batch_size = 1
    run_args.batch_size = 1
    run_args.phase_sample_steps = int(args.phase_sample_steps)
    run_args.phase_ensemble_size = int(args.phase_ensemble_size)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    models = load_all_models(run_args, device)
    selector = ReliabilityMLP(len(ckpt["feature_names"])).to(device)
    selector.load_state_dict(ckpt["model_state_dict"])
    selector.eval()
    mean_t = torch.from_numpy(ckpt["mean"].astype(np.float32)).to(device)
    std_t = torch.from_numpy(ckpt["std"].astype(np.float32)).to(device)
    mlp_gate = ckpt["summary"]["mlp_gate"]
    rule_gate = ckpt["summary"]["rule_gate"]

    loader = models["loaders_obj"]["loaders"][args.split]  # type: ignore[index]
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted_sample_id = args.sample_id
    if not wanted_sample_id:
        wanted_sample_id = f"new0612_obj{int(args.object_id):03d}_pose{int(args.pose_id):02d}"

    camera_matrix = None
    calibration_path = None
    if args.coordinate_mode == "camera_projective":
        calibration_path = Path(args.calibration_json) if args.calibration_json else default_calibration_path(run_args, wanted_sample_id)
        camera_matrix = load_camera_matrix(calibration_path)

    found: Dict[str, object] | None = None
    with torch.no_grad():
        for batch in loader:
            sample_ids = [str(x) for x in batch["sample_id"]]  # type: ignore[index]
            if wanted_sample_id not in sample_ids:
                continue
            j = sample_ids.index(wanted_sample_id)
            pack = forward_pack(batch, models, run_args, device)
            feats = pack["features"]
            b, c, h, w = feats.shape
            flat = feats.permute(0, 2, 3, 1).reshape(-1, c)
            prob = torch.sigmoid(selector((flat - mean_t) / std_t)).reshape(b, h, w).detach().cpu().numpy().astype(np.float32)

            base_n = pack["base_norm"].detach().cpu().numpy()[:, 0].astype(np.float32)
            x_n = pack["x_norm"].detach().cpu().numpy()[:, 0].astype(np.float32)
            anchor_n = pack["anchor_norm"].detach().cpu().numpy()[:, 0].astype(np.float32)
            refined_n = pack["refined_norm"].detach().cpu().numpy()[:, 0].astype(np.float32)
            unc = pack["phi_unc"].mean(dim=1).detach().cpu().numpy().astype(np.float32)
            delta_x = np.abs(refined_n - x_n).astype(np.float32)
            target = batch["depth_raw"].detach().cpu().numpy()[:, 0].astype(np.float32)  # type: ignore[index]
            mask = batch["object_mask"].detach().cpu().numpy()[:, 0].astype(bool)  # type: ignore[index]
            scale = batch["scale_mm"].detach().cpu().numpy().astype(np.float32)  # type: ignore[index]
            center = batch["center_mm"].detach().cpu().numpy().astype(np.float32)  # type: ignore[index]

            mlp_n, _, _ = final_norm(anchor_n[j], refined_n[j], prob[j], unc[j], delta_x[j], mlp_gate)
            rule_n, _, _ = final_norm(anchor_n[j], refined_n[j], prob[j], unc[j], delta_x[j], rule_gate)
            maps = {
                "base": to_mm(base_n[j], scale[j], center[j]),
                "x_phase": to_mm(x_n[j], scale[j], center[j]),
                "anchor_base_x_mean": to_mm(anchor_n[j], scale[j], center[j]),
                "diffusion_candidate": to_mm(refined_n[j], scale[j], center[j]),
                "ours_mlp_final": to_mm(mlp_n, scale[j], center[j]),
                "ours_rule_final": to_mm(rule_n, scale[j], center[j]),
                "gt": target[j],
            }

            metrics = {
                name: rmse_np(arr, target[j], mask[j])
                for name, arr in maps.items()
                if name != "gt"
            }
            gt_xyz = None
            backproject_check = None
            label_path = labels_path_for_sample(run_args, args.split, wanted_sample_id)
            if label_path is not None and label_path.exists():
                labels = np.load(label_path, allow_pickle=True)
                if "xyz_camera" in labels:
                    gt_xyz = labels["xyz_camera"].astype(np.float32)
                    if camera_matrix is not None:
                        gt_back = backproject_camera_projective(target[j], camera_matrix)
                        check_mask = mask[j] & np.all(np.isfinite(gt_xyz), axis=2) & np.all(np.isfinite(gt_back), axis=2)
                        if np.any(check_mask):
                            e = np.linalg.norm(gt_back[check_mask] - gt_xyz[check_mask], axis=1)
                            backproject_check = {
                                "mean": float(np.mean(e)),
                                "median": float(np.median(e)),
                                "p95": float(np.percentile(e, 95)),
                                "max": float(np.max(e)),
                            }
            files: Dict[str, object] = {}
            for name in args.outputs:
                if name not in maps:
                    raise KeyError(f"unknown output map: {name}")
                if args.coordinate_mode == "camera_projective":
                    if camera_matrix is None:
                        raise RuntimeError("camera matrix was not loaded")
                    path = out_dir / f"{wanted_sample_id}_{name}_camera_projective.ply"
                    n = write_camera_projective_ply(
                        path,
                        maps[name],
                        target[j],
                        mask[j],
                        prob[j],
                        unc[j],
                        maps["anchor_base_x_mean"],
                        maps["diffusion_candidate"],
                        camera_matrix,
                        gt_xyz,
                        stride=int(args.stride),
                        max_points=int(args.max_points),
                        comment=f"{wanted_sample_id} {name} split={args.split} anchor_mode={getattr(run_args, 'anchor_mode', None)}",
                    )
                    files[f"{name}_camera_projective"] = {"path": str(path), "vertices": n}
                else:
                    for suffix, z_scale in [("true_z", 1.0), ("z10_visual", 10.0)]:
                        path = out_dir / f"{wanted_sample_id}_{name}_{suffix}.ply"
                        n = write_ply(
                            path,
                            maps[name],
                            target[j],
                            mask[j],
                            prob[j],
                            unc[j],
                            maps["anchor_base_x_mean"],
                            maps["diffusion_candidate"],
                            z_scale=z_scale,
                            stride=int(args.stride),
                            max_points=int(args.max_points),
                            comment=f"{wanted_sample_id} {name} split={args.split} anchor_mode={getattr(run_args, 'anchor_mode', None)}",
                        )
                        files[f"{name}_{suffix}"] = {"path": str(path), "vertices": n}

            found = {
                "sample_id": wanted_sample_id,
                "split": args.split,
                "object_id": int(batch["object_id"][j].item()),  # type: ignore[index]
                "pose_id": int(batch["pose_id"][j].item()),  # type: ignore[index]
                "selector_ckpt": args.selector_ckpt,
                "anchor_mode": getattr(run_args, "anchor_mode", None),
                "mlp_gate": mlp_gate,
                "rule_gate": rule_gate,
                "rmse_mm": metrics,
                "mask_vertices_full": int(mask[j].sum()),
                "stride": int(args.stride),
                "max_points": int(args.max_points),
                "coordinate_mode": args.coordinate_mode,
                "calibration_json": str(calibration_path) if calibration_path is not None else None,
                "gt_xyz_label_path": str(label_path) if label_path is not None and label_path.exists() else None,
                "gt_backproject_check_vs_xyz_camera": backproject_check,
                "files": files,
                "coordinate_note": (
                    "camera_projective files use camera_matrix_3x4 and 0-based pixels to solve X/Y from predicted depth_z. "
                    "pixel_depth files use centered pixel x/y and z=depth_z or depth_z*10."
                ),
            }
            break

    if found is None:
        raise RuntimeError(f"sample {wanted_sample_id!r} not found in split {args.split!r}")
    (out_dir / f"{wanted_sample_id}_ply_summary.json").write_text(
        json.dumps(found, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return found


def parse_outputs(text: str) -> Iterable[str]:
    return [x.strip() for x in text.replace(",", " ").split() if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector_ckpt", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--split", default="ood")
    parser.add_argument("--sample_id", default="")
    parser.add_argument("--object_id", type=int, default=61)
    parser.add_argument("--pose_id", type=int, default=2)
    parser.add_argument(
        "--outputs",
        type=parse_outputs,
        default=parse_outputs("ours_mlp_final,ours_rule_final,anchor_base_x_mean,gt"),
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_points", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--phase_sample_steps", type=int, default=12)
    parser.add_argument("--phase_ensemble_size", type=int, default=3)
    parser.add_argument("--coordinate_mode", choices=["pixel_depth", "camera_projective"], default="pixel_depth")
    parser.add_argument("--calibration_json", default="")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    summary = export_sample(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
