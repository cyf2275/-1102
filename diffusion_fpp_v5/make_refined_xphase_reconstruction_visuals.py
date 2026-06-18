"""Visualize OOD 61-64 reconstructions for refined x-phase diffusion depth."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diagnose_single_frame3d_phase_residual import dir_dp_oracle, dir_dp_predict, load_model
from train_single_frame3d_full_pip_rcpc import create_loaders
from train_single_frame3d_physics_diffusion import forward_direct, load_base_model, pred_to_depth_mm, set_seed
from train_single_frame3d_refined_xphase_depth import depth_with_phi, load_phase_posterior, refined_phi
from train_single_frame3d_xphase_diffusion_rcpc import compact_batch, compact_item, dp_with_phi, rcpc_pred


def masked(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask.astype(bool), arr, np.nan)


def rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool) & np.isfinite(pred) & np.isfinite(target)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean((pred[m] - target[m]) ** 2)))


def robust_limits(arrays: List[np.ndarray], mask: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> Tuple[float, float]:
    vals = []
    m = mask.astype(bool)
    for arr in arrays:
        keep = m & np.isfinite(arr)
        if np.any(keep):
            vals.append(arr[keep])
    if not vals:
        return 0.0, 1.0
    x = np.concatenate(vals)
    a, b = np.percentile(x, [lo, hi])
    if not np.isfinite(a) or not np.isfinite(b) or abs(float(b - a)) < 1e-6:
        a, b = float(np.nanmin(x)), float(np.nanmax(x))
    if abs(float(b - a)) < 1e-6:
        b = a + 1.0
    return float(a), float(b)


def tensor_mm(pred_norm: torch.Tensor, batch: Dict[str, object]) -> np.ndarray:
    return pred_to_depth_mm(pred_norm, batch).detach().cpu().numpy()[:, 0]


@torch.no_grad()
def collect_predictions(args: argparse.Namespace, device: torch.device) -> List[Dict[str, object]]:
    loaders_obj = create_loaders(args)
    args.cond_channels = int(loaders_obj["cond_channels"])
    loader = loaders_obj["loaders"]["ood"]  # type: ignore[index]

    x_dir = Path(args.x_diag_dir) / "direction_x"
    base_model, _ = load_base_model(args.base_ckpt, args.cond_channels, device)
    phi_model = load_model(x_dir / "phi_predictor" / "checkpoints" / "best.pt", args.cond_channels, 4, args, device)
    old_dp = load_model(x_dir / "phase_depth_evidence" / "checkpoints" / "best.pt", args.cond_channels + 4, 1, args, device)
    oracle_dp = load_model(x_dir / "phase_depth_oracle_evidence" / "checkpoints" / "best.pt", args.cond_channels + 4, 1, args, device)
    refined_depth = load_model(Path(args.refined_depth_ckpt), args.cond_channels + 4, 1, args, device)
    posterior = load_phase_posterior(Path(args.phase_posterior_ckpt), args.cond_channels, args, device)

    summary = json.loads(Path(args.summary_path).read_text(encoding="utf-8"))
    gate = summary["gate"]

    out: List[Dict[str, object]] = []
    for batch in loader:
        base_norm = forward_direct(base_model, batch, device)[:, :1]
        phi_pred, phi_refined, phi_unc = refined_phi(
            posterior,
            phi_model,
            batch,
            args.phase_sample_steps,
            args.phase_ensemble_size,
        )
        x_pred_norm = dir_dp_predict(phi_model, old_dp, batch, "x", device)
        old_ref_norm = dp_with_phi(old_dp, batch, phi_refined, device)
        refined_norm = depth_with_phi(refined_depth, batch, phi_refined, device)
        oracle_norm = dir_dp_oracle(oracle_dp, batch, "x", device)

        base_mm = tensor_mm(base_norm, batch)
        x_pred_mm = tensor_mm(x_pred_norm, batch)
        old_ref_mm = tensor_mm(old_ref_norm, batch)
        refined_mm = tensor_mm(refined_norm, batch)
        oracle_mm = tensor_mm(oracle_norm, batch)
        target_mm = batch["depth_raw"].detach().cpu().numpy()[:, 0]  # type: ignore[index]
        mask = batch["object_mask"].detach().cpu().numpy()[:, 0]  # type: ignore[index]
        valid = batch["valid_mask"].detach().cpu().numpy()[:, 0]  # type: ignore[index]
        fringe = batch["fringe"].detach().cpu().numpy()[:, 0]  # type: ignore[index]

        for j, sid in enumerate(list(batch["sample_id"])):  # type: ignore[arg-type]
            item = compact_item(batch, j, base_norm, refined_norm, phi_unc)
            rcpc_norm, accepted = rcpc_pred(item, gate)
            rcpc_mm = pred_to_depth_mm(rcpc_norm, compact_batch(item)).detach().cpu().numpy()[0, 0]
            obj = int(batch["object_id"][j].item())  # type: ignore[index]
            pose = int(batch["pose_id"][j].item())  # type: ignore[index]
            sample = {
                "sample_id": str(sid),
                "object_id": obj,
                "pose_id": pose,
                "input": fringe[j],
                "target": target_mm[j],
                "mask": mask[j],
                "valid": valid[j],
                "direct_base": base_mm[j],
                "x_predicted_phase": x_pred_mm[j],
                "old_refined_map": old_ref_mm[j],
                "refined_depth": refined_mm[j],
                "rcpc": rcpc_mm,
                "x_oracle": oracle_mm[j],
                "rcpc_accepted": bool(accepted),
                "delta_mean": float(item["delta_mean"]),
                "unc_mean": float(item["unc_mean"]),
            }
            for key in ["direct_base", "x_predicted_phase", "old_refined_map", "refined_depth", "rcpc", "x_oracle"]:
                sample[f"{key}_rmse"] = rmse(sample[key], sample["target"], sample["mask"])  # type: ignore[arg-type]
            out.append(sample)
    out.sort(key=lambda x: (int(x["object_id"]), int(x["pose_id"])))
    return out


def write_metrics(samples: List[Dict[str, object]], out_dir: Path) -> None:
    keys = [
        "sample_id",
        "object_id",
        "pose_id",
        "rcpc_accepted",
        "delta_mean",
        "unc_mean",
        "direct_base_rmse",
        "x_predicted_phase_rmse",
        "old_refined_map_rmse",
        "refined_depth_rmse",
        "rcpc_rmse",
        "x_oracle_rmse",
    ]
    with (out_dir / "ood61_64_reconstruction_per_sample_rmse.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for s in samples:
            writer.writerow({k: s.get(k, "") for k in keys})


def save_overview(samples: List[Dict[str, object]], out_dir: Path) -> None:
    cols = [
        ("Input", "input", "gray"),
        ("GT", "target", "viridis"),
        ("Base", "direct_base", "viridis"),
        ("X phase", "x_predicted_phase", "viridis"),
        ("Refined depth", "refined_depth", "viridis"),
        ("RCPC", "rcpc", "viridis"),
        ("Oracle X", "x_oracle", "viridis"),
        ("Base err", "direct_base", "magma"),
        ("Refined err", "refined_depth", "magma"),
        ("RCPC err", "rcpc", "magma"),
    ]
    fig, axes = plt.subplots(len(samples), len(cols), figsize=(22, 2.2 * len(samples)), constrained_layout=True)
    if len(samples) == 1:
        axes = axes[None, :]
    for r, sample in enumerate(samples):
        target = sample["target"]  # type: ignore[assignment]
        mask = sample["mask"]  # type: ignore[assignment]
        preds = [sample[k] for k in ["target", "direct_base", "x_predicted_phase", "refined_depth", "rcpc", "x_oracle"]]
        vmin, vmax = robust_limits(preds, mask)  # type: ignore[arg-type]
        errs = [np.abs(sample[k] - target) for k in ["direct_base", "refined_depth", "rcpc"]]  # type: ignore[operator]
        _, emax = robust_limits(errs, mask, 50, 98)  # type: ignore[arg-type]
        emax = max(0.5, emax)
        for c, (title, key, cmap) in enumerate(cols):
            ax = axes[r, c]
            if key == "input":
                arr = sample[key]
                im = ax.imshow(arr, cmap=cmap)
            elif c >= 7:
                arr = np.abs(sample[key] - target)  # type: ignore[operator]
                im = ax.imshow(masked(arr, mask), cmap=cmap, vmin=0, vmax=emax)  # type: ignore[arg-type]
            else:
                arr = sample[key]
                im = ax.imshow(masked(arr, mask), cmap=cmap, vmin=vmin, vmax=vmax)  # type: ignore[arg-type]
            ax.axis("off")
            if r == 0:
                ax.set_title(title, fontsize=10)
        label = (
            f"obj{int(sample['object_id']):03d}/pose{int(sample['pose_id']):02d}\n"
            f"B {sample['direct_base_rmse']:.2f}  X {sample['x_predicted_phase_rmse']:.2f}  "
            f"R {sample['refined_depth_rmse']:.2f}  C {sample['rcpc_rmse']:.2f}  "
            f"use={int(sample['rcpc_accepted'])}"
        )
        axes[r, 0].set_ylabel(label, fontsize=8, rotation=0, labelpad=58, va="center")
    fig.suptitle("OOD 61-64 reconstruction: refined x-phase diffusion depth", fontsize=14)
    fig.savefig(out_dir / "ood61_64_refined_xphase_reconstruction_overview.png", dpi=150)
    plt.close(fig)


def save_sample_figures(samples: List[Dict[str, object]], out_dir: Path) -> None:
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        target = sample["target"]  # type: ignore[assignment]
        mask = sample["mask"]  # type: ignore[assignment]
        panels = [
            ("Input", sample["input"], "gray", None, None),
            ("GT", sample["target"], "viridis", None, None),
            (f"Base\n{sample['direct_base_rmse']:.3f}", sample["direct_base"], "viridis", None, None),
            (f"X phase\n{sample['x_predicted_phase_rmse']:.3f}", sample["x_predicted_phase"], "viridis", None, None),
            (f"Old refined\n{sample['old_refined_map_rmse']:.3f}", sample["old_refined_map"], "viridis", None, None),
            (f"New refined\n{sample['refined_depth_rmse']:.3f}", sample["refined_depth"], "viridis", None, None),
            (f"RCPC\n{sample['rcpc_rmse']:.3f}", sample["rcpc"], "viridis", None, None),
            (f"Oracle X\n{sample['x_oracle_rmse']:.3f}", sample["x_oracle"], "viridis", None, None),
        ]
        preds = [p[1] for p in panels[1:]]
        vmin, vmax = robust_limits(preds, mask)  # type: ignore[arg-type]
        fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
        axes_flat = axes.ravel()
        for i, (title, arr, cmap, lo, hi) in enumerate(panels):
            ax = axes_flat[i]
            if i == 0:
                ax.imshow(arr, cmap=cmap)
            else:
                ax.imshow(masked(arr, mask), cmap=cmap, vmin=vmin, vmax=vmax)  # type: ignore[arg-type]
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(
            f"obj{int(sample['object_id']):03d}/pose{int(sample['pose_id']):02d}  "
            f"RCPC accepted={sample['rcpc_accepted']}  "
            f"delta={sample['delta_mean']:.4f}  unc={sample['unc_mean']:.4f}",
            fontsize=12,
        )
        fig.savefig(sample_dir / f"obj{int(sample['object_id']):03d}_pose{int(sample['pose_id']):02d}_reconstruction.png", dpi=170)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--teacher_extra_root", required=True)
    parser.add_argument("--ood_root", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--x_diag_dir", required=True)
    parser.add_argument("--phase_posterior_ckpt", required=True)
    parser.add_argument("--refined_depth_ckpt", required=True)
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--ch_mult", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--time_emb_dim", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--phase_residual_scale", type=float, default=0.5)
    parser.add_argument("--phase_sample_steps", type=int, default=12)
    parser.add_argument("--phase_ensemble_size", type=int, default=3)
    parser.add_argument("--cache_features", action="store_true")
    parser.add_argument("--feature_cache_dir", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = Path(args.save_dir) / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_predictions(args, device)
    write_metrics(samples, out_dir)
    save_overview(samples, out_dir)
    save_sample_figures(samples, out_dir)
    summary = {
        "n": len(samples),
        "out_dir": str(out_dir),
        "overview": str(out_dir / "ood61_64_refined_xphase_reconstruction_overview.png"),
        "per_sample_csv": str(out_dir / "ood61_64_reconstruction_per_sample_rmse.csv"),
        "mean_rmse": {
            key: float(np.mean([float(s[f"{key}_rmse"]) for s in samples]))
            for key in ["direct_base", "x_predicted_phase", "old_refined_map", "refined_depth", "rcpc", "x_oracle"]
        },
    }
    (out_dir / "ood61_64_reconstruction_visual_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
