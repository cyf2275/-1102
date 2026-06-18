"""Diagnostics for refined x-phase depth failures.

This is inference-only. It checks whether diffusion-refined x phase improves
phase evidence but fails after phase-to-depth mapping, and whether a local
RCPC-style gate has enough signal to recover ordinary test performance.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diagnose_single_frame3d_phase_residual import dir_dp_oracle, dir_dp_predict, dir_phi_oracle, load_model
from train_single_frame3d_full_pip_rcpc import create_loaders
from train_single_frame3d_physics_diffusion import forward_direct, load_base_model, pred_to_depth_mm, set_seed
from train_single_frame3d_refined_xphase_depth import depth_with_phi, load_phase_posterior, refined_phi
from train_single_frame3d_xphase_diffusion_rcpc import compact_item, dp_with_phi, map_phi, rcpc_pred


METHODS = ["base", "x_pred", "old_refined", "refined", "rcpc", "oracle"]


def tensor_to_mm(pred_norm: torch.Tensor, batch: Dict[str, object]) -> np.ndarray:
    return pred_to_depth_mm(pred_norm, batch).detach().cpu().numpy()[:, 0].astype(np.float32)


def rmse_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool) & np.isfinite(pred) & np.isfinite(target)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean((pred[m] - target[m]) ** 2)))


def mae_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool) & np.isfinite(pred) & np.isfinite(target)
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.abs(pred[m] - target[m])))


def mse_sum(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Tuple[float, int]:
    m = mask.astype(bool) & np.isfinite(pred) & np.isfinite(target)
    if not np.any(m):
        return 0.0, 0
    err = pred[m] - target[m]
    return float(np.sum(err * err)), int(np.sum(m))


def grad_mag(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    gy, gx = np.gradient(arr)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def sincos_grad(sin_arr: np.ndarray, cos_arr: np.ndarray) -> np.ndarray:
    return np.sqrt(grad_mag(sin_arr) ** 2 + grad_mag(cos_arr) ** 2).astype(np.float32)


def erode_bool(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    m = mask.astype(bool)
    if radius <= 0:
        return m
    pad = np.pad(m, radius, mode="constant", constant_values=False)
    out = m.copy()
    h, w = m.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out &= pad[radius + dy: radius + dy + h, radius + dx: radius + dx + w]
    return out


def top_quantile(vals: np.ndarray, mask: np.ndarray, q: float = 75.0) -> np.ndarray:
    m = mask.astype(bool) & np.isfinite(vals)
    if not np.any(m):
        return np.zeros_like(mask, dtype=bool)
    if float(np.nanmax(vals[m]) - np.nanmin(vals[m])) < 1e-8:
        return np.zeros_like(mask, dtype=bool)
    tau = float(np.percentile(vals[m], q))
    return m & (vals >= tau)


def bottom_quantile(vals: np.ndarray, mask: np.ndarray, q: float = 25.0) -> np.ndarray:
    m = mask.astype(bool) & np.isfinite(vals)
    if not np.any(m):
        return np.zeros_like(mask, dtype=bool)
    if float(np.nanmax(vals[m]) - np.nanmin(vals[m])) < 1e-8:
        return np.zeros_like(mask, dtype=bool)
    tau = float(np.percentile(vals[m], q))
    return m & (vals <= tau)


def masked_mean(vals: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool) & np.isfinite(vals)
    if not np.any(m):
        return float("nan")
    return float(np.mean(vals[m]))


def collect_threshold_values(records: Iterable[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    deltas: List[np.ndarray] = []
    uncs: List[np.ndarray] = []
    for rec in records:
        mask = rec["mask"].astype(bool)  # type: ignore[index]
        deltas.append(rec["delta_norm"][mask].astype(np.float32))  # type: ignore[index]
        uncs.append(rec["unc_map"][mask].astype(np.float32))  # type: ignore[index]
    if not deltas:
        return np.asarray([0.0], dtype=np.float32), np.asarray([0.0], dtype=np.float32)
    return np.concatenate(deltas), np.concatenate(uncs)


def pixel_gate_pred(rec: Dict[str, object], gate: Dict[str, float]) -> np.ndarray:
    base = rec["base_norm"]  # type: ignore[assignment]
    cand = rec["refined_norm"]  # type: ignore[assignment]
    use = (rec["delta_norm"] <= gate["delta_max"]) & (rec["unc_map"] <= gate["unc_max"])  # type: ignore[operator]
    out = base + float(gate["alpha"]) * (cand - base) * use.astype(np.float32)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def norm_to_mm(norm: np.ndarray, rec: Dict[str, object]) -> np.ndarray:
    return norm * float(rec["scale_mm"]) + float(rec["center_mm"])


def eval_pixel_gate(records: List[Dict[str, object]], gate: Dict[str, float]) -> float:
    vals = []
    for rec in records:
        pred = norm_to_mm(pixel_gate_pred(rec, gate), rec)
        vals.append(rmse_np(pred, rec["target"], rec["mask"]))  # type: ignore[arg-type]
    return float(np.mean(vals)) if vals else float("nan")


def select_pixel_gate(val_records: List[Dict[str, object]]) -> Dict[str, float]:
    delta_vals, unc_vals = collect_threshold_values(val_records)
    qs = [10, 20, 35, 50, 65, 80, 90, 97, 100]
    delta_grid = sorted(set(float(x) for x in np.percentile(delta_vals, qs)))
    unc_grid = sorted(set(float(x) for x in np.percentile(unc_vals, qs)))
    best = {
        "alpha": 0.0,
        "delta_max": float(delta_grid[-1]),
        "unc_max": float(unc_grid[-1]),
        "val_rmse": float("inf"),
        "accepted_pixel_fraction": 0.0,
    }
    for alpha in [0.25, 0.5, 0.75, 1.0]:
        for delta_max in delta_grid:
            for unc_max in unc_grid:
                gate = {"alpha": alpha, "delta_max": delta_max, "unc_max": unc_max}
                score = eval_pixel_gate(val_records, gate)
                if score < best["val_rmse"]:
                    acc_num = 0
                    acc_den = 0
                    for rec in val_records:
                        m = rec["mask"].astype(bool)  # type: ignore[index]
                        use = (rec["delta_norm"] <= delta_max) & (rec["unc_map"] <= unc_max)  # type: ignore[operator]
                        acc_num += int(np.sum(use & m))
                        acc_den += int(np.sum(m))
                    best = {
                        "alpha": float(alpha),
                        "delta_max": float(delta_max),
                        "unc_max": float(unc_max),
                        "val_rmse": float(score),
                        "accepted_pixel_fraction": float(acc_num / max(1, acc_den)),
                    }
    return best


def region_masks(rec: Dict[str, object]) -> Dict[str, np.ndarray]:
    mask = rec["mask"].astype(bool)  # type: ignore[index]
    target = rec["target"]  # type: ignore[assignment]
    phi_true = rec["phi_true"]  # type: ignore[assignment]
    conf_x = np.clip(phi_true[3], 0.0, 1.0)
    order_x = phi_true[2]
    boundary = mask & (~erode_bool(mask, radius=3))
    interior = mask & (~boundary)
    depth_g = grad_mag(target)
    phase_g = sincos_grad(phi_true[0], phi_true[1])
    order_g = grad_mag(order_x)
    order_region = top_quantile(order_g, mask, 75.0) & (order_g > 1e-5)
    unc_map = rec["unc_map"]  # type: ignore[assignment]
    worse_x = mask & (np.abs(rec["refined"] - target) > np.abs(rec["x_pred"] - target))  # type: ignore[operator]
    return {
        "all": mask,
        "boundary": boundary,
        "interior": interior,
        "high_depth_grad": top_quantile(depth_g, mask, 75.0),
        "low_depth_grad": bottom_quantile(depth_g, mask, 25.0),
        "high_phase_grad": top_quantile(phase_g, mask, 75.0),
        "low_conf_x": bottom_quantile(conf_x, mask, 25.0),
        "high_conf_x": top_quantile(conf_x, mask, 75.0),
        "order_edge": order_region,
        "high_unc": top_quantile(unc_map, mask, 75.0),
        "low_unc": bottom_quantile(unc_map, mask, 25.0),
        "refined_worse_than_x": worse_x,
    }


def summarize_split(records: List[Dict[str, object]], pixel_gate: Dict[str, float]) -> Dict[str, object]:
    method_rmse: Dict[str, float] = {}
    method_mae: Dict[str, float] = {}
    for method in METHODS:
        method_rmse[method] = float(np.mean([r[f"{method}_rmse"] for r in records])) if records else float("nan")  # type: ignore[index]
        method_mae[method] = float(np.mean([r[f"{method}_mae"] for r in records])) if records else float("nan")  # type: ignore[index]

    pixel_gate_vals = []
    oracle_base_refined_vals = []
    oracle_x_refined_vals = []
    pixel_gate_accept = []
    phase_pred = []
    phase_refined = []
    for rec in records:
        pg_norm = pixel_gate_pred(rec, pixel_gate)
        pg_mm = norm_to_mm(pg_norm, rec)
        pixel_gate_vals.append(rmse_np(pg_mm, rec["target"], rec["mask"]))  # type: ignore[arg-type]
        m = rec["mask"].astype(bool)  # type: ignore[index]
        pixel_gate_accept.append(float(np.sum((np.abs(pg_norm - rec["base_norm"]) > 1e-7) & m) / max(1, np.sum(m))))  # type: ignore[operator]
        target = rec["target"]  # type: ignore[assignment]
        best_br = np.where(np.abs(rec["refined"] - target) < np.abs(rec["base"] - target), rec["refined"], rec["base"])  # type: ignore[operator]
        best_xr = np.where(np.abs(rec["refined"] - target) < np.abs(rec["x_pred"] - target), rec["refined"], rec["x_pred"])  # type: ignore[operator]
        oracle_base_refined_vals.append(rmse_np(best_br, target, rec["mask"]))  # type: ignore[arg-type]
        oracle_x_refined_vals.append(rmse_np(best_xr, target, rec["mask"]))  # type: ignore[arg-type]
        phase_pred.append(float(rec["phase_pred_err"]))
        phase_refined.append(float(rec["phase_refined_err"]))

    depth_gain_refined_vs_x = [float(r["x_pred_rmse"] - r["refined_rmse"]) for r in records]
    phase_gain = [float(r["phase_pred_err"] - r["phase_refined_err"]) for r in records]
    mismatch = [
        (float(r["phase_pred_err"] - r["phase_refined_err"]) > 0.0)
        and (float(r["x_pred_rmse"] - r["refined_rmse"]) < 0.0)
        for r in records
    ]
    corr = float("nan")
    if len(records) >= 3 and np.std(phase_gain) > 1e-12 and np.std(depth_gain_refined_vs_x) > 1e-12:
        corr = float(np.corrcoef(np.asarray(phase_gain), np.asarray(depth_gain_refined_vs_x))[0, 1])

    return {
        "n": len(records),
        "rmse": method_rmse,
        "mae": method_mae,
        "pixel_gate_rmse": float(np.mean(pixel_gate_vals)) if pixel_gate_vals else float("nan"),
        "pixel_gate_accept_fraction": float(np.mean(pixel_gate_accept)) if pixel_gate_accept else float("nan"),
        "oracle_local_base_refined_rmse": float(np.mean(oracle_base_refined_vals)) if oracle_base_refined_vals else float("nan"),
        "oracle_local_x_refined_rmse": float(np.mean(oracle_x_refined_vals)) if oracle_x_refined_vals else float("nan"),
        "phase_pred_err": float(np.mean(phase_pred)) if phase_pred else float("nan"),
        "phase_refined_err": float(np.mean(phase_refined)) if phase_refined else float("nan"),
        "phase_improved_fraction": float(np.mean([a > b for a, b in zip(phase_pred, phase_refined)])) if phase_pred else float("nan"),
        "depth_refined_better_than_x_fraction": float(np.mean([x > 0 for x in depth_gain_refined_vs_x])) if records else float("nan"),
        "phase_improved_but_depth_worse_fraction": float(np.mean(mismatch)) if mismatch else float("nan"),
        "phase_gain_depth_gain_corr": corr,
    }


def summarize_regions(records: List[Dict[str, object]]) -> Dict[str, object]:
    acc: Dict[str, Dict[str, Dict[str, float]]] = {}
    for rec in records:
        masks = region_masks(rec)
        target = rec["target"]  # type: ignore[assignment]
        for region, m in masks.items():
            slot = acc.setdefault(region, {method: {"sse": 0.0, "count": 0.0} for method in METHODS})
            for method in METHODS:
                sse, count = mse_sum(rec[method], target, m)  # type: ignore[arg-type]
                slot[method]["sse"] += sse
                slot[method]["count"] += count
            # Diagnostic excess relative to x phase candidate.
            ex_slot = slot.setdefault("excess_refined_vs_x", {"sum": 0.0, "count": 0.0})
            mm = m.astype(bool)
            if np.any(mm):
                ex = (rec["refined"][mm] - target[mm]) ** 2 - (rec["x_pred"][mm] - target[mm]) ** 2  # type: ignore[index]
                ex_slot["sum"] += float(np.sum(ex))
                ex_slot["count"] += int(np.sum(mm))
    out: Dict[str, object] = {}
    total_all = max(1.0, acc.get("all", {}).get("base", {}).get("count", 1.0))
    for region, data in acc.items():
        row: Dict[str, object] = {}
        for method in METHODS:
            count = data[method]["count"]
            row[f"{method}_rmse"] = float(np.sqrt(data[method]["sse"] / max(1.0, count)))
            row["pixel_fraction"] = float(count / total_all)
        ex = data.get("excess_refined_vs_x", {"sum": 0.0, "count": 0.0})
        row["excess_mse_refined_vs_x"] = float(ex["sum"] / max(1.0, ex["count"]))
        out[region] = row
    return out


@torch.no_grad()
def collect_records(args: argparse.Namespace, device: torch.device) -> Dict[str, List[Dict[str, object]]]:
    loaders_obj = create_loaders(args)
    args.cond_channels = int(loaders_obj["cond_channels"])
    loaders = loaders_obj["loaders"]  # type: ignore[assignment]

    x_dir = Path(args.x_diag_dir) / "direction_x"
    base_model, _ = load_base_model(args.base_ckpt, args.cond_channels, device)
    phi_model = load_model(x_dir / "phi_predictor" / "checkpoints" / "best.pt", args.cond_channels, 4, args, device)
    old_dp = load_model(x_dir / "phase_depth_evidence" / "checkpoints" / "best.pt", args.cond_channels + 4, 1, args, device)
    oracle_dp = load_model(x_dir / "phase_depth_oracle_evidence" / "checkpoints" / "best.pt", args.cond_channels + 4, 1, args, device)
    refined_depth = load_model(Path(args.refined_depth_ckpt), args.cond_channels + 4, 1, args, device)
    posterior = load_phase_posterior(Path(args.phase_posterior_ckpt), args.cond_channels, args, device)
    gate = json.loads(Path(args.summary_path).read_text(encoding="utf-8"))["gate"]

    out: Dict[str, List[Dict[str, object]]] = {}
    for split in ["val", "test", "ood"]:
        if split not in loaders:
            continue
        split_records: List[Dict[str, object]] = []
        for batch in loaders[split]:
            base_norm = forward_direct(base_model, batch, device)[:, :1]
            phi_pred, phi_refined, phi_unc = refined_phi(posterior, phi_model, batch, args.phase_sample_steps, args.phase_ensemble_size)
            phi_true = dir_phi_oracle(batch, "x", device)
            x_pred_norm = dir_dp_predict(phi_model, old_dp, batch, "x", device)
            old_ref_norm = dp_with_phi(old_dp, batch, phi_refined, device)
            refined_norm = depth_with_phi(refined_depth, batch, phi_refined, device)
            oracle_norm = dir_dp_oracle(oracle_dp, batch, "x", device)

            pred_mm = {
                "base": tensor_to_mm(base_norm, batch),
                "x_pred": tensor_to_mm(x_pred_norm, batch),
                "old_refined": tensor_to_mm(old_ref_norm, batch),
                "refined": tensor_to_mm(refined_norm, batch),
                "oracle": tensor_to_mm(oracle_norm, batch),
            }
            target = batch["depth_raw"].detach().cpu().numpy()[:, 0].astype(np.float32)  # type: ignore[index]
            mask = batch["object_mask"].detach().cpu().numpy()[:, 0].astype(bool)  # type: ignore[index]
            valid = batch["valid_mask"].detach().cpu().numpy()[:, 0].astype(bool)  # type: ignore[index]
            base_norm_np = base_norm.detach().cpu().numpy()[:, 0].astype(np.float32)
            refined_norm_np = refined_norm.detach().cpu().numpy()[:, 0].astype(np.float32)
            unc_np = phi_unc.detach().cpu().numpy().astype(np.float32)
            phi_pred_m = map_phi(phi_pred).detach().cpu().numpy().astype(np.float32)
            phi_ref_m = map_phi(phi_refined).detach().cpu().numpy().astype(np.float32)
            phi_true_m = map_phi(phi_true).detach().cpu().numpy().astype(np.float32)
            phi_true_raw = phi_true.detach().cpu().numpy().astype(np.float32)
            scale = batch["scale_mm"].detach().cpu().numpy().astype(np.float32)  # type: ignore[index]
            center = batch["center_mm"].detach().cpu().numpy().astype(np.float32)  # type: ignore[index]

            for j, sid in enumerate(list(batch["sample_id"])):  # type: ignore[arg-type]
                item = compact_item(batch, j, base_norm, refined_norm, phi_unc)
                rcpc_norm, accepted = rcpc_pred(item, gate)
                rcpc_mm = pred_to_depth_mm(rcpc_norm, {
                    "scale_mm": batch["scale_mm"][j:j + 1].detach().cpu(),  # type: ignore[index]
                    "center_mm": batch["center_mm"][j:j + 1].detach().cpu(),  # type: ignore[index]
                    "depth_raw": batch["depth_raw"][j:j + 1].detach().cpu(),  # type: ignore[index]
                    "object_mask": batch["object_mask"][j:j + 1].detach().cpu(),  # type: ignore[index]
                    "valid_mask": batch["valid_mask"][j:j + 1].detach().cpu(),  # type: ignore[index]
                }).detach().cpu().numpy()[0, 0].astype(np.float32)
                rec: Dict[str, object] = {
                    "split": split,
                    "sample_id": str(sid),
                    "object_id": int(batch["object_id"][j].item()),  # type: ignore[index]
                    "pose_id": int(batch["pose_id"][j].item()),  # type: ignore[index]
                    "target": target[j],
                    "mask": mask[j],
                    "valid": valid[j],
                    "base_norm": base_norm_np[j],
                    "refined_norm": refined_norm_np[j],
                    "scale_mm": float(scale[j]),
                    "center_mm": float(center[j]),
                    "delta_norm": np.abs(refined_norm_np[j] - base_norm_np[j]).astype(np.float32),
                    "unc_map": np.mean(unc_np[j], axis=0).astype(np.float32),
                    "phi_true": phi_true_raw[j],
                    "phi_pred_err_map": np.sqrt(np.mean((phi_pred_m[j] - phi_true_m[j]) ** 2, axis=0)).astype(np.float32),
                    "phi_refined_err_map": np.sqrt(np.mean((phi_ref_m[j] - phi_true_m[j]) ** 2, axis=0)).astype(np.float32),
                    "rcpc_accepted": bool(accepted),
                    "delta_mean": float(item["delta_mean"]),
                    "unc_mean": float(item["unc_mean"]),
                }
                for method in ["base", "x_pred", "old_refined", "refined", "oracle"]:
                    rec[method] = pred_mm[method][j]
                rec["rcpc"] = rcpc_mm
                for method in METHODS:
                    rec[f"{method}_rmse"] = rmse_np(rec[method], rec["target"], rec["mask"])  # type: ignore[arg-type]
                    rec[f"{method}_mae"] = mae_np(rec[method], rec["target"], rec["mask"])  # type: ignore[arg-type]
                rec["phase_pred_err"] = masked_mean(rec["phi_pred_err_map"], rec["mask"])  # type: ignore[arg-type]
                rec["phase_refined_err"] = masked_mean(rec["phi_refined_err_map"], rec["mask"])  # type: ignore[arg-type]
                rec["mean_conf_x"] = masked_mean(np.clip(phi_true_raw[j, 3], 0.0, 1.0), rec["mask"])  # type: ignore[arg-type]
                split_records.append(rec)
        out[split] = split_records
    return out


def write_per_sample(records_by_split: Dict[str, List[Dict[str, object]]], pixel_gate: Dict[str, float], out_dir: Path) -> None:
    keys = [
        "split",
        "sample_id",
        "object_id",
        "pose_id",
        "base_rmse",
        "x_pred_rmse",
        "old_refined_rmse",
        "refined_rmse",
        "rcpc_rmse",
        "oracle_rmse",
        "pixel_gate_rmse",
        "oracle_local_base_refined_rmse",
        "oracle_local_x_refined_rmse",
        "phase_pred_err",
        "phase_refined_err",
        "phase_gain",
        "refined_gain_vs_x",
        "refined_gain_vs_base",
        "phase_improved_but_depth_worse",
        "rcpc_accepted",
        "delta_mean",
        "unc_mean",
        "mean_conf_x",
        "refined_worse_than_x_pixel_fraction",
    ]
    with (out_dir / "refined_xphase_failure_per_sample.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for split, records in records_by_split.items():
            for rec in records:
                pg = norm_to_mm(pixel_gate_pred(rec, pixel_gate), rec)
                target = rec["target"]  # type: ignore[assignment]
                mask = rec["mask"]  # type: ignore[assignment]
                best_br = np.where(np.abs(rec["refined"] - target) < np.abs(rec["base"] - target), rec["refined"], rec["base"])  # type: ignore[operator]
                best_xr = np.where(np.abs(rec["refined"] - target) < np.abs(rec["x_pred"] - target), rec["refined"], rec["x_pred"])  # type: ignore[operator]
                worse_x = mask.astype(bool) & (np.abs(rec["refined"] - target) > np.abs(rec["x_pred"] - target))  # type: ignore[operator]
                row = {
                    "split": split,
                    "sample_id": rec["sample_id"],
                    "object_id": rec["object_id"],
                    "pose_id": rec["pose_id"],
                    "base_rmse": rec["base_rmse"],
                    "x_pred_rmse": rec["x_pred_rmse"],
                    "old_refined_rmse": rec["old_refined_rmse"],
                    "refined_rmse": rec["refined_rmse"],
                    "rcpc_rmse": rec["rcpc_rmse"],
                    "oracle_rmse": rec["oracle_rmse"],
                    "pixel_gate_rmse": rmse_np(pg, target, mask),
                    "oracle_local_base_refined_rmse": rmse_np(best_br, target, mask),
                    "oracle_local_x_refined_rmse": rmse_np(best_xr, target, mask),
                    "phase_pred_err": rec["phase_pred_err"],
                    "phase_refined_err": rec["phase_refined_err"],
                    "phase_gain": float(rec["phase_pred_err"] - rec["phase_refined_err"]),
                    "refined_gain_vs_x": float(rec["x_pred_rmse"] - rec["refined_rmse"]),
                    "refined_gain_vs_base": float(rec["base_rmse"] - rec["refined_rmse"]),
                    "phase_improved_but_depth_worse": bool((rec["phase_pred_err"] > rec["phase_refined_err"]) and (rec["x_pred_rmse"] < rec["refined_rmse"])),
                    "rcpc_accepted": rec["rcpc_accepted"],
                    "delta_mean": rec["delta_mean"],
                    "unc_mean": rec["unc_mean"],
                    "mean_conf_x": rec["mean_conf_x"],
                    "refined_worse_than_x_pixel_fraction": float(np.sum(worse_x) / max(1, np.sum(mask))),
                }
                writer.writerow(row)


def write_region_csv(region_summary: Dict[str, Dict[str, object]], out_dir: Path) -> None:
    keys = [
        "split",
        "region",
        "pixel_fraction",
        "base_rmse",
        "x_pred_rmse",
        "old_refined_rmse",
        "refined_rmse",
        "rcpc_rmse",
        "oracle_rmse",
        "excess_mse_refined_vs_x",
    ]
    with (out_dir / "refined_xphase_failure_region_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for split, data in region_summary.items():
            for region, metrics in data.items():  # type: ignore[union-attr]
                row = {"split": split, "region": region}
                row.update({k: metrics.get(k, "") for k in keys if k not in {"split", "region"}})  # type: ignore[attr-defined]
                writer.writerow(row)


def make_plots(summary: Dict[str, object], region_summary: Dict[str, Dict[str, object]], out_dir: Path) -> None:
    splits = [s for s in ["val", "test", "ood"] if s in summary["splits"]]  # type: ignore[operator]
    labels = ["base", "x_pred", "refined", "rcpc", "pixel_gate", "oracle_local_base_refined", "oracle"]
    display = ["Base", "X phase", "Refined", "Sample RCPC", "Pixel gate", "Oracle local", "True X"]
    vals = []
    for split in splits:
        data = summary["splits"][split]  # type: ignore[index]
        vals.append([
            data["rmse"]["base"],
            data["rmse"]["x_pred"],
            data["rmse"]["refined"],
            data["rmse"]["rcpc"],
            data["pixel_gate_rmse"],
            data["oracle_local_base_refined_rmse"],
            data["rmse"]["oracle"],
        ])
    x = np.arange(len(splits))
    width = 0.11
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, lab in enumerate(display):
        ax.bar(x + (i - 3) * width, [row[i] for row in vals], width, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Object RMSE")
    ax.set_title("Refined x-phase depth diagnostic")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(out_dir / "split_rmse_diagnostic.png", dpi=160)
    plt.close(fig)

    if "test" in region_summary:
        regions = [r for r in region_summary["test"].keys() if r not in {"all", "refined_worse_than_x"}]
        regions = sorted(regions, key=lambda r: float(region_summary["test"][r]["excess_mse_refined_vs_x"]), reverse=True)
        ex = [float(region_summary["test"][r]["excess_mse_refined_vs_x"]) for r in regions]
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(np.arange(len(regions)), ex)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(regions)))
        ax.set_xticklabels(regions, rotation=35, ha="right")
        ax.set_ylabel("Excess MSE: refined - x phase")
        ax.set_title("Where refined-depth candidate loses on ordinary test")
        fig.tight_layout()
        fig.savefig(out_dir / "test_region_excess_mse_refined_vs_x.png", dpi=160)
        plt.close(fig)


def compact_summary(summary: Dict[str, object]) -> str:
    lines = ["# Refined X-Phase Depth Failure Diagnostics", ""]
    lines.append("## Split Summary")
    lines.append("")
    lines.append("| split | base | x phase | refined | sample RCPC | pixel gate | local oracle base/refined | true x | phase pred | phase refined | mismatch |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split, data in summary["splits"].items():  # type: ignore[union-attr]
        lines.append(
            f"| {split} | {data['rmse']['base']:.4f} | {data['rmse']['x_pred']:.4f} | "
            f"{data['rmse']['refined']:.4f} | {data['rmse']['rcpc']:.4f} | "
            f"{data['pixel_gate_rmse']:.4f} | {data['oracle_local_base_refined_rmse']:.4f} | "
            f"{data['rmse']['oracle']:.4f} | {data['phase_pred_err']:.4f} | "
            f"{data['phase_refined_err']:.4f} | {data['phase_improved_but_depth_worse_fraction']:.2%} |"
        )
    lines.append("")
    lines.append("## Selected Pixel Gate")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["pixel_gate"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Main Diagnostic")
    lines.append("")
    lines.append(
        "If phase error decreases while depth RMSE gets worse, the weak link is not the diffusion "
        "posterior itself but the phase-to-depth adapter or the correction selection rule."
    )
    lines.append(
        "The local oracle columns show how much room exists if a pixel/patch-level reliability "
        "selector can keep only the helpful refined-phase regions."
    )
    return "\n".join(lines) + "\n"


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

    out_dir = Path(args.save_dir) / "diagnostics_refined_xphase"
    out_dir.mkdir(parents=True, exist_ok=True)

    records_by_split = collect_records(args, device)
    pixel_gate = select_pixel_gate(records_by_split.get("val", []))
    split_summary = {split: summarize_split(records, pixel_gate) for split, records in records_by_split.items()}
    region_summary = {split: summarize_regions(records) for split, records in records_by_split.items()}
    summary = {
        "out_dir": str(out_dir),
        "pixel_gate": pixel_gate,
        "splits": split_summary,
        "files": {
            "per_sample": str(out_dir / "refined_xphase_failure_per_sample.csv"),
            "regions": str(out_dir / "refined_xphase_failure_region_metrics.csv"),
            "report": str(out_dir / "refined_xphase_failure_diagnosis_report.md"),
            "split_plot": str(out_dir / "split_rmse_diagnostic.png"),
            "test_region_plot": str(out_dir / "test_region_excess_mse_refined_vs_x.png"),
        },
    }
    write_per_sample(records_by_split, pixel_gate, out_dir)
    write_region_csv(region_summary, out_dir)
    make_plots(summary, region_summary, out_dir)
    (out_dir / "refined_xphase_failure_diagnosis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "refined_xphase_failure_region_summary.json").write_text(json.dumps(region_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "refined_xphase_failure_diagnosis_report.md").write_text(compact_summary(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
