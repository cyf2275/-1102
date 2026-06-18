"""Evaluate direct fusion between a cached base-depth branch and a phase-depth branch."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from tqdm import tqdm

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from data.dataset_fpp_phase import create_fpp_phase_loaders
from eval_hierarchical_phase_fusion import (
    aux_predict01,
    build_aux_depth_model,
    masked_mean,
    parse_float_list,
)
from train_fpp_official_style_unet import METRIC_KEYS, summarize
from train_pip_lite import prediction_to_mm
from utils.metrics import compute_metrics


def save_rows(rows, path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def metric_row(pred_mm, target, mask):
    return compute_metrics(pred_mm, target, mask=mask)


@torch.no_grad()
def evaluate_split(args, split, aux_model, aux_args, aux_kind, aux_mode, loaders_depth, loaders_phase, device):
    weights = parse_float_list(args.phase_weights)
    branch_rows = {name: [] for name in ("base", "phase_branch")}
    fused_rows = {str(w): [] for w in weights}
    detail_rows = []
    gate_rows = []

    for depth_batch, phase_batch in tqdm(
        zip(loaders_depth[split], loaders_phase[split]),
        total=len(loaders_depth[split]),
        desc=f"base-phase fusion {split}",
    ):
        if not torch.equal(depth_batch["sample_index"], phase_batch["sample_index"]):
            raise RuntimeError("depth and phase loaders are not aligned")

        base = torch.clamp(depth_batch["base_height"].to(device, non_blocking=True), -1.0, 1.0)
        phase_branch = aux_predict01(aux_model, phase_batch, device, aux_args, aux_kind, aux_mode) * 2.0 - 1.0
        phase_branch = torch.clamp(phase_branch, -1.0, 1.0)
        target = depth_batch["height_raw"].to(device, non_blocking=True)
        mask = torch.clamp(depth_batch["mask"].to(device, non_blocking=True), 0.0, 1.0)
        edge = torch.clamp(depth_batch["edge_score"].to(device, non_blocking=True), 0.0, 1.0)
        conf = torch.clamp(depth_batch["phase_conf"].to(device, non_blocking=True), 0.0, 1.0)
        delta = torch.abs(phase_branch - base)

        pred_mm = {
            "base": prediction_to_mm(base, depth_batch, loaders_depth["height_scale"]),
            "phase_branch": prediction_to_mm(phase_branch, depth_batch, loaders_depth["height_scale"]),
        }
        for name, pred in pred_mm.items():
            for j in range(pred.shape[0]):
                branch_rows[name].append(metric_row(pred[j:j + 1], target[j:j + 1], mask[j:j + 1]))

        for w in weights:
            fused = torch.clamp((1.0 - w) * base + w * phase_branch, -1.0, 1.0)
            fused_mm = prediction_to_mm(fused, depth_batch, loaders_depth["height_scale"])
            for j in range(fused_mm.shape[0]):
                metrics = metric_row(fused_mm[j:j + 1], target[j:j + 1], mask[j:j + 1])
                fused_rows[str(w)].append(metrics)
                detail_rows.append({
                    "sample": len(gate_rows) + j,
                    "phase_weight": w,
                    **metrics,
                })

        edge_mean = masked_mean(edge, mask)
        delta_mean = masked_mean(delta, mask)
        conf_mean = masked_mean(conf, mask)
        for j in range(base.shape[0]):
            base_metrics = compute_metrics(pred_mm["base"][j:j + 1], target[j:j + 1], mask=mask[j:j + 1])
            phase_metrics = compute_metrics(pred_mm["phase_branch"][j:j + 1], target[j:j + 1], mask=mask[j:j + 1])
            row = {
                "sample": len(gate_rows),
                "branch": "base_phase",
                "pixel_selected_frac": 0.0,
                "edge_mean": float(edge_mean[j].detach().cpu()),
                "delta_mean": float(delta_mean[j].detach().cpu()),
                "phase_conf_mean": float(conf_mean[j].detach().cpu()),
            }
            for key in METRIC_KEYS:
                row[f"base_{key}"] = base_metrics[key]
                row[f"phase_branch_{key}"] = phase_metrics[key]
                row[f"diff_{key}"] = phase_metrics[key]
                row[f"pixel_gated_{key}"] = base_metrics[key]
                row[f"hierarchical_{key}"] = base_metrics[key]
            gate_rows.append(row)

    out = {
        "branches": {name: summarize(vals) for name, vals in branch_rows.items()},
        "weights": {},
    }
    for key, vals in fused_rows.items():
        item = summarize(vals)
        item["n"] = len(vals)
        out["weights"][key] = item
    return gate_rows, detail_rows, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase_depth_checkpoint", required=True)
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_pspquad_cache_960")
    parser.add_argument("--base_prefix", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--phase_weights", default="0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5")
    parser.add_argument("--splits", default="val test")
    parser.add_argument("--require_cache", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    aux_model, aux_args, aux_kind, aux_mode = build_aux_depth_model(args, device)
    phase_pred_prefix = getattr(aux_args, "phase_pred_prefix", None)

    loaders_depth = create_fpp_ml_bench_loaders(
        cache_dir=args.cache_dir,
        batch_size=1,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        include_ftp=False,
        image_h=args.image_size,
        image_w=args.image_size,
        require_cache=args.require_cache,
        base_prefix=args.base_prefix,
    )
    loaders_phase = create_fpp_phase_loaders(
        base_cache_dir=args.cache_dir,
        phase_cache_dir=args.phase_cache_dir,
        batch_size=1,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        image_h=args.image_size,
        image_w=args.image_size,
        phase_pred_prefix=phase_pred_prefix,
        require_cache=args.require_cache,
        preload_ram=bool(getattr(aux_args, "preload_ram", False)),
        train_minimal=bool(getattr(aux_args, "train_minimal", False)),
    )

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "direct base-depth plus phase-depth fusion",
        "base_prefix": args.base_prefix,
        "phase_depth_checkpoint": args.phase_depth_checkpoint,
        "aux_kind": aux_kind,
        "aux_mode": aux_mode,
        "phase_pred_prefix": phase_pred_prefix,
        "phase_weights": parse_float_list(args.phase_weights),
    }

    eval_splits = [split for split in str(args.splits).replace(",", " ").split() if split]
    for split in eval_splits:
        loader_split = "train_eval" if split == "train" else split
        rows, fused_rows, summary = evaluate_split(
            args, loader_split, aux_model, aux_args, aux_kind, aux_mode,
            loaders_depth, loaders_phase, device,
        )
        save_rows(
            rows,
            out_dir / f"{split}_hier_phase_rows.csv",
            [
                "sample", "branch", "pixel_selected_frac", "edge_mean", "delta_mean", "phase_conf_mean",
                *[f"{prefix}_{key}" for prefix in ("base", "diff", "pixel_gated", "hierarchical", "phase_branch")
                  for key in METRIC_KEYS],
            ],
        )
        save_rows(fused_rows, out_dir / f"{split}_fused_weight_rows.csv", ["sample", "phase_weight", *METRIC_KEYS])
        result[split] = summary

    if "val" in result and "test" in result:
        best_weight = min(
            result["val"]["weights"],
            key=lambda w: result["val"]["weights"][w]["rmse"]["mean"],
        )
        result["selected_by_val"] = {
            "phase_weight": float(best_weight),
            "val_rmse": result["val"]["weights"][best_weight]["rmse"]["mean"],
            "test_rmse": result["test"]["weights"][best_weight]["rmse"]["mean"],
            "test_edge_rmse": result["test"]["weights"][best_weight]["edge_rmse"]["mean"],
            "test_normal_deg": result["test"]["weights"][best_weight]["normal_deg"]["mean"],
        }

    with (out_dir / "base_phase_fusion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result.get("selected_by_val", {"splits": eval_splits}), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
