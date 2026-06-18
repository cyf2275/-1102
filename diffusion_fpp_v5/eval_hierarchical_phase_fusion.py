from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from tqdm import tqdm

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from data.dataset_fpp_phase import create_fpp_phase_loaders
from diffusion_pip import PIPDiffusion
from eval_adaptive_blend_features import _saved_arg, build_model
from eval_hierarchical_physical_gate import masked_mean
from eval_pixel_adaptive_gate import make_gate
from models import OfficialUNetFPPAdapter
from models.official_unet import OfficialUNetFPP
from train_fpp_official_style_unet import METRIC_KEYS, parse_channel_spec, summarize
from train_fpp_phase2depth_unet import input_channels, make_input
from train_fpp_psp_adapter_unet import cond_channel_count, make_cond, parse_indices
from train_pip_lite import prediction_to_mm
from utils.metrics import compute_metrics


def parse_float_list(text: str):
    return [float(x) for x in str(text).replace(",", " ").split() if x]


def metric_row(pred_mm, target, mask):
    return compute_metrics(pred_mm, target, mask=mask)


def save_rows(rows, path):
    keys = [
        "sample",
        "branch",
        "pixel_selected_frac",
        "edge_mean",
        "delta_mean",
        "phase_conf_mean",
    ]
    for prefix in ("base", "diff", "pixel_gated", "hierarchical", "phase_branch"):
        keys.extend(f"{prefix}_{key}" for key in METRIC_KEYS)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in keys})


def save_fused_rows(rows, path):
    keys = ["sample", "phase_weight"] + METRIC_KEYS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in keys})


def build_depth_diffusion(args, device):
    ckpt = torch.load(args.depth_checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})
    include_ftp = bool(_saved_arg(saved_args, "include_ftp", False))
    physics_indices = _saved_arg(saved_args, "physics_channel_indices", None)
    if physics_indices is None:
        physics_indices = parse_channel_spec(str(_saved_arg(saved_args, "physics_channels", "")), include_ftp)
    model_cond_channels = int(ckpt.get("model_cond_channels", len(physics_indices)))
    model = build_model(saved_args, model_cond_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    diffusion = PIPDiffusion(
        model,
        timesteps=int(_saved_arg(saved_args, "timesteps", 200)),
        image_h=args.image_size,
        image_w=args.image_size,
        device=device,
        cond_indices=physics_indices,
        target_mode=str(_saved_arg(saved_args, "target_mode", "base_residual")),
        residual_scale=float(_saved_arg(saved_args, "resolved_residual_scale", 1.0)),
        base_residual_gate=float(_saved_arg(saved_args, "base_residual_gate", 1.0)),
    )
    return diffusion, include_ftp


def build_aux_depth_model(args, device):
    ckpt = torch.load(args.phase_depth_checkpoint, map_location=device)
    saved = dict(ckpt.get("args", {}))
    saved.setdefault("phase_cache_dir", args.phase_cache_dir)
    saved.setdefault("phase_pred_prefix", None)
    saved.setdefault("batch_size", args.eval_batch_size)
    saved.setdefault("eval_batch_size", args.eval_batch_size)
    saved.setdefault("image_size", args.image_size)
    saved.setdefault("preload_ram", False)
    saved.setdefault("train_minimal", False)
    saved.setdefault("instr_channels", "1-6")
    saved["instr_channel_indices"] = saved.get("instr_channel_indices") or parse_indices(saved["instr_channels"])
    run_args = SimpleNamespace(**saved)

    input_mode = str(saved.get("input_mode", ""))
    if input_mode in {"gt_phase", "phase_pred", "gt_phase_plus_fringe", "phase_pred_plus_fringe"}:
        model = OfficialUNetFPP(
            in_channels=input_channels(input_mode),
            out_channels=1,
            dropout_rate=float(saved.get("dropout", 0.0)),
        ).to(device)
        model_kind = "phase2depth_unet"
        call_mode = input_mode
    else:
        cond_mode = str(saved.get("cond_mode", "phase_pred_xy"))
        run_args.cond_mode = cond_mode
        model = OfficialUNetFPPAdapter(
            cond_channels=cond_channel_count(cond_mode, run_args.instr_channel_indices),
            out_channels=1,
            dropout_rate=float(saved.get("dropout", 0.0)),
            adapter_hidden=int(saved.get("adapter_hidden", 32)),
        ).to(device)
        model_kind = "psp_adapter"
        call_mode = cond_mode

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, run_args, model_kind, call_mode


def aux_predict01(model, batch, device, run_args, model_kind, call_mode):
    if model_kind == "phase2depth_unet":
        return torch.clamp(model(make_input(batch, device, call_mode)), 0.0, 1.0)
    fringe = batch["fringe"].to(device, non_blocking=True)
    cond = make_cond(batch, device, run_args.cond_mode, run_args.instr_channel_indices)
    return torch.clamp(model(fringe, cond), 0.0, 1.0)


@torch.no_grad()
def evaluate_split(args, split, depth_diffusion, aux_model, aux_args, aux_kind, aux_mode,
                   loaders_depth, loaders_phase, device):
    weights = parse_float_list(args.phase_weights)
    rows = []
    blend_rows = {str(w): [] for w in weights}
    fused_detail_rows = []
    branch_rows = {name: [] for name in ("base", "diff", "pixel_gated", "hierarchical", "phase_branch")}
    branch_counts = {}

    pixel_cfg = {
        "alpha": args.pixel_alpha,
        "sample_edge_th": args.pixel_sample_edge_th,
        "edge_th": args.pixel_edge_th,
        "delta_min": args.pixel_delta_min,
        "conf_min": args.pixel_conf_min,
    }

    for depth_batch, phase_batch in tqdm(
        zip(loaders_depth[split], loaders_phase[split]),
        total=len(loaders_depth[split]),
        desc=f"hier-phase fusion {split}",
    ):
        if not torch.equal(depth_batch["sample_index"], phase_batch["sample_index"]):
            raise RuntimeError("depth and phase loaders are not aligned")

        base = torch.clamp(depth_batch["base_height"].to(device, non_blocking=True), -1.0, 1.0)
        diff = depth_diffusion.sample_ddim(
            depth_batch,
            steps=args.ddim_steps,
            ensemble_size=args.ensemble,
            start_from_base=True,
            start_ratio=args.start_ratio,
        )
        target = depth_batch["height_raw"].to(device, non_blocking=True)
        mask = torch.clamp(depth_batch["mask"].to(device, non_blocking=True), 0.0, 1.0)
        edge = torch.clamp(depth_batch["edge_score"].to(device, non_blocking=True), 0.0, 1.0)
        conf = torch.clamp(depth_batch["phase_conf"].to(device, non_blocking=True), 0.0, 1.0)
        delta = torch.abs(diff - base)

        pixel_gate = make_gate(base, diff, edge, conf, pixel_cfg, mask=mask)
        pixel_pred = torch.clamp(base + args.pixel_alpha * pixel_gate * (diff - base), -1.0, 1.0)

        edge_mean = masked_mean(edge, mask)
        delta_mean = masked_mean(delta, mask)
        conf_mean = masked_mean(conf, mask)
        high_override = (
            (edge_mean >= args.high_edge_min)
            & (edge_mean <= args.high_edge_max)
            & (delta_mean >= args.high_delta_min)
            & (delta_mean <= args.high_delta_max)
            & (conf_mean >= args.high_conf_min)
            & (conf_mean <= args.high_conf_max)
        )
        low_edge_sample = edge_mean <= args.pixel_sample_edge_th

        hierarchical = base.clone()
        sample_branch = []
        for j in range(base.shape[0]):
            if bool(high_override[j].item()):
                hierarchical[j:j + 1] = diff[j:j + 1]
                sample_branch.append("diff_high_override")
            elif bool(low_edge_sample[j].item()):
                hierarchical[j:j + 1] = pixel_pred[j:j + 1]
                sample_branch.append("pixel_low_edge")
            else:
                sample_branch.append("base")
            branch_counts[sample_branch[-1]] = branch_counts.get(sample_branch[-1], 0) + 1

        phase_branch = aux_predict01(aux_model, phase_batch, device, aux_args, aux_kind, aux_mode) * 2.0 - 1.0
        phase_branch = torch.clamp(phase_branch, -1.0, 1.0)

        pred_map = {
            "base": base,
            "diff": diff,
            "pixel_gated": pixel_pred,
            "hierarchical": hierarchical,
            "phase_branch": phase_branch,
        }
        pred_mm = {
            name: prediction_to_mm(pred, depth_batch, loaders_depth["height_scale"])
            for name, pred in pred_map.items()
        }
        for name, pred in pred_mm.items():
            for j in range(pred.shape[0]):
                branch_rows[name].append(metric_row(pred[j:j + 1], target[j:j + 1], mask[j:j + 1]))

        for w in weights:
            fused = torch.clamp((1.0 - w) * hierarchical + w * phase_branch, -1.0, 1.0)
            fused_mm = prediction_to_mm(fused, depth_batch, loaders_depth["height_scale"])
            for j in range(fused_mm.shape[0]):
                metrics = metric_row(fused_mm[j:j + 1], target[j:j + 1], mask[j:j + 1])
                blend_rows[str(w)].append(metrics)
                fused_detail_rows.append({
                    "sample": len(rows) + j,
                    "phase_weight": w,
                    **metrics,
                })

        for j in range(base.shape[0]):
            row = {
                "sample": len(rows),
                "branch": sample_branch[j],
                "pixel_selected_frac": float(
                    ((pixel_gate[j:j + 1] * mask[j:j + 1]).sum() / mask[j:j + 1].sum().clamp(min=1.0)).detach().cpu()
                ),
                "edge_mean": float(edge_mean[j].detach().cpu()),
                "delta_mean": float(delta_mean[j].detach().cpu()),
                "phase_conf_mean": float(conf_mean[j].detach().cpu()),
            }
            for prefix in pred_map:
                metrics = compute_metrics(pred_mm[prefix][j:j + 1], target[j:j + 1], mask=mask[j:j + 1])
                row.update({f"{prefix}_{key}": metrics[key] for key in METRIC_KEYS})
            rows.append(row)

    out = {
        "branches": {name: summarize(vals) for name, vals in branch_rows.items()},
        "branch_counts": branch_counts,
        "weights": {},
    }
    for key, vals in blend_rows.items():
        item = summarize(vals)
        item["n"] = len(vals)
        out["weights"][key] = item
    return rows, fused_detail_rows, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_checkpoint", required=True)
    parser.add_argument("--phase_depth_checkpoint", required=True)
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_pspquad_cache_960")
    parser.add_argument("--base_prefix", default="base_c4_adapter")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--start_ratio", type=float, default=0.05)
    parser.add_argument("--pixel_alpha", type=float, default=0.7)
    parser.add_argument("--pixel_sample_edge_th", type=float, default=0.47)
    parser.add_argument("--pixel_edge_th", type=float, default=1.0)
    parser.add_argument("--pixel_delta_min", type=float, default=0.12)
    parser.add_argument("--pixel_conf_min", type=float, default=0.0)
    parser.add_argument("--high_edge_min", type=float, default=0.58)
    parser.add_argument("--high_edge_max", type=float, default=0.62)
    parser.add_argument("--high_delta_min", type=float, default=0.09)
    parser.add_argument("--high_delta_max", type=float, default=0.105)
    parser.add_argument("--high_conf_min", type=float, default=0.76)
    parser.add_argument("--high_conf_max", type=float, default=0.80)
    parser.add_argument("--phase_weights", default="0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5")
    parser.add_argument("--splits", default="val test")
    parser.add_argument("--require_cache", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    depth_diffusion, include_ftp = build_depth_diffusion(args, device)
    aux_model, aux_args, aux_kind, aux_mode = build_aux_depth_model(args, device)
    phase_pred_prefix = getattr(aux_args, "phase_pred_prefix", None)

    loaders_depth = create_fpp_ml_bench_loaders(
        cache_dir=args.cache_dir,
        batch_size=1,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        include_ftp=include_ftp,
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
        "depth_checkpoint": args.depth_checkpoint,
        "phase_depth_checkpoint": args.phase_depth_checkpoint,
        "aux_kind": aux_kind,
        "aux_mode": aux_mode,
        "phase_pred_prefix": phase_pred_prefix,
        "phase_weights": parse_float_list(args.phase_weights),
    }
    eval_splits = [split for split in str(args.splits).replace(",", " ").split() if split]
    if not eval_splits:
        raise ValueError("--splits must contain at least one split")

    for split in eval_splits:
        loader_split = "train_eval" if split == "train" else split
        rows, fused_rows, summary = evaluate_split(
            args, loader_split, depth_diffusion, aux_model, aux_args, aux_kind, aux_mode,
            loaders_depth, loaders_phase, device,
        )
        save_rows(rows, out_dir / f"{split}_hier_phase_rows.csv")
        save_fused_rows(fused_rows, out_dir / f"{split}_fused_weight_rows.csv")
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
    with open(out_dir / "hierarchical_phase_fusion_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result.get("selected_by_val", {"splits": eval_splits}), ensure_ascii=False), flush=True)
    # Persistent DataLoader workers can keep this short evaluation process alive
    # after all result files are written. Exit explicitly so chained remote runs
    # can continue to the selection/report steps.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
