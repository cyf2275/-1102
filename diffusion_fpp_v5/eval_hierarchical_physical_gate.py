from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from tqdm import tqdm

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from diffusion_pip import PIPDiffusion
from eval_adaptive_blend_features import _saved_arg, build_model
from eval_pixel_adaptive_gate import make_gate
from train_fpp_official_style_unet import METRIC_KEYS, parse_channel_spec, summarize
from train_pip_lite import prediction_to_mm
from utils.metrics import compute_metrics


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = torch.clamp(mask.to(device=x.device, dtype=x.dtype), 0.0, 1.0)
    return (x * mask).flatten(1).sum(dim=1) / mask.flatten(1).sum(dim=1).clamp(min=1.0)


def save_rows(rows, path):
    keys = [
        "sample",
        "branch",
        "pixel_selected_frac",
        "edge_mean",
        "delta_mean",
        "phase_conf_mean",
    ]
    for prefix in ("base", "diff", "pixel_gated", "hierarchical"):
        keys.extend(f"{prefix}_{key}" for key in METRIC_KEYS)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in keys})


@torch.no_grad()
def evaluate_split(diffusion, loader, device, height_scale, args, split):
    rows = []
    pixel_cfg = {
        "alpha": args.pixel_alpha,
        "sample_edge_th": args.pixel_sample_edge_th,
        "edge_th": args.pixel_edge_th,
        "delta_min": args.pixel_delta_min,
        "conf_min": args.pixel_conf_min,
    }
    for batch in tqdm(loader, desc=f"hierarchical gate {split}"):
        base = torch.clamp(batch["base_height"].to(device, non_blocking=True), -1.0, 1.0)
        diff = diffusion.sample_ddim(
            batch,
            steps=args.ddim_steps,
            ensemble_size=args.ensemble,
            start_from_base=True,
            start_ratio=args.start_ratio,
        )
        target = batch["height_raw"].to(device, non_blocking=True)
        mask = torch.clamp(batch["mask"].to(device, non_blocking=True), 0.0, 1.0)
        edge = torch.clamp(batch["edge_score"].to(device, non_blocking=True), 0.0, 1.0)
        conf = torch.clamp(batch["phase_conf"].to(device, non_blocking=True), 0.0, 1.0)
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
        branch = []
        for j in range(base.shape[0]):
            if bool(high_override[j].item()):
                hierarchical[j:j + 1] = diff[j:j + 1]
                branch.append("diff_high_override")
            elif bool(low_edge_sample[j].item()):
                hierarchical[j:j + 1] = pixel_pred[j:j + 1]
                branch.append("pixel_low_edge")
            else:
                branch.append("base")

        base_mm = prediction_to_mm(base, batch, height_scale)
        diff_mm = prediction_to_mm(diff, batch, height_scale)
        pixel_mm = prediction_to_mm(pixel_pred, batch, height_scale)
        hierarchical_mm = prediction_to_mm(hierarchical, batch, height_scale)

        for j in range(base.shape[0]):
            single_mask = mask[j:j + 1]
            row = {
                "sample": len(rows),
                "branch": branch[j],
                "pixel_selected_frac": float(
                    ((pixel_gate[j:j + 1] * single_mask).sum() / single_mask.sum().clamp(min=1.0)).detach().cpu()
                ),
                "edge_mean": float(edge_mean[j].detach().cpu()),
                "delta_mean": float(delta_mean[j].detach().cpu()),
                "phase_conf_mean": float(conf_mean[j].detach().cpu()),
            }
            for prefix, pred_one in (
                ("base", base_mm),
                ("diff", diff_mm),
                ("pixel_gated", pixel_mm),
                ("hierarchical", hierarchical_mm),
            ):
                metrics = compute_metrics(pred_one[j:j + 1], target[j:j + 1], mask=single_mask)
                row.update({f"{prefix}_{key}": metrics[key] for key in METRIC_KEYS})
            rows.append(row)

    summaries = {}
    for prefix in ("base", "diff", "pixel_gated", "hierarchical"):
        prefixed = [{key: row[f"{prefix}_{key}"] for key in METRIC_KEYS} for row in rows]
        summaries[prefix] = summarize(prefixed)
    counts = {}
    for row in rows:
        counts[row["branch"]] = counts.get(row["branch"], 0) + 1
    summaries["branch_counts"] = counts
    summaries["pixel_selected_frac_mean"] = float(
        sum(row["pixel_selected_frac"] for row in rows) / max(len(rows), 1)
    )
    return rows, summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--base_prefix", default="base_c4_adapter")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--image_h", type=int, default=960)
    parser.add_argument("--image_w", type=int, default=960)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
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
    parser.add_argument("--require_cache", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})
    include_ftp = bool(_saved_arg(saved_args, "include_ftp", False))
    loaders = create_fpp_ml_bench_loaders(
        cache_dir=args.cache_dir,
        batch_size=1,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        include_ftp=include_ftp,
        image_h=args.image_h,
        image_w=args.image_w,
        require_cache=args.require_cache,
        base_prefix=args.base_prefix,
        phase_cache_dir=str(_saved_arg(saved_args, "phase_cache_dir", "/root/autodl-tmp/fpp_ml_phase_cache_960")),
        phase_pred_prefix=_saved_arg(saved_args, "phase_pred_prefix", None),
        append_phase_pred_to_cond=bool(_saved_arg(saved_args, "append_phase_pred_to_cond", False)),
    )
    physics_indices = _saved_arg(saved_args, "physics_channel_indices", None)
    if physics_indices is None:
        physics_indices = parse_channel_spec(str(_saved_arg(saved_args, "physics_channels", "")), include_ftp)
    model_cond_channels = int(ckpt.get(
        "model_cond_channels",
        len(physics_indices) if physics_indices is not None else loaders["cond_channels"],
    ))
    model = build_model(saved_args, model_cond_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    diffusion = PIPDiffusion(
        model,
        timesteps=int(_saved_arg(saved_args, "timesteps", 200)),
        image_h=args.image_h,
        image_w=args.image_w,
        device=device,
        cond_indices=physics_indices,
        target_mode=str(_saved_arg(saved_args, "target_mode", "base_residual")),
        residual_scale=float(_saved_arg(saved_args, "resolved_residual_scale", 1.0)),
        base_residual_gate=float(_saved_arg(saved_args, "base_residual_gate", 1.0)),
    )

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": args.checkpoint,
        "rules": {
            "pixel": {
                "alpha": args.pixel_alpha,
                "sample_edge_th": args.pixel_sample_edge_th,
                "edge_th": args.pixel_edge_th,
                "delta_min": args.pixel_delta_min,
                "conf_min": args.pixel_conf_min,
            },
            "high_override": {
                "edge": [args.high_edge_min, args.high_edge_max],
                "delta": [args.high_delta_min, args.high_delta_max],
                "phase_conf": [args.high_conf_min, args.high_conf_max],
                "branch": "raw_diffusion",
            },
        },
    }
    for split in ("val", "test"):
        rows, summary = evaluate_split(diffusion, loaders[split], device, loaders["height_scale"], args, split)
        save_rows(rows, out_dir / f"{split}_hierarchical_gate_metrics.csv")
        result[split] = summary
    with open(out_dir / "hierarchical_gate_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "val_base": result["val"]["base"]["rmse"]["mean"],
        "val_pixel": result["val"]["pixel_gated"]["rmse"]["mean"],
        "val_hierarchical": result["val"]["hierarchical"]["rmse"]["mean"],
        "test_base": result["test"]["base"]["rmse"]["mean"],
        "test_pixel": result["test"]["pixel_gated"]["rmse"]["mean"],
        "test_hierarchical": result["test"]["hierarchical"]["rmse"]["mean"],
        "test_branch_counts": result["test"]["branch_counts"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
