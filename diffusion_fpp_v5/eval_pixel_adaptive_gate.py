from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import torch
from tqdm import tqdm

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from diffusion_pip import PIPDiffusion
from eval_adaptive_blend_features import _saved_arg, build_model
from train_fpp_official_style_unet import METRIC_KEYS, parse_channel_spec, summarize
from train_pip_lite import prediction_to_mm
from utils.metrics import compute_metrics


def parse_float_list(text):
    return [float(x) for x in str(text).replace(",", " ").split() if x]


def save_rows(rows, path):
    keys = ["sample", "selected_frac"]
    for prefix in ("base", "diff", "pixel_gated"):
        keys.extend(f"{prefix}_{key}" for key in METRIC_KEYS)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in keys})


def make_gate(base, diff, edge, conf, cfg, mask=None):
    delta = torch.abs(diff - base)
    gate = torch.ones_like(base, dtype=torch.bool)
    if cfg.get("sample_edge_th", 1.0) < 1.0:
        if mask is None:
            edge_mean = edge.flatten(1).mean(dim=1).view(-1, 1, 1, 1)
        else:
            mask_f = torch.clamp(mask.to(device=edge.device, dtype=edge.dtype), 0.0, 1.0)
            edge_mean = ((edge * mask_f).flatten(1).sum(dim=1) /
                         mask_f.flatten(1).sum(dim=1).clamp(min=1.0)).view(-1, 1, 1, 1)
        gate = gate & (edge_mean <= cfg["sample_edge_th"])
    if cfg["edge_th"] < 1.0:
        gate = gate & (edge <= cfg["edge_th"])
    if cfg["delta_min"] > 0:
        gate = gate & (delta >= cfg["delta_min"])
    if cfg["conf_min"] > 0:
        gate = gate & (conf >= cfg["conf_min"])
    return gate.to(dtype=base.dtype)


@torch.no_grad()
def search_val_gate(diffusion, loader, device, height_scale, args, configs):
    accum = [
        {"rmse_sum": 0.0, "n": 0, "selected": 0.0, "valid": 0.0}
        for _ in configs
    ]
    base_rmse_sum = 0.0
    base_n = 0
    for batch in tqdm(loader, desc="pixel gate val search"):
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
        base_mm = prediction_to_mm(base, batch, height_scale)
        for j in range(base.shape[0]):
            valid = float(mask[j:j + 1].sum().detach().cpu())
            sse = float(((((base_mm[j:j + 1] - target[j:j + 1]) ** 2) * mask[j:j + 1]).sum()).detach().cpu())
            base_rmse_sum += (sse / max(valid, 1.0)) ** 0.5
            base_n += 1
        for i, cfg in enumerate(configs):
            gate = make_gate(base, diff, edge, conf, cfg, mask=mask)
            pred = torch.clamp(base + cfg["alpha"] * gate * (diff - base), -1.0, 1.0)
            pred_mm = prediction_to_mm(pred, batch, height_scale)
            for j in range(base.shape[0]):
                single_mask = mask[j:j + 1]
                valid = float(single_mask.sum().detach().cpu())
                sse = float(((((pred_mm[j:j + 1] - target[j:j + 1]) ** 2) * single_mask).sum()).detach().cpu())
                accum[i]["rmse_sum"] += (sse / max(valid, 1.0)) ** 0.5
                accum[i]["n"] += 1
                accum[i]["selected"] += float((gate[j:j + 1] * single_mask).sum().detach().cpu())
                accum[i]["valid"] += valid
    base_rmse = base_rmse_sum / max(base_n, 1)
    candidates = []
    for cfg, acc in zip(configs, accum):
        rmse = acc["rmse_sum"] / max(acc["n"], 1)
        selected_frac = acc["selected"] / max(acc["valid"], 1.0)
        candidates.append({**cfg, "val_rmse": rmse, "selected_frac": selected_frac})
    best = min(candidates, key=lambda x: x["val_rmse"])
    return base_rmse, best, candidates


@torch.no_grad()
def evaluate_split(diffusion, loader, device, height_scale, args, cfg, split):
    rows = []
    for batch in tqdm(loader, desc=f"pixel gate {split}"):
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
        gate = make_gate(base, diff, edge, conf, cfg, mask=mask)
        pred = torch.clamp(base + cfg["alpha"] * gate * (diff - base), -1.0, 1.0)
        base_mm = prediction_to_mm(base, batch, height_scale)
        diff_mm = prediction_to_mm(diff, batch, height_scale)
        pred_mm = prediction_to_mm(pred, batch, height_scale)
        for j in range(base.shape[0]):
            single_mask = mask[j:j + 1]
            row = {
                "sample": len(rows),
                "selected_frac": float(((gate[j:j + 1] * single_mask).sum() / single_mask.sum().clamp(min=1.0)).detach().cpu()),
            }
            for prefix, pred_one in (("base", base_mm), ("diff", diff_mm), ("pixel_gated", pred_mm)):
                metrics = compute_metrics(pred_one[j:j + 1], target[j:j + 1], mask=single_mask)
                row.update({f"{prefix}_{key}": metrics[key] for key in METRIC_KEYS})
            rows.append(row)
    summaries = {}
    for prefix in ("base", "diff", "pixel_gated"):
        prefixed = [{key: row[f"{prefix}_{key}"] for key in METRIC_KEYS} for row in rows]
        summaries[prefix] = summarize(prefixed)
    summaries["selected_frac_mean"] = float(sum(row["selected_frac"] for row in rows) / max(len(rows), 1))
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
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--start_ratio", type=float, default=0.05)
    parser.add_argument("--alphas", default="0.25 0.35 0.50")
    parser.add_argument("--edge_thresholds", default="0.25 0.35 0.4674050956964493 0.60 0.80 1.00")
    parser.add_argument("--sample_edge_thresholds", default="1.0")
    parser.add_argument("--delta_mins", default="0.0 0.02 0.05 0.08 0.12 0.16")
    parser.add_argument("--conf_mins", default="0.0 0.2 0.4")
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
    configs = [
        {
            "alpha": alpha,
            "sample_edge_th": sample_edge_th,
            "edge_th": edge_th,
            "delta_min": delta_min,
            "conf_min": conf_min,
        }
        for alpha, sample_edge_th, edge_th, delta_min, conf_min in itertools.product(
            parse_float_list(args.alphas),
            parse_float_list(args.sample_edge_thresholds),
            parse_float_list(args.edge_thresholds),
            parse_float_list(args.delta_mins),
            parse_float_list(args.conf_mins),
        )
    ]
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_base_rmse, best, candidates = search_val_gate(
        diffusion, loaders["val"], device, loaders["height_scale"], args, configs
    )
    val_rows, val_summary = evaluate_split(
        diffusion, loaders["val"], device, loaders["height_scale"], args, best, "val"
    )
    test_rows, test_summary = evaluate_split(
        diffusion, loaders["test"], device, loaders["height_scale"], args, best, "test"
    )
    save_rows(val_rows, out_dir / "val_pixel_gate_metrics.csv")
    save_rows(test_rows, out_dir / "test_pixel_gate_metrics.csv")
    result = {
        "checkpoint": args.checkpoint,
        "best_gate": best,
        "val_search_base_rmse": val_base_rmse,
        "val": val_summary,
        "test": test_summary,
        "candidate_count": len(candidates),
        "top_val_candidates": sorted(candidates, key=lambda x: x["val_rmse"])[:20],
    }
    with open(out_dir / "pixel_gate_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "best_gate": best,
        "val_base": val_summary["base"]["rmse"]["mean"],
        "val_pixel_gated": val_summary["pixel_gated"]["rmse"]["mean"],
        "test_base": test_summary["base"]["rmse"]["mean"],
        "test_pixel_gated": test_summary["pixel_gated"]["rmse"]["mean"],
        "test_selected_frac": test_summary["selected_frac_mean"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
