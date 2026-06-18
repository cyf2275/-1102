from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from diffusion_pip import PIPDiffusion
from eval_adaptive_blend_features import _saved_arg, build_model
from train_fpp_official_style_unet import METRIC_KEYS, parse_channel_spec, summarize
from train_pip_lite import prediction_to_mm
from utils.metrics import compute_metrics


def parse_floats(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_ints(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def box_blur(x, kernel):
    kernel = int(kernel)
    if kernel <= 1:
        return x
    pad = kernel // 2
    return F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=pad, count_include_pad=False)


def masked_rmse_mm(pred_norm, target_mm, batch, mask):
    pred_mm = prediction_to_mm(pred_norm, batch, 1.0)
    sq = (pred_mm - target_mm) ** 2
    return float(torch.sqrt((sq * mask).sum() / mask.sum().clamp(min=1.0)).detach().cpu())


def make_corrected(base, diff, edge, conf, alpha, kernel, edge_power, conf_power):
    residual = box_blur(diff - base, kernel)
    gate = torch.ones_like(residual)
    if edge_power > 0:
        gate = gate * torch.pow(1.0 - torch.clamp(edge, 0.0, 1.0), float(edge_power))
    if conf_power > 0:
        gate = gate * torch.pow(torch.clamp(conf, 0.0, 1.0), float(conf_power))
    return torch.clamp(base + float(alpha) * gate * residual, -1.0, 1.0)


def direct_summary(rows, prefix):
    out = {"n": len(rows)}
    for metric in METRIC_KEYS:
        vals = np.asarray([float(r[f"{prefix}_{metric}"]) for r in rows], dtype=np.float64)
        out[metric] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
        }
    return out


def save_rows(rows, path, prefix):
    keys = ["sample"] + [f"{prefix}_{key}" for key in METRIC_KEYS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in keys})


@torch.no_grad()
def eval_split(diffusion, loader, device, args, configs):
    score = {cfg: [] for cfg in configs}
    base_rows = []
    diff_rows = []
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"lowpass posterior {args.split}")):
        base = torch.clamp(batch["base_height"].to(device, non_blocking=True), -1.0, 1.0)
        diff = diffusion.sample_ddim(
            batch,
            steps=args.ddim_steps,
            ensemble_size=args.ensemble,
            start_from_base=True,
            start_ratio=args.start_ratio,
        )
        target_mm = batch["height_raw"].to(device, non_blocking=True)
        mask = torch.clamp(batch["mask"].to(device, non_blocking=True), 0.0, 1.0)
        edge = torch.clamp(batch["edge_score"].to(device, non_blocking=True), 0.0, 1.0)
        conf = torch.clamp(batch["phase_conf"].to(device, non_blocking=True), 0.0, 1.0)
        base_mm = prediction_to_mm(base, batch, 1.0)
        diff_mm = prediction_to_mm(diff, batch, 1.0)
        for j in range(base.shape[0]):
            sample = len(base_rows)
            single_mask = mask[j:j + 1]
            base_metrics = compute_metrics(base_mm[j:j + 1], target_mm[j:j + 1], mask=single_mask)
            diff_metrics = compute_metrics(diff_mm[j:j + 1], target_mm[j:j + 1], mask=single_mask)
            base_rows.append({"sample": sample, **{f"base_{k}": v for k, v in base_metrics.items()}})
            diff_rows.append({"sample": sample, **{f"diff_{k}": v for k, v in diff_metrics.items()}})

        blurred = {k: box_blur(diff - base, k) for k in sorted({cfg[1] for cfg in configs})}
        for cfg in configs:
            alpha, kernel, edge_power, conf_power = cfg
            residual = blurred[kernel]
            gate = torch.ones_like(residual)
            if edge_power > 0:
                gate = gate * torch.pow(1.0 - edge, float(edge_power))
            if conf_power > 0:
                gate = gate * torch.pow(conf, float(conf_power))
            pred = torch.clamp(base + float(alpha) * gate * residual, -1.0, 1.0)
            pred_mm = prediction_to_mm(pred, batch, 1.0)
            for j in range(pred.shape[0]):
                single_mask = mask[j:j + 1]
                rmse = masked_rmse_mm(pred[j:j + 1], target_mm[j:j + 1], batch, single_mask)
                score[cfg].append(rmse)
    return base_rows, diff_rows, {
        cfg: float(np.asarray(vals, dtype=np.float64).mean())
        for cfg, vals in score.items()
    }


@torch.no_grad()
def eval_selected_split(diffusion, loader, device, args, cfg):
    alpha, kernel, edge_power, conf_power = cfg
    selected_rows = []
    for batch in tqdm(loader, desc=f"selected lowpass posterior {args.split}"):
        base = torch.clamp(batch["base_height"].to(device, non_blocking=True), -1.0, 1.0)
        diff = diffusion.sample_ddim(
            batch,
            steps=args.ddim_steps,
            ensemble_size=args.ensemble,
            start_from_base=True,
            start_ratio=args.start_ratio,
        )
        edge = torch.clamp(batch["edge_score"].to(device, non_blocking=True), 0.0, 1.0)
        conf = torch.clamp(batch["phase_conf"].to(device, non_blocking=True), 0.0, 1.0)
        pred = make_corrected(base, diff, edge, conf, alpha, kernel, edge_power, conf_power)
        pred_mm = prediction_to_mm(pred, batch, 1.0)
        target_mm = batch["height_raw"].to(device, non_blocking=True)
        mask = torch.clamp(batch["mask"].to(device, non_blocking=True), 0.0, 1.0)
        for j in range(pred.shape[0]):
            sample = len(selected_rows)
            single_mask = mask[j:j + 1]
            metrics = compute_metrics(pred_mm[j:j + 1], target_mm[j:j + 1], mask=single_mask)
            selected_rows.append({"sample": sample, **{f"selected_{k}": v for k, v in metrics.items()}})
    return selected_rows


def summarize_selected(rows):
    out = {"n": len(rows)}
    for metric in METRIC_KEYS:
        vals = np.asarray([float(r[f"selected_{metric}"]) for r in rows], dtype=np.float64)
        out[metric] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--base_prefix", default="base_c4_adapter")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--image_h", type=int, default=960)
    parser.add_argument("--image_w", type=int, default=960)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--start_ratio", type=float, default=0.05)
    parser.add_argument("--alphas", default="0.10,0.20,0.35,0.50")
    parser.add_argument("--kernels", default="1,3,7,15")
    parser.add_argument("--edge_powers", default="0,1,2,4")
    parser.add_argument("--conf_powers", default="0,1")
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
    configs = list(product(
        parse_floats(args.alphas),
        parse_ints(args.kernels),
        parse_floats(args.edge_powers),
        parse_floats(args.conf_powers),
    ))
    base_rows, diff_rows, scores = eval_split(diffusion, loaders[args.split], device, args, configs)
    best_cfg = min(configs, key=lambda cfg: scores[cfg])
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_rows = [
        {
            "alpha": cfg[0],
            "kernel": cfg[1],
            "edge_power": cfg[2],
            "conf_power": cfg[3],
            "rmse": scores[cfg],
        }
        for cfg in configs
    ]
    with open(out_dir / f"{args.split}_lowpass_sweep.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "kernel", "edge_power", "conf_power", "rmse"])
        writer.writeheader()
        writer.writerows(sweep_rows)
    selected_rows = eval_selected_split(diffusion, loaders[args.split], device, args, best_cfg)
    save_rows(base_rows, out_dir / f"{args.split}_base_metrics.csv", "base")
    save_rows(diff_rows, out_dir / f"{args.split}_diff_metrics.csv", "diff")
    with open(out_dir / f"{args.split}_selected_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample"] + [f"selected_{k}" for k in METRIC_KEYS])
        writer.writeheader()
        writer.writerows(selected_rows)
    result = {
        "split": args.split,
        "selected": {
            "alpha": best_cfg[0],
            "kernel": best_cfg[1],
            "edge_power": best_cfg[2],
            "conf_power": best_cfg[3],
            "selection_rmse": scores[best_cfg],
        },
        "base": direct_summary(base_rows, "base"),
        "diff": direct_summary(diff_rows, "diff"),
        "selected_summary": summarize_selected(selected_rows),
    }
    with open(out_dir / f"{args.split}_lowpass_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
