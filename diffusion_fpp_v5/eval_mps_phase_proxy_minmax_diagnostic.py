from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.dataset_fpp_phase import create_fpp_phase_loaders
from models.single_frame_baselines import build_single_frame_baseline
from train_fpp_mps_xnet_phase_proxy_baseline import (
    get_train_global_phase_minmax,
    make_model_input,
    phase01_to_abs,
    phase_metrics,
    phase_xy_to_depth_mm,
    global01_to_abs_phase,
)
from train_fpp_official_style_unet import METRIC_KEYS, summarize
from utils.metrics import compute_metrics


def save_rows(rows, path):
    keys = ["split", "mode", "sample"] + METRIC_KEYS + ["phase_rmse", "phase_mae"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def masked_linear_fit(x, y, mask, eps=1e-6):
    """Fit y ~= a*x+b per sample under mask. Oracle diagnostic only."""
    m = mask > 0.5
    if int(m.sum()) < 16:
        return 1.0, 0.0
    xv = x[m].float()
    yv = y[m].float()
    x_mean = xv.mean()
    y_mean = yv.mean()
    denom = ((xv - x_mean) ** 2).mean().clamp_min(eps)
    a = ((xv - x_mean) * (yv - y_mean)).mean() / denom
    b = y_mean - a * x_mean
    return float(a.detach().cpu()), float(b.detach().cpu())


def build_args_from_checkpoint(ckpt_args, cli_args):
    merged = dict(ckpt_args or {})
    for key in (
        "base_cache_dir",
        "phase_cache_dir",
        "image_size",
        "eval_batch_size",
        "num_workers",
        "require_cache",
        "preload_ram",
    ):
        merged[key] = getattr(cli_args, key)
    if "input_mode" not in merged:
        merged["input_mode"] = cli_args.input_mode
    if "base_channels" not in merged:
        merged["base_channels"] = cli_args.base_channels
    if "dropout" not in merged:
        merged["dropout"] = 0.0
    return Namespace(**merged)


@torch.no_grad()
def evaluate(model, loader, device, args, split):
    model.eval()
    rows = []
    for batch in tqdm(loader, desc=f"{split} minmax diagnostic"):
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        cond = batch["cond"]
        target = batch["height_raw"]
        mask = batch["mask"]
        output = model(make_model_input(batch, args, device))
        pred01 = output["unwrapped"].float().clamp(0.0, 1.0)
        target_phase = phase01_to_abs(batch["phase_target"][:, 2:3], batch["phase_minmax"])

        phase_modes = {
            "valid_global_train_range": global01_to_abs_phase(pred01, args.global_phase_min, args.global_phase_max),
            "illegal_oracle_sample_minmax": phase01_to_abs(pred01, batch["phase_minmax"]),
        }

        # Per-sample affine alignment: stronger illegal oracle to diagnose whether
        # relative phase shape is useful despite wrong absolute scale/offset.
        affine_phase = []
        for j in range(pred01.shape[0]):
            pred_global = phase_modes["valid_global_train_range"][j:j + 1]
            a, b = masked_linear_fit(pred_global[0, 0], target_phase[j, 0], mask[j, 0])
            affine_phase.append(pred_global * a + b)
        phase_modes["illegal_oracle_sample_affine"] = torch.cat(affine_phase, dim=0)

        for mode, pred_phase in phase_modes.items():
            pred_mm = phase_xy_to_depth_mm(pred_phase, cond)
            for j in range(pred_mm.shape[0]):
                metrics = compute_metrics(pred_mm[j:j + 1], target[j:j + 1], mask=mask[j:j + 1])
                ph_rmse, ph_mae = phase_metrics(pred_phase[j:j + 1], target_phase[j:j + 1], mask[j:j + 1])
                rows.append(
                    {
                        "split": split,
                        "mode": mode,
                        "sample": int(batch["sample_index"][j].detach().cpu()) if "sample_index" in batch else len(rows),
                        **metrics,
                        "phase_rmse": ph_rmse,
                        "phase_mae": ph_mae,
                    }
                )
    return rows


def summarize_by_mode(rows):
    out = {}
    modes = sorted({r["mode"] for r in rows})
    for mode in modes:
        subset = [r for r in rows if r["mode"] == mode]
        summary = summarize(subset)
        for key in ("phase_rmse", "phase_mae"):
            vals = np.asarray([float(r[key]) for r in subset], dtype=np.float64)
            summary[key] = {
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "std": float(vals.std(ddof=0)),
            }
        summary["n"] = len(subset)
        out[mode] = summary
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an MPS-XNet-style phase proxy checkpoint under valid global "
            "phase scaling and illegal oracle phase scaling diagnostics."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--base_cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_phase_cache_960")
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--base_channels", type=int, default=8)
    parser.add_argument("--input_mode", choices=["fringe", "fringe_xy"], default="fringe_xy")
    parser.add_argument("--require_cache", action="store_true")
    parser.add_argument("--preload_ram", action="store_true")
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    args_cli = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    args = build_args_from_checkpoint(ckpt.get("args", {}), args_cli)

    loaders = create_fpp_phase_loaders(
        base_cache_dir=args.base_cache_dir,
        phase_cache_dir=args.phase_cache_dir,
        batch_size=1,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        image_h=args.image_size,
        image_w=args.image_size,
        require_cache=args.require_cache,
        preload_ram=args.preload_ram,
        train_minimal=True,
        train_extra_keys={"mask", "phase_minmax"},
    )
    if not hasattr(args, "global_phase_min") or not hasattr(args, "global_phase_max"):
        args.global_phase_min, args.global_phase_max = get_train_global_phase_minmax(loaders)

    in_channels = 3 if args.input_mode == "fringe_xy" else 1
    model = build_single_frame_baseline(
        "mps_xnet_phase",
        in_channels=in_channels,
        out_channels=1,
        base_channels=getattr(args, "base_channels", 8),
        dropout_rate=getattr(args, "dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    save_dir = Path(args_cli.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    splits = ["val", "test"] if args_cli.split == "both" else [args_cli.split]
    for split in splits:
        rows = evaluate(model, loaders[split], device, args, split)
        save_rows(rows, save_dir / f"{split}_mps_minmax_diagnostic_rows.csv")
        all_rows.extend(rows)
    save_rows(all_rows, save_dir / "mps_minmax_diagnostic_rows.csv")
    summary = {
        "checkpoint": str(args_cli.checkpoint),
        "global_phase_min": float(args.global_phase_min),
        "global_phase_max": float(args.global_phase_max),
        "input_mode": args.input_mode,
        "method_note": (
            "valid_global_train_range is the only valid single-frame evaluation. "
            "illegal_oracle_sample_minmax and illegal_oracle_sample_affine use "
            "teacher phase statistics from each evaluation sample and are only "
            "diagnostics for phase scale/offset failure, not publishable methods."
        ),
        "by_split_mode": {},
    }
    for split in splits:
        summary["by_split_mode"][split] = summarize_by_mode([r for r in all_rows if r["split"] == split])
    with open(save_dir / "mps_minmax_diagnostic_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
