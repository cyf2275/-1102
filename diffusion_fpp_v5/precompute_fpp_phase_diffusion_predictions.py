"""Precompute phase-restoration diffusion predictions for downstream depth models."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.dataset_fpp_phase import create_fpp_phase_loaders
from models import ConditionalUNetAdapter
from train_fpp_phase_diffusion import (
    PHASE_METRIC_KEYS,
    PhaseRestorationDiffusion,
    parse_channel_spec,
    parse_ch_mult,
    phase_metrics_tensor,
    phase_target,
)


def save_rows(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    out = {}
    for key in PHASE_METRIC_KEYS:
        if key not in rows[0]:
            continue
        vals = np.array([float(r[key]) for r in rows], dtype=np.float64)
        out[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
        }
    return out


def build_model_from_checkpoint(ckpt, device):
    args = ckpt.get("args", {})
    phase_indices = args.get("phase_channel_indices")
    if phase_indices is None:
        phase_channels = args.get("phase_channels", "0-12")
        phase_indices = parse_channel_spec(phase_channels, max_channel=12)
    ch_mult = args.get("ch_mult_tuple", None)
    if ch_mult is None:
        ch_mult = parse_ch_mult(args.get("ch_mult", "1,2,4,8,8"))
    target_channels = int(args.get("target_channels", 2))
    model = ConditionalUNetAdapter(
        in_channels=target_channels,
        cond_channels=len(phase_indices),
        out_channels=target_channels,
        base_ch=int(args.get("base_channels", 24)),
        ch_mult=tuple(ch_mult),
        dropout=float(args.get("dropout", 0.05)),
        adapter_hidden=int(args.get("adapter_hidden", 24)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    diffusion = PhaseRestorationDiffusion(
        model,
        timesteps=int(args.get("timesteps", 200)),
        image_h=int(args.get("image_size", 960)),
        image_w=int(args.get("image_size", 960)),
        device=device,
        cond_indices=phase_indices,
        phase_weight=float(args.get("phase_weight", 1.0)),
        grad_weight=float(args.get("grad_weight", 0.05)),
        unit_weight=float(args.get("unit_weight", 0.02)),
        target_channels=target_channels,
        uph_weight=float(args.get("uph_weight", 0.0)),
        uph_grad_weight=float(args.get("uph_grad_weight", 0.05)),
        uph_start_from=str(args.get("uph_start_from", "coord_auto")),
        uph_norm=str(args.get("uph_norm", "sample")),
        uph_global_min=float(args.get("uph_global_min", 0.0)),
        uph_global_max=float(args.get("uph_global_max", 1.0)),
        uph_representation=str(args.get("uph_representation", "absolute")),
        uph_prior_coef=args.get("uph_prior_coef", None),
        uph_prior_basis=str(args.get("uph_prior_basis", "xy2_phase")),
        uph_residual_scale=float(args.get("uph_residual_scale", 1.0) or 1.0),
    )
    return diffusion, args, phase_indices


@torch.no_grad()
def run_split(diffusion, loader, split, args, out_dir):
    sample_ds = loader.dataset
    n = len(sample_ds)
    image_size = int(args.image_size)
    target_channels = int(getattr(diffusion, "target_channels", 2))
    pred_path = out_dir / f"{args.output_prefix}_{split}_float16.npy"
    pred_map = np.lib.format.open_memmap(
        pred_path,
        mode="w+",
        dtype=np.float16,
        shape=(n, target_channels, image_size, image_size),
    )
    rows = []
    for batch in tqdm(loader, desc=f"precompute {split}"):
        pred = diffusion.sample_ddim(
            batch,
            steps=args.ddim_steps,
            ensemble_size=args.ensemble,
            start_from=args.sample_start_from,
            start_ratio=args.sample_start_ratio,
            progress=False,
        )
        target = diffusion.target(batch)
        mask = batch["mask"].to(diffusion.device, non_blocking=True)
        indices = batch["sample_index"].cpu().numpy().astype(int).tolist()
        pred_cpu = pred.detach().cpu().numpy().astype(np.float16)
        for j, idx in enumerate(indices):
            pred_map[idx] = pred_cpu[j]
            row = {"sample": int(idx)}
            row.update(phase_metrics_tensor(pred[j:j + 1], target[j:j + 1], mask=mask[j:j + 1]))
            rows.append(row)
    pred_map.flush()
    rows = sorted(rows, key=lambda r: r["sample"])
    save_rows(rows, out_dir / f"{args.output_prefix}_{split}_metrics.csv")
    summary = summarize(rows)
    summary["n"] = len(rows)
    summary["prediction_path"] = str(pred_path)
    with open(out_dir / f"{args.output_prefix}_{split}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_phase_cache_960")
    parser.add_argument("--output_prefix", default="phase_pred_e2c")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ddim_steps", type=int, default=10)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--sample_start_from", choices=["noise", "ftp", "hilbert"], default="ftp")
    parser.add_argument("--sample_start_ratio", type=float, default=0.7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    ckpt = torch.load(args.checkpoint, map_location=device)
    diffusion, ckpt_args, phase_indices = build_model_from_checkpoint(ckpt, device)
    print(f"Loaded {args.checkpoint}")
    print(f"Checkpoint epoch={ckpt.get('epoch')} phase_indices={phase_indices}")

    loaders = create_fpp_phase_loaders(
        base_cache_dir=args.base_cache_dir,
        phase_cache_dir=args.phase_cache_dir,
        batch_size=1,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_h=args.image_size,
        image_w=args.image_size,
        require_cache=True,
    )
    out_dir = Path(args.phase_cache_dir)
    all_summary = {"checkpoint": args.checkpoint, "args": vars(args), "splits": {}}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        loader_key = "train_eval" if split == "train" else split
        all_summary["splits"][split] = run_split(diffusion, loaders[loader_key], split, args, out_dir)
    with open(out_dir / f"{args.output_prefix}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(all_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
