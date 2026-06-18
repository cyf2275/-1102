"""Precompute base-depth predictions from official-style UNet variants.

This is the companion of precompute_fpp_base_predictions.py for
train_fpp_official_style_unet.py checkpoints whose input can be raw fringe,
physics instructions, or fringe plus physics instructions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from models import OfficialUNetFPP
from train_fpp_official_style_unet import (
    channel_names,
    input_channels,
    make_input,
    parse_channel_spec,
    prediction_to_mm,
    summarize,
)
from utils.metrics import compute_metrics


@torch.no_grad()
def run_split(model, loader, split, args, device, physics_channels, residual_hist=None):
    dataset = loader.dataset
    out_path = Path(args.cache_dir) / f"{args.prefix}_height_{split}_float16.npy"
    out = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(dataset), 1, args.image_size, args.image_size),
    )
    rows = []
    write_pos = 0
    for batch in tqdm(loader, desc=f"precompute official {split}"):
        x = make_input(batch, args.input_mode, device, physics_channels)
        pred01 = torch.clamp(model(x), 0.0, 1.0)
        pred_height = pred01 * 2.0 - 1.0
        batch_size = pred_height.shape[0]
        out[write_pos:write_pos + batch_size] = pred_height.detach().cpu().numpy().astype(np.float16)

        pred_mm = prediction_to_mm(pred01, batch)
        target_raw = batch["height_raw"].to(device, non_blocking=True)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        for j in range(batch_size):
            single_mask = mask[j:j + 1] if mask is not None else None
            metrics = compute_metrics(pred_mm[j:j + 1], target_raw[j:j + 1], mask=single_mask)
            rows.append({"sample": write_pos + j, **metrics})

        if residual_hist is not None:
            target = batch["height"].to(device, non_blocking=True)
            abs_residual = torch.abs(target - pred_height)
            if mask is not None:
                values = abs_residual[mask > 0.5].detach().float().cpu().numpy()
            else:
                values = abs_residual.detach().float().cpu().numpy().reshape(-1)
            update_histogram(residual_hist, values, (0.0, 2.0))
        write_pos += batch_size

    out.flush()
    del out
    return summarize(rows)


def update_histogram(hist, values, hist_range):
    if values.size == 0:
        return
    clipped = np.clip(values, hist_range[0], hist_range[1])
    h, _ = np.histogram(clipped, bins=hist.shape[0], range=hist_range)
    hist += h.astype(np.int64)


def quantile_from_hist(hist, hist_range, q):
    total = int(hist.sum())
    if total <= 0:
        return 1.0
    threshold = max(1, int(np.ceil(total * float(q))))
    idx = int(np.searchsorted(np.cumsum(hist), threshold, side="left"))
    idx = min(idx, hist.shape[0] - 1)
    lo, hi = hist_range
    return float(lo + (idx + 1) * (hi - lo) / hist.shape[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--input_mode", choices=["fringe", "physics", "fringe_plus_physics"], required=True)
    parser.add_argument("--include_ftp", action="store_true")
    parser.add_argument("--physics_channels", default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--require_cache", action="store_true")
    args = parser.parse_args()

    physics_channels = parse_channel_spec(args.physics_channels, args.include_ftp)
    in_ch = input_channels(args.input_mode, args.include_ftp, physics_channels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = OfficialUNetFPP(in_channels=in_ch, out_channels=1, dropout_rate=0.0).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    loaders = create_fpp_ml_bench_loaders(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_ftp=args.include_ftp,
        image_h=args.image_size,
        image_w=args.image_size,
        require_cache=args.require_cache,
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input mode: {args.input_mode} | channels={in_ch}")
    print(f"Physics channels: {physics_channels} | {channel_names(physics_channels)}")
    print(f"Cache prefix: {args.prefix}")

    residual_hist = np.zeros(20000, dtype=np.int64)
    split_summaries = {}
    for split in ("train", "val", "test"):
        hist = residual_hist if split == "train" else None
        split_summaries[split] = run_split(model, loaders[split], split, args, device, physics_channels, hist)

    stats = {
        "prefix": args.prefix,
        "checkpoint": args.checkpoint,
        "input_mode": args.input_mode,
        "include_ftp": args.include_ftp,
        "physics_channels": physics_channels,
        "physics_channel_names": channel_names(physics_channels),
        "image_size": args.image_size,
        "p95_abs_residual": quantile_from_hist(residual_hist, (0.0, 2.0), 0.95),
        "p99_abs_residual": quantile_from_hist(residual_hist, (0.0, 2.0), 0.99),
        "metrics": split_summaries,
    }
    stats["residual_scale"] = stats["p99_abs_residual"]
    stats_path = Path(args.cache_dir) / f"{args.prefix}_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
