from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from tqdm import tqdm

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from diffusion_pip import PIPDiffusion
from models import ConditionalUNet, ConditionalUNetAdapter
from train_fpp_official_style_unet import METRIC_KEYS, parse_channel_spec, summarize
from train_pip_lite import prediction_to_mm
from utils.metrics import compute_metrics


def _saved_arg(saved_args, key, default=None):
    if isinstance(saved_args, dict):
        return saved_args.get(key, default)
    return default


def build_model(saved_args, model_cond_channels):
    base_ch = int(_saved_arg(saved_args, "base_channels", 48))
    injection = str(_saved_arg(saved_args, "condition_injection", "concat"))
    if injection == "adapter":
        return ConditionalUNetAdapter(
            cond_channels=model_cond_channels,
            base_ch=base_ch,
            ch_mult=(1, 2, 4, 8),
            dropout=0.05,
            adapter_hidden=int(_saved_arg(saved_args, "adapter_hidden", 32)),
        )
    return ConditionalUNet(
        cond_channels=model_cond_channels,
        base_ch=base_ch,
        ch_mult=(1, 2, 4, 8),
        dropout=0.05,
    )


def save_summary(rows, path):
    keys = ["split", "start_ratio", "alpha", "n"] + METRIC_KEYS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_rows(rows, path):
    keys = ["sample"] + METRIC_KEYS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in keys})


@torch.no_grad()
def evaluate_blend(diffusion, loader, device, height_scale, start_ratio, alphas, args):
    diffusion.model.eval()
    rows_by_alpha = {float(alpha): [] for alpha in alphas}
    for batch in tqdm(loader, desc=f"blend sr={start_ratio:.2f}"):
        base = torch.clamp(batch["base_height"].to(device, non_blocking=True), -1.0, 1.0)
        pred = diffusion.sample_ddim(
            batch,
            steps=args.ddim_steps,
            ensemble_size=args.ensemble,
            progress=False,
            start_from_base=True,
            start_ratio=start_ratio,
        )
        target_raw = batch["height_raw"].to(device, non_blocking=True)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        for alpha in rows_by_alpha:
            blended = torch.clamp(base + alpha * (pred - base), -1.0, 1.0)
            blended_mm = prediction_to_mm(blended, batch, height_scale)
            for j in range(blended_mm.shape[0]):
                single_mask = mask[j:j + 1] if mask is not None else None
                rows_by_alpha[alpha].append({
                    "sample": len(rows_by_alpha[alpha]),
                    **compute_metrics(blended_mm[j:j + 1], target_raw[j:j + 1], mask=single_mask),
                })
    return rows_by_alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--base_prefix", default="base_c4_adapter")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--image_h", type=int, default=0)
    parser.add_argument("--image_w", type=int, default=0)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--start_ratios", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.35, 0.55])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
    parser.add_argument("--require_cache", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})
    image_h = int(args.image_h or _saved_arg(saved_args, "image_h", 960))
    image_w = int(args.image_w or _saved_arg(saved_args, "image_w", 960))
    include_ftp = bool(_saved_arg(saved_args, "include_ftp", False))

    loaders = create_fpp_ml_bench_loaders(
        cache_dir=args.cache_dir,
        batch_size=1,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        include_ftp=include_ftp,
        image_h=image_h,
        image_w=image_w,
        require_cache=args.require_cache,
        base_prefix=args.base_prefix,
    )
    loader = loaders[args.split]

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
        image_h=image_h,
        image_w=image_w,
        device=device,
        cond_indices=physics_indices,
        target_mode=str(_saved_arg(saved_args, "target_mode", "base_residual")),
        residual_scale=float(_saved_arg(saved_args, "resolved_residual_scale", 1.0)),
        base_residual_gate=float(_saved_arg(saved_args, "base_residual_gate", 1.0)),
    )

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for ratio in args.start_ratios:
        rows_by_alpha = evaluate_blend(diffusion, loader, device, loaders["height_scale"], float(ratio), args.alphas, args)
        for alpha, rows in rows_by_alpha.items():
            tag = f"{args.split}_sr{float(ratio):.2f}_alpha{float(alpha):.2f}"
            save_rows(rows, out_dir / f"{tag}_per_sample_metrics.csv")
            metrics = summarize(rows)
            row = {
                "split": args.split,
                "start_ratio": float(ratio),
                "alpha": float(alpha),
                "n": len(rows),
                **{key: metrics[key]["mean"] for key in METRIC_KEYS},
            }
            summary_rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    save_summary(summary_rows, out_dir / f"blend_summary_{args.split}.csv")
    with open(out_dir / f"blend_summary_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
