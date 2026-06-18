from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from diffusion_pip import PIPDiffusion
from models import ConditionalUNet, ConditionalUNetAdapter
from train_fpp_official_style_unet import parse_channel_spec
from train_pip_lite import evaluate_split, save_rows, summarize


METRIC_KEYS = ["rmse", "mae", "edge_rmse", "normal_deg", "ssim"]


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


def write_summary(rows, path):
    keys = ["mode", "start_ratio", "split", "n"] + METRIC_KEYS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--base_prefix", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--image_h", type=int, default=0)
    parser.add_argument("--image_w", type=int, default=0)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--start_ratios", type=float, nargs="+", default=[0.55, 0.75, 0.85])
    parser.add_argument("--no_random", action="store_true")
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
    loader = loaders["train_eval" if args.split == "train" else args.split]

    physics_indices = _saved_arg(saved_args, "physics_channel_indices", None)
    if physics_indices is None:
        spec = str(_saved_arg(saved_args, "physics_channels", ""))
        physics_indices = parse_channel_spec(spec, include_ftp)
    model_cond_channels = int(ckpt.get(
        "model_cond_channels",
        len(physics_indices) if physics_indices is not None else loaders["cond_channels"],
    ))
    model = build_model(saved_args, model_cond_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    target_mode = str(_saved_arg(saved_args, "target_mode", "full_x0"))
    if target_mode not in {"full_x0", "base_residual"}:
        raise ValueError("sampling sweep is only defined for full_x0/base_residual checkpoints")
    diffusion = PIPDiffusion(
        model,
        timesteps=int(_saved_arg(saved_args, "timesteps", 200)),
        image_h=image_h,
        image_w=image_w,
        device=device,
        cond_indices=physics_indices,
        target_mode=target_mode,
        residual_scale=float(_saved_arg(saved_args, "resolved_residual_scale", 1.0)),
        base_residual_gate=float(_saved_arg(saved_args, "base_residual_gate", 1.0)),
    )

    out_root = Path(args.save_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    settings = []
    if not args.no_random and target_mode == "full_x0":
        settings.append(("random", None))
    settings.extend(("base", float(r)) for r in args.start_ratios)

    for mode, ratio in settings:
        eval_args = SimpleNamespace(
            dataset="fpp_ml_bench",
            ddim_steps=args.ddim_steps,
            ensemble=args.ensemble,
            sample_start_from_base=(mode == "base" or target_mode == "base_residual"),
            sample_start_ratio=float(ratio if ratio is not None else 1.0),
            target_mode=target_mode,
        )
        tag = "random" if mode == "random" else f"base_sr{ratio:.2f}"
        rows = evaluate_split(
            diffusion,
            loader,
            device,
            loaders["height_scale"],
            args.split,
            eval_args,
        )
        run_dir = out_root / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        save_rows(rows, run_dir / "per_sample_metrics.csv")
        metrics = summarize(rows)
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        summary_rows.append({
            "mode": mode,
            "start_ratio": "" if ratio is None else float(ratio),
            "split": args.split,
            "n": len(rows),
            **{key: metrics[key]["mean"] for key in METRIC_KEYS},
        })
        print(json.dumps(summary_rows[-1], ensure_ascii=False))

    write_summary(summary_rows, out_root / "summary.csv")
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
