from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from models import CoarseLowpassNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute CoarseNet low-pass depth/uncertainty channels for PIP-Full conditioning."
    )
    parser.add_argument("--coarse_checkpoint", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--phase_cache_dir", required=True)
    parser.add_argument("--phase_pred_prefix", default="")
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--coarse_mode", choices=["depth_unc", "depth_only"], default="depth_unc")
    parser.add_argument("--image_h", type=int, default=960)
    parser.add_argument("--image_w", type=int, default=960)
    parser.add_argument("--lowpass_factor", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--require_cache", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def write_split(model, loader, split: str, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    out_path = Path(args.phase_cache_dir) / f"{args.output_prefix}_{split}_float16.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first = next(iter(loader))
    phase_ch = int(first["phase_pred"].shape[1]) if "phase_pred" in first else 0
    coarse_ch = 2 if args.coarse_mode == "depth_unc" else 1
    out_ch = phase_ch + coarse_ch
    n = len(loader.dataset)
    h, w = int(args.image_h), int(args.image_w)
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, out_ch, h, w))

    depth_vals = []
    unc_vals = []
    written = 0
    for batch in tqdm(loader, desc=f"precompute coarse cond {split}"):
        cond = batch["cond"].to(device, non_blocking=True)
        depth_low, log_var = model(cond)
        # Keep condition channels bounded and roughly zero-centered.
        unc_norm = torch.tanh(log_var / 4.0)
        parts = []
        if "phase_pred" in batch:
            parts.append(batch["phase_pred"].to(device, non_blocking=True))
        parts.append(depth_low)
        if args.coarse_mode == "depth_unc":
            parts.append(unc_norm)
        feat = torch.cat(parts, dim=1).detach().cpu().numpy().astype(np.float16)
        sample_index = batch["sample_index"].detach().cpu().numpy().astype(np.int64)
        arr[sample_index] = feat
        written += int(feat.shape[0])
        depth_vals.append(depth_low.detach().float().cpu())
        unc_vals.append(unc_norm.detach().float().cpu())
    arr.flush()
    depth_cat = torch.cat([x.flatten() for x in depth_vals])
    unc_cat = torch.cat([x.flatten() for x in unc_vals])
    return {
        "split": split,
        "path": str(out_path),
        "shape": [n, out_ch, h, w],
        "phase_pred_channels": phase_ch,
        "coarse_mode": args.coarse_mode,
        "coarse_channels": coarse_ch,
        "written": written,
        "depth_low_mean": float(depth_cat.mean().item()),
        "depth_low_std": float(depth_cat.std(unbiased=False).item()),
        "unc_norm_mean": float(unc_cat.mean().item()),
        "unc_norm_std": float(unc_cat.std(unbiased=False).item()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.coarse_checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})
    cond_channels = int(ckpt.get("cond_channels", 9))
    model = CoarseLowpassNet(cond_channels, base_ch=int(saved_args.get("base_channels", 32))).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loaders = create_fpp_ml_bench_loaders(
        cache_dir=args.cache_dir,
        batch_size=args.eval_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        include_ftp=bool(saved_args.get("include_ftp", False)),
        image_h=args.image_h,
        image_w=args.image_w,
        lowpass_factor=args.lowpass_factor,
        require_cache=args.require_cache,
        phase_cache_dir=args.phase_cache_dir,
        phase_pred_prefix=args.phase_pred_prefix or None,
        append_phase_pred_to_cond=False,
    )

    summary = {
        "coarse_checkpoint": args.coarse_checkpoint,
        "phase_pred_prefix": args.phase_pred_prefix,
        "output_prefix": args.output_prefix,
        "coarse_mode": args.coarse_mode,
        "uncertainty_channel": "tanh(log_var / 4)",
        "splits": {},
    }
    for split, loader_key in (("train", "train_eval"), ("val", "val"), ("test", "test")):
        summary["splits"][split] = write_split(model, loaders[loader_key], split, args, device)

    out_json = Path(args.phase_cache_dir) / f"{args.output_prefix}_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
