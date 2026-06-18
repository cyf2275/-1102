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
    return saved_args.get(key, default) if isinstance(saved_args, dict) else default


def build_model(saved_args, model_cond_channels):
    if str(_saved_arg(saved_args, "condition_injection", "concat")) == "adapter":
        return ConditionalUNetAdapter(
            cond_channels=model_cond_channels,
            base_ch=int(_saved_arg(saved_args, "base_channels", 48)),
            ch_mult=(1, 2, 4, 8),
            dropout=0.05,
            adapter_hidden=int(_saved_arg(saved_args, "adapter_hidden", 32)),
        )
    return ConditionalUNet(
        cond_channels=model_cond_channels,
        base_ch=int(_saved_arg(saved_args, "base_channels", 48)),
        ch_mult=(1, 2, 4, 8),
        dropout=0.05,
    )


def masked_scalar(x, mask):
    mask = torch.clamp(mask.to(device=x.device, dtype=x.dtype), 0.0, 1.0)
    return float((x * mask).sum().detach().cpu() / mask.sum().clamp(min=1.0).detach().cpu())


def save_rows(rows, path):
    keys = [
        "sample",
        "delta_mean",
        "delta_edge_mean",
        "delta_lowconf_mean",
        "phase_conf_mean",
        "edge_mean",
    ]
    for prefix in ("base", "blend", "diff"):
        keys.extend(f"{prefix}_{key}" for key in METRIC_KEYS)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in keys})


@torch.no_grad()
def evaluate(diffusion, loader, device, height_scale, args):
    rows = []
    for batch in tqdm(loader, desc=f"adaptive features {args.split}"):
        base = torch.clamp(batch["base_height"].to(device, non_blocking=True), -1.0, 1.0)
        diff = diffusion.sample_ddim(
            batch,
            steps=args.ddim_steps,
            ensemble_size=args.ensemble,
            start_from_base=True,
            start_ratio=args.start_ratio,
        )
        blend = torch.clamp(base + args.alpha * (diff - base), -1.0, 1.0)
        base_mm = prediction_to_mm(base, batch, height_scale)
        diff_mm = prediction_to_mm(diff, batch, height_scale)
        blend_mm = prediction_to_mm(blend, batch, height_scale)
        target = batch["height_raw"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        edge = torch.clamp(batch["edge_score"].to(device, non_blocking=True), 0.0, 1.0)
        conf = torch.clamp(batch["phase_conf"].to(device, non_blocking=True), 0.0, 1.0)
        delta = torch.abs(diff - base)
        for j in range(base.shape[0]):
            single_mask = mask[j:j + 1]
            row = {
                "sample": len(rows),
                "delta_mean": masked_scalar(delta[j:j + 1], single_mask),
                "delta_edge_mean": masked_scalar(delta[j:j + 1] * edge[j:j + 1], single_mask),
                "delta_lowconf_mean": masked_scalar(delta[j:j + 1] * (1.0 - conf[j:j + 1]), single_mask),
                "phase_conf_mean": masked_scalar(conf[j:j + 1], single_mask),
                "edge_mean": masked_scalar(edge[j:j + 1], single_mask),
            }
            for prefix, pred in (("base", base_mm), ("blend", blend_mm), ("diff", diff_mm)):
                metrics = compute_metrics(pred[j:j + 1], target[j:j + 1], mask=single_mask)
                row.update({f"{prefix}_{key}": metrics[key] for key in METRIC_KEYS})
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--base_prefix", default="base_c4_adapter")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--image_h", type=int, default=960)
    parser.add_argument("--image_w", type=int, default=960)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--start_ratio", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.35)
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
    rows = evaluate(diffusion, loaders[args.split], device, loaders["height_scale"], args)
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rows(rows, out_dir / f"{args.split}_adaptive_features.csv")

    summaries = {}
    for prefix in ("base", "blend", "diff"):
        prefixed = [
            {key: row[f"{prefix}_{key}"] for key in METRIC_KEYS}
            for row in rows
        ]
        summaries[prefix] = summarize(prefixed)
    with open(out_dir / f"{args.split}_adaptive_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(json.dumps({"split": args.split, **{k: summaries[k]["rmse"]["mean"] for k in summaries}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
