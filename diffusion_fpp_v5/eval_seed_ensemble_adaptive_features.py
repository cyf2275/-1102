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
from train_fpp_official_style_unet import METRIC_KEYS, parse_channel_spec, summarize
from train_pip_lite import prediction_to_mm
from utils.metrics import compute_metrics


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
    for prefix in ("base", "ensemble", "diff"):
        keys.extend(f"{prefix}_{key}" for key in METRIC_KEYS)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in keys})


@torch.no_grad()
def evaluate(args, loaders, split, saved_args, physics_indices, model_cond_channels, device):
    loader = loaders["train_eval" if split == "train" else split]
    base_store = []
    target_store = []
    mask_store = []
    edge_store = []
    conf_store = []
    minmax_store = []
    pred_sum = None

    for ckpt_idx, ckpt_path in enumerate(args.checkpoints):
        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt_args = ckpt.get("args", saved_args)
        model = build_model(ckpt_args, model_cond_channels).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        diffusion = PIPDiffusion(
            model,
            timesteps=int(_saved_arg(ckpt_args, "timesteps", 200)),
            image_h=args.image_h,
            image_w=args.image_w,
            device=device,
            cond_indices=physics_indices,
            target_mode=str(_saved_arg(ckpt_args, "target_mode", "base_residual")),
            residual_scale=float(_saved_arg(ckpt_args, "resolved_residual_scale", 1.0)),
            base_residual_gate=float(_saved_arg(ckpt_args, "base_residual_gate", 1.0)),
        )
        preds = []
        for batch in tqdm(loader, desc=f"{Path(ckpt_path).parts[-3]} {split}"):
            pred = diffusion.sample_ddim(
                batch,
                steps=args.ddim_steps,
                ensemble_size=1,
                start_from_base=True,
                start_ratio=args.start_ratio,
            ).detach().cpu()
            preds.append(pred)
            if ckpt_idx == 0:
                base_store.append(torch.clamp(batch["base_height"], -1.0, 1.0).cpu())
                target_store.append(batch["height_raw"].cpu())
                mask_store.append(torch.clamp(batch["mask"], 0.0, 1.0).cpu())
                edge_store.append(torch.clamp(batch["edge_score"], 0.0, 1.0).cpu())
                conf_store.append(torch.clamp(batch["phase_conf"], 0.0, 1.0).cpu())
                minmax_store.append(batch["depth_minmax"].cpu())
        pred_tensor = torch.cat(preds, dim=0)
        pred_sum = pred_tensor if pred_sum is None else pred_sum + pred_tensor
        del model, diffusion
        if device.type == "cuda":
            torch.cuda.empty_cache()

    diff_all = pred_sum / float(len(args.checkpoints))
    base_all = torch.cat(base_store, dim=0)
    target_all = torch.cat(target_store, dim=0)
    mask_all = torch.cat(mask_store, dim=0)
    edge_all = torch.cat(edge_store, dim=0)
    conf_all = torch.cat(conf_store, dim=0)
    minmax_all = torch.cat(minmax_store, dim=0)

    rows = []
    for i in range(diff_all.shape[0]):
        base = base_all[i:i + 1]
        diff_raw = diff_all[i:i + 1]
        ensemble = torch.clamp(base + args.alpha * (diff_raw - base), -1.0, 1.0)
        mask = mask_all[i:i + 1]
        edge = edge_all[i:i + 1]
        conf = conf_all[i:i + 1]
        delta = torch.abs(diff_raw - base)
        batch_stub = {"depth_minmax": minmax_all[i:i + 1]}
        target = target_all[i:i + 1]
        row = {
            "sample": i,
            "delta_mean": masked_scalar(delta, mask),
            "delta_edge_mean": masked_scalar(delta * edge, mask),
            "delta_lowconf_mean": masked_scalar(delta * (1.0 - conf), mask),
            "phase_conf_mean": masked_scalar(conf, mask),
            "edge_mean": masked_scalar(edge, mask),
        }
        for prefix, pred in (("base", base), ("ensemble", ensemble), ("diff", diff_raw)):
            pred_mm = prediction_to_mm(pred, batch_stub, loaders["height_scale"])
            metrics = compute_metrics(pred_mm, target, mask=mask)
            row.update({f"{prefix}_{key}": metrics[key] for key in METRIC_KEYS})
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--base_prefix", default="base_c4_adapter")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    parser.add_argument("--image_h", type=int, default=960)
    parser.add_argument("--image_w", type=int, default=960)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--start_ratio", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--require_cache", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first_ckpt = torch.load(args.checkpoints[0], map_location=device)
    saved_args = first_ckpt.get("args", {})
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
    model_cond_channels = int(first_ckpt.get(
        "model_cond_channels",
        len(physics_indices) if physics_indices is not None else loaders["cond_channels"],
    ))

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    for split in args.splits:
        rows = evaluate(args, loaders, split, saved_args, physics_indices, model_cond_channels, device)
        save_rows(rows, out_dir / f"{split}_adaptive_features.csv")
        summaries = {}
        for prefix in ("base", "ensemble", "diff"):
            summaries[prefix] = summarize([
                {key: row[f"{prefix}_{key}"] for key in METRIC_KEYS}
                for row in rows
            ])
        all_summaries[split] = summaries
        with open(out_dir / f"{split}_adaptive_summary.json", "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
    with open(out_dir / "adaptive_summary_all.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        split: {prefix: all_summaries[split][prefix]["rmse"]["mean"] for prefix in all_summaries[split]}
        for split in args.splits
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
