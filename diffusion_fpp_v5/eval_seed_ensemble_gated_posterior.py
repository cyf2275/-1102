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


def masked_mean_scalar(x, mask):
    mask = torch.clamp(mask.to(dtype=x.dtype), 0.0, 1.0)
    return float((x * mask).sum() / mask.sum().clamp(min=1.0))


def save_rows(rows, path):
    keys = ["sample", "use_diffusion", "edge_mean"] + [
        f"{prefix}_{key}"
        for prefix in ("base", "ensemble", "gated")
        for key in METRIC_KEYS
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in keys})


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--base_prefix", default="base_c4_adapter")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--image_h", type=int, default=960)
    parser.add_argument("--image_w", type=int, default=960)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--start_ratio", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--edge_threshold", type=float, default=0.4674050956964493)
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
    loader = loaders[args.split]
    physics_indices = _saved_arg(saved_args, "physics_channel_indices", None)
    if physics_indices is None:
        physics_indices = parse_channel_spec(str(_saved_arg(saved_args, "physics_channels", "")), include_ftp)
    model_cond_channels = int(first_ckpt.get(
        "model_cond_channels",
        len(physics_indices) if physics_indices is not None else loaders["cond_channels"],
    ))

    base_store = []
    target_store = []
    mask_store = []
    edge_mean_store = []
    pred_sum = None
    count = 0

    for ckpt_path in args.checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device)
        model = build_model(ckpt.get("args", saved_args), model_cond_channels).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        diffusion = PIPDiffusion(
            model,
            timesteps=int(_saved_arg(ckpt.get("args", saved_args), "timesteps", 200)),
            image_h=args.image_h,
            image_w=args.image_w,
            device=device,
            cond_indices=physics_indices,
            target_mode=str(_saved_arg(ckpt.get("args", saved_args), "target_mode", "base_residual")),
            residual_scale=float(_saved_arg(ckpt.get("args", saved_args), "resolved_residual_scale", 1.0)),
            base_residual_gate=float(_saved_arg(ckpt.get("args", saved_args), "base_residual_gate", 1.0)),
        )
        preds = []
        if count == 0:
            base_store.clear()
            target_store.clear()
            mask_store.clear()
            edge_mean_store.clear()
        for batch in tqdm(loader, desc=f"{Path(ckpt_path).parts[-3]} {args.split}"):
            pred = diffusion.sample_ddim(
                batch,
                steps=args.ddim_steps,
                ensemble_size=1,
                start_from_base=True,
                start_ratio=args.start_ratio,
            ).detach().cpu()
            preds.append(pred)
            if count == 0:
                base = torch.clamp(batch["base_height"], -1.0, 1.0).cpu()
                mask = batch["mask"].cpu()
                edge = torch.clamp(batch["edge_score"], 0.0, 1.0).cpu()
                base_store.append(base)
                target_store.append(batch["height_raw"].cpu())
                mask_store.append(mask)
                edge_mean_store.append(masked_mean_scalar(edge, mask))
        pred_tensor = torch.cat(preds, dim=0)
        pred_sum = pred_tensor if pred_sum is None else pred_sum + pred_tensor
        count += 1
        del model, diffusion
        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_diff = pred_sum / float(count)
    base_all = torch.cat(base_store, dim=0)
    target_all = torch.cat(target_store, dim=0)
    mask_all = torch.cat(mask_store, dim=0)
    edge_means = edge_mean_store

    rows = []
    for i in range(avg_diff.shape[0]):
        base = base_all[i:i + 1]
        diff = avg_diff[i:i + 1]
        ensemble = torch.clamp(base + args.alpha * (diff - base), -1.0, 1.0)
        use = edge_means[i] <= args.edge_threshold
        gated = ensemble if use else base
        batch_stub = {
            "depth_minmax": loader.dataset[i]["depth_minmax"].unsqueeze(0),
        }
        base_mm = prediction_to_mm(base, batch_stub, loaders["height_scale"])
        ensemble_mm = prediction_to_mm(ensemble, batch_stub, loaders["height_scale"])
        gated_mm = prediction_to_mm(gated, batch_stub, loaders["height_scale"])
        target = target_all[i:i + 1]
        mask = mask_all[i:i + 1]
        row = {
            "sample": i,
            "use_diffusion": int(use),
            "edge_mean": edge_means[i],
        }
        for prefix, pred_mm in (("base", base_mm), ("ensemble", ensemble_mm), ("gated", gated_mm)):
            metrics = compute_metrics(pred_mm, target, mask=mask)
            row.update({f"{prefix}_{key}": metrics[key] for key in METRIC_KEYS})
        rows.append(row)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rows(rows, out_dir / f"{args.split}_seed_ensemble_gated_metrics.csv")
    summary = {
        "split": args.split,
        "n": len(rows),
        "num_checkpoints": len(args.checkpoints),
        "alpha": args.alpha,
        "edge_threshold": args.edge_threshold,
        "selected": sum(row["use_diffusion"] for row in rows),
    }
    for prefix in ("base", "ensemble", "gated"):
        summary[prefix] = summarize([
            {key: row[f"{prefix}_{key}"] for key in METRIC_KEYS}
            for row in rows
        ])
    with open(out_dir / f"{args.split}_seed_ensemble_gated_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "split": args.split,
        "base_rmse": summary["base"]["rmse"]["mean"],
        "ensemble_rmse": summary["ensemble"]["rmse"]["mean"],
        "gated_rmse": summary["gated"]["rmse"]["mean"],
        "selected": summary["selected"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
