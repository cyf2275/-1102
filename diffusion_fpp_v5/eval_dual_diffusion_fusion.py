from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from data.dataset_fpp_phase import create_fpp_phase_loaders
from diffusion_pip import PIPDiffusion
from eval_adaptive_blend_features import _saved_arg, build_model
from train_fpp_official_style_unet import METRIC_KEYS, parse_channel_spec, summarize
from train_fpp_phase2depth_unet import input_channels, make_input
from train_pip_lite import prediction_to_mm
from models.official_unet import OfficialUNetFPP
from utils.metrics import compute_metrics


def parse_weights(text):
    return [float(x) for x in str(text).replace(",", " ").split() if x]


def masked_mean(x, mask):
    mask = torch.clamp(mask.to(device=x.device, dtype=x.dtype), 0.0, 1.0)
    return (x * mask).sum().detach() / mask.sum().clamp(min=1.0).detach()


def pixel_gate(base, diff, edge, conf, mask, args):
    delta = torch.abs(diff - base)
    mask_f = torch.clamp(mask.to(device=edge.device, dtype=edge.dtype), 0.0, 1.0)
    edge_mean = ((edge * mask_f).flatten(1).sum(dim=1) /
                 mask_f.flatten(1).sum(dim=1).clamp(min=1.0)).view(-1, 1, 1, 1)
    gate = torch.ones_like(base, dtype=torch.bool)
    if args.sample_edge_th < 1.0:
        gate = gate & (edge_mean <= args.sample_edge_th)
    if args.edge_th < 1.0:
        gate = gate & (edge <= args.edge_th)
    if args.delta_min > 0:
        gate = gate & (delta >= args.delta_min)
    if args.conf_min > 0:
        gate = gate & (conf >= args.conf_min)
    return gate.to(dtype=base.dtype)


def build_depth_diffusion(args, device):
    ckpt = torch.load(args.depth_checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})
    include_ftp = bool(_saved_arg(saved_args, "include_ftp", False))
    physics_indices = _saved_arg(saved_args, "physics_channel_indices", None)
    if physics_indices is None:
        physics_indices = parse_channel_spec(str(_saved_arg(saved_args, "physics_channels", "")), include_ftp)
    model_cond_channels = int(ckpt.get("model_cond_channels", len(physics_indices)))
    model = build_model(saved_args, model_cond_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    diffusion = PIPDiffusion(
        model,
        timesteps=int(_saved_arg(saved_args, "timesteps", 200)),
        image_h=args.image_size,
        image_w=args.image_size,
        device=device,
        cond_indices=physics_indices,
        target_mode=str(_saved_arg(saved_args, "target_mode", "base_residual")),
        residual_scale=float(_saved_arg(saved_args, "resolved_residual_scale", 1.0)),
        base_residual_gate=float(_saved_arg(saved_args, "base_residual_gate", 1.0)),
    )
    return diffusion, include_ftp


def build_phase_model(args, device):
    ckpt = torch.load(args.phase_checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})
    mode = str(saved_args.get("input_mode", "phase_pred_plus_fringe"))
    model = OfficialUNetFPP(
        in_channels=input_channels(mode),
        out_channels=1,
        dropout_rate=float(saved_args.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, saved_args, mode


@torch.no_grad()
def evaluate_split(args, split, depth_diffusion, phase_model, phase_mode, loaders_depth, loaders_phase, device):
    weights = parse_weights(args.phase_weights)
    rows_by_weight = {float(w): [] for w in weights}
    base_rows = []
    depth_rows = []
    phase_rows = []
    selected_fracs = []

    depth_loader = loaders_depth[split]
    phase_loader = loaders_phase[split]
    for depth_batch, phase_batch in tqdm(zip(depth_loader, phase_loader), total=len(depth_loader), desc=f"fusion {split}"):
        if not torch.equal(depth_batch["sample_index"], phase_batch["sample_index"]):
            raise RuntimeError("depth and phase loaders are not aligned")
        base = torch.clamp(depth_batch["base_height"].to(device, non_blocking=True), -1.0, 1.0)
        diff = depth_diffusion.sample_ddim(
            depth_batch,
            steps=args.ddim_steps,
            ensemble_size=1,
            start_from_base=True,
            start_ratio=args.start_ratio,
        )
        mask = torch.clamp(depth_batch["mask"].to(device, non_blocking=True), 0.0, 1.0)
        edge = torch.clamp(depth_batch["edge_score"].to(device, non_blocking=True), 0.0, 1.0)
        conf = torch.clamp(depth_batch["phase_conf"].to(device, non_blocking=True), 0.0, 1.0)
        gate = pixel_gate(base, diff, edge, conf, mask, args)
        depth_branch = torch.clamp(base + args.depth_alpha * gate * (diff - base), -1.0, 1.0)
        selected_fracs.extend([
            float(masked_mean(gate[j:j + 1], mask[j:j + 1]).cpu())
            for j in range(gate.shape[0])
        ])

        x_phase = make_input(phase_batch, device, phase_mode)
        phase_pred01 = torch.clamp(phase_model(x_phase), 0.0, 1.0)
        phase_branch = phase_pred01 * 2.0 - 1.0

        target = depth_batch["height_raw"].to(device, non_blocking=True)
        for name, pred, store in (
            ("base", base, base_rows),
            ("depth_branch", depth_branch, depth_rows),
            ("phase_branch", phase_branch, phase_rows),
        ):
            pred_mm = prediction_to_mm(pred, depth_batch, loaders_depth["height_scale"])
            for j in range(pred_mm.shape[0]):
                store.append(compute_metrics(pred_mm[j:j + 1], target[j:j + 1], mask=mask[j:j + 1]))

        for w, rows in rows_by_weight.items():
            fused = torch.clamp((1.0 - w) * depth_branch + w * phase_branch, -1.0, 1.0)
            fused_mm = prediction_to_mm(fused, depth_batch, loaders_depth["height_scale"])
            for j in range(fused_mm.shape[0]):
                rows.append(compute_metrics(fused_mm[j:j + 1], target[j:j + 1], mask=mask[j:j + 1]))

    summaries = {
        "base": summarize(base_rows),
        "depth_branch": summarize(depth_rows),
        "phase_branch": summarize(phase_rows),
        "selected_frac_mean": float(np.mean(selected_fracs)) if selected_fracs else 0.0,
        "weights": {},
    }
    for w, rows in rows_by_weight.items():
        item = summarize(rows)
        item["n"] = len(rows)
        summaries["weights"][str(w)] = item
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_checkpoint", required=True)
    parser.add_argument("--phase_checkpoint", required=True)
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_pspquad_cache_960")
    parser.add_argument("--base_prefix", default="base_c4_adapter")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--start_ratio", type=float, default=0.05)
    parser.add_argument("--depth_alpha", type=float, default=0.7)
    parser.add_argument("--sample_edge_th", type=float, default=0.47)
    parser.add_argument("--edge_th", type=float, default=1.0)
    parser.add_argument("--delta_min", type=float, default=0.12)
    parser.add_argument("--conf_min", type=float, default=0.0)
    parser.add_argument("--phase_weights", default="0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0")
    parser.add_argument("--require_cache", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    depth_diffusion, include_ftp = build_depth_diffusion(args, device)
    phase_model, phase_args, phase_mode = build_phase_model(args, device)
    phase_pred_prefix = phase_args.get("phase_pred_prefix")

    loaders_depth = create_fpp_ml_bench_loaders(
        cache_dir=args.cache_dir,
        batch_size=1,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        include_ftp=include_ftp,
        image_h=args.image_size,
        image_w=args.image_size,
        require_cache=args.require_cache,
        base_prefix=args.base_prefix,
    )
    loaders_phase = create_fpp_phase_loaders(
        base_cache_dir=args.cache_dir,
        phase_cache_dir=args.phase_cache_dir,
        batch_size=1,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        image_h=args.image_size,
        image_w=args.image_size,
        phase_pred_prefix=phase_pred_prefix,
        require_cache=args.require_cache,
    )

    out = {
        "depth_checkpoint": args.depth_checkpoint,
        "phase_checkpoint": args.phase_checkpoint,
        "phase_mode": phase_mode,
        "phase_pred_prefix": phase_pred_prefix,
        "gate": {
            "depth_alpha": args.depth_alpha,
            "sample_edge_th": args.sample_edge_th,
            "edge_th": args.edge_th,
            "delta_min": args.delta_min,
            "conf_min": args.conf_min,
        },
    }
    for split in ("val", "test"):
        out[split] = evaluate_split(args, split, depth_diffusion, phase_model, phase_mode,
                                    loaders_depth, loaders_phase, device)
    best_weight = min(
        out["val"]["weights"],
        key=lambda w: out["val"]["weights"][w]["rmse"]["mean"],
    )
    out["selected_by_val"] = {
        "phase_weight": float(best_weight),
        "val_rmse": out["val"]["weights"][best_weight]["rmse"]["mean"],
        "test_rmse": out["test"]["weights"][best_weight]["rmse"]["mean"],
    }
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "dual_fusion_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out["selected_by_val"], ensure_ascii=False))


if __name__ == "__main__":
    main()
