from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from data.dataset_fpp_phase import create_fpp_phase_loaders
from eval_hierarchical_phase_fusion import (
    aux_predict01,
    build_aux_depth_model,
    build_depth_diffusion,
    masked_mean,
)
from eval_pixel_adaptive_gate import make_gate
from train_pip_lite import prediction_to_mm


def parse_int_list(text: str) -> set[int]:
    return {int(x) for x in str(text).replace(",", " ").split() if x.strip()}


def choose_weight(edge_mean, delta_mean, conf_mean, args) -> float:
    selected = edge_mean >= args.edge_tau
    if args.delta_max >= 0:
        selected = selected and delta_mean <= args.delta_max
    if args.phase_conf_max >= 0:
        selected = selected and conf_mean <= args.phase_conf_max
    return args.high_weight if selected else args.low_weight


def save_visual(path, batch, target_mm, before_mm, after_mm, edge, conf, title):
    fringe = batch["fringe"][0, 0].detach().cpu()
    mask = batch["mask"][0, 0].detach().cpu() > 0.5
    target = target_mm[0, 0].detach().cpu()
    before = before_mm[0, 0].detach().cpu()
    after = after_mm[0, 0].detach().cpu()
    target_disp = torch.where(mask, target, torch.zeros_like(target))
    before_disp = torch.where(mask, before, torch.zeros_like(before))
    after_disp = torch.where(mask, after, torch.zeros_like(after))
    before_err = torch.where(mask, torch.abs(before - target), torch.nan)
    after_err = torch.where(mask, torch.abs(after - target), torch.nan)
    edge_img = edge[0, 0].detach().cpu()
    conf_img = conf[0, 0].detach().cpu()

    vmax_depth = float(torch.nanquantile(target[mask], 0.98)) if mask.any() else 1.0
    vmax_err = float(torch.nanquantile(before_err, 0.95)) if mask.any() else 1.0
    vmax_err = max(vmax_err, 1e-3)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    panels = [
        (fringe, "fringe", "gray", None, None),
        (target_disp, "GT depth", "viridis", 0.0, vmax_depth),
        (before_disp, "D47 depth posterior", "viridis", 0.0, vmax_depth),
        (after_disp, "gated phase fusion", "viridis", 0.0, vmax_depth),
        (before_err, "D47 abs error", "magma", 0.0, vmax_err),
        (after_err, "fused abs error", "magma", 0.0, vmax_err),
        (edge_img, "edge score", "magma", 0.0, 1.0),
        (conf_img, "phase confidence", "viridis", 0.0, 1.0),
    ]
    for ax, (img, name, cmap, vmin, vmax) in zip(axes.flat, panels):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(title + " | background masked to GT")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_checkpoint", required=True)
    parser.add_argument("--phase_depth_checkpoint", required=True)
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_pspquad_cache_960")
    parser.add_argument("--base_prefix", default="base_c4_adapter")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--start_ratio", type=float, default=0.05)
    parser.add_argument("--pixel_alpha", type=float, default=0.7)
    parser.add_argument("--pixel_sample_edge_th", type=float, default=0.47)
    parser.add_argument("--pixel_edge_th", type=float, default=1.0)
    parser.add_argument("--pixel_delta_min", type=float, default=0.12)
    parser.add_argument("--pixel_conf_min", type=float, default=0.0)
    parser.add_argument("--high_edge_min", type=float, default=0.58)
    parser.add_argument("--high_edge_max", type=float, default=0.62)
    parser.add_argument("--high_delta_min", type=float, default=0.09)
    parser.add_argument("--high_delta_max", type=float, default=0.105)
    parser.add_argument("--high_conf_min", type=float, default=0.76)
    parser.add_argument("--high_conf_max", type=float, default=0.80)
    parser.add_argument("--edge_tau", type=float, default=0.42)
    parser.add_argument("--delta_max", type=float, default=0.11)
    parser.add_argument("--phase_conf_max", type=float, default=0.78)
    parser.add_argument("--low_weight", type=float, default=0.0)
    parser.add_argument("--high_weight", type=float, default=0.6)
    parser.add_argument("--require_cache", action="store_true")
    args = parser.parse_args()

    wanted = parse_int_list(args.samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    depth_diffusion, include_ftp = build_depth_diffusion(args, device)
    aux_model, aux_args, aux_kind, aux_mode = build_aux_depth_model(args, device)
    phase_pred_prefix = getattr(aux_args, "phase_pred_prefix", None)

    loaders_depth = create_fpp_ml_bench_loaders(
        cache_dir=args.cache_dir,
        batch_size=1,
        eval_batch_size=1,
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
        eval_batch_size=1,
        num_workers=args.num_workers,
        image_h=args.image_size,
        image_w=args.image_size,
        phase_pred_prefix=phase_pred_prefix,
        require_cache=args.require_cache,
        preload_ram=bool(getattr(aux_args, "preload_ram", False)),
        train_minimal=False,
    )

    pixel_cfg = {
        "alpha": args.pixel_alpha,
        "sample_edge_th": args.pixel_sample_edge_th,
        "edge_th": args.pixel_edge_th,
        "delta_min": args.pixel_delta_min,
        "conf_min": args.pixel_conf_min,
    }

    split_key = "train_eval" if args.split == "train" else args.split
    saved = 0
    for depth_batch, phase_batch in zip(loaders_depth[split_key], loaders_phase[split_key]):
        sample = int(depth_batch["sample_index"][0].item())
        if sample not in wanted:
            continue
        if not torch.equal(depth_batch["sample_index"], phase_batch["sample_index"]):
            raise RuntimeError("depth and phase loaders are not aligned")

        base = torch.clamp(depth_batch["base_height"].to(device, non_blocking=True), -1.0, 1.0)
        diff = depth_diffusion.sample_ddim(
            depth_batch,
            steps=args.ddim_steps,
            ensemble_size=args.ensemble,
            start_from_base=True,
            start_ratio=args.start_ratio,
        )
        mask = torch.clamp(depth_batch["mask"].to(device, non_blocking=True), 0.0, 1.0)
        edge = torch.clamp(depth_batch["edge_score"].to(device, non_blocking=True), 0.0, 1.0)
        conf = torch.clamp(depth_batch["phase_conf"].to(device, non_blocking=True), 0.0, 1.0)
        delta = torch.abs(diff - base)
        pixel_gate = make_gate(base, diff, edge, conf, pixel_cfg, mask=mask)
        pixel_pred = torch.clamp(base + args.pixel_alpha * pixel_gate * (diff - base), -1.0, 1.0)

        edge_mean = float(masked_mean(edge, mask)[0].detach().cpu())
        delta_mean = float(masked_mean(delta, mask)[0].detach().cpu())
        conf_mean = float(masked_mean(conf, mask)[0].detach().cpu())
        high_override = (
            (edge_mean >= args.high_edge_min)
            and (edge_mean <= args.high_edge_max)
            and (delta_mean >= args.high_delta_min)
            and (delta_mean <= args.high_delta_max)
            and (conf_mean >= args.high_conf_min)
            and (conf_mean <= args.high_conf_max)
        )
        if high_override:
            hierarchical = diff
            branch = "diff_high_override"
        elif edge_mean <= args.pixel_sample_edge_th:
            hierarchical = pixel_pred
            branch = "pixel_low_edge"
        else:
            hierarchical = base
            branch = "base"

        phase_branch = aux_predict01(aux_model, phase_batch, device, aux_args, aux_kind, aux_mode) * 2.0 - 1.0
        phase_branch = torch.clamp(phase_branch, -1.0, 1.0)
        weight = choose_weight(edge_mean, delta_mean, conf_mean, args)
        fused = torch.clamp((1.0 - weight) * hierarchical + weight * phase_branch, -1.0, 1.0)

        target_mm = depth_batch["height_raw"].to(device, non_blocking=True)
        before_mm = prediction_to_mm(hierarchical, depth_batch, loaders_depth["height_scale"])
        after_mm = prediction_to_mm(fused, depth_batch, loaders_depth["height_scale"])
        title = (
            f"{args.split} sample {sample} | branch={branch} | w={weight:.2f} | "
            f"edge={edge_mean:.3f} delta={delta_mean:.3f} conf={conf_mean:.3f}"
        )
        save_visual(
            Path(args.save_dir) / f"{args.split}_{sample:03d}_w{weight:.2f}.png",
            depth_batch,
            target_mm,
            before_mm,
            after_mm,
            edge,
            conf,
            title,
        )
        saved += 1
    print(f"saved {saved} visuals to {args.save_dir}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
