"""Train PIP-lite: phase-instructed x0 diffusion on Nguyen/Wang data."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from data.dataset_pip import create_pip_loaders
from diffusion_pip import PIPDiffusion
from models import ConditionalUNet, ConditionalUNetAdapter, PointwisePhaseProjectionHead
from physics_features_pip import FEATURE_ORDER
from train_fpp_official_style_unet import parse_channel_spec
from utils.metrics import compute_metrics
from utils.visualization import save_comparison


METRIC_KEYS = ["rmse", "mae", "edge_rmse", "normal_deg", "ssim"]
HARD_TEST_SAMPLES = {18, 19, 32, 33, 34, 35}


def mean_std(rows, key):
    vals = np.array([r[key] for r in rows], dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)


def summarize(rows):
    return {key: {"mean": mean_std(rows, key)[0], "std": mean_std(rows, key)[1]} for key in METRIC_KEYS}


def save_rows(rows, path):
    keys = ["sample", "hard"] + METRIC_KEYS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in keys})


def load_phase_head(path, device):
    if not path:
        return None
    ckpt = torch.load(path, map_location=device)
    args = ckpt.get("args", {})
    head = PointwisePhaseProjectionHead(
        hidden_dim=int(args.get("hidden_dim", 64)),
        num_layers=int(args.get("num_layers", 4)),
    ).to(device)
    head.load_state_dict(ckpt["model_state_dict"])
    head.phase_depth_input = str(args.get("depth_input", "height_norm"))
    head.raw_depth_center = float(args.get("raw_depth_center", 0.0))
    head.raw_depth_scale = float(args.get("raw_depth_scale", 1.0) or 1.0)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def resolve_residual_scale(args):
    if args.target_mode not in {"residual", "base_residual"}:
        return float(args.residual_scale)
    if args.residual_scale > 0:
        return float(args.residual_scale)
    if not args.base_prefix:
        raise ValueError("--target_mode residual requires --base_prefix")
    stats_path = Path(args.cache_dir) / f"{args.base_prefix}_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"missing residual stats file: {stats_path}. "
            "Run precompute_fpp_base_predictions.py first or pass --residual_scale."
        )
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    for key in ("p99_abs_residual", "residual_scale"):
        value = float(stats.get(key, 0.0))
        if value > 0:
            return value
    raise ValueError(f"{stats_path} does not contain a positive p99_abs_residual/residual_scale")


def zero_initialize_prediction_head(model):
    head = getattr(model, "out", None)
    if isinstance(head, torch.nn.Conv2d):
        torch.nn.init.zeros_(head.weight)
        if head.bias is not None:
            torch.nn.init.zeros_(head.bias)
        return True
    return False


def prediction_to_mm(pred, batch, height_scale):
    pred_01 = torch.clamp((pred + 1.0) * 0.5, 0.0, 1.0)
    if "depth_minmax" in batch:
        minmax = batch["depth_minmax"].to(pred.device, non_blocking=True)
        depth_min = minmax[:, 0].view(-1, 1, 1, 1)
        depth_max = minmax[:, 1].view(-1, 1, 1, 1)
        return pred_01 * (depth_max - depth_min).clamp(min=1e-6) + depth_min
    return pred_01 * float(height_scale)


@torch.no_grad()
def evaluate_split(diffusion, loader, device, height_scale, split_name, args,
                   out_dir=None, save_images=False, guidance=None):
    diffusion.model.eval()
    rows = []
    if save_images and out_dir is not None:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
        (out_dir / "hard_samples").mkdir(parents=True, exist_ok=True)
    for idx, batch in enumerate(tqdm(loader, desc=f"eval {split_name}")):
        fringe = batch["fringe"].to(device, non_blocking=True)
        target_raw = batch["height_raw"].to(device, non_blocking=True)
        use_base_start = (
            bool(getattr(args, "sample_start_from_base", False))
            or getattr(args, "target_mode", "full_x0") == "base_residual"
        )
        pred = diffusion.sample_ddim(batch, steps=args.ddim_steps,
                                      ensemble_size=args.ensemble,
                                      guidance=guidance, progress=False,
                                      start_from_base=use_base_start,
                                      start_ratio=args.sample_start_ratio)
        pred_mm = prediction_to_mm(pred, batch, height_scale)
        metric_mask = batch.get("mask")
        if metric_mask is not None:
            metric_mask = metric_mask.to(device, non_blocking=True)
        for j in range(pred_mm.shape[0]):
            sample_idx = len(rows)
            single_mask = metric_mask[j:j + 1] if metric_mask is not None else None
            metrics = compute_metrics(pred_mm[j:j + 1], target_raw[j:j + 1], mask=single_mask)
            hard = int(getattr(args, "dataset", "nguyen") == "nguyen" and
                       split_name == "test" and sample_idx in HARD_TEST_SAMPLES)
            row = {"sample": sample_idx, "hard": hard, **metrics}
            rows.append(row)
            if save_images and out_dir is not None:
                if sample_idx < 8:
                    save_comparison(fringe[j:j + 1], target_raw[j:j + 1], pred_mm[j:j + 1],
                                    out_dir / "samples" / f"sample_{sample_idx:02d}.png",
                                    title=f"PIP-lite RMSE {metrics['rmse']:.2f}mm",
                                    mask=single_mask)
                if hard:
                    save_comparison(fringe[j:j + 1], target_raw[j:j + 1], pred_mm[j:j + 1],
                                    out_dir / "hard_samples" / f"sample_{sample_idx:02d}.png",
                                    title=f"PIP-lite hard RMSE {metrics['rmse']:.2f}mm",
                                    mask=single_mask)
    return rows


def write_eval_outputs(rows, out_dir, height_scale, checkpoint, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rows(rows, out_dir / "per_sample_metrics.csv")
    hard_rows = [r for r in rows if r["hard"]]
    summary = summarize(rows)
    summary["n"] = len(rows)
    summary["height_scale"] = height_scale
    summary["checkpoint"] = str(checkpoint)
    summary["sampling"] = {
        "ddim_steps": args.ddim_steps,
        "ensemble": args.ensemble,
        "sample_start_from_base": bool(getattr(args, "sample_start_from_base", False)),
        "sample_start_ratio": float(getattr(args, "sample_start_ratio", 1.0)),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    hard_summary = summarize(hard_rows) if hard_rows else {}
    hard_summary["n"] = len(hard_rows)
    hard_summary["hard_samples"] = sorted(HARD_TEST_SAMPLES)
    with open(out_dir / "hard_sample_summary.json", "w", encoding="utf-8") as f:
        json.dump(hard_summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["nguyen", "fpp_ml_bench"], default="nguyen")
    parser.add_argument("--data_dir", default="/root/diffusion_fpp_v5/data")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/diffusion_fpp_pip_cache")
    parser.add_argument("--save_dir", default="/root/diffusion_fpp_v5/results/pip_lite")
    parser.add_argument("--phase_head", default="")
    parser.add_argument("--include_ftp", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random seed. Negative leaves framework defaults unchanged.")
    parser.add_argument("--base_channels", type=int, default=48)
    parser.add_argument("--condition_injection", choices=["concat", "adapter"], default="concat")
    parser.add_argument("--physics_channels", default="",
                        help="Comma/range channel spec in PIP feature order. Empty means all cached channels.")
    parser.add_argument("--adapter_hidden", type=int, default=32)
    parser.add_argument("--target_mode", choices=["full_x0", "residual", "base_residual"], default="full_x0")
    parser.add_argument("--base_prefix", default="",
                        help="Prefix for cached base predictions in FPP-ML-Bench residual mode.")
    parser.add_argument("--residual_scale", type=float, default=0.0,
                        help="Scale for normalized residual target. <=0 reads {base_prefix}_stats.json.")
    parser.add_argument("--base_residual_gate", type=float, default=1.0,
                        help="Multiplier on residual_scale for target_mode=base_residual.")
    parser.add_argument("--disable_zero_residual_init", action="store_true",
                        help="Disable zero output-head initialization for residual target mode.")
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ensemble", type=int, default=3)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save latest.pt every N epochs. Use 0 to disable periodic latest checkpoints.")
    parser.add_argument("--save_epoch_checkpoints", action="store_true",
                        help="When saving latest.pt, also keep an epoch_NNN.pt checkpoint for posterior selection.")
    parser.add_argument("--resume", type=str, default="", help="Optional latest.pt checkpoint to resume from.")
    parser.add_argument("--lambda_oriented", type=float, default=0.08)
    parser.add_argument("--lambda_edge", type=float, default=0.03)
    parser.add_argument("--lambda_normal", type=float, default=0.01)
    parser.add_argument("--lambda_phase", type=float, default=0.05)
    parser.add_argument("--image_h", type=int, default=480)
    parser.add_argument("--image_w", type=int, default=640)
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_phase_cache_960")
    parser.add_argument("--phase_pred_prefix", default="")
    parser.add_argument("--append_phase_pred_to_cond", action="store_true")
    parser.add_argument("--require_cache", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--train_epoch_repeats", type=int, default=1,
                        help="Repeat FPP train samples with replacement within each epoch.")
    parser.add_argument("--train_subset", type=int, default=0,
                        help="Use the first N FPP train samples for overfit diagnostics. 0 uses all.")
    parser.add_argument("--train_crop_size", type=int, default=0,
                        help="Random square crop size for FPP training only. 0 keeps full images.")
    parser.add_argument("--train_crop_h", type=int, default=0,
                        help="Random crop height for FPP training only. Overrides --train_crop_size when >0.")
    parser.add_argument("--train_crop_w", type=int, default=0,
                        help="Random crop width for FPP training only. Overrides --train_crop_size when >0.")
    parser.add_argument("--eval_train_subset", action="store_true",
                        help="Evaluate the training subset instead of val during training.")
    parser.add_argument("--skip_final_test", action="store_true",
                        help="Skip final test evaluation after training.")
    parser.add_argument("--sample_start_from_base", action="store_true",
                        help="Start full_x0 DDIM sampling from cached base_height noised at --sample_start_ratio.")
    parser.add_argument("--sample_start_ratio", type=float, default=1.0)
    parser.add_argument("--train_start_from_base", action="store_true",
                        help="Train full_x0 denoising from cached base_height noised at sampled timesteps.")
    parser.add_argument("--train_t_min_ratio", type=float, default=0.0)
    parser.add_argument("--train_t_max_ratio", type=float, default=1.0)
    parser.add_argument("--base_error_loss_weight", type=float, default=0.0,
                        help="Extra residual loss weight on pixels where cached base prediction is wrong.")
    parser.add_argument("--base_error_loss_gamma", type=float, default=1.0,
                        help="Exponent for normalized cached-base error weighting.")
    parser.add_argument("--low_edge_loss_weight", type=float, default=0.0,
                        help="Extra residual loss weight in low-edge regions, matching the stable D8 gate.")
    parser.add_argument("--low_edge_threshold", type=float, default=0.467,
                        help="Edge threshold used to define low-edge training regions.")
    parser.add_argument("--blend_loss_alpha", type=float, default=0.0,
                        help="If >0 in residual modes, train depth losses on base + alpha*(x0-base).")
    args = parser.parse_args()
    if args.sample_start_from_base and args.target_mode not in {"full_x0", "base_residual"}:
        raise ValueError("--sample_start_from_base is only valid with --target_mode full_x0/base_residual")
    if args.train_start_from_base and args.target_mode not in {"full_x0", "base_residual"}:
        raise ValueError("--train_start_from_base is only valid with --target_mode full_x0/base_residual")
    if (
        args.sample_start_from_base
        or args.train_start_from_base
        or args.target_mode == "base_residual"
    ) and not args.base_prefix:
        raise ValueError("base-start/base_residual modes require --base_prefix")
    args.physics_channel_indices = parse_channel_spec(args.physics_channels, args.include_ftp)
    args.physics_channel_names = (
        [FEATURE_ORDER[idx] for idx in args.physics_channel_indices]
        if args.physics_channel_indices is not None else None
    )
    args.resolved_residual_scale = resolve_residual_scale(args)
    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    save_dir = Path(args.save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    if args.dataset == "fpp_ml_bench":
        loaders = create_fpp_ml_bench_loaders(
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            require_cache=args.require_cache,
            include_ftp=args.include_ftp,
            image_h=args.image_h,
            image_w=args.image_w,
            train_epoch_repeats=args.train_epoch_repeats,
            train_subset=args.train_subset,
            base_prefix=args.base_prefix if (
                args.target_mode in {"residual", "base_residual"}
                or args.sample_start_from_base
                or args.train_start_from_base
            ) else None,
            phase_cache_dir=args.phase_cache_dir,
            phase_pred_prefix=args.phase_pred_prefix or None,
            append_phase_pred_to_cond=args.append_phase_pred_to_cond,
            train_crop_h=args.train_crop_h or args.train_crop_size,
            train_crop_w=args.train_crop_w or args.train_crop_size,
        )
    else:
        if args.target_mode == "residual":
            raise ValueError("residual target mode is currently wired for FPP-ML-Bench base prediction caches")
        loaders = create_pip_loaders(
            args.data_dir,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            cache_dir=args.cache_dir,
            require_cache=args.require_cache,
            include_ftp=args.include_ftp,
            image_h=args.image_h,
            image_w=args.image_w,
        )
    height_scale = loaders["height_scale"]
    cond_channels = loaders["cond_channels"]
    model_cond_channels = len(args.physics_channel_indices) if args.physics_channel_indices is not None else cond_channels
    phase_head = load_phase_head(args.phase_head, device)
    if phase_head is None:
        args.lambda_phase = 0.0

    print(f"Device: {device}")
    print(f"Height scale: {height_scale:.6f}mm")
    print(f"Cond channels: {cond_channels} | model cond channels: {model_cond_channels}")
    print(f"Condition injection: {args.condition_injection}")
    print(f"Target mode: {args.target_mode}")
    if args.target_mode in {"residual", "base_residual"}:
        print(f"Base prefix: {args.base_prefix} | residual_scale={args.resolved_residual_scale:.6f}")
    if args.target_mode == "base_residual":
        print(f"Base residual gate: {args.base_residual_gate:.3f}")
    if args.sample_start_from_base:
        print(f"Sampling starts from cached base: {args.base_prefix} | start_ratio={args.sample_start_ratio:.3f}")
    if args.train_start_from_base or args.target_mode == "base_residual":
        print(
            "Training x_t starts from cached base: "
            f"{args.base_prefix} | t_ratio=[{args.train_t_min_ratio:.3f}, {args.train_t_max_ratio:.3f}]"
        )
    if args.base_error_loss_weight > 0 or args.low_edge_loss_weight > 0:
        print(
            "Residual training weights: "
            f"base_error={args.base_error_loss_weight:.3f}, "
            f"gamma={args.base_error_loss_gamma:.3f}, "
            f"low_edge={args.low_edge_loss_weight:.3f}, "
            f"edge_threshold={args.low_edge_threshold:.3f}"
        )
    if args.physics_channel_indices is not None:
        print(f"Physics channels: {args.physics_channel_indices} | {args.physics_channel_names}")
    print(f"Train {len(loaders['train'].dataset)} | Val {len(loaders['val'].dataset)} | Test {len(loaders['test'].dataset)}")

    if args.condition_injection == "adapter":
        model = ConditionalUNetAdapter(
            cond_channels=model_cond_channels,
            base_ch=args.base_channels,
            ch_mult=(1, 2, 4, 8),
            dropout=0.05,
            adapter_hidden=args.adapter_hidden,
        ).to(device)
    else:
        model = ConditionalUNet(cond_channels=model_cond_channels, base_ch=args.base_channels,
                                ch_mult=(1, 2, 4, 8), dropout=0.05).to(device)
    if args.target_mode in {"residual", "base_residual"} and not args.disable_zero_residual_init and not args.resume:
        if zero_initialize_prediction_head(model):
            print("Residual mode: zero-initialized prediction head, so initial output preserves base depth.")
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    diffusion = PIPDiffusion(
        model,
        timesteps=args.timesteps,
        image_h=args.image_h,
        image_w=args.image_w,
        device=device,
        phase_head=phase_head,
        lambda_oriented=args.lambda_oriented,
        lambda_edge=args.lambda_edge,
        lambda_normal=args.lambda_normal,
        lambda_phase=args.lambda_phase,
        cond_indices=args.physics_channel_indices,
        target_mode=args.target_mode,
        residual_scale=args.resolved_residual_scale,
        base_residual_gate=args.base_residual_gate,
        train_start_from_base=args.train_start_from_base,
        train_t_min_ratio=args.train_t_min_ratio,
        train_t_max_ratio=args.train_t_max_ratio,
        base_error_loss_weight=args.base_error_loss_weight,
        base_error_loss_gamma=args.base_error_loss_gamma,
        low_edge_loss_weight=args.low_edge_loss_weight,
        low_edge_threshold=args.low_edge_threshold,
        blend_loss_alpha=args.blend_loss_alpha,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=(device.type == "cuda"))
    best = float("inf")
    history = []
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            sch.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        best = float(ckpt.get("best_val_rmse", ckpt.get("best", best)))
        history = ckpt.get("history", history)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for ep in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"PIP-lite {ep}/{args.epochs}"):
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                loss = diffusion.p_loss(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.item())
            seen += 1
            if args.max_train_batches and seen >= args.max_train_batches:
                break
        sch.step()
        log = {
            "epoch": ep,
            "train_loss": total / max(1, seen),
            "lr": sch.get_last_lr()[0],
            "seconds": time.time() - t0,
        }

        if ep == 1 or ep % args.eval_every == 0:
            eval_key = "train_eval" if args.eval_train_subset else "val"
            eval_name = "train" if args.eval_train_subset else "val"
            val_rows = evaluate_split(diffusion, loaders[eval_key], device, height_scale, eval_name, args)
            val_summary = summarize(val_rows)
            log.update({f"val_{k}": val_summary[k]["mean"] for k in METRIC_KEYS})
            val_rmse = val_summary["rmse"]["mean"]
            if val_rmse < best:
                best = val_rmse
                ckpt = {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "height_scale": height_scale,
                    "best_val_rmse": best,
                    "cond_channels": cond_channels,
                    "model_cond_channels": model_cond_channels,
                    "include_ftp": args.include_ftp,
                    "phase_head": args.phase_head,
                }
                torch.save(ckpt, save_dir / "checkpoints" / "best.pt")
                print(f"  -> best full-{eval_name} RMSE {best:.3f}mm")
                first = next(iter(loaders[eval_key]))
                pred = diffusion.sample_ddim(
                    first,
                    steps=args.ddim_steps,
                    ensemble_size=args.ensemble,
                    start_from_base=args.sample_start_from_base or args.target_mode == "base_residual",
                    start_ratio=args.sample_start_ratio,
                )
                pred_mm = prediction_to_mm(pred, first, height_scale)
                first_mask = first.get("mask")
                if first_mask is not None:
                    first_mask = first_mask.to(device, non_blocking=True)
                save_comparison(first["fringe"].to(device), first["height_raw"].to(device), pred_mm,
                                save_dir / "visualizations" / f"val_ep{ep:03d}.png",
                                title=f"PIP-lite val RMSE {best:.2f}mm",
                                mask=first_mask)

        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        save_latest = args.save_every > 0 and (
            ep == 1 or ep == args.epochs or ep % args.save_every == 0 or ep % args.eval_every == 0
        )
        if save_latest:
            payload = {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sch.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "args": vars(args),
                "height_scale": height_scale,
                "best_val_rmse": best,
                "history": history,
            }
            torch.save(payload, save_dir / "checkpoints" / "latest.pt")
            if args.save_epoch_checkpoints:
                torch.save(payload, save_dir / "checkpoints" / f"epoch_{ep:03d}.pt")

    best_path = save_dir / "checkpoints" / "best.pt"
    if best_path.exists() and not args.skip_final_test:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_rows = evaluate_split(diffusion, loaders["test"], device, height_scale, "test", args,
                                   out_dir=save_dir / "evaluation", save_images=True)
        summary = write_eval_outputs(test_rows, save_dir / "evaluation", height_scale, best_path, args)
        print("Final test:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
