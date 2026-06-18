"""Train low-pass CoarseNet and uncertainty for PIP-DiffFPP."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset_pip import create_pip_loaders
from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from diffusion_pip import charbonnier, grad_xy_padded
from models import CoarseLowpassNet, heteroscedastic_l1
from utils.metrics import compute_metrics
from utils.visualization import save_comparison


def low_freq_grad_loss(pred, target):
    pdx, pdy = grad_xy_padded(pred)
    tdx, tdy = grad_xy_padded(target)
    return F.l1_loss(pdx, tdx) + F.l1_loss(pdy, tdy)


def depth_norm_to_metric(depth_norm, batch, height_scale):
    depth01 = torch.clamp((depth_norm + 1.0) * 0.5, 0.0, 1.0)
    if "depth_minmax" not in batch:
        return depth01 * height_scale
    minmax = batch["depth_minmax"].to(depth_norm.device, non_blocking=True)
    dmin = minmax[:, 0].view(-1, 1, 1, 1)
    dmax = minmax[:, 1].view(-1, 1, 1, 1)
    return depth01 * (dmax - dmin).clamp(min=1e-6) + dmin


def masked_heteroscedastic_l1(pred, target, log_var, mask=None):
    if mask is None:
        return heteroscedastic_l1(pred, target, log_var)
    weight = torch.clamp(mask.to(device=pred.device, dtype=pred.dtype), 0.0, 1.0)
    loss = torch.exp(-log_var) * torch.abs(pred - target) + log_var
    return (loss * weight).sum() / weight.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model, loader, device, height_scale):
    model.eval()
    rows = []
    corr_num = corr_den_x = corr_den_y = 0.0
    for idx, batch in enumerate(tqdm(loader, desc="eval coarse")):
        cond = batch["cond"].to(device, non_blocking=True)
        target_low = batch["height_low"].to(device, non_blocking=True)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        pred, log_var = model(cond)
        pred_mm = depth_norm_to_metric(pred, batch, height_scale)
        target_mm = depth_norm_to_metric(target_low, batch, height_scale)
        metrics = compute_metrics(pred_mm, target_mm, mask=mask)
        rows.append(metrics)
        err_map = torch.abs(pred - target_low)
        unc_map = torch.exp(log_var)
        if mask is not None:
            valid = torch.clamp(mask, 0.0, 1.0) > 0.5
            err = err_map[valid]
            unc = unc_map[valid]
        else:
            err = err_map.flatten()
            unc = unc_map.flatten()
        if err.numel() == 0:
            continue
        err = err - err.mean()
        unc = unc - unc.mean()
        corr_num += float((err * unc).sum().item())
        corr_den_x += float((err * err).sum().item())
        corr_den_y += float((unc * unc).sum().item())
    summary = {}
    for key in ("rmse", "mae", "edge_rmse", "normal_deg", "ssim"):
        vals = torch.tensor([r[key] for r in rows], dtype=torch.float32)
        summary[key] = float(vals.mean().item())
    summary["uncertainty_error_corr"] = corr_num / max((corr_den_x * corr_den_y) ** 0.5, 1e-8)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["nguyen", "fpp_ml_bench"], default="nguyen")
    parser.add_argument("--data_dir", default="/root/diffusion_fpp_v5/data")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/diffusion_fpp_pip_cache")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_pspquad_cache_960")
    parser.add_argument("--phase_pred_prefix", default="")
    parser.add_argument("--append_phase_pred_to_cond", action="store_true")
    parser.add_argument("--base_prefix", default="")
    parser.add_argument("--save_dir", default="/root/diffusion_fpp_v5/results/pip_coarse_lowpass")
    parser.add_argument("--include_ftp", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--lambda_grad", type=float, default=0.1)
    parser.add_argument("--lambda_unc", type=float, default=0.05)
    parser.add_argument("--lowpass_factor", type=int, default=8)
    parser.add_argument("--image_h", type=int, default=480)
    parser.add_argument("--image_w", type=int, default=640)
    parser.add_argument("--require_cache", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--train_epoch_repeats", type=int, default=1)
    parser.add_argument("--train_crop_h", type=int, default=0)
    parser.add_argument("--train_crop_w", type=int, default=0)
    args = parser.parse_args()

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
            include_ftp=args.include_ftp,
            image_h=args.image_h,
            image_w=args.image_w,
            lowpass_factor=args.lowpass_factor,
            require_cache=args.require_cache,
            train_epoch_repeats=args.train_epoch_repeats,
            base_prefix=args.base_prefix or None,
            phase_cache_dir=args.phase_cache_dir,
            phase_pred_prefix=args.phase_pred_prefix or None,
            append_phase_pred_to_cond=args.append_phase_pred_to_cond,
            train_crop_h=args.train_crop_h,
            train_crop_w=args.train_crop_w,
        )
    else:
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
            lowpass_factor=args.lowpass_factor,
        )
    model = CoarseLowpassNet(loaders["cond_channels"], base_ch=args.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=(device.type == "cuda"))
    best = float("inf")
    history = []
    print(f"Device: {device}")
    print(f"Cond channels: {loaders['cond_channels']}")
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"coarse {ep}/{args.epochs}"):
            cond = batch["cond"].to(device, non_blocking=True)
            target = batch["height_low"].to(device, non_blocking=True)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                pred, log_var = model(cond)
                loss = charbonnier(pred, target, mask=mask)
                loss = loss + args.lambda_grad * low_freq_grad_loss(pred, target)
                loss = loss + args.lambda_unc * masked_heteroscedastic_l1(pred, target, log_var, mask=mask)
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
            summary = evaluate(model, loaders["val"], device, loaders["height_scale"])
            log.update({f"val_{k}": v for k, v in summary.items()})
            if summary["rmse"] < best:
                best = summary["rmse"]
                torch.save({
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_val_lowpass_rmse": best,
                    "cond_channels": loaders["cond_channels"],
                }, save_dir / "checkpoints" / "best.pt")
                first = next(iter(loaders["val"]))
                pred, _ = model(first["cond"].to(device))
                pred_mm = depth_norm_to_metric(pred, first, loaders["height_scale"])
                target_mm = depth_norm_to_metric(first["height_low"].to(device), first, loaders["height_scale"])
                save_comparison(first["fringe"].to(device), target_mm, pred_mm,
                                save_dir / "visualizations" / f"val_ep{ep:03d}.png",
                                title=f"Coarse low-pass RMSE {best:.2f}mm")
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        torch.save({"epoch": ep, "model_state_dict": model.state_dict(), "args": vars(args)},
                   save_dir / "checkpoints" / "latest.pt")

    best_path = save_dir / "checkpoints" / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        final = {
            "checkpoint": str(best_path),
            "best_val_lowpass_rmse": ckpt.get("best_val_lowpass_rmse"),
            "val": evaluate(model, loaders["val"], device, loaders["height_scale"]),
            "test": evaluate(model, loaders["test"], device, loaders["height_scale"]),
            "args": vars(args),
        }
        with open(save_dir / "coarse_lowpass_summary.json", "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2, ensure_ascii=False)
        print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
