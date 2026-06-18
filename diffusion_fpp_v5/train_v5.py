"""Train fringe-only physics-conditioned full-height diffusion v5."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data import create_loaders
from diffusion import PhysicsConditionedDiffusion
from models import CoarsePredictor, ConditionalUNet
from utils.metrics import compute_metrics
from utils.visualization import save_comparison


def append_coarse(cond, coarse_model):
    with torch.no_grad():
        coarse = coarse_model(cond)
    return torch.cat([cond, coarse], dim=1), coarse


@torch.no_grad()
def evaluate_subset(diffusion, coarse_model, loader, device, height_scale, args, max_samples=4, save_dir=None, tag="val"):
    diffusion.model.eval()
    if coarse_model is not None:
        coarse_model.eval()
    rows = []
    for idx, batch in enumerate(loader):
        if idx >= max_samples:
            break
        cond = batch["cond"].to(device, non_blocking=True)
        target = batch["height"].to(device, non_blocking=True)
        target_raw = batch["height_raw"].to(device, non_blocking=True)
        fringe = batch["fringe"].to(device, non_blocking=True)
        coarse = None
        if coarse_model is not None:
            cond, coarse = append_coarse(cond, coarse_model)
        pred = diffusion.sample_ddim(cond, steps=args.ddim_steps, ensemble_size=args.ensemble,
                                     coarse=coarse, start_ratio=args.start_ratio, progress=False)
        pred_mm = torch.clamp((pred + 1.0) * 0.5, 0.0, 1.0) * height_scale
        metrics = compute_metrics(pred_mm, target_raw)
        rows.append(metrics)
        if save_dir is not None and idx == 0:
            save_comparison(fringe, target_raw, pred_mm, save_dir / f"{tag}_sample.png",
                            title=f"{tag} RMSE {metrics['rmse']:.2f}mm")
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}


def train_coarse(model, loaders, device, save_dir, epochs, lr, height_scale, max_batches=0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    best = float("inf")
    ckpt = save_dir / "checkpoints" / "coarse_best.pt"
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"Coarse {ep}/{epochs}"):
            cond = batch["cond"].to(device, non_blocking=True)
            target = batch["height"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(cond)
            loss = F.l1_loss(pred, target) + 0.5 * F.mse_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            seen += 1
            if max_batches and seen >= max_batches:
                break
        sch.step()

        val = evaluate_coarse(model, loaders["val"], device, height_scale)
        denom = max(1, seen)
        print(f"Coarse {ep:03d} | train={total/denom:.5f} | val_rmse={val:.2f}mm")
        if val < best:
            best = val
            torch.save({"model_state_dict": model.state_dict(), "height_scale": height_scale,
                        "best_rmse": best, "epoch": ep}, ckpt)
            print(f"  -> best coarse {best:.2f}mm")
    if ckpt.exists():
        model.load_state_dict(torch.load(str(ckpt), map_location=device)["model_state_dict"])
    return best


@torch.no_grad()
def evaluate_coarse(model, loader, device, height_scale):
    model.eval()
    rmses = []
    for batch in loader:
        cond = batch["cond"].to(device, non_blocking=True)
        target_raw = batch["height_raw"].to(device, non_blocking=True)
        pred = model(cond)
        pred_mm = torch.clamp((pred + 1.0) * 0.5, 0.0, 1.0) * height_scale
        rmses.append(float(torch.sqrt(torch.mean((pred_mm - target_raw) ** 2)).item()))
    return float(np.mean(rmses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/root/diffusion_fpp_v5/data")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/diffusion_fpp_v5_cache")
    parser.add_argument("--save_dir", default="/root/diffusion_fpp_v5/results/fringe_physics")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--coarse_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--coarse_lr", type=float, default=1e-3)
    parser.add_argument("--base_channels", type=int, default=48)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ensemble", type=int, default=3)
    parser.add_argument("--start_ratio", type=float, default=0.55)
    parser.add_argument("--lambda_grad", type=float, default=0.2)
    parser.add_argument("--lambda_fft", type=float, default=0.05)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--eval_samples", type=int, default=4)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_coarse_batches", type=int, default=0)
    parser.add_argument("--no_coarse", action="store_true")
    parser.add_argument("--require_cache", action="store_true")
    parser.add_argument("--image_h", type=int, default=480)
    parser.add_argument("--image_w", type=int, default=640)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    save_dir = Path(args.save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    loaders = create_loaders(args.data_dir, batch_size=args.batch_size,
                             num_workers=args.num_workers, image_h=args.image_h,
                             image_w=args.image_w, cache_dir=args.cache_dir,
                             require_cache=args.require_cache)
    height_scale = loaders["height_scale"]
    print(f"Device: {device}")
    print(f"Height scale: {height_scale:.3f}mm")
    print(f"Train {len(loaders['train'].dataset)} | Val {len(loaders['val'].dataset)} | Test {len(loaders['test'].dataset)}")

    coarse_model = None
    cond_channels = 7
    if not args.no_coarse:
        coarse_model = CoarsePredictor(in_channels=7, base_ch=32).to(device)
        if args.coarse_epochs > 0:
            train_coarse(coarse_model, loaders, device, save_dir, args.coarse_epochs,
                         args.coarse_lr, height_scale, max_batches=args.max_coarse_batches)
        coarse_model.eval().requires_grad_(False)
        cond_channels = 8

    model = ConditionalUNet(cond_channels=cond_channels, base_ch=args.base_channels,
                            ch_mult=(1, 2, 4, 8), dropout=0.05).to(device)
    print(f"Diffusion params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    diffusion = PhysicsConditionedDiffusion(model, timesteps=args.timesteps,
                                            image_h=args.image_h, image_w=args.image_w,
                                            device=device, lambda_grad=args.lambda_grad,
                                            lambda_fft=args.lambda_fft)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=(device.type == "cuda"))
    best = float("inf")
    history = []

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"Diff {ep}/{args.epochs}"):
            cond = batch["cond"].to(device, non_blocking=True)
            target = batch["height"].to(device, non_blocking=True)
            if coarse_model is not None:
                cond, _ = append_coarse(cond, coarse_model)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                loss = diffusion.p_loss(target, cond)
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

        log = {"epoch": ep, "train_loss": total / max(1, seen),
               "lr": sch.get_last_lr()[0], "seconds": time.time() - t0}
        if ep == 1 or ep % args.eval_every == 0:
            metrics = evaluate_subset(diffusion, coarse_model, loaders["val"], device,
                                      height_scale, args, max_samples=args.eval_samples,
                                      save_dir=save_dir / "visualizations", tag=f"val_ep{ep:03d}")
            log.update({f"val_{k}": v for k, v in metrics.items()})
            if metrics and metrics["rmse"] < best:
                best = metrics["rmse"]
                torch.save({
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "coarse_state_dict": coarse_model.state_dict() if coarse_model is not None else None,
                    "args": vars(args),
                    "height_scale": height_scale,
                    "best_val_rmse": best,
                }, save_dir / "checkpoints" / "best.pt")
                print(f"  -> best val rmse {best:.2f}mm")
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"Done. Best sampled val RMSE: {best:.2f}mm")


if __name__ == "__main__":
    main()
