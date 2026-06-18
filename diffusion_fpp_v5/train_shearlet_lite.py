"""Train Nguyen/Wang x0 diffusion with shearlet-lite physical instructions."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset_v35 import create_v35_loaders
from diffusion_v35 import PhaseEdgeDiffusion
from models import ConditionalUNet
from precompute_features_shearlet_lite import FEATURE_ORDER
from utils.metrics import compute_metrics
from utils.visualization import save_comparison


METRIC_KEYS = ["rmse", "mae", "edge_rmse", "normal_deg", "ssim"]
HARD_TEST_SAMPLES = {18, 19, 32, 33, 34, 35}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


@torch.no_grad()
def evaluate_split(diffusion, loader, device, height_scale, split_name, args, out_dir=None, save_images=False):
    diffusion.model.eval()
    rows = []
    if save_images and out_dir is not None:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
        (out_dir / "hard_samples").mkdir(parents=True, exist_ok=True)
    for idx, batch in enumerate(tqdm(loader, desc=f"eval {split_name}")):
        cond = batch["cond"].to(device, non_blocking=True)
        fringe = batch["fringe"].to(device, non_blocking=True)
        target_raw = batch["height_raw"].to(device, non_blocking=True)
        pred = diffusion.sample_ddim(cond, steps=args.ddim_steps, ensemble_size=args.ensemble, progress=False)
        pred_mm = torch.clamp((pred + 1.0) * 0.5, 0.0, 1.0) * height_scale
        metrics = compute_metrics(pred_mm, target_raw)
        hard = int(split_name == "test" and idx in HARD_TEST_SAMPLES)
        rows.append({"sample": idx, "hard": hard, **metrics})
        if save_images and out_dir is not None:
            if idx < 8:
                save_comparison(
                    fringe,
                    target_raw,
                    pred_mm,
                    out_dir / "samples" / f"sample_{idx:02d}.png",
                    title=f"shearlet-lite RMSE {metrics['rmse']:.2f}mm",
                )
            if hard:
                save_comparison(
                    fringe,
                    target_raw,
                    pred_mm,
                    out_dir / "hard_samples" / f"sample_{idx:02d}.png",
                    title=f"shearlet-lite hard RMSE {metrics['rmse']:.2f}mm",
                )
    return rows


def write_eval_outputs(rows, out_dir, height_scale, checkpoint, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rows(rows, out_dir / "per_sample_metrics.csv")
    hard_rows = [r for r in rows if r["hard"]]
    summary = summarize(rows)
    summary["n"] = len(rows)
    summary["height_scale"] = height_scale
    summary["checkpoint"] = str(checkpoint)
    summary["sampling"] = {"ddim_steps": args.ddim_steps, "ensemble": args.ensemble}
    summary["feature_order"] = FEATURE_ORDER
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
    parser.add_argument("--data_dir", default="/root/diffusion_fpp_v5/data")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/diffusion_fpp_shearlet_lite_cache")
    parser.add_argument("--cache_prefix", default="physics_shearlet_lite")
    parser.add_argument("--save_dir", default="/root/diffusion_fpp_v5/results/shearlet_lite_nguyen")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=48)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ensemble", type=int, default=3)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--lambda_grad", type=float, default=0.2)
    parser.add_argument("--lambda_edge", type=float, default=0.12)
    parser.add_argument("--lambda_normal", type=float, default=0.04)
    parser.add_argument("--image_h", type=int, default=480)
    parser.add_argument("--image_w", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--require_cache", action="store_true")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    save_dir = Path(args.save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    with open(save_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    loaders = create_v35_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        cache_dir=args.cache_dir,
        require_cache=args.require_cache,
        image_h=args.image_h,
        image_w=args.image_w,
        cache_prefix=args.cache_prefix,
        seed=args.seed,
    )
    height_scale = loaders["height_scale"]
    first = next(iter(loaders["train"]))
    cond_channels = int(first["cond"].shape[1])
    print(f"Device: {device}")
    print(f"Height scale: {height_scale:.6f}mm")
    print(f"Condition channels: {cond_channels}")
    print(f"Train {len(loaders['train'].dataset)} | Val {len(loaders['val'].dataset)} | Test {len(loaders['test'].dataset)}")

    model = ConditionalUNet(cond_channels=cond_channels, base_ch=args.base_channels, ch_mult=(1, 2, 4, 8), dropout=0.05).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    diffusion = PhaseEdgeDiffusion(
        model,
        timesteps=args.timesteps,
        image_h=args.image_h,
        image_w=args.image_w,
        device=device,
        lambda_grad=args.lambda_grad,
        lambda_edge=args.lambda_edge,
        lambda_normal=args.lambda_normal,
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
        if "scaler_state_dict" in ckpt and device.type == "cuda":
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        best = float(ckpt.get("best_val_rmse", best))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        hist_path = Path(args.resume).parents[1] / "history.json"
        if hist_path.exists():
            history = json.loads(hist_path.read_text(encoding="utf-8"))
        print(f"Resumed from {args.resume} at epoch {start_epoch}; best={best:.3f}")

    for ep in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"shearlet-lite {ep}/{args.epochs}"):
            cond = batch["cond"].to(device, non_blocking=True)
            target = batch["height"].to(device, non_blocking=True)
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

        log = {
            "epoch": ep,
            "train_loss": total / max(1, seen),
            "lr": sch.get_last_lr()[0],
            "seconds": time.time() - t0,
        }
        if ep == 1 or ep % args.eval_every == 0:
            val_rows = evaluate_split(diffusion, loaders["val"], device, height_scale, "val", args)
            val_summary = summarize(val_rows)
            log.update({f"val_{k}": val_summary[k]["mean"] for k in METRIC_KEYS})
            val_rmse = val_summary["rmse"]["mean"]
            if val_rmse < best:
                best = val_rmse
                torch.save(
                    {
                        "epoch": ep,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "scheduler_state_dict": sch.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "args": vars(args),
                        "height_scale": height_scale,
                        "best_val_rmse": best,
                        "feature_order": FEATURE_ORDER,
                    },
                    save_dir / "checkpoints" / "best.pt",
                )
                print(f"  -> best full-val RMSE {best:.3f}mm")
                first_val = next(iter(loaders["val"]))
                cond = first_val["cond"].to(device)
                fringe = first_val["fringe"].to(device)
                target_raw = first_val["height_raw"].to(device)
                pred = diffusion.sample_ddim(cond, steps=args.ddim_steps, ensemble_size=args.ensemble)
                pred_mm = torch.clamp((pred + 1.0) * 0.5, 0.0, 1.0) * height_scale
                save_comparison(
                    fringe,
                    target_raw,
                    pred_mm,
                    save_dir / "visualizations" / f"val_ep{ep:03d}.png",
                    title=f"shearlet-lite val RMSE {best:.2f}mm",
                )

        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        torch.save(
            {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sch.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "args": vars(args),
                "height_scale": height_scale,
                "best_val_rmse": best,
                "feature_order": FEATURE_ORDER,
            },
            save_dir / "checkpoints" / "latest.pt",
        )

    best_path = save_dir / "checkpoints" / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_rows = evaluate_split(
            diffusion,
            loaders["test"],
            device,
            height_scale,
            "test",
            args,
            out_dir=save_dir / "evaluation",
            save_images=True,
        )
        summary = write_eval_outputs(test_rows, save_dir / "evaluation", height_scale, best_path, args)
        print("Final test:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
