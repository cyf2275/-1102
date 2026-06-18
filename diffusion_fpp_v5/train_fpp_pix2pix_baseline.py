from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from models.single_frame_baselines import PatchGANDiscriminatorFPP, Pix2PixGeneratorFPP
from train_fpp_official_style_unet import METRIC_KEYS, prediction_to_mm, summarize
from utils.metrics import compute_metrics
from utils.visualization import save_comparison


def save_rows(rows, path):
    keys = ["sample"] + METRIC_KEYS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in keys})


def masked_l1(pred, target, mask, eps=1e-6):
    mask = mask.to(dtype=pred.dtype, device=pred.device)
    return (torch.abs(pred - target) * mask).sum() / mask.sum().clamp_min(eps)


def masked_smooth_l1(pred, target, mask, eps=1e-6):
    mask = mask.to(dtype=pred.dtype, device=pred.device)
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(eps)


def checkpoint_state(ep, gen, disc, opt_g, opt_d, sch_g, sch_d, scaler, args, best_val_rmse, history):
    return {
        "epoch": ep,
        "generator_state_dict": gen.state_dict(),
        "discriminator_state_dict": disc.state_dict(),
        "optimizer_g_state_dict": opt_g.state_dict(),
        "optimizer_d_state_dict": opt_d.state_dict(),
        "scheduler_g_state_dict": sch_g.state_dict(),
        "scheduler_d_state_dict": sch_d.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": vars(args),
        "best_val_rmse": best_val_rmse,
        "history": history,
    }


@torch.no_grad()
def evaluate_metrics(gen, loader, device, args, out_dir=None, save_images=False):
    gen.eval()
    rows = []
    if save_images and out_dir is not None:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc="eval metrics"):
        fringe = batch["fringe"].to(device, non_blocking=True)
        pred = gen(fringe)
        pred_mm = prediction_to_mm(pred, batch)
        target = batch["height_raw"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        for j in range(pred.shape[0]):
            sample_idx = len(rows)
            metrics = compute_metrics(pred_mm[j:j + 1], target[j:j + 1], mask=mask[j:j + 1])
            rows.append({"sample": sample_idx, **metrics})
            if save_images and out_dir is not None and sample_idx < 8:
                save_comparison(
                    fringe[j:j + 1],
                    target[j:j + 1],
                    pred_mm[j:j + 1],
                    out_dir / "samples" / f"sample_{sample_idx:02d}.png",
                    title=f"pix2pix RMSE {metrics['rmse']:.2f}mm",
                    mask=mask[j:j + 1],
                )
    return rows


def write_eval_outputs(rows, out_dir, checkpoint, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rows(rows, out_dir / "per_sample_metrics.csv")
    summary = summarize(rows)
    summary["n"] = len(rows)
    summary["checkpoint"] = str(checkpoint)
    summary["args"] = vars(args)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--gen_channels", type=int, default=64)
    parser.add_argument("--disc_channels", type=int, default=64)
    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_gan", type=float, default=0.01)
    parser.add_argument("--lambda_smooth_l1", type=float, default=0.5)
    parser.add_argument("--label_smooth", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_metrics_every", type=int, default=1)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--require_cache", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=244)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    loaders = create_fpp_ml_bench_loaders(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        image_h=args.image_size,
        image_w=args.image_size,
        require_cache=args.require_cache,
    )

    save_dir = Path(args.save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    gen = Pix2PixGeneratorFPP(1, 1, args.gen_channels, args.dropout).to(device)
    disc = PatchGANDiscriminatorFPP(2, args.disc_channels).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr_g, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_d, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
    sch_g = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_g, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    sch_d = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_d, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))

    print(f"Device: {device}")
    print(f"pix2pix params: G={sum(p.numel() for p in gen.parameters())/1e6:.2f}M D={sum(p.numel() for p in disc.parameters())/1e6:.2f}M")
    print(f"Train {len(loaders['train'].dataset)} | Val {len(loaders['val'].dataset)} | Test {len(loaders['test'].dataset)}")

    history = []
    best_val_rmse = float("inf")
    bce = torch.nn.BCEWithLogitsLoss()
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        gen.train()
        disc.train()
        totals = {"g_loss": 0.0, "d_loss": 0.0, "l1": 0.0, "smooth_l1": 0.0, "gan_g": 0.0}
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"pix2pix {ep}/{args.epochs}"):
            fringe = batch["fringe"].to(device, non_blocking=True)
            target = batch["height_01"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            opt_d.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                fake = gen(fringe).detach()
                real_logits = disc(fringe, target)
                fake_logits = disc(fringe, fake)
                real_labels = torch.full_like(real_logits, args.label_smooth)
                fake_labels = torch.zeros_like(fake_logits)
                d_loss = 0.5 * (bce(real_logits, real_labels) + bce(fake_logits, fake_labels))
            scaler.scale(d_loss).backward()
            scaler.step(opt_d)

            opt_g.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                fake = gen(fringe)
                fake_logits = disc(fringe, fake)
                gan_g = bce(fake_logits, torch.ones_like(fake_logits))
                l1 = masked_l1(fake, target, mask)
                smooth_l1 = masked_smooth_l1(fake, target, mask)
                g_loss = args.lambda_l1 * l1 + args.lambda_smooth_l1 * smooth_l1 + args.lambda_gan * gan_g
            scaler.scale(g_loss).backward()
            scaler.step(opt_g)
            scaler.update()

            totals["g_loss"] += float(g_loss.item())
            totals["d_loss"] += float(d_loss.item())
            totals["l1"] += float(l1.item())
            totals["smooth_l1"] += float(smooth_l1.item())
            totals["gan_g"] += float(gan_g.item())
            seen += 1
            if args.max_train_batches and seen >= args.max_train_batches:
                break

        train = {k: v / max(1, seen) for k, v in totals.items()}
        log = {
            "epoch": ep,
            **{f"train_{k}": v for k, v in train.items()},
            "lr_g": opt_g.param_groups[0]["lr"],
            "lr_d": opt_d.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        }

        if ep == 1 or ep == args.epochs or ep % max(1, args.eval_metrics_every) == 0:
            val_rows = evaluate_metrics(gen, loaders["val"], device, args)
            val_summary = summarize(val_rows)
            log.update({f"val_{k}": val_summary[k]["mean"] for k in METRIC_KEYS})
            val_rmse = val_summary["rmse"]["mean"]
            sch_g.step(val_rmse)
            sch_d.step(val_rmse)
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                torch.save(
                    checkpoint_state(ep, gen, disc, opt_g, opt_d, sch_g, sch_d, scaler, args, best_val_rmse, history),
                    save_dir / "checkpoints" / "best_rmse.pt",
                )

        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    best_path = save_dir / "checkpoints" / "best_rmse.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        gen.load_state_dict(ckpt["generator_state_dict"])
        rows = evaluate_metrics(gen, loaders["test"], device, args, out_dir=save_dir / "evaluation", save_images=True)
        summary = write_eval_outputs(rows, save_dir / "evaluation", best_path, args)
        print("Final test:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
