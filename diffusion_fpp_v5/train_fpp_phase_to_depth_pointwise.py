from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data.dataset_fpp_phase import create_fpp_phase_loaders
from train_fpp_official_style_unet import METRIC_KEYS, HybridL1Loss, prediction_to_mm, summarize
from utils.metrics import compute_metrics
from utils.visualization import save_comparison


class PointwiseDepthDecoder(nn.Module):
    def __init__(self, in_channels=5, hidden=96, layers=4):
        super().__init__()
        layers = max(2, int(layers))
        hidden = int(hidden)
        blocks = []
        ch = in_channels
        for _ in range(layers - 1):
            blocks.extend([nn.Conv2d(ch, hidden, 1), nn.SiLU()])
            ch = hidden
        blocks.append(nn.Conv2d(ch, 1, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


def make_input(batch, device, mode="gt_phase"):
    cond = batch["cond"].to(device, non_blocking=True)
    xy = cond[:, 11:13]
    if mode == "gt_phase":
        phase = batch["phase_target"].to(device, non_blocking=True)
        return torch.cat([phase, xy], dim=1)
    if mode == "phase_pred":
        phase = batch["phase_pred"].to(device, non_blocking=True)
        if phase.shape[1] < 3:
            raise ValueError(f"phase_pred mode requires at least 3 channels, got {phase.shape[1]}")
        return torch.cat([phase[:, :3], xy], dim=1)
    if mode == "ftp_instr":
        ftp_sin = cond[:, 5:6]
        ftp_cos = cond[:, 6:7]
        ftp_residual_01 = torch.clamp(cond[:, 7:8] * 0.5 + 0.5, 0.0, 1.0)
        return torch.cat([ftp_sin, ftp_cos, ftp_residual_01, xy], dim=1)
    if mode == "hilbert_instr":
        h_sin = cond[:, 1:2]
        h_cos = cond[:, 2:3]
        h_residual_01 = torch.clamp(cond[:, 3:4] * 0.5 + 0.5, 0.0, 1.0)
        return torch.cat([h_sin, h_cos, h_residual_01, xy], dim=1)
    raise ValueError(f"unknown input mode: {mode}")


def save_rows(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample"] + METRIC_KEYS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in ["sample"] + METRIC_KEYS})


@torch.no_grad()
def evaluate(model, loader, device, input_mode, out_dir=None, save_images=False):
    model.eval()
    rows = []
    if save_images and out_dir is not None:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc="eval"):
        x = make_input(batch, device, mode=input_mode)
        pred = model(x)
        pred_mm = prediction_to_mm(pred, batch)
        target_raw = batch["height_raw"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        fringe = batch["fringe"].to(device, non_blocking=True)
        for j in range(pred.shape[0]):
            sample_idx = len(rows)
            metrics = compute_metrics(pred_mm[j:j + 1], target_raw[j:j + 1], mask=mask[j:j + 1])
            rows.append({"sample": sample_idx, **metrics})
            if save_images and out_dir is not None and sample_idx < 8:
                save_comparison(
                    fringe[j:j + 1],
                    target_raw[j:j + 1],
                    pred_mm[j:j + 1],
                    out_dir / "samples" / f"sample_{sample_idx:02d}.png",
                    title=f"{input_mode} pointwise RMSE {metrics['rmse']:.2f}mm",
                    mask=mask[j:j + 1],
                )
    return rows


def checkpoint_state(ep, model, optimizer, args, best_val_rmse, history):
    return {
        "epoch": ep,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "best_val_rmse": best_val_rmse,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_phase_cache_960")
    parser.add_argument("--save_dir", default="/root/autodl-tmp/diffusion_fpp_v5/results/fpp960_gt_phase_pointwise_decoder")
    parser.add_argument("--phase_pred_prefix", default=None)
    parser.add_argument("--input_mode", choices=["gt_phase", "phase_pred", "ftp_instr", "hilbert_instr"], default="gt_phase")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--train_crop_h", type=int, default=0)
    parser.add_argument("--train_crop_w", type=int, default=0)
    parser.add_argument("--train_epoch_repeats", type=int, default=1)
    parser.add_argument("--train_subset", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    loaders = create_fpp_phase_loaders(
        base_cache_dir=args.base_cache_dir,
        phase_cache_dir=args.phase_cache_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        image_h=args.image_size,
        image_w=args.image_size,
        train_crop_h=args.train_crop_h,
        train_crop_w=args.train_crop_w,
        train_epoch_repeats=args.train_epoch_repeats,
        train_subset=args.train_subset,
        phase_pred_prefix=args.phase_pred_prefix,
        require_cache=True,
    )
    if args.input_mode == "phase_pred" and int(loaders.get("phase_pred_channels", 0)) < 3:
        raise ValueError("input_mode=phase_pred requires --phase_pred_prefix with at least 3 channels")

    save_dir = Path(args.save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    model = PointwiseDepthDecoder(in_channels=5, hidden=args.hidden, layers=args.layers).to(device)
    criterion = HybridL1Loss(alpha=args.alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    history = []
    best_val_rmse = float("inf")

    print(f"Device: {device}")
    print(f"Input mode: {args.input_mode}")
    print(f"Phase pred prefix: {args.phase_pred_prefix} | channels={loaders.get('phase_pred_channels', 0)}")
    print(f"Train {len(loaders['train'].dataset)} | Val {len(loaders['val'].dataset)} | Test {len(loaders['test'].dataset)}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.4f}M")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"pointwise {ep}/{args.epochs}"):
            x = make_input(batch, device, mode=args.input_mode)
            target = batch["height_01"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += float(loss.item())
            seen += 1
            if args.max_train_batches and seen >= args.max_train_batches:
                break
        log = {
            "epoch": ep,
            "train_loss": total / max(1, seen),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        }
        if ep == 1 or ep % args.eval_every == 0:
            val_rows = evaluate(model, loaders["val"], device, args.input_mode)
            val_summary = summarize(val_rows)
            log.update({f"val_{k}": val_summary[k]["mean"] for k in METRIC_KEYS})
            if val_summary["rmse"]["mean"] < best_val_rmse:
                best_val_rmse = val_summary["rmse"]["mean"]
                torch.save(
                    checkpoint_state(ep, model, optimizer, args, best_val_rmse, history),
                    save_dir / "checkpoints" / "best_rmse.pt",
                )
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    best_path = save_dir / "checkpoints" / "best_rmse.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_rows = evaluate(model, loaders["test"], device, args.input_mode, out_dir=save_dir / "evaluation", save_images=True)
        save_rows(test_rows, save_dir / "evaluation" / "per_sample_metrics.csv")
        summary = summarize(test_rows)
        summary["n"] = len(test_rows)
        summary["checkpoint"] = str(best_path)
        summary["args"] = vars(args)
        with open(save_dir / "evaluation" / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("Final test:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
