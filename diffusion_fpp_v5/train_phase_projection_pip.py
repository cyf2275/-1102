"""Train the point-wise depth-to-phase projection head for PIP-DiffFPP."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from data.dataset_pip import create_pip_loaders
from models import PointwisePhaseProjectionHead


def phase_loss(pred, target, conf, mask=None):
    if mask is not None:
        conf = conf * torch.clamp(mask.to(device=conf.device, dtype=conf.dtype), 0.0, 1.0)
    err = torch.abs(pred - target).sum(dim=1, keepdim=True)
    return (err * conf).sum() / conf.sum().clamp(min=1.0)


def estimate_raw_depth_stats(loader):
    raw_min = float("inf")
    raw_max = float("-inf")
    for batch in tqdm(loader, desc="estimate raw depth stats"):
        raw = batch["height_raw"].float()
        mask = batch.get("mask")
        if mask is not None:
            valid = mask > 0.5
            if valid.any():
                vals = raw[valid]
            else:
                vals = raw.reshape(-1)
        else:
            vals = raw.reshape(-1)
        raw_min = min(raw_min, float(vals.min().item()))
        raw_max = max(raw_max, float(vals.max().item()))
    if not torch.isfinite(torch.tensor(raw_min)) or not torch.isfinite(torch.tensor(raw_max)):
        return 0.0, 1.0
    center = 0.5 * (raw_min + raw_max)
    scale = max(0.5 * (raw_max - raw_min), 1.0)
    return float(center), float(scale)


def depth_input(batch, args, device):
    mode = getattr(args, "depth_input", "height_norm")
    if mode == "height_norm":
        return batch["height"].to(device, non_blocking=True)
    if mode == "depth01":
        return batch["height_01"].to(device, non_blocking=True) * 2.0 - 1.0
    if mode == "raw_mm":
        raw = batch["height_raw"].to(device, non_blocking=True)
        center = float(getattr(args, "raw_depth_center", 0.0))
        scale = max(float(getattr(args, "raw_depth_scale", 1.0)), 1e-6)
        return torch.clamp((raw - center) / scale, -2.0, 2.0)
    raise ValueError(f"unknown depth_input: {mode}")


@torch.no_grad()
def evaluate(head, loader, device, args, max_batches=0):
    head.eval()
    rows = []
    for i, batch in enumerate(tqdm(loader, desc="eval P(D)")):
        d = depth_input(batch, args, device)
        xy = batch["xy"].to(device, non_blocking=True)
        target = torch.cat([
            batch["phase_sin"].to(device, non_blocking=True),
            batch["phase_cos"].to(device, non_blocking=True),
        ], dim=1)
        conf = batch["phase_conf"].to(device, non_blocking=True)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        normal = phase_loss(head(d, xy), target, conf, mask=mask)
        zero = phase_loss(head(torch.zeros_like(d), xy), target, conf, mask=mask)
        flat = d.flatten(2)
        perm = torch.randperm(flat.shape[-1], device=device)
        shuffled = flat[:, :, perm].view_as(d)
        shuffled_loss = phase_loss(head(shuffled, xy), target, conf, mask=mask)
        rows.append({
            "sample": i,
            "normal": float(normal.item()),
            "zero_depth": float(zero.item()),
            "shuffled_depth": float(shuffled_loss.item()),
        })
        if max_batches and i + 1 >= max_batches:
            break
    mean = {}
    for key in ("normal", "zero_depth", "shuffled_depth"):
        vals = torch.tensor([r[key] for r in rows], dtype=torch.float32)
        mean[key] = float(vals.mean().item())
    mean["zero_over_normal"] = mean["zero_depth"] / max(mean["normal"], 1e-8)
    mean["shuffled_over_normal"] = mean["shuffled_depth"] / max(mean["normal"], 1e-8)
    return rows, mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["nguyen", "fpp_ml_bench"], default="nguyen")
    parser.add_argument("--data_dir", default="/root/diffusion_fpp_v5/data")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/diffusion_fpp_pip_cache")
    parser.add_argument("--save_dir", default="/root/diffusion_fpp_v5/results/pip_phase_projection")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--depth_input", choices=["height_norm", "depth01", "raw_mm"], default="height_norm")
    parser.add_argument("--raw_depth_center", type=float, default=0.0)
    parser.add_argument("--raw_depth_scale", type=float, default=0.0,
                        help="Only for --depth_input raw_mm. <=0 estimates train min/max and saves center/scale.")
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--image_h", type=int, default=480)
    parser.add_argument("--image_w", type=int, default=640)
    parser.add_argument("--require_cache", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    save_dir = Path(args.save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    if args.dataset == "fpp_ml_bench":
        loaders = create_fpp_ml_bench_loaders(
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            require_cache=args.require_cache,
            include_ftp=False,
            image_h=args.image_h,
            image_w=args.image_w,
        )
    else:
        loaders = create_pip_loaders(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cache_dir=args.cache_dir,
            require_cache=args.require_cache,
            include_ftp=False,
            image_h=args.image_h,
            image_w=args.image_w,
        )
    if args.depth_input == "raw_mm" and args.raw_depth_scale <= 0:
        args.raw_depth_center, args.raw_depth_scale = estimate_raw_depth_stats(loaders["train"])
    elif args.depth_input != "raw_mm":
        args.raw_depth_center, args.raw_depth_scale = 0.0, 1.0
    head = PointwisePhaseProjectionHead(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=(device.type == "cuda"))
    best = float("inf")
    history = []
    print(f"Device: {device}")
    print(f"Params: {sum(p.numel() for p in head.parameters())}")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        head.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"P(D) {ep}/{args.epochs}"):
            d = depth_input(batch, args, device)
            xy = batch["xy"].to(device, non_blocking=True)
            target = torch.cat([
                batch["phase_sin"].to(device, non_blocking=True),
                batch["phase_cos"].to(device, non_blocking=True),
            ], dim=1)
            conf = batch["phase_conf"].to(device, non_blocking=True)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                loss = phase_loss(head(d, xy), target, conf, mask=mask)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
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
            _, mean = evaluate(head, loaders["val"], device, args)
            log.update({f"val_{k}": v for k, v in mean.items()})
            if mean["normal"] < best:
                best = mean["normal"]
                torch.save({
                    "epoch": ep,
                    "model_state_dict": head.state_dict(),
                    "args": vars(args),
                    "best_val_phase_loss": best,
                }, save_dir / "checkpoints" / "best.pt")
                print(f"  -> best val phase loss {best:.6f}")
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        torch.save({"epoch": ep, "model_state_dict": head.state_dict(), "args": vars(args)},
                   save_dir / "checkpoints" / "latest.pt")

    best_path = save_dir / "checkpoints" / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        head.load_state_dict(ckpt["model_state_dict"])
    rows, mean = evaluate(head, loaders["val"], device, args)
    with open(save_dir / "scrambled_depth_test.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "normal", "zero_depth", "shuffled_depth"])
        writer.writeheader()
        writer.writerows(rows)
    with open(save_dir / "scrambled_depth_summary.json", "w", encoding="utf-8") as f:
        json.dump(mean, f, indent=2, ensure_ascii=False)
    print("Scrambled depth test:")
    print(json.dumps(mean, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
