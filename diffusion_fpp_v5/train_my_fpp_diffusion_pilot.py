"""Small-scale residual posterior pilot for the real-capture my_fpp dataset.

This is intentionally a pilot, not the main decision experiment. It should run
after input ablations and compare only raw against the best legal physics input.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset_my_fpp import canonical_input_mode, create_my_fpp_loaders, is_legal_single_frame_mode, smoke_summary
from models import ConditionalUNet
from train_my_fpp_input_ablation import (
    gradient_loss,
    masked_mse,
    prediction_to_height_mm,
    save_checkpoint,
    set_seed,
    summarize,
    train_weight,
    write_eval_outputs,
)
from utils.visualization import save_comparison


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class X0Diffusion:
    def __init__(self, model: ConditionalUNet, timesteps: int, device: torch.device) -> None:
        self.model = model
        self.timesteps = int(timesteps)
        self.device = device
        betas = cosine_beta_schedule(self.timesteps).to(device)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        self.sqrt_acp = torch.sqrt(acp)
        self.sqrt_om = torch.sqrt(1.0 - acp)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self.sqrt_acp[t].view(-1, 1, 1, 1)
        so = self.sqrt_om[t].view(-1, 1, 1, 1)
        return sa * x0 + so * noise

    def training_loss(self, batch: Dict[str, object], args: argparse.Namespace) -> torch.Tensor:
        cond = batch["cond"].to(self.device, non_blocking=True).float()  # type: ignore[index]
        target = batch["height"].to(self.device, non_blocking=True).float()  # type: ignore[index]
        b = target.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.device)
        noisy = self.q_sample(target, t)
        pred = torch.tanh(self.model(noisy, t, cond))
        weight = train_weight(batch, self.device, args.object_mask_weight)
        loss = torch.sqrt((pred - target) ** 2 + 1e-6)
        loss = (loss * weight).sum() / weight.sum().clamp(min=1.0)
        loss = loss + args.lambda_mse * masked_mse(pred, target, weight=weight)
        if args.lambda_grad > 0:
            loss = loss + args.lambda_grad * gradient_loss(pred, target, weight=weight)
        return loss

    @torch.no_grad()
    def sample_ddim(self, batch: Dict[str, object], steps: int = 20, ensemble_size: int = 1) -> torch.Tensor:
        cond = batch["cond"].to(self.device, non_blocking=True).float()  # type: ignore[index]
        b, _, h, w = cond.shape
        outputs = []
        seq = torch.linspace(self.timesteps - 1, 0, steps, device=self.device).long()
        for _ in range(max(1, int(ensemble_size))):
            x = torch.randn((b, 1, h, w), device=self.device)
            for t_val in seq:
                t = torch.full((b,), int(t_val.item()), device=self.device, dtype=torch.long)
                x0 = torch.tanh(self.model(x, t, cond))
                if int(t_val.item()) == 0:
                    x = x0
                    continue
                prev_t = max(int(t_val.item()) - max(1, self.timesteps // max(1, steps)), 0)
                a_prev = self.sqrt_acp[prev_t].view(1, 1, 1, 1)
                om_prev = self.sqrt_om[prev_t].view(1, 1, 1, 1)
                eps = (x - self.sqrt_acp[t].view(-1, 1, 1, 1) * x0) / self.sqrt_om[t].view(-1, 1, 1, 1).clamp(min=1e-6)
                x = a_prev * x0 + om_prev * eps
            outputs.append(torch.clamp(x, -1.0, 1.0))
        return torch.stack(outputs, dim=0).mean(dim=0)


@torch.no_grad()
def evaluate_pilot(diffusion: X0Diffusion, loader, device: torch.device, args: argparse.Namespace, out_dir: Path | None = None, save_images: bool = False) -> List[Dict[str, object]]:
    from train_my_fpp_input_ablation import METRIC_KEYS, metric_row

    diffusion.model.eval()
    rows: List[Dict[str, object]] = []
    if save_images and out_dir is not None:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc="eval diffusion pilot", leave=False):
        pred = diffusion.sample_ddim(batch, steps=args.sample_steps, ensemble_size=args.ensemble_size)
        pred_mm = prediction_to_height_mm(pred, batch)
        target_mm = batch["height_raw"].to(device, non_blocking=True).float()  # type: ignore[index]
        object_mask = batch["object_mask"].to(device, non_blocking=True).float()  # type: ignore[index]
        valid_mask = batch["valid_mask"].to(device, non_blocking=True).float()  # type: ignore[index]
        fringe = batch["fringe"].to(device, non_blocking=True).float()  # type: ignore[index]
        sample_ids = batch["sample_id"]  # type: ignore[index]
        object_ids = batch["object_id"].detach().cpu().numpy().tolist()  # type: ignore[index]
        pose_ids = batch["pose_id"].detach().cpu().numpy().tolist()  # type: ignore[index]
        for j in range(pred.shape[0]):
            object_metrics = metric_row(pred_mm[j:j + 1], target_mm[j:j + 1], object_mask[j:j + 1])
            valid_metrics = metric_row(pred_mm[j:j + 1], target_mm[j:j + 1], valid_mask[j:j + 1])
            row: Dict[str, object] = {
                "sample_id": sample_ids[j],
                "object_id": int(object_ids[j]),
                "pose_id": int(pose_ids[j]),
                "legal_single_frame": bool(args.legal_single_frame),
                "config": args.config,
            }
            for key in METRIC_KEYS:
                row[f"object_{key}"] = object_metrics[key]
                row[f"valid_{key}"] = valid_metrics[key]
            rows.append(row)
            if save_images and out_dir is not None and len(rows) <= args.save_eval_images:
                save_comparison(
                    fringe[j:j + 1],
                    target_mm[j:j + 1],
                    pred_mm[j:j + 1],
                    out_dir / "samples" / f"{len(rows):02d}_{sample_ids[j]}.png",
                    title=f"diffusion pilot object RMSE {object_metrics['rmse']:.3f}mm",
                    mask=object_mask[j:j + 1],
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="my_fpp_dataset_v1")
    parser.add_argument("--processed_dir", default="")
    parser.add_argument("--split_dir", default="")
    parser.add_argument("--save_dir", default="cloud_results/A_20260611_my_fpp_physics_validation/runs/diffusion_pilot_debug")
    parser.add_argument("--config", default="raw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_epoch_repeats", type=int, default=1)
    parser.add_argument("--image_h", type=int, default=240)
    parser.add_argument("--image_w", type=int, default=320)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--ch_mult", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--time_emb_dim", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--ensemble_size", type=int, default=1)
    parser.add_argument("--lambda_mse", type=float, default=0.5)
    parser.add_argument("--lambda_grad", type=float, default=0.10)
    parser.add_argument("--object_mask_weight", type=float, default=3.0)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--save_eval_images", type=int, default=8)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--cache_features", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--smoke_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.config = canonical_input_mode(args.config)
    if args.config not in {"raw", "raw_xy", "raw_single_phys"}:
        raise ValueError("diffusion pilot only supports raw, raw_xy, or raw_single_phys")
    args.input_mode = args.config
    args.legal_single_frame = is_legal_single_frame_mode(args.config)
    args.experiment_role = "small-scale residual posterior pilot"
    if not args.legal_single_frame:
        raise ValueError("diffusion pilot only supports legal single-frame configs")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    loaders = create_my_fpp_loaders(
        data_root=args.data_root,
        processed_dir=args.processed_dir or None,
        split_dir=args.split_dir or None,
        input_mode=args.input_mode,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        image_h=args.image_h,
        image_w=args.image_w,
        train_epoch_repeats=args.train_epoch_repeats,
        cache_features=args.cache_features,
    )
    args.channel_names = loaders["channel_names"]
    args.cond_channels = int(loaders["cond_channels"])
    args.height_stats = loaders["stats"]
    with (save_dir / "loader_smoke_summary.json").open("w", encoding="utf-8") as f:
        json.dump(smoke_summary(loaders), f, indent=2, ensure_ascii=False)
    if args.smoke_only:
        print((save_dir / "loader_smoke_summary.json").read_text(encoding="utf-8"))
        return

    model = ConditionalUNet(
        in_channels=1,
        cond_channels=args.cond_channels,
        out_channels=1,
        base_ch=args.base_channels,
        ch_mult=tuple(args.ch_mult),
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        time_emb_dim=args.time_emb_dim,
    ).to(device)
    diffusion = X0Diffusion(model, timesteps=args.timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))
    best = float("inf")
    history: List[Dict[str, object]] = []
    (save_dir / "checkpoints").mkdir(exist_ok=True)
    (save_dir / "visualizations").mkdir(exist_ok=True)

    print(f"Device: {device}")
    print(f"Config: {args.config} | role={args.experiment_role}")
    print(f"Channels ({args.cond_channels}): {args.channel_names}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"diffusion-pilot {args.config} {ep}/{args.epochs}"):  # type: ignore[index]
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                loss = diffusion.training_loss(batch, args)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item())
            seen += 1
            if args.max_train_batches and seen >= args.max_train_batches:
                break
        scheduler.step()
        log: Dict[str, object] = {
            "epoch": ep,
            "train_loss": total / max(1, seen),
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - t0,
        }
        if ep == 1 or ep % args.eval_every == 0:
            val_rows = evaluate_pilot(diffusion, loaders["val"], device, args)  # type: ignore[index]
            val_summary = summarize(val_rows)
            log["val_object_rmse"] = val_summary["object"]["rmse"]["mean"]  # type: ignore[index]
            log["val_valid_rmse"] = val_summary["valid"]["rmse"]["mean"]  # type: ignore[index]
            val_rmse = float(log["val_object_rmse"])
            if val_rmse < best:
                best = val_rmse
                save_checkpoint(save_dir / "checkpoints" / "best.pt", ep, model, optimizer, scaler, args, best, history)
                first = next(iter(loaders["val"]))  # type: ignore[index]
                pred = diffusion.sample_ddim(first, steps=args.sample_steps, ensemble_size=1)
                pred_mm = prediction_to_height_mm(pred, first)
                save_comparison(
                    first["fringe"].to(device),  # type: ignore[index]
                    first["height_raw"].to(device),  # type: ignore[index]
                    pred_mm,
                    save_dir / "visualizations" / f"val_ep{ep:03d}.png",
                    title=f"diffusion pilot val object RMSE {val_rmse:.3f}mm",
                    mask=first["object_mask"].to(device),  # type: ignore[index]
                )
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        with (save_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        if args.save_every > 0 and (ep == 1 or ep == args.epochs or ep % args.save_every == 0):
            save_checkpoint(save_dir / "checkpoints" / "latest.pt", ep, model, optimizer, scaler, args, best, history)

    best_path = save_dir / "checkpoints" / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    test_rows = evaluate_pilot(diffusion, loaders["test"], device, args, out_dir=save_dir / "evaluation", save_images=True)  # type: ignore[index]
    summary = write_eval_outputs(test_rows, save_dir / "evaluation", best_path, args)
    summary["experiment_role"] = args.experiment_role
    with (save_dir / "evaluation" / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Final test:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
