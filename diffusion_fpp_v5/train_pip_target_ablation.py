"""Full x0 vs normalized residual vs hybrid target ablation for PIP-lite."""
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
from diffusion_pip import (
    PIPDiffusion,
    charbonnier,
    confidence_edge_loss,
    normal_loss,
    oriented_gradient_loss,
)
from models import ConditionalUNet, PointwisePhaseProjectionHead
from train_pip_lite import evaluate_split, load_phase_head, write_eval_outputs, METRIC_KEYS, summarize
from utils.visualization import save_comparison


@torch.no_grad()
def estimate_residual_scale(loader, device, max_batches=0):
    vals = []
    for i, batch in enumerate(tqdm(loader, desc="estimate residual scale")):
        h = batch["height"].to(device)
        low = batch["height_low"].to(device)
        vals.append(torch.abs(h - low).flatten().detach().cpu())
        if max_batches and i + 1 >= max_batches:
            break
    all_vals = torch.cat(vals)
    return float(torch.quantile(all_vals, 0.99).clamp(min=1e-3).item())


def make_target(batch, target_mode, residual_scale):
    height = batch["height"]
    low = batch["height_low"]
    if target_mode == "full":
        return height
    if target_mode == "residual":
        return torch.clamp((height - low) / residual_scale, -1.0, 1.0)
    if target_mode == "hybrid":
        return height
    raise ValueError(f"unknown target mode: {target_mode}")


def final_from_model_output(out, batch, target_mode, residual_scale):
    if target_mode == "residual":
        return torch.clamp(batch["height_low"].to(out.device) + out * residual_scale, -1.0, 1.0)
    return out


def target_loss(diffusion, batch, target_mode, residual_scale, lambda_hybrid=0.05):
    cond = batch["cond"].to(diffusion.device, non_blocking=True)
    target = make_target(batch, target_mode, residual_scale).to(diffusion.device, non_blocking=True)
    b = target.shape[0]
    t = torch.randint(0, diffusion.timesteps, (b,), device=diffusion.device, dtype=torch.long)
    x_t = diffusion.q_sample(target, t)
    out = diffusion.model(x_t, t, cond)
    final = final_from_model_output(out, batch, target_mode, residual_scale)
    height = batch["height"].to(diffusion.device, non_blocking=True)
    loss = charbonnier(final, height) + 0.5 * F.mse_loss(final, height)
    loss = loss + diffusion.lambda_oriented * oriented_gradient_loss(
        final, height,
        batch["phase_sin"].to(diffusion.device, non_blocking=True),
        batch["phase_cos"].to(diffusion.device, non_blocking=True),
        batch["phase_conf"].to(diffusion.device, non_blocking=True),
    )
    loss = loss + diffusion.lambda_edge * confidence_edge_loss(
        final, height,
        batch["edge_score"].to(diffusion.device, non_blocking=True),
        batch["phase_conf"].to(diffusion.device, non_blocking=True),
    )
    loss = loss + diffusion.lambda_normal * normal_loss(final, height)
    if diffusion.lambda_phase > 0 and diffusion.phase_head is not None:
        loss = loss + diffusion.lambda_phase * diffusion.phase_consistency_loss(final, batch)
    if target_mode == "hybrid":
        residual = torch.clamp((height - batch["height_low"].to(diffusion.device)) / residual_scale, -1.0, 1.0)
        pred_residual = torch.clamp((final - batch["height_low"].to(diffusion.device)) / residual_scale, -1.0, 1.0)
        loss = loss + float(lambda_hybrid) * F.l1_loss(pred_residual, residual)
    return loss


class TargetAblationSampler:
    def __init__(self, diffusion, target_mode, residual_scale):
        self.diffusion = diffusion
        self.model = diffusion.model
        self.target_mode = target_mode
        self.residual_scale = residual_scale

    @torch.no_grad()
    def sample_ddim(self, batch, steps=50, ensemble_size=1, progress=False):
        raw = self.diffusion.sample_ddim(batch, steps=steps, ensemble_size=ensemble_size, progress=progress)
        return final_from_model_output(raw, batch, self.target_mode, self.residual_scale)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/root/diffusion_fpp_v5/data")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/diffusion_fpp_pip_cache")
    parser.add_argument("--save_dir", default="/root/diffusion_fpp_v5/results/pip_target_ablation")
    parser.add_argument("--target_mode", choices=["full", "residual", "hybrid"], default="full")
    parser.add_argument("--phase_head", default="")
    parser.add_argument("--include_ftp", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=48)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ensemble", type=int, default=3)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--lambda_oriented", type=float, default=0.08)
    parser.add_argument("--lambda_edge", type=float, default=0.03)
    parser.add_argument("--lambda_normal", type=float, default=0.01)
    parser.add_argument("--lambda_phase", type=float, default=0.05)
    parser.add_argument("--lambda_hybrid", type=float, default=0.05)
    parser.add_argument("--image_h", type=int, default=480)
    parser.add_argument("--image_w", type=int, default=640)
    parser.add_argument("--require_cache", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    save_dir = Path(args.save_dir) / args.target_mode
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    loaders = create_pip_loaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
        cache_dir=args.cache_dir, require_cache=args.require_cache,
        include_ftp=args.include_ftp, image_h=args.image_h, image_w=args.image_w)
    height_scale = loaders["height_scale"]
    residual_scale = estimate_residual_scale(loaders["train"], device, max_batches=0)
    phase_head = load_phase_head(args.phase_head, device) if args.phase_head else None
    if phase_head is None:
        args.lambda_phase = 0.0
    model = ConditionalUNet(cond_channels=loaders["cond_channels"], base_ch=args.base_channels,
                            ch_mult=(1, 2, 4, 8), dropout=0.05).to(device)
    diffusion = PIPDiffusion(
        model, timesteps=args.timesteps, image_h=args.image_h, image_w=args.image_w,
        device=device, phase_head=phase_head, lambda_oriented=args.lambda_oriented,
        lambda_edge=args.lambda_edge, lambda_normal=args.lambda_normal,
        lambda_phase=args.lambda_phase)
    sampler = TargetAblationSampler(diffusion, args.target_mode, residual_scale)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=(device.type == "cuda"))
    best = float("inf")
    history = []
    print(f"Device: {device}")
    print(f"Target mode: {args.target_mode} | residual_scale={residual_scale:.6f}")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"{args.target_mode} {ep}/{args.epochs}"):
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                loss = target_loss(diffusion, batch, args.target_mode, residual_scale, args.lambda_hybrid)
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
        log = {"epoch": ep, "train_loss": total / max(1, seen), "lr": sch.get_last_lr()[0], "seconds": time.time() - t0}
        if ep == 1 or ep % args.eval_every == 0:
            val_rows = evaluate_split(sampler, loaders["val"], device, height_scale, "val", args)
            summary = summarize(val_rows)
            log.update({f"val_{k}": summary[k]["mean"] for k in METRIC_KEYS})
            if summary["rmse"]["mean"] < best:
                best = summary["rmse"]["mean"]
                torch.save({
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "height_scale": height_scale,
                    "residual_scale": residual_scale,
                    "best_val_rmse": best,
                }, save_dir / "checkpoints" / "best.pt")
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        torch.save({"epoch": ep, "model_state_dict": model.state_dict(), "args": vars(args),
                    "residual_scale": residual_scale}, save_dir / "checkpoints" / "latest.pt")

    best_path = save_dir / "checkpoints" / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        rows = evaluate_split(sampler, loaders["test"], device, height_scale, "test", args,
                              out_dir=save_dir / "evaluation", save_images=True)
        summary = write_eval_outputs(rows, save_dir / "evaluation", height_scale, best_path, args)
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
