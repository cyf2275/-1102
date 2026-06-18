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
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset_fpp_phase import create_fpp_phase_loaders
from models import OfficialUNetFPPAdapter
from train_fpp_official_style_unet import HybridL1Loss, METRIC_KEYS, prediction_to_mm, summarize
from utils.metrics import compute_metrics
from utils.visualization import save_comparison


DEFAULT_PHASE_PROXY_COEFFS = [
    45.452732235850426,
    0.030703432236726596,
    -0.001926472379255077,
    66.69305714209537,
    -5.830587969787353e-06,
    -2.542049065133751e-06,
    2.4473742115755103e-05,
    -0.0025881002126868897,
    0.00997907576305477,
    0.003892213471562195,
]
PHASE_PROXY_METRIC_KEYS = ["phase_proxy_mae_rad", "phase_proxy_rmse_rad"]


def parse_indices(spec):
    if spec is None:
        return []
    text = str(spec).strip().lower()
    if text in ("", "none"):
        return []
    selected = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_text, hi_text = part.split("-", 1)
            lo, hi = int(lo_text), int(hi_text)
            selected.extend(range(lo, hi + 1))
        else:
            selected.append(int(part))
    out = []
    for idx in selected:
        if idx not in out:
            out.append(idx)
    return out


def make_cond(batch, device, mode, instr_indices):
    cond_parts = []
    if mode.startswith("phase_pred"):
        if "phase_pred" not in batch:
            raise ValueError("phase_pred cond mode requires --phase_pred_prefix")
        phase = batch["phase_pred"].to(device, non_blocking=True)
        cond_parts.append(phase[:, :3])
    elif mode.startswith("gt_psp"):
        cond_parts.append(batch["phase_target"].to(device, non_blocking=True)[:, :3])
    else:
        raise ValueError(f"unknown cond mode: {mode}")

    instr = batch["cond"].to(device, non_blocking=True)
    if mode.endswith("_instr_xy"):
        if instr_indices:
            cond_parts.append(instr[:, instr_indices])
        cond_parts.append(instr[:, 11:13])
    elif mode.endswith("_instr"):
        cond_parts.append(instr[:, instr_indices])
    elif mode.endswith("_xy"):
        cond_parts.append(instr[:, 11:13])
    return torch.cat(cond_parts, dim=1)


def cond_channel_count(mode, instr_indices):
    if mode.startswith("phase_pred") or mode.startswith("gt_psp"):
        count = 3
    else:
        raise ValueError(f"unknown cond mode: {mode}")
    if mode.endswith("_instr_xy"):
        count += len(instr_indices) + 2
    elif mode.endswith("_instr"):
        count += len(instr_indices)
    elif mode.endswith("_xy"):
        count += 2
    return count


def parse_float_list(spec):
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        return [float(v) for v in spec]
    return [float(x) for x in str(spec).replace(",", " ").split() if x]


def load_phase_proxy_coeffs(args):
    if args.phase_proxy_summary:
        with open(args.phase_proxy_summary, "r", encoding="utf-8") as f:
            summary = json.load(f)
        coeffs = summary["models"]["depth_xy_to_phase"]["coef"]
    else:
        coeffs = parse_float_list(args.phase_proxy_coeffs)
        if not coeffs:
            coeffs = DEFAULT_PHASE_PROXY_COEFFS
    if len(coeffs) != 10:
        raise ValueError(f"phase proxy expects 10 degree-2 coefficients, got {len(coeffs)}")
    return [float(v) for v in coeffs]


class TeacherPhaseProxyLoss(torch.nn.Module):
    """Teacher-phase proxy consistency: f_phi(depth_mm, x, y) -> unwrapped phase."""

    def __init__(self, coeffs, x_channel=11, y_channel=12, eps=1e-8):
        super().__init__()
        self.register_buffer("coeffs", torch.tensor(coeffs, dtype=torch.float32).view(1, 10, 1, 1))
        self.x_channel = int(x_channel)
        self.y_channel = int(y_channel)
        self.eps = float(eps)

    def depth_to_mm(self, pred_norm, batch):
        pred_01 = torch.clamp(pred_norm.float(), 0.0, 1.0)
        minmax = batch["depth_minmax"].to(pred_norm.device, non_blocking=True).float()
        depth_min = minmax[:, 0].view(-1, 1, 1, 1)
        depth_max = minmax[:, 1].view(-1, 1, 1, 1)
        return pred_01 * (depth_max - depth_min).clamp(min=self.eps) + depth_min

    def xy(self, batch, device):
        cond = batch["cond"].to(device, non_blocking=True).float()
        return cond[:, self.x_channel:self.x_channel + 1], cond[:, self.y_channel:self.y_channel + 1]

    def teacher_phase(self, batch, device):
        teacher01 = batch["teacher_uph01"].to(device, non_blocking=True).float()
        minmax = batch["teacher_uph_minmax"].to(device, non_blocking=True).float()
        lo = minmax[:, 0].view(-1, 1, 1, 1)
        hi = minmax[:, 1].view(-1, 1, 1, 1)
        return teacher01 * (hi - lo).clamp(min=self.eps) + lo

    def predict_phase_from_depth(self, depth_mm, x, y):
        feats = torch.stack(
            [
                torch.ones_like(depth_mm),
                depth_mm,
                x,
                y,
                depth_mm * depth_mm,
                depth_mm * x,
                depth_mm * y,
                x * x,
                x * y,
                y * y,
            ],
            dim=1,
        ).squeeze(2)
        coeffs = self.coeffs.to(dtype=feats.dtype, device=feats.device)
        return (feats * coeffs).sum(dim=1, keepdim=True)

    def error_map(self, pred_norm, batch, device):
        depth_mm = self.depth_to_mm(pred_norm, batch)
        x, y = self.xy(batch, device)
        pred_phase = self.predict_phase_from_depth(depth_mm, x, y)
        target_phase = self.teacher_phase(batch, device)
        return pred_phase - target_phase

    def valid_mask(self, pred_norm, batch, device):
        if "mask" in batch:
            return torch.clamp(batch["mask"].to(device, non_blocking=True).float(), 0.0, 1.0)
        return (batch["height_01"].to(device, non_blocking=True).float() > 0.0).float()

    def forward(self, pred_norm, batch, device):
        err = torch.abs(self.error_map(pred_norm.float(), batch, device))
        mask = self.valid_mask(pred_norm, batch, device)
        return (err * mask).sum() / mask.sum().clamp(min=self.eps)

    @torch.no_grad()
    def batch_metrics(self, pred_norm, batch, device):
        err = self.error_map(pred_norm.float(), batch, device)
        mask = self.valid_mask(pred_norm, batch, device)
        rows = []
        for j in range(err.shape[0]):
            m = mask[j:j + 1]
            e = err[j:j + 1]
            denom = m.sum().clamp(min=self.eps)
            mae = (torch.abs(e) * m).sum() / denom
            rmse = torch.sqrt(((e * e) * m).sum() / denom)
            rows.append(
                {
                    "phase_proxy_mae_rad": float(mae.detach().cpu()),
                    "phase_proxy_rmse_rad": float(rmse.detach().cpu()),
                }
            )
        return rows


def save_rows(rows, path):
    keys = ["sample"] + METRIC_KEYS
    for key in PHASE_PROXY_METRIC_KEYS:
        if rows and key in rows[0]:
            keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in keys})


def summarize_with_phase(rows):
    summary = summarize(rows)
    for key in PHASE_PROXY_METRIC_KEYS:
        if rows and key in rows[0]:
            vals = np.array([r[key] for r in rows], dtype=np.float64)
            summary[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
            }
    return summary


def checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best_val_loss, best_val_rmse, history):
    return {
        "epoch": ep,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": vars(args),
        "best_val_loss": best_val_loss,
        "best_val_rmse": best_val_rmse,
        "history": history,
    }


def configure_train_scope(model, args):
    """Select which part of the adapter UNet is allowed to update."""
    if args.freeze_backbone:
        model.freeze_backbone()
        return "adapter"

    scope = args.train_scope
    if scope == "all":
        for param in model.parameters():
            param.requires_grad_(True)
        return scope

    for param in model.parameters():
        param.requires_grad_(False)

    def should_train(name):
        is_adapter = name.startswith("adapter")
        is_out = name.startswith("backbone.out")
        is_decoder = (
            name.startswith("backbone.bottleneck")
            or name.startswith("backbone.up")
            or is_out
        )
        if scope == "adapter":
            return is_adapter
        if scope == "adapter_out":
            return is_adapter or is_out
        if scope == "adapter_decoder":
            return is_adapter or is_decoder
        if scope == "decoder":
            return is_decoder
        raise ValueError(f"unknown train_scope: {scope}")

    for name, param in model.named_parameters():
        param.requires_grad_(should_train(name))
    return scope


def build_optimizer(model, args):
    adapter_lr = args.lr if args.adapter_lr is None else args.adapter_lr
    backbone_lr = args.lr if args.backbone_lr is None else args.backbone_lr
    adapter_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("adapter"):
            adapter_params.append(param)
        else:
            backbone_params.append(param)
    groups = []
    if adapter_params:
        groups.append({"params": adapter_params, "lr": adapter_lr, "name": "adapter"})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
    if not groups:
        raise RuntimeError("no trainable parameters selected")
    return torch.optim.RMSprop(groups, lr=args.lr, weight_decay=args.weight_decay)


def optimizer_lrs(optimizer):
    return {
        group.get("name", f"group{idx}"): group["lr"]
        for idx, group in enumerate(optimizer.param_groups)
    }


def compute_model_loss(model, batch, criterion, device, args, phase_proxy=None):
    fringe = batch["fringe"].to(device, non_blocking=True)
    cond = make_cond(batch, device, args.cond_mode, args.instr_channel_indices)
    target = batch["height_01"].to(device, non_blocking=True)
    pred = model(fringe, cond)
    depth_loss = criterion(pred, target)
    if phase_proxy is None or args.phase_proxy_loss_weight <= 0.0:
        phase_loss = pred.new_tensor(0.0)
    else:
        # Keep proxy polynomial in fp32; depth^2 overflows in fp16.
        with autocast(enabled=False):
            phase_loss = phase_proxy(pred.float(), batch, device)
    loss = depth_loss + float(args.phase_proxy_loss_weight) * phase_loss
    return pred, loss, depth_loss, phase_loss


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, args, phase_proxy=None):
    model.eval()
    total = 0.0
    total_depth = 0.0
    total_phase = 0.0
    seen = 0
    for batch in tqdm(loader, desc="val loss", leave=False):
        _, loss, depth_loss, phase_loss = compute_model_loss(model, batch, criterion, device, args, phase_proxy)
        total += float(loss.item())
        total_depth += float(depth_loss.item())
        total_phase += float(phase_loss.item())
        seen += 1
    denom = max(1, seen)
    return {
        "loss": total / denom,
        "depth_loss": total_depth / denom,
        "phase_proxy_loss": total_phase / denom,
    }


@torch.no_grad()
def evaluate_metrics(model, loader, device, args, out_dir=None, save_images=False, phase_proxy=None):
    model.eval()
    rows = []
    if save_images and out_dir is not None:
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc="eval metrics"):
        fringe = batch["fringe"].to(device, non_blocking=True)
        cond = make_cond(batch, device, args.cond_mode, args.instr_channel_indices)
        pred = model(fringe, cond)
        pred_mm = prediction_to_mm(pred, batch)
        target_raw = batch["height_raw"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        phase_rows = None
        if phase_proxy is not None and "teacher_uph01" in batch:
            phase_rows = phase_proxy.batch_metrics(pred, batch, device)
        for j in range(pred.shape[0]):
            sample_idx = len(rows)
            metrics = compute_metrics(pred_mm[j:j + 1], target_raw[j:j + 1], mask=mask[j:j + 1])
            row = {"sample": sample_idx, **metrics}
            if phase_rows is not None:
                row.update(phase_rows[j])
            rows.append(row)
            if save_images and out_dir is not None and sample_idx < 8:
                save_comparison(
                    fringe[j:j + 1],
                    target_raw[j:j + 1],
                    pred_mm[j:j + 1],
                    out_dir / "samples" / f"sample_{sample_idx:02d}.png",
                    title=f"{args.cond_mode} adapter RMSE {metrics['rmse']:.2f}mm",
                    mask=mask[j:j + 1],
                )
    return rows


def write_eval_outputs(rows, out_dir, checkpoint, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rows(rows, out_dir / "per_sample_metrics.csv")
    summary = summarize_with_phase(rows)
    summary["n"] = len(rows)
    summary["checkpoint"] = str(checkpoint)
    summary["args"] = vars(args)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_pspquad_cache_960")
    parser.add_argument(
        "--teacher_phase_cache_dir",
        default="",
        help="Optional source phase cache containing gt_unwrapped_01 for teacher-phase proxy supervision.",
    )
    parser.add_argument("--phase_pred_prefix", default=None)
    parser.add_argument("--save_dir", default="results/fpp960_psp_adapter")
    parser.add_argument("--base_checkpoint", default="")
    parser.add_argument(
        "--cond_mode",
        choices=[
            "phase_pred",
            "phase_pred_xy",
            "phase_pred_instr",
            "phase_pred_instr_xy",
            "gt_psp",
            "gt_psp_xy",
            "gt_psp_instr",
            "gt_psp_instr_xy",
        ],
        default="phase_pred_xy",
    )
    parser.add_argument("--instr_channels", default="1-6")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument(
        "--train_scope",
        choices=["all", "adapter", "adapter_out", "adapter_decoder", "decoder"],
        default="all",
        help="Which parameters to train when --freeze_backbone is not used.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adapter_lr", type=float, default=None)
    parser.add_argument("--backbone_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--adapter_hidden", type=int, default=32)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_metrics_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--eval_initial", action="store_true")
    parser.add_argument("--preload_ram", action="store_true")
    parser.add_argument("--train_minimal", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase_proxy_loss_weight", type=float, default=0.0)
    parser.add_argument(
        "--phase_proxy_coeffs",
        default=" ".join(str(v) for v in DEFAULT_PHASE_PROXY_COEFFS),
        help="Degree-2 coeffs for f_phi(depth_mm,x,y)->teacher unwrapped phase.",
    )
    parser.add_argument(
        "--phase_proxy_summary",
        default="",
        help="Optional measurement_proxy_consistency_summary.json; overrides --phase_proxy_coeffs.",
    )
    parser.add_argument("--phase_proxy_x_channel", type=int, default=11)
    parser.add_argument("--phase_proxy_y_channel", type=int, default=12)
    args = parser.parse_args()
    args.instr_channel_indices = parse_indices(args.instr_channels)
    args.teacher_phase_cache_dir = str(args.teacher_phase_cache_dir).strip()
    args.phase_proxy_coeff_list = load_phase_proxy_coeffs(args)

    if args.phase_proxy_loss_weight > 0.0 and not args.teacher_phase_cache_dir:
        raise ValueError("--phase_proxy_loss_weight > 0 requires --teacher_phase_cache_dir")

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

    phase_proxy_train_keys = []
    if args.phase_proxy_loss_weight > 0.0:
        phase_proxy_train_keys = ["mask", "depth_minmax", "teacher_uph01", "teacher_uph_minmax"]

    loaders = create_fpp_phase_loaders(
        base_cache_dir=args.base_cache_dir,
        phase_cache_dir=args.phase_cache_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        image_h=args.image_size,
        image_w=args.image_size,
        phase_pred_prefix=args.phase_pred_prefix,
        teacher_phase_cache_dir=args.teacher_phase_cache_dir or None,
        require_cache=True,
        preload_ram=args.preload_ram,
        train_minimal=args.train_minimal,
        train_extra_keys=phase_proxy_train_keys,
    )

    cond_channels = cond_channel_count(args.cond_mode, args.instr_channel_indices)

    save_dir = Path(args.save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    model = OfficialUNetFPPAdapter(
        cond_channels=cond_channels,
        out_channels=1,
        dropout_rate=args.dropout,
        adapter_hidden=args.adapter_hidden,
    ).to(device)
    if args.base_checkpoint:
        ckpt = torch.load(args.base_checkpoint, map_location=device)
        model.load_backbone_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
        print(f"Loaded raw backbone from {args.base_checkpoint}")
    active_scope = configure_train_scope(model, args)

    trainable = [p for p in model.parameters() if p.requires_grad]
    criterion = HybridL1Loss(alpha=args.alpha)
    phase_proxy = None
    if args.teacher_phase_cache_dir:
        phase_proxy = TeacherPhaseProxyLoss(
            args.phase_proxy_coeff_list,
            x_channel=args.phase_proxy_x_channel,
            y_channel=args.phase_proxy_y_channel,
        ).to(device)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, min_lr=1e-6)
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))

    print(f"Device: {device}")
    print(f"Cond mode: {args.cond_mode} | cond_channels={cond_channels} | instr={args.instr_channel_indices}")
    print(f"Train scope: {active_scope} | optimizer_lrs={optimizer_lrs(optimizer)}")
    print(
        "Phase proxy: "
        f"teacher_cache={args.teacher_phase_cache_dir or 'disabled'} | "
        f"weight={args.phase_proxy_loss_weight} | coeffs={args.phase_proxy_coeff_list}"
    )
    print(f"Train {len(loaders['train'].dataset)} | Val {len(loaders['val'].dataset)} | Test {len(loaders['test'].dataset)}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | trainable {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    history = []
    best_val_loss = float("inf")
    best_val_rmse = float("inf")
    if args.eval_initial:
        val_loss_dict = evaluate_loss(model, loaders["val"], criterion, device, args, phase_proxy=phase_proxy)
        val_loss = val_loss_dict["loss"]
        val_rows = evaluate_metrics(model, loaders["val"], device, args, phase_proxy=phase_proxy)
        val_summary = summarize_with_phase(val_rows)
        best_val_loss = val_loss
        best_val_rmse = val_summary["rmse"]["mean"]
        log = {
            "epoch": 0,
            "train_loss": None,
            "val_loss": val_loss,
            "val_depth_loss": val_loss_dict["depth_loss"],
            "val_phase_proxy_loss": val_loss_dict["phase_proxy_loss"],
            "lr": optimizer.param_groups[0]["lr"],
            "lrs": optimizer_lrs(optimizer),
            "seconds": 0.0,
            **{f"val_{k}": val_summary[k]["mean"] for k in METRIC_KEYS},
        }
        history.append(log)
        torch.save(checkpoint_state(0, model, optimizer, scheduler, scaler, args, best_val_loss, best_val_rmse, history), save_dir / "checkpoints" / "best_rmse.pt")
        torch.save(checkpoint_state(0, model, optimizer, scheduler, scaler, args, best_val_loss, best_val_rmse, history), save_dir / "checkpoints" / "best.pt")
        print(json.dumps(log, ensure_ascii=False))

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        total_depth = 0.0
        total_phase = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"psp adapter {ep}/{args.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                fringe = batch["fringe"].to(device, non_blocking=True)
                cond = make_cond(batch, device, args.cond_mode, args.instr_channel_indices)
                target = batch["height_01"].to(device, non_blocking=True)
                pred = model(fringe, cond)
                depth_loss = criterion(pred, target)
            if phase_proxy is not None and args.phase_proxy_loss_weight > 0.0:
                with autocast(enabled=False):
                    phase_loss = phase_proxy(pred.float(), batch, device)
            else:
                phase_loss = depth_loss.new_tensor(0.0)
            loss = depth_loss + float(args.phase_proxy_loss_weight) * phase_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item())
            total_depth += float(depth_loss.item())
            total_phase += float(phase_loss.item())
            seen += 1
            if args.max_train_batches and seen >= args.max_train_batches:
                break
        train_loss = total / max(1, seen)
        train_depth_loss = total_depth / max(1, seen)
        train_phase_loss = total_phase / max(1, seen)
        do_eval = ep == 1 or ep == args.epochs or ep % max(1, args.eval_every) == 0
        val_loss = None
        val_loss_dict = None
        if do_eval:
            val_loss_dict = evaluate_loss(model, loaders["val"], criterion, device, args, phase_proxy=phase_proxy)
            val_loss = val_loss_dict["loss"]
            scheduler.step(val_loss)
        log = {
            "epoch": ep,
            "train_loss": train_loss,
            "train_depth_loss": train_depth_loss,
            "train_phase_proxy_loss": train_phase_loss,
            "val_loss": val_loss,
            "val_depth_loss": val_loss_dict["depth_loss"] if val_loss_dict else None,
            "val_phase_proxy_loss": val_loss_dict["phase_proxy_loss"] if val_loss_dict else None,
            "lr": optimizer.param_groups[0]["lr"],
            "lrs": optimizer_lrs(optimizer),
            "seconds": time.time() - t0,
        }
        improved_rmse = False
        if do_eval and (ep == 1 or ep == args.epochs or ep % max(1, args.eval_metrics_every) == 0):
            val_rows = evaluate_metrics(model, loaders["val"], device, args, phase_proxy=phase_proxy)
            val_summary = summarize_with_phase(val_rows)
            log.update({f"val_{k}": val_summary[k]["mean"] for k in METRIC_KEYS})
            for key in PHASE_PROXY_METRIC_KEYS:
                if key in val_summary:
                    log[f"val_{key}"] = val_summary[key]["mean"]
            if val_summary["rmse"]["mean"] < best_val_rmse:
                best_val_rmse = val_summary["rmse"]["mean"]
                improved_rmse = True
        history.append(log)
        if improved_rmse:
            torch.save(checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best_val_loss, best_val_rmse, history), save_dir / "checkpoints" / "best_rmse.pt")
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best_val_loss, best_val_rmse, history), save_dir / "checkpoints" / "best.pt")
        print(json.dumps(log, ensure_ascii=False))
        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        if args.save_every > 0 and (ep == 1 or ep == args.epochs or ep % args.save_every == 0):
            torch.save(checkpoint_state(ep, model, optimizer, scheduler, scaler, args, best_val_loss, best_val_rmse, history), save_dir / "checkpoints" / "latest.pt")

    best_path = save_dir / "checkpoints" / "best_rmse.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        rows = evaluate_metrics(
            model,
            loaders["test"],
            device,
            args,
            out_dir=save_dir / "evaluation",
            save_images=True,
            phase_proxy=phase_proxy,
        )
        summary = write_eval_outputs(rows, save_dir / "evaluation", best_path, args)
        print("Final test:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    # See eval_hierarchical_phase_fusion.py: persistent worker cleanup can keep
    # chained cloud scripts idle after all artifacts are already written.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
