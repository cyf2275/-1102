"""Backbone baselines for the self-built single_frame_3d depth_z dataset.

This entry point is separate from the older FPP-ML-Bench cache baselines. It
uses the current manifest-driven self-built dataset, keeps test-time input to
`input_vertical_0120.bmp`, and evaluates both regular test and optional OOD
61-64 samples under object/valid-mask RMSE.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.single_frame_baselines import (
    PatchGANDiscriminatorFPP,
    Pix2PixGeneratorFPP,
    build_single_frame_baseline,
)
from train_single_frame3d_full_pip_rcpc import FullTeacherDataset
from train_single_frame3d_physics_diffusion import (
    METRIC_KEYS,
    SingleFrame3DDataset,
    charbonnier,
    collate_single_frame,
    gradient_loss,
    masked_mse,
    normalization_from_stats,
    pred_to_depth_mm,
    row_from_prediction,
    save_rows,
    set_seed,
    summarize_rows,
    train_weight,
)


ARCHES = ["unet", "resunet", "attention_unet", "unetpp", "mps_xnet", "pix2pix"]


def split_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).replace(",", " ").split() if x.strip()]


def loader_common(num_workers: int, collate_fn=None) -> Dict[str, object]:
    out: Dict[str, object] = {
        "num_workers": int(num_workers),
        "pin_memory": True,
    }
    if collate_fn is not None:
        out["collate_fn"] = collate_fn
    if int(num_workers) > 0:
        out["persistent_workers"] = True
        out["prefetch_factor"] = 2
    return out


def make_loaders(args: argparse.Namespace) -> Dict[str, object]:
    norm = normalization_from_stats(args.data_root)

    def collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
        return collate_single_frame(batch, image_h=args.image_h, image_w=args.image_w)

    datasets: Dict[str, object] = {
        "train": SingleFrame3DDataset(args.data_root, "train", "raw", norm=norm),
        "val": SingleFrame3DDataset(args.data_root, "val", "raw", norm=norm),
        "test": SingleFrame3DDataset(args.data_root, "test", "raw", norm=norm),
    }
    common = loader_common(args.num_workers, collate_fn=collate)
    loaders: Dict[str, object] = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **common,
        ),
        "val": DataLoader(datasets["val"], batch_size=args.eval_batch_size, shuffle=False, **common),
        "test": DataLoader(datasets["test"], batch_size=args.eval_batch_size, shuffle=False, **common),
    }
    if args.ood_root:
        if not args.teacher_extra_root:
            raise ValueError("--teacher_extra_root is required when --ood_root is used")
        datasets["ood"] = FullTeacherDataset(
            args.data_root,
            args.teacher_extra_root,
            "ood",
            norm=norm,
            ood_root=args.ood_root,
            cache_features=False,
        )
        loaders["ood"] = DataLoader(
            datasets["ood"],
            batch_size=args.eval_batch_size,
            shuffle=False,
            **loader_common(args.num_workers),
        )
    return {
        "datasets": datasets,
        "loaders": loaders,
        "norm": norm,
        "split_counts": {k: len(v) for k, v in datasets.items()},  # type: ignore[arg-type]
    }


def build_arch(args: argparse.Namespace, device: torch.device) -> Tuple[torch.nn.Module, Optional[torch.nn.Module]]:
    arch = args.arch
    if arch == "unet":
        from models.unet import ConditionalUNet

        model = ConditionalUNet(
            in_channels=1,
            cond_channels=1,
            out_channels=1,
            base_ch=args.unet_base_channels,
            ch_mult=tuple(args.unet_ch_mult),
            num_res_blocks=args.unet_num_res_blocks,
            dropout=args.dropout,
            time_emb_dim=args.unet_time_emb_dim,
        ).to(device)
        return model, None
    if arch == "pix2pix":
        gen = Pix2PixGeneratorFPP(1, 1, args.pix2pix_gen_channels, args.dropout).to(device)
        disc = PatchGANDiscriminatorFPP(2, args.pix2pix_disc_channels).to(device)
        return gen, disc
    model = build_single_frame_baseline(
        arch,
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels or None,
        dropout_rate=args.dropout,
    ).to(device)
    return model, None


def phase_x_targets(batch: Dict[str, object], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phase = batch["phase_target"].to(device, non_blocking=True).float()  # type: ignore[index]
    conf = batch["phase_conf"].to(device, non_blocking=True).float()  # type: ignore[index]
    sin_x = phase[:, 2:3]
    cos_x = phase[:, 3:4]
    conf_x = conf[:, 2:3]
    wrapped = torch.atan2(sin_x, cos_x)
    wrapped01 = (wrapped + math.pi) / (2.0 * math.pi)
    return sin_x, cos_x, torch.clamp(wrapped01, 0.0, 1.0) * conf_x + wrapped01.detach() * (1.0 - conf_x)


def masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.to(dtype=value.dtype, device=value.device)
    return (value * mask).sum() / mask.sum().clamp_min(eps)


def model_depth_norm(
    model: torch.nn.Module,
    batch: Dict[str, object],
    device: torch.device,
    arch: str,
    return_output: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, object]:
    fringe = batch["fringe"].to(device, non_blocking=True).float()  # type: ignore[index]
    if arch == "unet":
        zeros = torch.zeros((fringe.shape[0], 1, fringe.shape[-2], fringe.shape[-1]), device=device)
        t = torch.zeros((fringe.shape[0],), dtype=torch.long, device=device)
        pred = torch.tanh(model(zeros, t, fringe))
        return (pred, pred) if return_output else pred
    output = model(fringe)
    if isinstance(output, dict):
        depth = output["depth"]
    else:
        depth = output
    if arch == "pix2pix":
        pred = torch.clamp(depth, 0.0, 1.0) * 2.0 - 1.0
    else:
        pred = torch.tanh(depth)
    return (pred, output) if return_output else pred


def depth_loss(pred_norm: torch.Tensor, batch: Dict[str, object], device: torch.device, args: argparse.Namespace) -> torch.Tensor:
    target = batch["depth"].to(device, non_blocking=True).float()  # type: ignore[index]
    weight = train_weight(batch, device, args.object_mask_weight)
    loss = charbonnier(pred_norm, target, weight=weight)
    loss = loss + args.lambda_mse * masked_mse(pred_norm, target, weight=weight)
    if args.lambda_grad > 0:
        loss = loss + args.lambda_grad * gradient_loss(pred_norm, target, weight=weight)
    return loss


def mps_aux_loss(output: object, batch: Dict[str, object], device: torch.device, args: argparse.Namespace) -> torch.Tensor:
    if not isinstance(output, dict) or args.mps_aux_weight <= 0:
        return torch.tensor(0.0, device=device)
    valid = batch["valid_mask"].to(device, non_blocking=True).float()  # type: ignore[index]
    sin_x, cos_x, wrapped01 = phase_x_targets(batch, device)
    fenzi = torch.tanh(output["fenzi"])
    fenmu = torch.tanh(output["fenmu"])
    wrapped = torch.sigmoid(output["wrapped"])
    return args.mps_aux_weight * (
        masked_mean(torch.abs(fenzi - sin_x), valid)
        + masked_mean(torch.abs(fenmu - cos_x), valid)
        + masked_mean(torch.abs(wrapped - wrapped01), valid)
    )


def pix2pix_step(
    gen: torch.nn.Module,
    disc: torch.nn.Module,
    batch: Dict[str, object],
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    scaler: GradScaler,
    bce: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    fringe = batch["fringe"].to(device, non_blocking=True).float()  # type: ignore[index]
    target_norm = batch["depth"].to(device, non_blocking=True).float()  # type: ignore[index]
    target01 = torch.clamp((target_norm + 1.0) * 0.5, 0.0, 1.0)

    opt_d.zero_grad(set_to_none=True)
    with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
        fake01 = gen(fringe).detach()
        real_logits = disc(fringe, target01)
        fake_logits = disc(fringe, fake01)
        d_loss = 0.5 * (
            bce(real_logits, torch.full_like(real_logits, args.label_smooth))
            + bce(fake_logits, torch.zeros_like(fake_logits))
        )
    scaler.scale(d_loss).backward()
    scaler.step(opt_d)

    opt_g.zero_grad(set_to_none=True)
    with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
        fake01 = gen(fringe)
        fake_norm = torch.clamp(fake01, 0.0, 1.0) * 2.0 - 1.0
        fake_logits_for_g = disc(fringe, fake01)
        gan_g = bce(fake_logits_for_g, torch.ones_like(fake_logits_for_g))
        recon = depth_loss(fake_norm, batch, device, args)
        g_loss = recon + args.lambda_gan * gan_g
    scaler.scale(g_loss).backward()
    scaler.unscale_(opt_g)
    torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
    scaler.step(opt_g)
    scaler.update()
    return {
        "loss": float(g_loss.item()),
        "recon": float(recon.item()),
        "gan_g": float(gan_g.item()),
        "d_loss": float(d_loss.item()),
    }


def checkpoint_state(
    epoch: int,
    model: torch.nn.Module,
    disc: Optional[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    optimizer_d: Optional[torch.optim.Optimizer],
    scaler: GradScaler,
    args: argparse.Namespace,
    best: float,
    history: List[Dict[str, object]],
) -> Dict[str, object]:
    state: Dict[str, object] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": vars(args),
        "best_val_object_rmse": best,
        "history": history,
    }
    if disc is not None:
        state["discriminator_state_dict"] = disc.state_dict()
    if optimizer_d is not None:
        state["optimizer_d_state_dict"] = optimizer_d.state_dict()
    return state


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Iterable[Dict[str, object]],
    device: torch.device,
    args: argparse.Namespace,
    split: str,
) -> List[Dict[str, object]]:
    model.eval()
    rows: List[Dict[str, object]] = []
    for batch in tqdm(loader, desc=f"eval {args.arch} {split}", leave=False):
        pred = model_depth_norm(model, batch, device, args.arch)
        assert torch.is_tensor(pred)
        for j in range(pred.shape[0]):
            rows.append(row_from_prediction(pred, batch, j, config=args.arch, mode=split))
    return rows


@torch.no_grad()
def save_visuals(
    model: torch.nn.Module,
    loader: Iterable[Dict[str, object]],
    device: torch.device,
    args: argparse.Namespace,
    split: str,
    out_dir: Path,
    max_images: int,
) -> None:
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for batch in loader:
        pred = model_depth_norm(model, batch, device, args.arch)
        assert torch.is_tensor(pred)
        pred_mm = pred_to_depth_mm(pred, batch)
        target = batch["depth_raw"].to(device, non_blocking=True).float()  # type: ignore[index]
        mask = batch["object_mask"].to(device, non_blocking=True).bool()  # type: ignore[index]
        fringe = batch["fringe"].to(device, non_blocking=True).float()  # type: ignore[index]
        for j in range(pred.shape[0]):
            if saved >= max_images:
                return
            m = mask[j, 0].detach().cpu().numpy().astype(bool)
            gt = target[j, 0].detach().cpu().numpy()
            pd = pred_mm[j, 0].detach().cpu().numpy()
            inp = fringe[j, 0].detach().cpu().numpy()
            err = np.abs(pd - gt)
            vals = np.concatenate([gt[m], pd[m]]) if np.any(m) else np.array([0.0, 1.0])
            vmin, vmax = np.percentile(vals[np.isfinite(vals)], [1, 99])
            ev = np.percentile(err[m & np.isfinite(err)], 95) if np.any(m & np.isfinite(err)) else 1.0
            fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
            panels = [
                (inp, "input", "gray", None, None),
                (np.where(m, gt, np.nan), "GT depth_z", "viridis", vmin, vmax),
                (np.where(m, pd, np.nan), f"{args.arch} pred", "viridis", vmin, vmax),
                (np.where(m, err, np.nan), "abs error", "magma", 0.0, max(float(ev), 0.1)),
            ]
            for ax, (arr, title, cmap, lo, hi) in zip(axes, panels):
                im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
                ax.set_title(title, fontsize=9)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            sample_id = str(batch["sample_id"][j])  # type: ignore[index]
            rmse = float(np.sqrt(np.mean((pd[m] - gt[m]) ** 2))) if np.any(m) else float("nan")
            fig.suptitle(f"{split} | {sample_id} | object RMSE {rmse:.3f}", fontsize=10)
            fig.savefig(out_dir / f"{split}_{saved:02d}_{sample_id}.png", dpi=180)
            plt.close(fig)
            saved += 1


def smoke_check(args: argparse.Namespace, loaders_obj: Dict[str, object], device: torch.device) -> Dict[str, object]:
    model, disc = build_arch(args, device)
    batch = next(iter(loaders_obj["loaders"]["train"]))  # type: ignore[index]
    pred, output = model_depth_norm(model, batch, device, args.arch, return_output=True)  # type: ignore[assignment]
    assert torch.is_tensor(pred)
    target = batch["depth"].to(device).float()  # type: ignore[index]
    loss = depth_loss(pred, batch, device, args)
    if args.arch == "mps_xnet":
        loss = loss + mps_aux_loss(output, batch, device, args)
    return {
        "arch": args.arch,
        "device": str(device),
        "split_counts": loaders_obj["split_counts"],
        "normalization": loaders_obj["norm"],
        "batch_shapes": {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)},
        "pred_shape": list(pred.shape),
        "target_shape": list(target.shape),
        "pred_nan": int(torch.isnan(pred).sum().item()),
        "loss": float(loss.item()),
        "disc_params": int(sum(p.numel() for p in disc.parameters())) if disc is not None else 0,
    }


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def train_one(args: argparse.Namespace) -> None:
    if args.arch not in ARCHES:
        raise ValueError(f"unknown arch {args.arch!r}; expected one of {ARCHES}")
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    loaders_obj = make_loaders(args)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    args.normalization = loaders_obj["norm"]
    args.split_counts = loaders_obj["split_counts"]
    smoke = smoke_check(args, loaders_obj, device)
    write_json(save_dir / "loader_smoke_summary.json", smoke)
    if args.smoke_only:
        print(json.dumps(smoke, indent=2, ensure_ascii=False), flush=True)
        return

    model, disc = build_arch(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_d = None
    if disc is not None:
        optimizer_d = torch.optim.AdamW(disc.parameters(), lr=args.lr_d, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))
    bce = torch.nn.BCEWithLogitsLoss()
    best = float("inf")
    history: List[Dict[str, object]] = []

    print(json.dumps({
        "arch": args.arch,
        "device": str(device),
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "split_counts": args.split_counts,
        "quick_screening": True,
    }, ensure_ascii=False), flush=True)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        if disc is not None:
            disc.train()
        totals: Dict[str, float] = {"loss": 0.0}
        seen = 0
        for batch in tqdm(loaders_obj["loaders"]["train"], desc=f"{args.arch} {ep}/{args.epochs}"):  # type: ignore[index]
            if args.arch == "pix2pix":
                assert disc is not None and optimizer_d is not None
                parts = pix2pix_step(model, disc, batch, optimizer, optimizer_d, scaler, bce, device, args)
                for k, v in parts.items():
                    totals[k] = totals.get(k, 0.0) + v
            else:
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                    pred, output = model_depth_norm(model, batch, device, args.arch, return_output=True)  # type: ignore[assignment]
                    assert torch.is_tensor(pred)
                    loss = depth_loss(pred, batch, device, args)
                    if args.arch == "mps_xnet":
                        loss = loss + mps_aux_loss(output, batch, device, args)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                totals["loss"] = totals.get("loss", 0.0) + float(loss.item())
            seen += 1
            if args.max_train_batches and seen >= args.max_train_batches:
                break

        log: Dict[str, object] = {
            "epoch": ep,
            "seconds": time.time() - t0,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v / max(1, seen) for k, v in totals.items()},
        }
        if ep == 1 or ep % args.eval_every == 0 or ep == args.epochs:
            val_rows = evaluate(model, loaders_obj["loaders"]["val"], device, args, "val")  # type: ignore[index]
            val_summary = summarize_rows(val_rows)
            val_rmse = float(val_summary["object"]["rmse"]["mean"])  # type: ignore[index]
            log["val_object_rmse"] = val_rmse
            log["val_valid_rmse"] = float(val_summary["valid"]["rmse"]["mean"])  # type: ignore[index]
            if val_rmse < best:
                best = val_rmse
                torch.save(
                    checkpoint_state(ep, model, disc, optimizer, optimizer_d, scaler, args, best, history),
                    save_dir / "checkpoints" / "best.pt",
                )
        history.append(log)
        print(json.dumps(log, ensure_ascii=False), flush=True)

    best_path = save_dir / "checkpoints" / "best.pt"
    if best_path.exists():
        ckpt = torch.load(str(best_path), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if disc is not None and "discriminator_state_dict" in ckpt:
            disc.load_state_dict(ckpt["discriminator_state_dict"])

    eval_dir = save_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    split_summaries: Dict[str, object] = {}
    for split in ["val", "test"] + (["ood"] if "ood" in loaders_obj["loaders"] else []):  # type: ignore[operator]
        rows = evaluate(model, loaders_obj["loaders"][split], device, args, split)  # type: ignore[index]
        save_rows(rows, eval_dir / (f"{split}_per_sample_metrics.csv" if split != "test" else "per_sample_metrics.csv"))
        split_summaries[split] = summarize_rows(rows)
        if split in {"test", "ood"}:
            save_visuals(
                model,
                loaders_obj["loaders"][split],  # type: ignore[index]
                device,
                args,
                split,
                save_dir / "visualizations" / split,
                max_images=args.max_visuals,
            )

    summary = {
        "stage": "single_frame3d_backbone_baseline_quick1seed",
        "arch": args.arch,
        "seed": args.seed,
        "epochs": args.epochs,
        "quick_screening": True,
        "paper_final": False,
        "legal_single_frame": True,
        "input": "input_vertical_0120.bmp",
        "target": "depth_z",
        "normalization": args.normalization,
        "split_counts": args.split_counts,
        "best_val_object_rmse": best,
        "checkpoint": str(best_path),
        "splits": split_summaries,
        "history": history,
        "args": vars(args),
    }
    write_json(eval_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--teacher_extra_root", default="")
    parser.add_argument("--ood_root", default="")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--arch", choices=ARCHES, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_h", type=int, default=480)
    parser.add_argument("--image_w", type=int, default=640)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--object_mask_weight", type=float, default=3.0)
    parser.add_argument("--lambda_mse", type=float, default=0.5)
    parser.add_argument("--lambda_grad", type=float, default=0.05)
    parser.add_argument("--mps_aux_weight", type=float, default=0.05)
    parser.add_argument("--lambda_gan", type=float, default=0.01)
    parser.add_argument("--label_smooth", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--base_channels", type=int, default=0)
    parser.add_argument("--unet_base_channels", type=int, default=32)
    parser.add_argument("--unet_ch_mult", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--unet_num_res_blocks", type=int, default=1)
    parser.add_argument("--unet_time_emb_dim", type=int, default=128)
    parser.add_argument("--pix2pix_gen_channels", type=int, default=64)
    parser.add_argument("--pix2pix_disc_channels", type=int, default=64)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_visuals", type=int, default=5)
    parser.add_argument("--smoke_only", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train_one(args)


if __name__ == "__main__":
    main()
