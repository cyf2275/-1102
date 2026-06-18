"""Train a depth candidate network for diffusion-refined x-phase evidence.

This script freezes the single-frame x-phase predictor and the x-phase
diffusion posterior, then trains a dedicated depth adapter on refined x-phase
evidence. It tests whether the previous phase-to-depth adapter was the weak
link after moving diffusion from depth residual space to phase posterior space.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from diagnose_single_frame3d_phase_residual import (
    dir_dp_oracle,
    dir_dp_predict,
    dir_phi_oracle,
    dir_phi_predict,
    load_model,
)
from models.unet import ConditionalUNet
from train_single_frame3d_full_pip_rcpc import create_loaders, zero_time_forward
from train_single_frame3d_physics_diffusion import (
    build_model,
    charbonnier,
    forward_direct,
    gradient_loss,
    load_base_model,
    masked_mse,
    pred_to_depth_mm,
    row_from_prediction,
    save_checkpoint,
    set_seed,
    summarize_rows,
    train_weight,
)
from train_single_frame3d_xphase_diffusion_rcpc import (
    PhaseResidualPosterior,
    compact_batch,
    compact_item,
    dp_with_phi,
    map_phi,
    phase_weight,
    rcpc_pred,
    save_rows_csv,
    select_gate,
)


def phase_model_args(path: Path, fallback: argparse.Namespace) -> argparse.Namespace:
    ckpt = torch.load(str(path), map_location="cpu")
    saved = ckpt.get("args", {})
    if not isinstance(saved, dict):
        saved = {}
    return argparse.Namespace(
        base_channels=int(saved.get("base_channels", fallback.base_channels)),
        ch_mult=list(saved.get("ch_mult", fallback.ch_mult)),
        num_res_blocks=int(saved.get("num_res_blocks", fallback.num_res_blocks)),
        dropout=float(saved.get("dropout", fallback.dropout)),
        time_emb_dim=int(saved.get("time_emb_dim", fallback.time_emb_dim)),
        timesteps=int(saved.get("timesteps", fallback.timesteps)),
        phase_residual_scale=float(saved.get("phase_residual_scale", fallback.phase_residual_scale)),
    )


def load_phase_posterior(path: Path, cond_channels: int, fallback: argparse.Namespace, device: torch.device) -> PhaseResidualPosterior:
    ns = phase_model_args(path, fallback)
    model = ConditionalUNet(
        in_channels=4,
        cond_channels=cond_channels + 4,
        out_channels=4,
        base_ch=ns.base_channels,
        ch_mult=tuple(ns.ch_mult),
        num_res_blocks=ns.num_res_blocks,
        dropout=ns.dropout,
        time_emb_dim=ns.time_emb_dim,
    ).to(device)
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return PhaseResidualPosterior(model, ns.timesteps, ns.phase_residual_scale, device)


@torch.no_grad()
def refined_phi(
    posterior: PhaseResidualPosterior,
    phi_model: torch.nn.Module,
    batch: Dict[str, object],
    steps: int,
    ensemble_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phi_pred, phi_refined, phi_unc = posterior.sample(phi_model, batch, steps, ensemble_size)
    return phi_pred.detach(), phi_refined.detach(), phi_unc.detach()


def depth_with_phi(model: torch.nn.Module, batch: Dict[str, object], phi: torch.Tensor, device: torch.device) -> torch.Tensor:
    cond = batch["cond"].to(device, non_blocking=True).float()  # type: ignore[index]
    return zero_time_forward(model, torch.cat([cond, phi.detach()], dim=1))[:, :1]


@torch.no_grad()
def eval_depth_fast(
    model: torch.nn.Module,
    loader: Iterable[Dict[str, object]],
    posterior: PhaseResidualPosterior,
    phi_model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    model.eval()
    vals: List[float] = []
    for batch in loader:
        _, phi_refined, _ = refined_phi(posterior, phi_model, batch, args.eval_phase_sample_steps, args.eval_phase_ensemble_size)
        pred = pred_to_depth_mm(depth_with_phi(model, batch, phi_refined, device), batch)
        target = batch["depth_raw"].to(device, non_blocking=True).float()  # type: ignore[index]
        mask = batch["object_mask"].to(device, non_blocking=True).float()  # type: ignore[index]
        count = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        rmse = torch.sqrt((((pred - target) ** 2) * mask).sum(dim=(1, 2, 3)) / count)
        vals.extend(float(x) for x in rmse.detach().cpu().tolist())
    return float(np.mean(vals)) if vals else float("nan")


def train_refined_depth(
    args: argparse.Namespace,
    loaders: Dict[str, object],
    posterior: PhaseResidualPosterior,
    phi_model: torch.nn.Module,
    device: torch.device,
) -> torch.nn.Module:
    save_dir = Path(args.save_dir) / "refined_xphase_depth"
    ckpt_path = save_dir / "checkpoints" / "best.pt"
    if args.reuse_checkpoints and ckpt_path.exists():
        return load_model(ckpt_path, args.cond_channels + 4, 1, args, device)

    model = build_model(args.cond_channels + 4, 1, args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.depth_epochs))
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))
    best = float("inf")
    history: List[Dict[str, object]] = []

    posterior.model.eval()
    phi_model.eval()
    for ep in range(1, args.depth_epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"refined-x depth {ep}/{args.depth_epochs}"):
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                _, phi_refined, _ = refined_phi(posterior, phi_model, batch, args.train_phase_sample_steps, args.train_phase_ensemble_size)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                pred = depth_with_phi(model, batch, phi_refined, device)
                target = batch["depth"].to(device, non_blocking=True).float()  # type: ignore[index]
                weight = train_weight(batch, device, args.object_mask_weight)
                loss = charbonnier(pred, target, weight=weight)
                loss = loss + args.lambda_mse * masked_mse(pred, target, weight=weight)
                if args.lambda_grad > 0:
                    loss = loss + args.lambda_grad * gradient_loss(pred, target, weight=weight)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.item())
            seen += 1
        sched.step()
        do_val = ep == 1 or ep == args.depth_epochs or ep % max(1, args.eval_interval) == 0
        val_rmse = eval_depth_fast(model, loaders["val"], posterior, phi_model, args, device) if do_val else float("nan")
        log = {
            "stage": "refined_xphase_depth",
            "epoch": ep,
            "train_loss": total / max(1, seen),
            "val_object_rmse": val_rmse,
            "seconds": time.time() - t0,
        }
        history.append(log)
        print(json.dumps(log, ensure_ascii=False), flush=True)
        if do_val and val_rmse < best:
            best = val_rmse
            save_checkpoint(ckpt_path, ep, model, opt, scaler, args, best, history)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def row_from_item(item: Dict[str, object], pred: torch.Tensor, mode: str) -> Dict[str, object]:
    return row_from_prediction(pred, compact_batch(item), 0, "refined_xphase_depth", mode)


@torch.no_grad()
def collect_split(
    loader: Iterable[Dict[str, object]],
    base_model: torch.nn.Module,
    phi_model: torch.nn.Module,
    posterior: PhaseResidualPosterior,
    old_dp_model: torch.nn.Module,
    refined_depth_model: torch.nn.Module,
    oracle_dp_model: torch.nn.Module | None,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    rows: Dict[str, List[Dict[str, object]]] = {
        "direct_base": [],
        "x_predicted_phase": [],
        "old_depth_map_on_refined_phase": [],
        "refined_xphase_depth_candidate": [],
    }
    if oracle_dp_model is not None:
        rows["x_oracle_phase"] = []
    items: List[Dict[str, object]] = []
    phase_errors: List[float] = []
    refined_phase_errors: List[float] = []

    base_model.eval()
    phi_model.eval()
    posterior.model.eval()
    old_dp_model.eval()
    refined_depth_model.eval()
    if oracle_dp_model is not None:
        oracle_dp_model.eval()

    for batch in tqdm(loader, desc="collect refined-x depth", leave=False):
        base = forward_direct(base_model, batch, device)[:, :1]
        phi_pred, phi_refined, phi_unc = refined_phi(posterior, phi_model, batch, args.eval_phase_sample_steps, args.eval_phase_ensemble_size)
        phi_true = dir_phi_oracle(batch, "x", device)
        old_pred = dir_dp_predict(phi_model, old_dp_model, batch, "x", device)
        old_refined = dp_with_phi(old_dp_model, batch, phi_refined, device)
        new_refined = depth_with_phi(refined_depth_model, batch, phi_refined, device)

        w = phase_weight(batch, device)
        e0 = torch.sqrt((((map_phi(phi_pred) - map_phi(phi_true)) ** 2) * w).sum(dim=(1, 2, 3)) / w.sum(dim=(1, 2, 3)).clamp_min(1.0))
        e1 = torch.sqrt((((map_phi(phi_refined) - map_phi(phi_true)) ** 2) * w).sum(dim=(1, 2, 3)) / w.sum(dim=(1, 2, 3)).clamp_min(1.0))
        phase_errors.extend(float(x) for x in e0.detach().cpu().tolist())
        refined_phase_errors.extend(float(x) for x in e1.detach().cpu().tolist())

        oracle_pred = dir_dp_oracle(oracle_dp_model, batch, "x", device) if oracle_dp_model is not None else None
        for j in range(base.shape[0]):
            rows["direct_base"].append(row_from_prediction(base, batch, j, "refined_xphase_depth", "direct_base"))
            rows["x_predicted_phase"].append(row_from_prediction(old_pred, batch, j, "refined_xphase_depth", "x_predicted_phase"))
            rows["old_depth_map_on_refined_phase"].append(row_from_prediction(old_refined, batch, j, "refined_xphase_depth", "old_depth_map_on_refined_phase"))
            rows["refined_xphase_depth_candidate"].append(row_from_prediction(new_refined, batch, j, "refined_xphase_depth", "refined_xphase_depth_candidate"))
            if oracle_pred is not None:
                rows["x_oracle_phase"].append(row_from_prediction(oracle_pred, batch, j, "refined_xphase_depth", "x_oracle_phase"))
            items.append(compact_item(batch, j, base, new_refined, phi_unc))

    return {
        "rows": rows,
        "items": items,
        "phase_error_mean": float(np.mean(phase_errors)) if phase_errors else float("nan"),
        "refined_phase_error_mean": float(np.mean(refined_phase_errors)) if refined_phase_errors else float("nan"),
    }


def make_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Refined X-Phase Depth Candidate Pilot",
        "",
        "A dedicated depth candidate is trained on diffusion-refined x-phase evidence.",
        "",
        "| split | branch | object RMSE | valid RMSE |",
        "|---|---|---:|---:|",
    ]
    for split, data in summary["splits"].items():  # type: ignore[union-attr]
        for branch, metrics in data.items():  # type: ignore[union-attr]
            if not isinstance(metrics, dict) or "object" not in metrics:
                continue
            lines.append(f"| {split} | {branch} | {metrics['object']['rmse']['mean']:.4f} | {metrics['valid']['rmse']['mean']:.4f} |")
    lines += ["", "## Gate", "", "```json", json.dumps(summary["gate"], indent=2, ensure_ascii=False), "```"]
    lines += ["", "## Phase Error", "", "```json", json.dumps(summary["phase_error"], indent=2, ensure_ascii=False), "```"]
    return "\n".join(lines) + "\n"


def write_rows_csv(rows: List[Dict[str, object]], path: Path) -> None:
    keys = ["split", "sample_id", "object_id", "pose_id", "config", "mode", "legal_single_frame"]
    for roi in ("object", "valid"):
        for metric in ("rmse", "mae", "edge_rmse", "normal_deg", "ssim"):
            keys.append(f"{roi}_{metric}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--teacher_extra_root", required=True)
    parser.add_argument("--ood_root", default="")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--x_diag_dir", required=True)
    parser.add_argument("--phase_posterior_ckpt", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--eval_batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--depth_epochs", type=int, default=30)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--ch_mult", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--time_emb_dim", type=int, default=128)
    parser.add_argument("--object_mask_weight", type=float, default=3.0)
    parser.add_argument("--lambda_mse", type=float, default=0.2)
    parser.add_argument("--lambda_grad", type=float, default=0.1)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--phase_residual_scale", type=float, default=0.5)
    parser.add_argument("--train_phase_sample_steps", type=int, default=8)
    parser.add_argument("--train_phase_ensemble_size", type=int, default=1)
    parser.add_argument("--eval_phase_sample_steps", type=int, default=12)
    parser.add_argument("--eval_phase_ensemble_size", type=int, default=3)
    parser.add_argument("--cache_features", action="store_true")
    parser.add_argument("--feature_cache_dir", default="")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--reuse_checkpoints", action="store_true")
    parser.add_argument("--smoke_only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    loaders_obj = create_loaders(args)
    args.cond_channels = int(loaders_obj["cond_channels"])
    args.normalization = loaders_obj["norm"]
    args.split_counts = loaders_obj["split_counts"]
    smoke = {
        "device": str(device),
        "cond_channels": args.cond_channels,
        "split_counts": args.split_counts,
        "normalization": args.normalization,
        "phase_posterior_ckpt": args.phase_posterior_ckpt,
    }
    (save_dir / "refined_xphase_depth_smoke.json").write_text(json.dumps(smoke, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(smoke, indent=2, ensure_ascii=False), flush=True)
    if args.smoke_only:
        return

    loaders: Dict[str, object] = loaders_obj["loaders"]  # type: ignore[assignment]
    x_dir = Path(args.x_diag_dir) / "direction_x"
    phi_model = load_model(x_dir / "phi_predictor" / "checkpoints" / "best.pt", args.cond_channels, 4, args, device)
    posterior = load_phase_posterior(Path(args.phase_posterior_ckpt), args.cond_channels, args, device)

    refined_depth = train_refined_depth(args, loaders, posterior, phi_model, device)

    # Load evaluation-only models after training to avoid wasting memory during
    # the refined depth adapter backward pass.
    base_model, base_args = load_base_model(args.base_ckpt, args.cond_channels, device)
    old_dp = load_model(x_dir / "phase_depth_evidence" / "checkpoints" / "best.pt", args.cond_channels + 4, 1, args, device)
    oracle_dp_path = x_dir / "phase_depth_oracle_evidence" / "checkpoints" / "best.pt"
    oracle_dp = load_model(oracle_dp_path, args.cond_channels + 4, 1, args, device) if oracle_dp_path.exists() else None

    collected = {}
    for split in [s for s in ("val", "test", "ood") if s in loaders]:
        collected[split] = collect_split(loaders[split], base_model, phi_model, posterior, old_dp, refined_depth, oracle_dp, args, device)  # type: ignore[index]

    gate = select_gate(collected["val"]["items"])
    summary: Dict[str, object] = {
        "stage": "refined_xphase_depth",
        "seed": args.seed,
        "legal_single_frame": True,
        "note": "The refined depth candidate uses only single-frame predicted x phase refined by a frozen phase diffusion posterior at test time.",
        "base_ckpt": args.base_ckpt,
        "x_diag_dir": args.x_diag_dir,
        "phase_posterior_ckpt": args.phase_posterior_ckpt,
        "base_args": base_args,
        "split_counts": args.split_counts,
        "normalization": args.normalization,
        "gate": gate,
        "phase_error": {},
        "splits": {},
    }
    all_rows: List[Dict[str, object]] = []
    for split, block in collected.items():
        rows_by_mode: Dict[str, List[Dict[str, object]]] = dict(block["rows"])
        rcpc_rows = []
        acc = 0
        for item in block["items"]:
            pred, use = rcpc_pred(item, gate)
            acc += int(use)
            rcpc_rows.append(row_from_item(item, pred, "RCPC_refined_xphase_depth"))
        rows_by_mode["RCPC_refined_xphase_depth"] = rcpc_rows
        rows = []
        for mode_rows in rows_by_mode.values():
            rows.extend(mode_rows)
        rows_with_split = [{**r, "split": split} for r in rows]
        write_rows_csv(rows_with_split, save_dir / f"{split}_refined_xphase_depth_metrics.csv")
        all_rows.extend(rows_with_split)
        summary["phase_error"][split] = {  # type: ignore[index]
            "pred_phi_rmse_mapped": block["phase_error_mean"],
            "refined_phi_rmse_mapped": block["refined_phase_error_mean"],
        }
        summary["splits"][split] = {mode: summarize_rows(mode_rows) for mode, mode_rows in sorted(rows_by_mode.items())}  # type: ignore[index]
        summary["splits"][split]["rcpc_accept"] = float(acc) / max(1, len(block["items"]))  # type: ignore[index]

    write_rows_csv(all_rows, save_dir / "refined_xphase_depth_metrics_all.csv")
    (save_dir / "refined_xphase_depth_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (save_dir / "refined_xphase_depth_report.md").write_text(make_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
