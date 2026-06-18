"""x-phase diffusion posterior pilot for self-built single-frame 3D data.

This experiment moves diffusion from depth residual space to x-phase evidence
space. Formal test-time inputs remain legal: input_vertical_0120.bmp plus
single-frame derived features. The teacher x phase/order/confidence arrays are
used as train-time targets only.
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
    DIR_INDEX,
    dir_dp_oracle,
    dir_dp_predict,
    dir_phi_oracle,
    dir_phi_predict,
    load_model,
)
from models.unet import ConditionalUNet
from train_single_frame3d_full_pip_rcpc import create_loaders, zero_time_forward
from train_single_frame3d_physics_diffusion import (
    charbonnier,
    cosine_beta_schedule,
    forward_direct,
    load_base_model,
    masked_mse,
    pred_to_depth_mm,
    row_from_prediction,
    save_checkpoint,
    set_seed,
    summarize_rows,
)


def map_phi(phi: torch.Tensor) -> torch.Tensor:
    out = phi.clone()
    out[:, 2:4] = out[:, 2:4] * 2.0 - 1.0
    return torch.clamp(out, -1.0, 1.0)


def unmap_phi(phi_mapped: torch.Tensor) -> torch.Tensor:
    out = torch.clamp(phi_mapped, -1.0, 1.0).clone()
    out[:, 2:4] = (out[:, 2:4] + 1.0) * 0.5
    return out


def phase_weight(batch: Dict[str, object], device: torch.device) -> torch.Tensor:
    idx = DIR_INDEX["x"]
    valid = batch["valid_mask"].to(device, non_blocking=True).float()  # type: ignore[index]
    return batch["phi_weight"].to(device, non_blocking=True).float()[:, idx] * valid  # type: ignore[index]


def dp_with_phi(dp_model: torch.nn.Module, batch: Dict[str, object], phi: torch.Tensor, device: torch.device) -> torch.Tensor:
    cond = batch["cond"].to(device, non_blocking=True).float()  # type: ignore[index]
    return zero_time_forward(dp_model, torch.cat([cond, phi], dim=1))[:, :1]


class PhaseResidualPosterior:
    def __init__(self, model: torch.nn.Module, timesteps: int, residual_scale: float, device: torch.device) -> None:
        self.model = model
        self.timesteps = int(timesteps)
        self.residual_scale = float(residual_scale)
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

    def pred_phi(self, phi_model: torch.nn.Module, batch: Dict[str, object]) -> torch.Tensor:
        return dir_phi_predict(phi_model, batch, "x", self.device)

    def target_phi(self, batch: Dict[str, object]) -> torch.Tensor:
        return dir_phi_oracle(batch, "x", self.device)

    def cond(self, batch: Dict[str, object], pred_phi: torch.Tensor) -> torch.Tensor:
        base_cond = batch["cond"].to(self.device, non_blocking=True).float()  # type: ignore[index]
        return torch.cat([base_cond, map_phi(pred_phi.detach())], dim=1)

    def target_residual(self, target_phi: torch.Tensor, pred_phi: torch.Tensor) -> torch.Tensor:
        target_m = map_phi(target_phi)
        pred_m = map_phi(pred_phi)
        return torch.clamp((target_m - pred_m.detach()) / max(self.residual_scale, 1e-6), -1.0, 1.0)

    def training_loss(self, phi_model: torch.nn.Module, batch: Dict[str, object], args: argparse.Namespace) -> torch.Tensor:
        pred_phi = self.pred_phi(phi_model, batch)
        target_phi = self.target_phi(batch)
        target_res = self.target_residual(target_phi, pred_phi)
        cond = self.cond(batch, pred_phi)
        t = torch.randint(0, self.timesteps, (target_res.shape[0],), device=self.device)
        noisy = self.q_sample(target_res, t)
        pred_res = torch.tanh(self.model(noisy, t, cond))
        weight = phase_weight(batch, self.device)
        loss = charbonnier(pred_res, target_res, weight=weight)
        loss = loss + args.lambda_mse * masked_mse(pred_res, target_res, weight=weight)
        if args.lambda_final > 0:
            refined = torch.clamp(map_phi(pred_phi) + self.residual_scale * pred_res, -1.0, 1.0)
            loss = loss + args.lambda_final * charbonnier(refined, map_phi(target_phi), weight=weight)
        return loss

    @torch.no_grad()
    def sample(self, phi_model: torch.nn.Module, batch: Dict[str, object], steps: int, ensemble_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_phi = self.pred_phi(phi_model, batch)
        pred_m = map_phi(pred_phi)
        cond = self.cond(batch, pred_phi)
        b, _, h, w = pred_m.shape
        seq = torch.linspace(self.timesteps - 1, 0, int(steps), device=self.device).long()
        stride = max(1, self.timesteps // max(1, int(steps)))
        refined = []
        for _ in range(max(1, int(ensemble_size))):
            x = torch.randn((b, 4, h, w), device=self.device)
            for t_val in seq:
                t_int = int(t_val.item())
                t = torch.full((b,), t_int, dtype=torch.long, device=self.device)
                x0 = torch.tanh(self.model(x, t, cond))
                if t_int == 0:
                    x = x0
                    continue
                prev_t = max(t_int - stride, 0)
                eps = (x - self.sqrt_acp[t].view(-1, 1, 1, 1) * x0) / self.sqrt_om[t].view(-1, 1, 1, 1).clamp(min=1e-6)
                x = self.sqrt_acp[prev_t].view(1, 1, 1, 1) * x0 + self.sqrt_om[prev_t].view(1, 1, 1, 1) * eps
            refined.append(torch.clamp(pred_m + self.residual_scale * torch.clamp(x, -1.0, 1.0), -1.0, 1.0))
        stack = torch.stack(refined, dim=0)
        mean_m = stack.mean(dim=0)
        unc = stack.std(dim=0, unbiased=False) if stack.shape[0] > 1 else torch.zeros_like(mean_m)
        return pred_phi, unmap_phi(mean_m), unc


@torch.no_grad()
def eval_phase_loss(posterior: PhaseResidualPosterior, phi_model: torch.nn.Module, loader: Iterable[Dict[str, object]], args: argparse.Namespace) -> float:
    vals: List[float] = []
    posterior.model.eval()
    for batch in loader:
        with autocast(enabled=(posterior.device.type == "cuda" and not args.no_amp)):
            vals.append(float(posterior.training_loss(phi_model, batch, args).item()))
    return float(np.mean(vals)) if vals else float("nan")


def train_phase_posterior(args: argparse.Namespace, loaders: Dict[str, object], phi_model: torch.nn.Module, device: torch.device) -> PhaseResidualPosterior:
    save_dir = Path(args.save_dir) / "x_phase_diffusion"
    ckpt_path = save_dir / "checkpoints" / "best.pt"
    model = ConditionalUNet(
        in_channels=4,
        cond_channels=args.cond_channels + 4,
        out_channels=4,
        base_ch=args.base_channels,
        ch_mult=tuple(args.ch_mult),
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        time_emb_dim=args.time_emb_dim,
    ).to(device)
    posterior = PhaseResidualPosterior(model, args.timesteps, args.phase_residual_scale, device)
    if args.reuse_checkpoints and ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return posterior
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.phase_epochs))
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))
    best = float("inf")
    history: List[Dict[str, object]] = []
    for ep in range(1, args.phase_epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"x-phase diffusion {ep}/{args.phase_epochs}"):
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                loss = posterior.training_loss(phi_model, batch, args)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.item())
            seen += 1
        sched.step()
        do_val = ep == 1 or ep == args.phase_epochs or ep % max(1, args.val_interval) == 0
        val = eval_phase_loss(posterior, phi_model, loaders["val"], args) if do_val else float("nan")
        log = {"stage": "x_phase_diffusion", "epoch": ep, "train_loss": total / max(1, seen), "val_loss": val, "seconds": time.time() - t0}
        history.append(log)
        print(json.dumps(log, ensure_ascii=False), flush=True)
        if do_val and val < best:
            best = val
            save_checkpoint(ckpt_path, ep, model, opt, scaler, args, best, history)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return posterior


def compact_item(batch: Dict[str, object], j: int, pred: torch.Tensor, candidate: torch.Tensor, unc: torch.Tensor) -> Dict[str, object]:
    return {
        "sample_id": batch["sample_id"][j],  # type: ignore[index]
        "object_id": int(batch["object_id"][j].item()),  # type: ignore[index]
        "pose_id": int(batch["pose_id"][j].item()),  # type: ignore[index]
        "scale_mm": batch["scale_mm"][j:j + 1].detach().cpu(),  # type: ignore[index]
        "center_mm": batch["center_mm"][j:j + 1].detach().cpu(),  # type: ignore[index]
        "depth_raw": batch["depth_raw"][j:j + 1].detach().cpu(),  # type: ignore[index]
        "object_mask": batch["object_mask"][j:j + 1].detach().cpu(),  # type: ignore[index]
        "valid_mask": batch["valid_mask"][j:j + 1].detach().cpu(),  # type: ignore[index]
        "base": pred[j:j + 1].detach().cpu(),
        "candidate": candidate[j:j + 1].detach().cpu(),
        "unc_mean": masked_mean(unc[j:j + 1].detach().cpu(), batch["object_mask"][j:j + 1].detach().cpu()),  # type: ignore[index]
        "delta_mean": masked_mean(torch.abs(candidate[j:j + 1].detach().cpu() - pred[j:j + 1].detach().cpu()), batch["object_mask"][j:j + 1].detach().cpu()),  # type: ignore[index]
    }


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask > 0.5
    if not bool(m.any()):
        return float(x.mean().item())
    return float(x[m.expand_as(x)].mean().item())


def compact_batch(item: Dict[str, object]) -> Dict[str, object]:
    return {
        "sample_id": [item["sample_id"]],
        "object_id": torch.tensor([int(item["object_id"])]),
        "pose_id": torch.tensor([int(item["pose_id"])]),
        "scale_mm": item["scale_mm"],
        "center_mm": item["center_mm"],
        "depth_raw": item["depth_raw"],
        "object_mask": item["object_mask"],
        "valid_mask": item["valid_mask"],
    }


def fast_rmse(item: Dict[str, object], pred_norm: torch.Tensor) -> float:
    batch = compact_batch(item)
    pred_mm = pred_to_depth_mm(pred_norm, batch)
    target = batch["depth_raw"].float()  # type: ignore[union-attr]
    mask = batch["object_mask"].float()  # type: ignore[union-attr]
    count = mask.sum().clamp_min(1.0)
    return float(torch.sqrt((((pred_mm - target) ** 2) * mask).sum() / count).item())


def rcpc_pred(item: Dict[str, object], gate: Dict[str, float]) -> Tuple[torch.Tensor, bool]:
    use = float(item["delta_mean"]) <= gate["delta_max"] and float(item["unc_mean"]) <= gate["unc_max"]
    base = item["base"]  # type: ignore[assignment]
    cand = item["candidate"]  # type: ignore[assignment]
    out = torch.clamp(base + (float(gate["alpha"]) * (cand - base) if use else 0.0), -1.0, 1.0)
    return out, bool(use)


def select_gate(items: List[Dict[str, object]]) -> Dict[str, float]:
    deltas = np.asarray([float(x["delta_mean"]) for x in items], dtype=np.float32)
    uncs = np.asarray([float(x["unc_mean"]) for x in items], dtype=np.float32)
    delta_grid = sorted(set(float(x) for x in np.quantile(deltas, [0.2, 0.4, 0.6, 0.8, 1.0])))
    unc_grid = sorted(set(float(x) for x in np.quantile(uncs, [0.2, 0.4, 0.6, 0.8, 1.0])))
    best = {"object_rmse": float("inf"), "alpha": 0.0, "delta_max": float(delta_grid[-1]), "unc_max": float(unc_grid[-1]), "accepted_fraction": 0.0}
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for delta_max in delta_grid:
            for unc_max in unc_grid:
                gate = {"alpha": alpha, "delta_max": delta_max, "unc_max": unc_max}
                vals = []
                acc = 0
                for item in items:
                    pred, use = rcpc_pred(item, gate)
                    acc += int(use)
                    vals.append(fast_rmse(item, pred))
                rmse = float(np.mean(vals)) if vals else float("nan")
                if rmse < best["object_rmse"]:
                    best = {**gate, "object_rmse": rmse, "accepted_fraction": float(acc) / max(1, len(items))}
    return best


def row_from_item(item: Dict[str, object], pred: torch.Tensor, mode: str) -> Dict[str, object]:
    return row_from_prediction(pred, compact_batch(item), 0, "xphase_diffusion_rcpc", mode)


@torch.no_grad()
def collect_split(
    loader: Iterable[Dict[str, object]],
    base_model: torch.nn.Module,
    phi_model: torch.nn.Module,
    dp_pred: torch.nn.Module,
    dp_oracle: torch.nn.Module,
    posterior: PhaseResidualPosterior,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    rows = {"direct_base": [], "x_predicted_phase": [], "x_oracle_phase": [], "x_phase_diffusion_predDP": [], "x_phase_diffusion_oracleDP": []}
    items_pred: List[Dict[str, object]] = []
    items_oracle: List[Dict[str, object]] = []
    phase_errors: List[float] = []
    refined_phase_errors: List[float] = []
    for batch in tqdm(loader, desc="collect x-phase", leave=False):
        base = forward_direct(base_model, batch, device)[:, :1]
        phi_pred = dir_phi_predict(phi_model, batch, "x", device)
        phi_true = dir_phi_oracle(batch, "x", device)
        _, phi_refined, phi_unc = posterior.sample(phi_model, batch, args.phase_sample_steps, args.phase_ensemble_size)
        d_pred = dp_with_phi(dp_pred, batch, phi_pred, device)
        d_oracle = dp_with_phi(dp_oracle, batch, phi_true, device)
        d_ref_pred = dp_with_phi(dp_pred, batch, phi_refined, device)
        d_ref_oracle = dp_with_phi(dp_oracle, batch, phi_refined, device)
        phase_weight_mask = phase_weight(batch, device)
        e0 = torch.sqrt((((map_phi(phi_pred) - map_phi(phi_true)) ** 2) * phase_weight_mask).sum(dim=(1, 2, 3)) / phase_weight_mask.sum(dim=(1, 2, 3)).clamp_min(1.0))
        e1 = torch.sqrt((((map_phi(phi_refined) - map_phi(phi_true)) ** 2) * phase_weight_mask).sum(dim=(1, 2, 3)) / phase_weight_mask.sum(dim=(1, 2, 3)).clamp_min(1.0))
        phase_errors.extend(float(x) for x in e0.detach().cpu().tolist())
        refined_phase_errors.extend(float(x) for x in e1.detach().cpu().tolist())
        for j in range(base.shape[0]):
            rows["direct_base"].append(row_from_prediction(base, batch, j, "xphase_diffusion_rcpc", "direct_base"))
            rows["x_predicted_phase"].append(row_from_prediction(d_pred, batch, j, "xphase_diffusion_rcpc", "x_predicted_phase"))
            rows["x_oracle_phase"].append(row_from_prediction(d_oracle, batch, j, "x_oracle_phase", "x_oracle_phase"))
            rows["x_phase_diffusion_predDP"].append(row_from_prediction(d_ref_pred, batch, j, "xphase_diffusion_rcpc", "x_phase_diffusion_predDP"))
            rows["x_phase_diffusion_oracleDP"].append(row_from_prediction(d_ref_oracle, batch, j, "xphase_diffusion_rcpc", "x_phase_diffusion_oracleDP"))
            items_pred.append(compact_item(batch, j, base, d_ref_pred, phi_unc))
            items_oracle.append(compact_item(batch, j, base, d_ref_oracle, phi_unc))
    return {
        "rows": rows,
        "items_pred": items_pred,
        "items_oracle": items_oracle,
        "phase_error_mean": float(np.mean(phase_errors)) if phase_errors else float("nan"),
        "refined_phase_error_mean": float(np.mean(refined_phase_errors)) if refined_phase_errors else float("nan"),
    }


def save_rows_csv(rows: List[Dict[str, object]], path: Path) -> None:
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


def make_report(summary: Dict[str, object]) -> str:
    lines = [
        "# X-Phase Diffusion RCPC Pilot",
        "",
        "Diffusion is trained in x-phase evidence residual space, not depth residual space.",
        "",
        "| split | branch | object RMSE | valid RMSE |",
        "|---|---|---:|---:|",
    ]
    for split, data in summary["splits"].items():  # type: ignore[union-attr]
        for branch, metrics in data.items():  # type: ignore[union-attr]
            if not isinstance(metrics, dict) or "object" not in metrics:
                continue
            lines.append(f"| {split} | {branch} | {metrics['object']['rmse']['mean']:.4f} | {metrics['valid']['rmse']['mean']:.4f} |")
    lines += ["", "## Gates", "", "```json", json.dumps(summary["gates"], indent=2, ensure_ascii=False), "```"]
    lines += ["", "## Phase Error", "", "```json", json.dumps(summary["phase_error"], indent=2, ensure_ascii=False), "```"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--teacher_extra_root", required=True)
    parser.add_argument("--ood_root", default="")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--x_diag_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--eval_batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--phase_epochs", type=int, default=30)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--phase_residual_scale", type=float, default=0.5)
    parser.add_argument("--phase_sample_steps", type=int, default=12)
    parser.add_argument("--phase_ensemble_size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--ch_mult", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--time_emb_dim", type=int, default=128)
    parser.add_argument("--lambda_mse", type=float, default=0.2)
    parser.add_argument("--lambda_final", type=float, default=0.5)
    parser.add_argument("--val_interval", type=int, default=5)
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
    smoke = {"device": str(device), "cond_channels": args.cond_channels, "split_counts": args.split_counts, "normalization": args.normalization}
    (save_dir / "xphase_diffusion_smoke.json").write_text(json.dumps(smoke, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(smoke, indent=2, ensure_ascii=False), flush=True)
    if args.smoke_only:
        return

    loaders: Dict[str, object] = loaders_obj["loaders"]  # type: ignore[assignment]
    x_dir = Path(args.x_diag_dir) / "direction_x"
    base_model, base_args = load_base_model(args.base_ckpt, args.cond_channels, device)
    phi_model = load_model(x_dir / "phi_predictor" / "checkpoints" / "best.pt", args.cond_channels, 4, args, device)
    dp_pred = load_model(x_dir / "phase_depth_evidence" / "checkpoints" / "best.pt", args.cond_channels + 4, 1, args, device)
    dp_oracle = load_model(x_dir / "phase_depth_oracle_evidence" / "checkpoints" / "best.pt", args.cond_channels + 4, 1, args, device)

    posterior = train_phase_posterior(args, loaders, phi_model, device)

    collected = {}
    for split in [s for s in ("val", "test", "ood") if s in loaders]:
        collected[split] = collect_split(loaders[split], base_model, phi_model, dp_pred, dp_oracle, posterior, args, device)  # type: ignore[index]

    gate_pred = select_gate(collected["val"]["items_pred"])
    gate_oracle = select_gate(collected["val"]["items_oracle"])
    summary: Dict[str, object] = {
        "stage": "xphase_diffusion_rcpc",
        "seed": args.seed,
        "legal_single_frame": True,
        "note": "x_oracle_phase uses teacher x phase and is diagnostic; x_phase_diffusion branches use only single-frame predicted/refined phase at test time.",
        "base_ckpt": args.base_ckpt,
        "x_diag_dir": args.x_diag_dir,
        "base_args": base_args,
        "split_counts": args.split_counts,
        "normalization": args.normalization,
        "gates": {"predDP": gate_pred, "oracleDP": gate_oracle},
        "phase_error": {},
        "splits": {},
    }
    all_rows: List[Dict[str, object]] = []
    for split, block in collected.items():
        rows_by_mode: Dict[str, List[Dict[str, object]]] = dict(block["rows"])
        rcpc_pred_rows = []
        acc = 0
        for item in block["items_pred"]:
            pred, use = rcpc_pred(item, gate_pred)
            acc += int(use)
            rcpc_pred_rows.append(row_from_item(item, pred, "RCPC_phase_diff_predDP"))
        rows_by_mode["RCPC_phase_diff_predDP"] = rcpc_pred_rows
        rcpc_oracle_rows = []
        acc2 = 0
        for item in block["items_oracle"]:
            pred, use = rcpc_pred(item, gate_oracle)
            acc2 += int(use)
            rcpc_oracle_rows.append(row_from_item(item, pred, "RCPC_phase_diff_oracleDP"))
        rows_by_mode["RCPC_phase_diff_oracleDP"] = rcpc_oracle_rows
        rows = []
        for mode_rows in rows_by_mode.values():
            rows.extend(mode_rows)
        rows_with_split = [{**r, "split": split} for r in rows]
        save_rows_csv(rows_with_split, save_dir / f"{split}_xphase_diffusion_metrics.csv")
        all_rows.extend(rows_with_split)
        summary["phase_error"][split] = {  # type: ignore[index]
            "pred_phi_rmse_mapped": block["phase_error_mean"],
            "refined_phi_rmse_mapped": block["refined_phase_error_mean"],
        }
        summary["splits"][split] = {mode: summarize_rows(mode_rows) for mode, mode_rows in sorted(rows_by_mode.items())}  # type: ignore[index]
        summary["splits"][split]["rcpc_accept_predDP"] = float(acc) / max(1, len(block["items_pred"]))  # type: ignore[index]
        summary["splits"][split]["rcpc_accept_oracleDP"] = float(acc2) / max(1, len(block["items_oracle"]))  # type: ignore[index]
    save_rows_csv(all_rows, save_dir / "xphase_diffusion_metrics_all.csv")
    (save_dir / "xphase_diffusion_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (save_dir / "xphase_diffusion_report.md").write_text(make_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
