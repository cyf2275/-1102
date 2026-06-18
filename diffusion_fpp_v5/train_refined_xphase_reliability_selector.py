"""Local reliability selector for refined x-phase depth candidates.

This script covers two follow-up directions:

1. A fast rule-based local RCPC search.
2. A lightweight pixel reliability MLP that decides where to use the
   diffusion-refined x-phase depth candidate.

The selector uses only test-time legal maps derived from the single frame and
model outputs. Ground-truth depth is used only to build train/validation labels
and to evaluate metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diagnose_single_frame3d_phase_residual import dir_dp_oracle, dir_dp_predict, load_model
from train_single_frame3d_full_pip_rcpc import create_loaders
from train_single_frame3d_physics_diffusion import forward_direct, load_base_model, pred_to_depth_mm, set_seed
from train_single_frame3d_refined_xphase_depth import depth_with_phi, load_phase_posterior, refined_phi
from train_single_frame3d_xphase_diffusion_rcpc import compact_item, rcpc_pred


FEATURE_NAMES = [
    "ref_minus_x",
    "abs_ref_minus_x",
    "ref_minus_base",
    "abs_ref_minus_base",
    "base_minus_x",
    "unc_mean",
    "unc_max",
    "pred_conf_x",
    "refined_conf_x",
    "abs_conf_delta",
    "pred_phase_grad",
    "refined_phase_grad",
    "abs_phase_grad_delta",
    "x_depth_grad",
    "refined_depth_grad",
    "depth_delta_grad",
    "fringe_grad",
    "coord_x",
    "coord_y",
]


def grad_mag(x: torch.Tensor) -> torch.Tensor:
    dx = F.pad(x[..., 1:] - x[..., :-1], (0, 1, 0, 0))
    dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx * dx + dy * dy + 1e-12)


def phase_grad(phi: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(grad_mag(phi[:, 0:1]) ** 2 + grad_mag(phi[:, 1:2]) ** 2 + 1e-12)


def tensor_mm(pred_norm: torch.Tensor, batch: Dict[str, object]) -> torch.Tensor:
    return pred_to_depth_mm(pred_norm, batch).float()


def rmse_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool) & np.isfinite(pred) & np.isfinite(target)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean((pred[m] - target[m]) ** 2)))


def norm_to_mm(norm: np.ndarray, scale: float, center: float) -> np.ndarray:
    return norm.astype(np.float32) * float(scale) + float(center)


def build_features(
    batch: Dict[str, object],
    base_norm: torch.Tensor,
    x_norm: torch.Tensor,
    refined_norm: torch.Tensor,
    phi_pred: torch.Tensor,
    phi_refined: torch.Tensor,
    phi_unc: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    fringe = batch["fringe"].to(device, non_blocking=True).float()  # type: ignore[index]
    xy = batch["xy"].to(device, non_blocking=True).float()  # type: ignore[index]
    ref_minus_x = refined_norm - x_norm
    ref_minus_base = refined_norm - base_norm
    base_minus_x = base_norm - x_norm
    unc_mean = phi_unc.mean(dim=1, keepdim=True)
    unc_max = phi_unc.max(dim=1, keepdim=True).values
    pred_conf = torch.clamp(phi_pred[:, 3:4], 0.0, 1.0)
    refined_conf = torch.clamp(phi_refined[:, 3:4], 0.0, 1.0)
    pred_pg = phase_grad(phi_pred[:, 0:2])
    refined_pg = phase_grad(phi_refined[:, 0:2])
    x_grad = grad_mag(x_norm)
    refined_grad = grad_mag(refined_norm)
    delta_grad = grad_mag(ref_minus_x)
    fringe_grad = grad_mag(fringe)
    feats = [
        ref_minus_x,
        torch.abs(ref_minus_x),
        ref_minus_base,
        torch.abs(ref_minus_base),
        base_minus_x,
        unc_mean,
        unc_max,
        pred_conf,
        refined_conf,
        torch.abs(refined_conf - pred_conf),
        pred_pg,
        refined_pg,
        torch.abs(refined_pg - pred_pg),
        x_grad,
        refined_grad,
        delta_grad,
        fringe_grad,
        xy[:, 0:1],
        xy[:, 1:2],
    ]
    return torch.cat(feats, dim=1)


class ReliabilityMLP(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def balanced_indices(label: torch.Tensor, mask: torch.Tensor, max_count: int) -> torch.Tensor:
    valid = torch.nonzero(mask.reshape(-1) > 0.5, as_tuple=False).flatten()
    if valid.numel() == 0:
        return valid
    lab = label.reshape(-1)
    pos = valid[lab[valid] > 0.5]
    neg = valid[lab[valid] <= 0.5]
    half = max(1, max_count // 2)
    picks: List[torch.Tensor] = []
    for pool, cap in [(pos, half), (neg, max_count - min(half, pos.numel()))]:
        if pool.numel() == 0 or cap <= 0:
            continue
        if pool.numel() > cap:
            pool = pool[torch.randperm(pool.numel(), device=pool.device)[:cap]]
        picks.append(pool)
    if not picks:
        pool = valid
        if pool.numel() > max_count:
            pool = pool[torch.randperm(pool.numel(), device=pool.device)[:max_count]]
        return pool
    idx = torch.cat(picks)
    if idx.numel() > max_count:
        idx = idx[torch.randperm(idx.numel(), device=idx.device)[:max_count]]
    return idx


def sample_train_pixels(
    feats: torch.Tensor,
    anchor_norm: torch.Tensor,
    refined_norm: torch.Tensor,
    batch: Dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = batch["depth_raw"].to(device, non_blocking=True).float()  # type: ignore[index]
    mask = batch["object_mask"].to(device, non_blocking=True).float()  # type: ignore[index]
    anchor_mm = tensor_mm(anchor_norm, batch)
    refined_mm = tensor_mm(refined_norm, batch)
    err_anchor = torch.abs(anchor_mm - target)
    err_ref = torch.abs(refined_mm - target)
    label = (err_ref + float(args.label_margin_mm) < err_anchor).float()
    gap = torch.clamp(torch.abs(err_anchor - err_ref) / max(float(args.weight_scale_mm), 1e-6), 0.05, 3.0)
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ws: List[np.ndarray] = []
    for j in range(feats.shape[0]):
        idx = balanced_indices(label[j], mask[j], int(args.train_pixels_per_sample))
        if idx.numel() == 0:
            continue
        f = feats[j].permute(1, 2, 0).reshape(-1, feats.shape[1])[idx]
        xs.append(f.detach().cpu().numpy().astype(np.float32))
        ys.append(label[j].reshape(-1)[idx].detach().cpu().numpy().astype(np.float32))
        ws.append(gap[j].reshape(-1)[idx].detach().cpu().numpy().astype(np.float32))
    if not xs:
        return np.empty((0, feats.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(ws, axis=0)


def load_all_models(args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    loaders_obj = create_loaders(args)
    args.cond_channels = int(loaders_obj["cond_channels"])
    x_dir = Path(args.x_diag_dir) / "direction_x"
    base_model, _ = load_base_model(args.base_ckpt, args.cond_channels, device)
    phi_model = load_model(x_dir / "phi_predictor" / "checkpoints" / "best.pt", args.cond_channels, 4, args, device)
    old_dp = load_model(x_dir / "phase_depth_evidence" / "checkpoints" / "best.pt", args.cond_channels + 4, 1, args, device)
    oracle_dp = load_model(x_dir / "phase_depth_oracle_evidence" / "checkpoints" / "best.pt", args.cond_channels + 4, 1, args, device)
    refined_depth = load_model(Path(args.refined_depth_ckpt), args.cond_channels + 4, 1, args, device)
    posterior = load_phase_posterior(Path(args.phase_posterior_ckpt), args.cond_channels, args, device)
    gate = json.loads(Path(args.summary_path).read_text(encoding="utf-8"))["gate"]
    return {
        "loaders_obj": loaders_obj,
        "base": base_model,
        "phi": phi_model,
        "old_dp": old_dp,
        "oracle_dp": oracle_dp,
        "refined_depth": refined_depth,
        "posterior": posterior,
        "sample_gate": gate,
    }


@torch.no_grad()
def forward_pack(batch: Dict[str, object], models: Dict[str, object], args: argparse.Namespace, device: torch.device) -> Dict[str, torch.Tensor]:
    base_norm = forward_direct(models["base"], batch, device)[:, :1]  # type: ignore[arg-type]
    phi_pred, phi_refined, phi_unc = refined_phi(
        models["posterior"],  # type: ignore[arg-type]
        models["phi"],  # type: ignore[arg-type]
        batch,
        args.phase_sample_steps,
        args.phase_ensemble_size,
    )
    x_norm = dir_dp_predict(models["phi"], models["old_dp"], batch, "x", device)  # type: ignore[arg-type]
    refined_norm = depth_with_phi(models["refined_depth"], batch, phi_refined, device)  # type: ignore[arg-type]
    oracle_norm = dir_dp_oracle(models["oracle_dp"], batch, "x", device)  # type: ignore[arg-type]
    feats = build_features(batch, base_norm, x_norm, refined_norm, phi_pred, phi_refined, phi_unc, device)
    return {
        "base_norm": base_norm,
        "x_norm": x_norm,
        "anchor_norm": choose_anchor_norm(base_norm, x_norm, args.anchor_mode),
        "refined_norm": refined_norm,
        "oracle_norm": oracle_norm,
        "phi_unc": phi_unc,
        "features": feats,
    }


def choose_anchor_norm(base_norm: torch.Tensor, x_norm: torch.Tensor, anchor_mode: str) -> torch.Tensor:
    if anchor_mode == "x_phase":
        return x_norm
    if anchor_mode == "base":
        return base_norm
    if anchor_mode == "base_x_mean":
        return torch.clamp(0.5 * (base_norm + x_norm), -1.0, 1.0)
    raise ValueError(f"unknown anchor_mode: {anchor_mode}")


def collect_training_set(models: Dict[str, object], args: argparse.Namespace, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = models["loaders_obj"]["loaders"]["train"]  # type: ignore[index]
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ws: List[np.ndarray] = []
    total = 0
    for batch in loader:
        pack = forward_pack(batch, models, args, device)
        x, y, w = sample_train_pixels(pack["features"], pack["anchor_norm"], pack["refined_norm"], batch, args, device)
        if x.shape[0] == 0:
            continue
        remain = int(args.max_train_pixels) - total
        if remain <= 0:
            break
        if x.shape[0] > remain:
            sel = np.random.choice(x.shape[0], size=remain, replace=False)
            x, y, w = x[sel], y[sel], w[sel]
        xs.append(x)
        ys.append(y)
        ws.append(w)
        total += x.shape[0]
        if total >= int(args.max_train_pixels):
            break
    if not xs:
        raise RuntimeError("no training pixels sampled")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(ws, axis=0)


def train_mlp(x: np.ndarray, y: np.ndarray, w: np.ndarray, args: argparse.Namespace, device: torch.device) -> Tuple[ReliabilityMLP, np.ndarray, np.ndarray, List[Dict[str, float]]]:
    mean = x.mean(axis=0).astype(np.float32)
    std = (x.std(axis=0) + 1e-6).astype(np.float32)
    x_std = (x - mean[None, :]) / std[None, :]
    xt = torch.from_numpy(x_std).to(device)
    yt = torch.from_numpy(y).to(device)
    wt = torch.from_numpy(w).to(device)
    model = ReliabilityMLP(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.selector_lr), weight_decay=float(args.selector_weight_decay))
    history: List[Dict[str, float]] = []
    n = xt.shape[0]
    for ep in range(1, int(args.selector_epochs) + 1):
        t0 = time.time()
        model.train()
        order = torch.randperm(n, device=device)
        losses = []
        accs = []
        for start in range(0, n, int(args.selector_batch_pixels)):
            idx = order[start:start + int(args.selector_batch_pixels)]
            logits = model(xt[idx])
            loss_raw = F.binary_cross_entropy_with_logits(logits, yt[idx], reduction="none")
            loss = (loss_raw * wt[idx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
            pred = (torch.sigmoid(logits) > 0.5).float()
            accs.append(float((pred == yt[idx]).float().mean().item()))
        row = {
            "epoch": float(ep),
            "loss": float(np.mean(losses)),
            "train_pixel_acc": float(np.mean(accs)),
            "seconds": float(time.time() - t0),
        }
        history.append(row)
        print(json.dumps({"stage": "reliability_selector", **row}, ensure_ascii=False), flush=True)
    model.eval()
    return model, mean, std, history


def rmse_from_norm(norm: np.ndarray, entry: Dict[str, object]) -> float:
    pred = norm_to_mm(norm, float(entry["scale_mm"]), float(entry["center_mm"]))
    return rmse_np(pred, entry["target"], entry["mask"])  # type: ignore[arg-type]


def local_oracle(entry: Dict[str, object], a_key: str, b_key: str) -> float:
    target = entry["target"]  # type: ignore[assignment]
    a = entry[a_key]  # type: ignore[assignment]
    b = entry[b_key]  # type: ignore[assignment]
    best = np.where(np.abs(b - target) < np.abs(a - target), b, a)
    return rmse_np(best, target, entry["mask"])  # type: ignore[arg-type]


def summarize_entries(entries: List[Dict[str, object]], pred_key: str | None = None) -> float:
    if pred_key is None:
        return float("nan")
    vals = [float(e[pred_key]) for e in entries]
    return float(np.mean(vals)) if vals else float("nan")


def eval_entries(entries: List[Dict[str, object]], gate: Dict[str, object]) -> Tuple[float, float]:
    vals: List[float] = []
    accepts: List[float] = []
    for e in entries:
        anchor = e.get("anchor_norm", e["x_norm"])  # type: ignore[assignment]
        ref = e["refined_norm"]  # type: ignore[assignment]
        alpha = float(gate["alpha"])
        if gate["kind"] == "rule":
            use = np.ones_like(anchor, dtype=bool)
            if "unc_max" in gate:
                use &= e["unc"] <= float(gate["unc_max"])  # type: ignore[operator]
            if "delta_max" in gate:
                use &= e["delta_x"] <= float(gate["delta_max"])  # type: ignore[operator]
            final = np.clip(anchor + alpha * (ref - anchor) * use.astype(np.float32), -1.0, 1.0)
            accepts.append(float(np.sum(use & e["mask"]) / max(1, np.sum(e["mask"]))))  # type: ignore[operator]
        elif gate["kind"] == "mlp_hard":
            use = e["prob"] >= float(gate["threshold"])  # type: ignore[operator]
            final = np.clip(anchor + alpha * (ref - anchor) * use.astype(np.float32), -1.0, 1.0)
            accepts.append(float(np.sum(use & e["mask"]) / max(1, np.sum(e["mask"]))))  # type: ignore[operator]
        elif gate["kind"] == "mlp_soft":
            prob = e["prob"].astype(np.float32)  # type: ignore[union-attr]
            final = np.clip(anchor + alpha * (ref - anchor) * prob, -1.0, 1.0)
            accepts.append(float(np.mean(prob[e["mask"].astype(bool)])))  # type: ignore[index]
        else:
            raise ValueError(str(gate["kind"]))
        vals.append(rmse_from_norm(final, e))
    return (float(np.mean(vals)) if vals else float("nan"), float(np.mean(accepts)) if accepts else float("nan"))


def search_rule_gate(val_entries: List[Dict[str, object]]) -> Dict[str, object]:
    unc_vals = np.concatenate([e["unc"][e["mask"].astype(bool)] for e in val_entries]).astype(np.float32)  # type: ignore[index]
    delta_vals = np.concatenate([e["delta_x"][e["mask"].astype(bool)] for e in val_entries]).astype(np.float32)  # type: ignore[index]
    unc_grid = sorted(set(float(x) for x in np.percentile(unc_vals, [10, 20, 35, 50, 65, 80, 90, 97, 100])))
    delta_grid = sorted(set(float(x) for x in np.percentile(delta_vals, [10, 20, 35, 50, 65, 80, 90, 97, 100])))
    candidates: List[Dict[str, object]] = []
    for alpha in [0.25, 0.5, 0.75, 1.0]:
        for t in unc_grid:
            candidates.append({"kind": "rule", "rule": "unc", "alpha": alpha, "unc_max": t})
        for t in delta_grid:
            candidates.append({"kind": "rule", "rule": "delta_x", "alpha": alpha, "delta_max": t})
        for tu in unc_grid:
            for td in delta_grid:
                candidates.append({"kind": "rule", "rule": "unc_and_delta_x", "alpha": alpha, "unc_max": tu, "delta_max": td})
    best: Dict[str, object] = {"kind": "rule", "rule": "none", "alpha": 0.0, "val_rmse": float("inf"), "accept": 0.0}
    for gate in candidates:
        rmse, acc = eval_entries(val_entries, gate)
        if rmse < float(best["val_rmse"]):
            best = {**gate, "val_rmse": rmse, "accept": acc}
    return best


def search_mlp_gate(val_entries: List[Dict[str, object]]) -> Dict[str, object]:
    candidates: List[Dict[str, object]] = []
    for alpha in [0.25, 0.5, 0.75, 1.0]:
        candidates.append({"kind": "mlp_soft", "alpha": alpha, "threshold": -1.0})
        for t in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
            candidates.append({"kind": "mlp_hard", "alpha": alpha, "threshold": t})
    best: Dict[str, object] = {"kind": "mlp_hard", "alpha": 0.0, "threshold": 1.0, "val_rmse": float("inf"), "accept": 0.0}
    for gate in candidates:
        rmse, acc = eval_entries(val_entries, gate)
        if rmse < float(best["val_rmse"]):
            best = {**gate, "val_rmse": rmse, "accept": acc}
    return best


@torch.no_grad()
def collect_eval_entries(
    split: str,
    models: Dict[str, object],
    selector: ReliabilityMLP,
    mean: np.ndarray,
    std: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> List[Dict[str, object]]:
    loader = models["loaders_obj"]["loaders"][split]  # type: ignore[index]
    entries: List[Dict[str, object]] = []
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)
    for batch in loader:
        pack = forward_pack(batch, models, args, device)
        feats = pack["features"]
        b, c, h, w = feats.shape
        flat = feats.permute(0, 2, 3, 1).reshape(-1, c)
        prob = torch.sigmoid(selector((flat - mean_t) / std_t)).reshape(b, h, w).detach().cpu().numpy().astype(np.float32)
        base_norm = pack["base_norm"].detach().cpu().numpy()[:, 0].astype(np.float32)
        x_norm = pack["x_norm"].detach().cpu().numpy()[:, 0].astype(np.float32)
        anchor_norm = pack["anchor_norm"].detach().cpu().numpy()[:, 0].astype(np.float32)
        refined_norm = pack["refined_norm"].detach().cpu().numpy()[:, 0].astype(np.float32)
        oracle_norm = pack["oracle_norm"].detach().cpu().numpy()[:, 0].astype(np.float32)
        unc = pack["phi_unc"].mean(dim=1).detach().cpu().numpy().astype(np.float32)
        delta_x = np.abs(refined_norm - x_norm).astype(np.float32)
        target = batch["depth_raw"].detach().cpu().numpy()[:, 0].astype(np.float32)  # type: ignore[index]
        mask = batch["object_mask"].detach().cpu().numpy()[:, 0].astype(bool)  # type: ignore[index]
        scale = batch["scale_mm"].detach().cpu().numpy().astype(np.float32)  # type: ignore[index]
        center = batch["center_mm"].detach().cpu().numpy().astype(np.float32)  # type: ignore[index]
        base_mm = tensor_mm(pack["base_norm"], batch).detach().cpu().numpy()[:, 0].astype(np.float32)
        x_mm = tensor_mm(pack["x_norm"], batch).detach().cpu().numpy()[:, 0].astype(np.float32)
        anchor_mm = tensor_mm(pack["anchor_norm"], batch).detach().cpu().numpy()[:, 0].astype(np.float32)
        refined_mm = tensor_mm(pack["refined_norm"], batch).detach().cpu().numpy()[:, 0].astype(np.float32)
        oracle_mm = tensor_mm(pack["oracle_norm"], batch).detach().cpu().numpy()[:, 0].astype(np.float32)
        for j, sid in enumerate(list(batch["sample_id"])):  # type: ignore[arg-type]
            item = compact_item(batch, j, pack["base_norm"], pack["refined_norm"], pack["phi_unc"])
            sample_rcpc_norm, sample_use = rcpc_pred(item, models["sample_gate"])  # type: ignore[arg-type]
            sample_rcpc_mm = pred_to_depth_mm(sample_rcpc_norm, {
                "scale_mm": batch["scale_mm"][j:j + 1].detach().cpu(),  # type: ignore[index]
                "center_mm": batch["center_mm"][j:j + 1].detach().cpu(),  # type: ignore[index]
                "depth_raw": batch["depth_raw"][j:j + 1].detach().cpu(),  # type: ignore[index]
                "object_mask": batch["object_mask"][j:j + 1].detach().cpu(),  # type: ignore[index]
                "valid_mask": batch["valid_mask"][j:j + 1].detach().cpu(),  # type: ignore[index]
            }).detach().cpu().numpy()[0, 0].astype(np.float32)
            entry: Dict[str, object] = {
                "split": split,
                "sample_id": str(sid),
                "object_id": int(batch["object_id"][j].item()),  # type: ignore[index]
                "pose_id": int(batch["pose_id"][j].item()),  # type: ignore[index]
                "scale_mm": float(scale[j]),
                "center_mm": float(center[j]),
                "target": target[j],
                "mask": mask[j],
                "base_norm": base_norm[j],
                "x_norm": x_norm[j],
                "anchor_norm": anchor_norm[j],
                "refined_norm": refined_norm[j],
                "oracle_norm": oracle_norm[j],
                "prob": prob[j],
                "unc": unc[j],
                "delta_x": delta_x[j],
                "base": base_mm[j],
                "x": x_mm[j],
                "anchor": anchor_mm[j],
                "refined": refined_mm[j],
                "oracle": oracle_mm[j],
                "sample_rcpc": sample_rcpc_mm,
                "sample_rcpc_used": bool(sample_use),
            }
            for name in ["base", "x", "anchor", "refined", "oracle", "sample_rcpc"]:
                entry[f"{name}_rmse"] = rmse_np(entry[name], target[j], mask[j])  # type: ignore[arg-type]
            entry["local_oracle_x_refined_rmse"] = local_oracle(entry, "x", "refined")
            entry["local_oracle_base_refined_rmse"] = local_oracle(entry, "base", "refined")
            entry["local_oracle_anchor_refined_rmse"] = local_oracle(entry, "anchor", "refined")
            entries.append(entry)
    return entries


def split_metrics(entries: List[Dict[str, object]], gate: Dict[str, object]) -> Dict[str, object]:
    rule_rmse, rule_acc = eval_entries(entries, gate) if gate["kind"] == "rule" else (float("nan"), float("nan"))
    mlp_rmse, mlp_acc = eval_entries(entries, gate) if str(gate["kind"]).startswith("mlp") else (float("nan"), float("nan"))
    return {
        "n": len(entries),
        "anchor": summarize_entries(entries, "anchor_rmse"),
        "base": summarize_entries(entries, "base_rmse"),
        "x_phase": summarize_entries(entries, "x_rmse"),
        "refined": summarize_entries(entries, "refined_rmse"),
        "sample_rcpc": summarize_entries(entries, "sample_rcpc_rmse"),
        "true_x_oracle": summarize_entries(entries, "oracle_rmse"),
        "local_oracle_x_refined": summarize_entries(entries, "local_oracle_x_refined_rmse"),
        "local_oracle_base_refined": summarize_entries(entries, "local_oracle_base_refined_rmse"),
        "local_oracle_anchor_refined": summarize_entries(entries, "local_oracle_anchor_refined_rmse"),
        "gate_rmse": mlp_rmse if str(gate["kind"]).startswith("mlp") else rule_rmse,
        "gate_accept": mlp_acc if str(gate["kind"]).startswith("mlp") else rule_acc,
    }


def write_rows(entries_by_split: Dict[str, List[Dict[str, object]]], rule_gate: Dict[str, object], mlp_gate: Dict[str, object], out_dir: Path) -> None:
    keys = [
        "split",
        "sample_id",
        "object_id",
        "pose_id",
        "base_rmse",
        "x_rmse",
        "anchor_rmse",
        "refined_rmse",
        "sample_rcpc_rmse",
        "rule_rmse",
        "mlp_rmse",
        "oracle_rmse",
        "local_oracle_x_refined_rmse",
        "local_oracle_anchor_refined_rmse",
        "rule_accept",
        "mlp_accept",
        "sample_rcpc_used",
    ]
    with (out_dir / "reliability_selector_per_sample.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for split, entries in entries_by_split.items():
            for e in entries:
                rule_rmse, rule_acc = eval_entries([e], rule_gate)
                mlp_rmse, mlp_acc = eval_entries([e], mlp_gate)
                writer.writerow({
                    "split": split,
                    "sample_id": e["sample_id"],
                    "object_id": e["object_id"],
                    "pose_id": e["pose_id"],
                    "base_rmse": e["base_rmse"],
                    "x_rmse": e["x_rmse"],
                    "anchor_rmse": e["anchor_rmse"],
                    "refined_rmse": e["refined_rmse"],
                    "sample_rcpc_rmse": e["sample_rcpc_rmse"],
                    "rule_rmse": rule_rmse,
                    "mlp_rmse": mlp_rmse,
                    "oracle_rmse": e["oracle_rmse"],
                    "local_oracle_x_refined_rmse": e["local_oracle_x_refined_rmse"],
                    "local_oracle_anchor_refined_rmse": e["local_oracle_anchor_refined_rmse"],
                    "rule_accept": rule_acc,
                    "mlp_accept": mlp_acc,
                    "sample_rcpc_used": e["sample_rcpc_used"],
                })


def make_plot(summary: Dict[str, object], out_dir: Path) -> None:
    splits = [s for s in ["val", "test", "ood"] if s in summary["splits"]]  # type: ignore[operator]
    labels = ["Anchor", "Base", "X phase", "Refined", "Sample RCPC", "Rule local", "MLP local", "True X", "Local oracle"]
    values = []
    for split in splits:
        sp = summary["splits"][split]  # type: ignore[index]
        values.append([
            sp["anchor"],
            sp["base"],
            sp["x_phase"],
            sp["refined"],
            sp["sample_rcpc"],
            sp["rule_gate"]["gate_rmse"],
            sp["mlp_gate"]["gate_rmse"],
            sp["true_x_oracle"],
            sp["local_oracle_anchor_refined"],
        ])
    x = np.arange(len(splits))
    width = 0.085
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, label in enumerate(labels):
        ax.bar(x + (i - 4.0) * width, [row[i] for row in values], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Object RMSE")
    ax.set_title("Local RCPC and reliability selector")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "reliability_selector_rmse.png", dpi=160)
    plt.close(fig)


def make_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Local Reliability Selector Report",
        "",
        "This experiment tests fast rule-based local RCPC and a lightweight MLP selector.",
        f"Anchor mode: `{summary['anchor_mode']}`.",
        "",
        "## Gates",
        "",
        "Rule gate:",
        "",
        "```json",
        json.dumps(summary["rule_gate"], indent=2, ensure_ascii=False),
        "```",
        "",
        "MLP gate:",
        "",
        "```json",
        json.dumps(summary["mlp_gate"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## RMSE",
        "",
        "| split | anchor | base | x phase | refined | sample RCPC | rule local | MLP local | true x | local oracle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, data in summary["splits"].items():  # type: ignore[union-attr]
        lines.append(
            f"| {split} | {data['anchor']:.4f} | {data['base']:.4f} | {data['x_phase']:.4f} | {data['refined']:.4f} | "
            f"{data['sample_rcpc']:.4f} | {data['rule_gate']['gate_rmse']:.4f} | "
            f"{data['mlp_gate']['gate_rmse']:.4f} | {data['true_x_oracle']:.4f} | "
            f"{data['local_oracle_anchor_refined']:.4f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "If MLP local improves ordinary test while keeping OOD gains, the next step is seed replication.",
        "If it rejects most refined candidates on OOD, the selector needs OOD-aware reliability features or a patch-level objective.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--teacher_extra_root", required=True)
    parser.add_argument("--ood_root", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--x_diag_dir", required=True)
    parser.add_argument("--phase_posterior_ckpt", required=True)
    parser.add_argument("--refined_depth_ckpt", required=True)
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--ch_mult", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--time_emb_dim", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--phase_residual_scale", type=float, default=0.5)
    parser.add_argument("--phase_sample_steps", type=int, default=12)
    parser.add_argument("--phase_ensemble_size", type=int, default=3)
    parser.add_argument("--cache_features", action="store_true")
    parser.add_argument("--feature_cache_dir", default="")
    parser.add_argument("--train_pixels_per_sample", type=int, default=2048)
    parser.add_argument("--max_train_pixels", type=int, default=700000)
    parser.add_argument("--selector_epochs", type=int, default=25)
    parser.add_argument("--selector_batch_pixels", type=int, default=65536)
    parser.add_argument("--selector_lr", type=float, default=1e-3)
    parser.add_argument("--selector_weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_margin_mm", type=float, default=0.0)
    parser.add_argument("--weight_scale_mm", type=float, default=1.0)
    parser.add_argument("--anchor_mode", choices=["x_phase", "base", "base_x_mean"], default="x_phase")
    args = parser.parse_args()

    set_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    out_dir = Path(args.save_dir) / f"reliability_selector_seed{int(args.seed)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    models = load_all_models(args, device)
    x_train, y_train, w_train = collect_training_set(models, args, device)
    train_stats = {
        "pixels": int(x_train.shape[0]),
        "features": FEATURE_NAMES,
        "positive_fraction": float(y_train.mean()),
        "weight_mean": float(w_train.mean()),
    }
    print(json.dumps({"stage": "selector_train_data", **train_stats}, ensure_ascii=False), flush=True)
    selector, mean, std, history = train_mlp(x_train, y_train, w_train, args, device)

    entries_by_split: Dict[str, List[Dict[str, object]]] = {}
    for split in ["val", "test", "ood"]:
        if split in models["loaders_obj"]["loaders"]:  # type: ignore[index]
            entries_by_split[split] = collect_eval_entries(split, models, selector, mean, std, args, device)

    rule_gate = search_rule_gate(entries_by_split["val"])
    mlp_gate = search_mlp_gate(entries_by_split["val"])
    split_summary: Dict[str, object] = {}
    for split, entries in entries_by_split.items():
        split_summary[split] = {
            **split_metrics(entries, {"kind": "mlp_soft", "alpha": 0.0}),
            "rule_gate": split_metrics(entries, rule_gate),
            "mlp_gate": split_metrics(entries, mlp_gate),
        }

    summary = {
        "stage": "local_reliability_selector",
        "seed": args.seed,
        "anchor_mode": args.anchor_mode,
        "elapsed_seconds": time.time() - t_start,
        "train_stats": train_stats,
        "train_history": history,
        "rule_gate": rule_gate,
        "mlp_gate": mlp_gate,
        "splits": split_summary,
        "files": {
            "summary": str(out_dir / "reliability_selector_summary.json"),
            "report": str(out_dir / "reliability_selector_report.md"),
            "per_sample": str(out_dir / "reliability_selector_per_sample.csv"),
            "plot": str(out_dir / "reliability_selector_rmse.png"),
            "checkpoint": str(out_dir / "reliability_selector.pt"),
        },
    }
    torch.save({
        "model_state_dict": selector.state_dict(),
        "feature_names": FEATURE_NAMES,
        "mean": mean,
        "std": std,
        "args": vars(args),
        "summary": summary,
    }, out_dir / "reliability_selector.pt")
    write_rows(entries_by_split, rule_gate, mlp_gate, out_dir)
    make_plot(summary, out_dir)
    (out_dir / "reliability_selector_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "reliability_selector_report.md").write_text(make_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
