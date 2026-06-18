"""Pilot full PIP/RCPC validation on single_frame_3d teacher-extra data.

This script intentionally keeps test-time inputs legal: formal predictions use
only input_vertical_0120.bmp and single-frame derived features. PMP fields from
teacher_extra.npz are used as train-time supervision and diagnostics only.
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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm

from models.unet import ConditionalUNet
from physics_features_pip import FEATURE_ORDER, build_pip_features
from train_single_frame3d_physics_diffusion import (
    ResidualPosterior,
    build_model,
    charbonnier,
    finite_float,
    forward_direct,
    gradient_loss,
    load_base_model,
    masked_mse,
    normalization_from_stats,
    pred_to_depth_mm,
    read_manifest,
    robust_unit,
    row_from_prediction,
    save_checkpoint,
    set_seed,
    summarize_rows,
    train_weight,
    xy_maps,
)


TEACHER_FIELDS = {
    "phase_y",
    "phase_x",
    "bc_y",
    "bc_x",
    "wrapped_phase_y",
    "wrapped_phase_x",
    "phase_conf_y",
    "phase_conf_x",
    "fringe_order_y",
    "fringe_order_x",
}


def sample_paths(root: Path, row: Dict[str, str]) -> Dict[str, Path]:
    domain = str(row["domain"])
    obj = int(row["object_id"])
    pose = int(row["pose_id"])
    sample_dir = root / "samples" / domain / f"obj{obj:03d}" / f"pose{pose:02d}"
    return {
        "sample_dir": sample_dir,
        "primary": sample_dir / "input_vertical_0120.bmp",
        "labels": sample_dir / "labels.npz",
    }


def teacher_extra_path(extra_root: Path, row: Dict[str, str]) -> Path:
    domain = str(row["domain"])
    obj = int(row["object_id"])
    pose = int(row["pose_id"])
    return extra_root / "samples" / domain / f"obj{obj:03d}" / f"pose{pose:02d}" / "teacher_extra.npz"


def load_gray(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing image: {path}")
    return (np.asarray(Image.open(path).convert("L")).astype(np.float32) / 255.0).astype(np.float32)


def object_pose_from_row(row: Dict[str, str]) -> Tuple[int, int]:
    return int(row["object_id"]), int(row["pose_id"])


class FullTeacherDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        teacher_extra_root: str | Path,
        split: str,
        norm: Optional[Dict[str, float]] = None,
        ood_root: str | Path | None = None,
        cache_features: bool = False,
        feature_cache_dir: str | Path | None = None,
    ) -> None:
        self.root = Path(data_root)
        self.extra_root = Path(teacher_extra_root)
        self.ood_root = Path(ood_root) if ood_root else None
        self.split = str(split)
        self.norm = norm or normalization_from_stats(self.root)
        self.center_mm = float(self.norm["center_mm"])
        self.scale_mm = float(self.norm["scale_mm"])
        self.cache_features = bool(cache_features)
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        self.feature_cache: Dict[str, Tuple[np.ndarray, Dict[str, float]]] = {}
        if self.split == "ood":
            rows = []
            manifest = self.extra_root / "teacher_extra_manifest.csv"
            with manifest.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("split") == "extra_unlisted":
                        rows.append(dict(row))
            if self.ood_root is None:
                raise ValueError("--ood_root is required for split=ood")
            self.rows = rows
        else:
            self.rows = read_manifest(self.root / f"{self.split}_manifest.csv")
        self.channel_names = ["input_vertical_0120"] + [name for name in FEATURE_ORDER[1:]]
        self.dp_channel_names = self.channel_names + [
            "pred_wrapped_y_sin",
            "pred_wrapped_y_cos",
            "pred_wrapped_x_sin",
            "pred_wrapped_x_cos",
            "pred_order_y01",
            "pred_order_x01",
            "pred_conf_y",
            "pred_conf_x",
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def _phys(self, sample_id: str, raw: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        if self.cache_features and sample_id in self.feature_cache:
            return self.feature_cache[sample_id]
        cache_path = self.feature_cache_dir / f"{sample_id}.npz" if self.feature_cache_dir else None
        if cache_path is not None and cache_path.exists():
            with np.load(cache_path) as z:
                features = z["features"].astype(np.float32)
                carrier = {
                    "dx": float(z["carrier_dx"][()]) if "carrier_dx" in z.files else math.nan,
                    "dy": float(z["carrier_dy"][()]) if "carrier_dy" in z.files else math.nan,
                    "spectral_confidence": float(z["carrier_conf"][()]) if "carrier_conf" in z.files else math.nan,
                }
            if self.cache_features:
                self.feature_cache[sample_id] = (features, carrier)
            return features, carrier
        features, carrier = build_pip_features(raw[None, :, :].astype(np.float32))
        if self.cache_features:
            self.feature_cache[sample_id] = (features, carrier)
        return features, carrier

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[int(idx)]
        sample_id = str(row.get("sample_id") or row.get("\ufeffsample_id"))
        paths = sample_paths(self.ood_root if self.split == "ood" else self.root, row)
        raw = load_gray(paths["primary"])
        with np.load(paths["labels"]) as z:
            depth_z = finite_float(z["depth_z"])
            valid_mask = z["valid_mask"].astype(np.float32)
            object_mask = z["object_mask"].astype(np.float32)
            phase_y = finite_float(z["phase_y"])
            phase_x = finite_float(z["phase_x"])
            bc_y = robust_unit(finite_float(z["bc_y"]))
            bc_x = robust_unit(finite_float(z["bc_x"]))
        tpath = teacher_extra_path(self.extra_root, row)
        if not tpath.exists():
            raise FileNotFoundError(f"missing teacher_extra: {tpath}")
        with np.load(tpath) as t:
            wrapped_y = finite_float(t["wrapped_phase_y"])
            wrapped_x = finite_float(t["wrapped_phase_x"])
            conf_y = np.clip(finite_float(t["phase_conf_y"]), 0.0, 1.0)
            conf_x = np.clip(finite_float(t["phase_conf_x"]), 0.0, 1.0)
            order_y = finite_float(t["fringe_order_y"])
            order_x = finite_float(t["fringe_order_x"])

        h, w = raw.shape
        xy = xy_maps(h, w)
        features, carrier = self._phys(sample_id, raw)
        cond = np.concatenate([raw[None, :, :], features[1:]], axis=0).astype(np.float32)
        phase_target = np.stack([
            np.sin(phase_y),
            np.cos(phase_y),
            np.sin(phase_x),
            np.cos(phase_x),
        ], axis=0).astype(np.float32)
        phase_conf = np.stack([bc_y, bc_y, bc_x, bc_x], axis=0).astype(np.float32)
        phi_target = np.stack([
            np.sin(wrapped_y),
            np.cos(wrapped_y),
            np.sin(wrapped_x),
            np.cos(wrapped_x),
            np.clip(order_y, 0.0, 1.0),
            np.clip(order_x, 0.0, 1.0),
            conf_y,
            conf_x,
        ], axis=0).astype(np.float32)
        phi_weight = np.stack([conf_y, conf_y, conf_x, conf_x, conf_y, conf_x, valid_mask, valid_mask], axis=0).astype(np.float32)
        depth_norm = np.clip((depth_z - self.center_mm) / self.scale_mm, -1.0, 1.0).astype(np.float32)
        obj, pose = object_pose_from_row(row)
        return {
            "sample_id": sample_id,
            "object_id": torch.tensor(obj, dtype=torch.long),
            "pose_id": torch.tensor(pose, dtype=torch.long),
            "cond": torch.from_numpy(cond),
            "fringe": torch.from_numpy(raw[None, :, :].astype(np.float32)),
            "depth": torch.from_numpy(depth_norm[None, :, :]),
            "depth_raw": torch.from_numpy(depth_z[None, :, :]),
            "valid_mask": torch.from_numpy(valid_mask[None, :, :].astype(np.float32)),
            "object_mask": torch.from_numpy(object_mask[None, :, :].astype(np.float32)),
            "phase_target": torch.from_numpy(phase_target),
            "phase_conf": torch.from_numpy(phase_conf),
            "phi_target": torch.from_numpy(phi_target),
            "phi_weight": torch.from_numpy(phi_weight),
            "teacher_conf": torch.from_numpy(((conf_y + conf_x) * 0.5)[None, :, :].astype(np.float32)),
            "xy": torch.from_numpy(xy.astype(np.float32)),
            "scale_mm": torch.tensor(self.scale_mm, dtype=torch.float32),
            "center_mm": torch.tensor(self.center_mm, dtype=torch.float32),
            "carrier_dx": torch.tensor(float(carrier.get("dx", math.nan)), dtype=torch.float32),
            "carrier_dy": torch.tensor(float(carrier.get("dy", math.nan)), dtype=torch.float32),
            "legal_single_frame": True,
        }


def create_loaders(args: argparse.Namespace) -> Dict[str, object]:
    norm = normalization_from_stats(args.data_root)
    out: Dict[str, object] = {}
    datasets = {
        "train": FullTeacherDataset(args.data_root, args.teacher_extra_root, "train", norm=norm, ood_root=args.ood_root, cache_features=args.cache_features, feature_cache_dir=args.feature_cache_dir or None),
        "val": FullTeacherDataset(args.data_root, args.teacher_extra_root, "val", norm=norm, ood_root=args.ood_root, cache_features=args.cache_features, feature_cache_dir=args.feature_cache_dir or None),
        "test": FullTeacherDataset(args.data_root, args.teacher_extra_root, "test", norm=norm, ood_root=args.ood_root, cache_features=args.cache_features, feature_cache_dir=args.feature_cache_dir or None),
    }
    if args.ood_root:
        datasets["ood"] = FullTeacherDataset(args.data_root, args.teacher_extra_root, "ood", norm=norm, ood_root=args.ood_root, cache_features=args.cache_features, feature_cache_dir=args.feature_cache_dir or None)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, sampler=RandomSampler(datasets["train"]), num_workers=args.num_workers, pin_memory=True),
        "val": DataLoader(datasets["val"], batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
        "test": DataLoader(datasets["test"], batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
    }
    if "ood" in datasets:
        loaders["ood"] = DataLoader(datasets["ood"], batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    out["datasets"] = datasets
    out["loaders"] = loaders
    out["norm"] = norm
    out["cond_channels"] = len(datasets["train"].channel_names)
    out["dp_cond_channels"] = len(datasets["train"].dp_channel_names)
    out["channel_names"] = datasets["train"].channel_names
    out["dp_channel_names"] = datasets["train"].dp_channel_names
    out["split_counts"] = {k: len(v) for k, v in datasets.items()}
    return out


def zero_time_forward(model: torch.nn.Module, cond: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros((cond.shape[0], 1, cond.shape[-2], cond.shape[-1]), device=cond.device)
    t = torch.zeros((cond.shape[0],), dtype=torch.long, device=cond.device)
    return torch.tanh(model(zeros, t, cond))


def phi_predict(phi_model: torch.nn.Module, batch: Dict[str, object], device: torch.device) -> torch.Tensor:
    cond = batch["cond"].to(device, non_blocking=True).float()  # type: ignore[index]
    raw = zero_time_forward(phi_model, cond)
    out = raw.clone()
    out[:, 4:8] = (out[:, 4:8] + 1.0) * 0.5
    return out


def dp_predict(phi_model: torch.nn.Module, dp_model: torch.nn.Module, batch: Dict[str, object], device: torch.device) -> torch.Tensor:
    cond = batch["cond"].to(device, non_blocking=True).float()  # type: ignore[index]
    phi = phi_predict(phi_model, batch, device).detach()
    return zero_time_forward(dp_model, torch.cat([cond, phi], dim=1))[:, :1]


def phi_loss(pred: torch.Tensor, batch: Dict[str, object], device: torch.device) -> torch.Tensor:
    target = batch["phi_target"].to(device, non_blocking=True).float()  # type: ignore[index]
    weight = batch["phi_weight"].to(device, non_blocking=True).float()  # type: ignore[index]
    valid = batch["valid_mask"].to(device, non_blocking=True).float()  # type: ignore[index]
    pred_sc = pred[:, :4]
    pred_rest = pred[:, 4:8]
    loss_sc = charbonnier(pred_sc, target[:, :4], weight=weight[:, :4] * valid)
    loss_rest = masked_mse(pred_rest, target[:, 4:8], weight=weight[:, 4:8] * valid)
    return loss_sc + loss_rest


def train_phi(args: argparse.Namespace, loaders: Dict[str, object], device: torch.device) -> torch.nn.Module:
    model = build_model(args.cond_channels, 8, args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.phi_epochs))
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))
    best = float("inf")
    history: List[Dict[str, object]] = []
    save_dir = Path(args.save_dir) / "phi_predictor"
    for ep in range(1, args.phi_epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"P_phi {ep}/{args.phi_epochs}"):
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                pred = phi_predict(model, batch, device)
                loss = phi_loss(pred, batch, device)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.item())
            seen += 1
        sched.step()
        val = eval_phi_loss(model, loaders["val"], device)
        log = {"epoch": ep, "train_loss": total / max(1, seen), "val_loss": val, "seconds": time.time() - t0}
        history.append(log)
        print(json.dumps({"stage": "phi", **log}, ensure_ascii=False), flush=True)
        if val < best:
            best = val
            save_checkpoint(save_dir / "checkpoints" / "best.pt", ep, model, opt, scaler, args, best, history)
    ckpt = torch.load(str(save_dir / "checkpoints" / "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def eval_phi_loss(model: torch.nn.Module, loader: Iterable[Dict[str, object]], device: torch.device) -> float:
    model.eval()
    vals = []
    for batch in loader:
        vals.append(float(phi_loss(phi_predict(model, batch, device), batch, device).item()))
    return float(np.mean(vals)) if vals else float("nan")


def train_dp(args: argparse.Namespace, loaders: Dict[str, object], phi_model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    model = build_model(args.dp_cond_channels, 1, args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.dp_epochs))
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))
    best = float("inf")
    history: List[Dict[str, object]] = []
    save_dir = Path(args.save_dir) / "phase_depth_evidence"
    for ep in range(1, args.dp_epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"D_p {ep}/{args.dp_epochs}"):
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                pred = dp_predict(phi_model, model, batch, device)
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
        val = eval_dp_fast(phi_model, model, loaders["val"], device)
        log = {"epoch": ep, "train_loss": total / max(1, seen), "val_object_rmse": val, "seconds": time.time() - t0}
        history.append(log)
        print(json.dumps({"stage": "dp", **log}, ensure_ascii=False), flush=True)
        if val < best:
            best = val
            save_checkpoint(save_dir / "checkpoints" / "best.pt", ep, model, opt, scaler, args, best, history)
    ckpt = torch.load(str(save_dir / "checkpoints" / "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def eval_dp_fast(phi_model: torch.nn.Module, dp_model: torch.nn.Module, loader: Iterable[Dict[str, object]], device: torch.device) -> float:
    vals = []
    for batch in loader:
        pred = pred_to_depth_mm(dp_predict(phi_model, dp_model, batch, device), batch)
        target = batch["depth_raw"].to(device, non_blocking=True).float()  # type: ignore[index]
        mask = batch["object_mask"].to(device, non_blocking=True).float()  # type: ignore[index]
        count = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        rmse = torch.sqrt((((pred - target) ** 2) * mask).sum(dim=(1, 2, 3)) / count)
        vals.extend(float(x) for x in rmse.detach().cpu().tolist())
    return float(np.mean(vals)) if vals else float("nan")


def load_residual(path: str | Path, cond_channels: int, device: torch.device) -> Tuple[ResidualPosterior, Dict[str, object]]:
    ckpt = torch.load(str(path), map_location=device)
    saved_args = ckpt.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = {}
    ns = argparse.Namespace(
        base_channels=int(saved_args.get("base_channels", 32)),
        ch_mult=list(saved_args.get("ch_mult", [1, 2, 4, 8])),
        num_res_blocks=int(saved_args.get("num_res_blocks", 1)),
        dropout=float(saved_args.get("dropout", 0.05)),
        time_emb_dim=int(saved_args.get("time_emb_dim", 128)),
    )
    model = build_model(cond_channels + 1, 1, ns).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    posterior = ResidualPosterior(
        model,
        timesteps=int(saved_args.get("timesteps", 200)),
        residual_scale=float(saved_args.get("residual_scale", 0.25)),
        device=device,
    )
    return posterior, saved_args


def masked_mean_np(arr: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 0.5
    if not np.any(m):
        return float(np.mean(arr))
    return float(np.mean(arr[m]))


def raw_edge_score(fringe: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    dx = fringe[..., :, 1:] - fringe[..., :, :-1]
    dy = fringe[..., 1:, :] - fringe[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    mag = torch.sqrt(dx * dx + dy * dy)
    out = []
    for j in range(mag.shape[0]):
        out.append(float((mag[j:j + 1] * mask[j:j + 1]).sum().item() / mask[j:j + 1].sum().clamp_min(1.0).item()))
    return np.asarray(out, dtype=np.float32)


@torch.no_grad()
def collect_items(
    split: str,
    loader: Iterable[Dict[str, object]],
    base_model: torch.nn.Module,
    posterior: ResidualPosterior,
    phi_model: torch.nn.Module,
    dp_model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for batch in tqdm(loader, desc=f"collect {split}", leave=False):
        base = forward_direct(base_model, batch, device)[:, :1]
        dp = dp_predict(phi_model, dp_model, batch, device)
        _, dd, unc = posterior.sample(batch, base_model, steps=args.sample_steps, ensemble_size=args.ensemble_size)
        phi = phi_predict(phi_model, batch, device)
        pred_conf = torch.clamp((phi[:, 6:7] + phi[:, 7:8]) * 0.5, 0.0, 1.0)
        mask = batch["object_mask"].to(device, non_blocking=True).float()  # type: ignore[index]
        edge_vals = raw_edge_score(batch["fringe"].to(device, non_blocking=True).float(), mask)  # type: ignore[index]
        for j in range(base.shape[0]):
            item = {
                "sample_id": batch["sample_id"][j],  # type: ignore[index]
                "object_id": int(batch["object_id"][j].item()),  # type: ignore[index]
                "pose_id": int(batch["pose_id"][j].item()),  # type: ignore[index]
                "scale_mm": batch["scale_mm"][j:j + 1].detach().cpu(),  # type: ignore[index]
                "center_mm": batch["center_mm"][j:j + 1].detach().cpu(),  # type: ignore[index]
                "depth_raw": batch["depth_raw"][j:j + 1].detach().cpu(),  # type: ignore[index]
                "object_mask": batch["object_mask"][j:j + 1].detach().cpu(),  # type: ignore[index]
                "valid_mask": batch["valid_mask"][j:j + 1].detach().cpu(),  # type: ignore[index]
                "base": base[j:j + 1].detach().cpu(),
                "dp": dp[j:j + 1].detach().cpu(),
                "dd": dd[j:j + 1].detach().cpu(),
                "unc": unc[j:j + 1].detach().cpu(),
                "edge_mean": float(edge_vals[j]),
                "delta_mean": masked_mean_np(torch.abs(dp[j:j + 1] - dd[j:j + 1]).detach().cpu().numpy()[0, 0], batch["object_mask"][j].numpy()[0]),  # type: ignore[index]
                "unc_mean": masked_mean_np(unc[j:j + 1].detach().cpu().numpy()[0, 0], batch["object_mask"][j].numpy()[0]),  # type: ignore[index]
                "pred_conf_mean": masked_mean_np(pred_conf[j:j + 1].detach().cpu().numpy()[0, 0], batch["object_mask"][j].numpy()[0]),  # type: ignore[index]
                "teacher_conf_mean": masked_mean_np(batch["teacher_conf"][j].numpy()[0], batch["object_mask"][j].numpy()[0]),  # type: ignore[index]
            }
            items.append(item)
    return items


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


def item_row(item: Dict[str, object], pred_key: str, mode: str) -> Dict[str, object]:
    pred = item[pred_key] if isinstance(pred_key, str) else pred_key
    return row_from_prediction(pred, compact_batch(item), 0, "full_pip_rcpc", mode)  # type: ignore[arg-type]


def rows_for_items(items: List[Dict[str, object]], key: str, mode: str) -> List[Dict[str, object]]:
    return [item_row(item, key, mode) for item in items]


def rcpc_prediction(item: Dict[str, object], gate: Dict[str, float]) -> Tuple[torch.Tensor, bool]:
    use = (
        float(item["delta_mean"]) <= gate["delta_max"]
        and float(item["unc_mean"]) <= gate["unc_max"]
        and float(item["pred_conf_mean"]) >= gate["conf_min"]
    )
    dd = item["dd"]  # type: ignore[assignment]
    dp = item["dp"]  # type: ignore[assignment]
    alpha = float(gate["alpha"])
    pred = torch.clamp(dd + (alpha * (dp - dd) if use else 0.0), -1.0, 1.0)
    return pred, bool(use)


def gate_rows(items: List[Dict[str, object]], gate: Dict[str, float], mode: str = "rcpc") -> Tuple[List[Dict[str, object]], float]:
    rows = []
    accepted = 0
    for item in items:
        pred, use = rcpc_prediction(item, gate)
        accepted += int(use)
        rows.append(row_from_prediction(pred, compact_batch(item), 0, "full_pip_rcpc", mode))
    return rows, float(accepted) / max(1, len(items))


def fast_object_rmse_item(item: Dict[str, object], pred: torch.Tensor) -> float:
    target = item["depth_raw"].float()  # type: ignore[union-attr]
    mask = item["object_mask"].float()  # type: ignore[union-attr]
    scale = item["scale_mm"].view(1, 1, 1, 1).float()  # type: ignore[union-attr]
    center = item["center_mm"].view(1, 1, 1, 1).float()  # type: ignore[union-attr]
    pred_mm = torch.clamp(pred.float(), -1.0, 1.0) * scale + center
    count = mask.sum().clamp_min(1.0)
    return float(torch.sqrt((((pred_mm - target) ** 2) * mask).sum() / count).item())


def gate_fast_rmse(items: List[Dict[str, object]], gate: Dict[str, float]) -> Tuple[float, float]:
    vals = []
    accepted = 0
    for item in items:
        pred, use = rcpc_prediction(item, gate)
        accepted += int(use)
        vals.append(fast_object_rmse_item(item, pred))
    return (float(np.mean(vals)) if vals else float("nan"), float(accepted) / max(1, len(items)))


def select_gate(val_items: List[Dict[str, object]]) -> Dict[str, object]:
    deltas = np.asarray([float(x["delta_mean"]) for x in val_items], dtype=np.float32)
    uncs = np.asarray([float(x["unc_mean"]) for x in val_items], dtype=np.float32)
    confs = np.asarray([float(x["pred_conf_mean"]) for x in val_items], dtype=np.float32)
    delta_grid = sorted(set(float(x) for x in np.quantile(deltas, [0.2, 0.4, 0.6, 0.8, 1.0])))
    unc_grid = sorted(set(float(x) for x in np.quantile(uncs, [0.2, 0.4, 0.6, 0.8, 1.0])))
    conf_grid = sorted(set([0.0] + [float(x) for x in np.quantile(confs, [0.0, 0.2, 0.4, 0.6, 0.8])]))
    best: Dict[str, object] = {"object_rmse": float("inf")}
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for delta_max in delta_grid:
            for unc_max in unc_grid:
                for conf_min in conf_grid:
                    gate = {"alpha": alpha, "delta_max": delta_max, "unc_max": unc_max, "conf_min": conf_min}
                    rmse, acc = gate_fast_rmse(val_items, gate)
                    if rmse < float(best["object_rmse"]):
                        best = {**gate, "object_rmse": rmse, "accepted_fraction": acc, "selection_metric": "fast_object_rmse_mean"}
    return best


def save_rows_csv(rows: List[Dict[str, object]], path: Path) -> None:
    keys = ["sample_id", "object_id", "pose_id", "config", "mode", "legal_single_frame"]
    for roi in ("object", "valid"):
        for metric in ("rmse", "mae", "edge_rmse", "normal_deg", "ssim"):
            keys.append(f"{roi}_{metric}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def leakage_audit(args: argparse.Namespace, loaders: Dict[str, object], device: torch.device) -> Dict[str, object]:
    cond_names = list(args.channel_names)
    dp_names = list(args.dp_channel_names)
    formal = set(cond_names)
    dp_formal = set(dp_names)
    teacher_in_cond = sorted((formal | dp_formal) & TEACHER_FIELDS)
    return {
        "legal_test_time_input": "input_vertical_0120.bmp plus single-frame derived Hilbert/FTP/DWT/gradient/xy features",
        "cond_channel_names": cond_names,
        "dp_channel_names": dp_names,
        "teacher_fields": sorted(TEACHER_FIELDS),
        "teacher_fields_in_formal_cond": teacher_in_cond,
        "leakage_detected": bool(teacher_in_cond),
        "note": "P_phi predictions are legal because they are generated from cond only; teacher_extra arrays are targets/diagnostics only.",
    }


def smoke(args: argparse.Namespace, loaders_obj: Dict[str, object], device: torch.device) -> None:
    out = {
        "split_counts": loaders_obj["split_counts"],
        "cond_channels": loaders_obj["cond_channels"],
        "dp_cond_channels": loaders_obj["dp_cond_channels"],
        "channel_names": loaders_obj["channel_names"],
        "dp_channel_names": loaders_obj["dp_channel_names"],
        "normalization": loaders_obj["norm"],
    }
    for split in ["train", "val", "test"] + (["ood"] if "ood" in loaders_obj["loaders"] else []):  # type: ignore[operator]
        batch = next(iter(loaders_obj["loaders"][split]))  # type: ignore[index]
        out[f"{split}_batch"] = {
            "cond_shape": list(batch["cond"].shape),  # type: ignore[index]
            "phi_target_shape": list(batch["phi_target"].shape),  # type: ignore[index]
            "depth_shape": list(batch["depth"].shape),  # type: ignore[index]
            "cond_finite": bool(torch.isfinite(batch["cond"]).all().item()),  # type: ignore[index]
            "phi_finite": bool(torch.isfinite(batch["phi_target"]).all().item()),  # type: ignore[index]
            "object_pixels": float(batch["object_mask"].sum().item()),  # type: ignore[index]
        }
    out["leakage_audit"] = leakage_audit(args, loaders_obj, device)
    path = Path(args.save_dir) / "smoke_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if out["leakage_audit"]["leakage_detected"]:
        raise RuntimeError("formal cond contains teacher fields")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--teacher_extra_root", required=True)
    parser.add_argument("--ood_root", default="")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--residual_ckpt", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--phi_epochs", type=int, default=30)
    parser.add_argument("--dp_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--ch_mult", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--time_emb_dim", type=int, default=128)
    parser.add_argument("--object_mask_weight", type=float, default=3.0)
    parser.add_argument("--lambda_mse", type=float, default=0.2)
    parser.add_argument("--lambda_grad", type=float, default=0.05)
    parser.add_argument("--sample_steps", type=int, default=12)
    parser.add_argument("--ensemble_size", type=int, default=3)
    parser.add_argument("--cache_features", action="store_true")
    parser.add_argument("--feature_cache_dir", default="")
    parser.add_argument("--no_amp", action="store_true")
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
    loaders_obj = create_loaders(args)
    args.cond_channels = int(loaders_obj["cond_channels"])
    args.dp_cond_channels = int(loaders_obj["dp_cond_channels"])
    args.channel_names = loaders_obj["channel_names"]
    args.dp_channel_names = loaders_obj["dp_channel_names"]
    args.normalization = loaders_obj["norm"]
    args.split_counts = loaders_obj["split_counts"]
    smoke(args, loaders_obj, device)
    if args.smoke_only:
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    base_model, base_args = load_base_model(args.base_ckpt, args.cond_channels, device)
    posterior, residual_args = load_residual(args.residual_ckpt, args.cond_channels, device)
    phi_model = train_phi(args, loaders_obj["loaders"], device)  # type: ignore[arg-type]
    dp_model = train_dp(args, loaders_obj["loaders"], phi_model, device)  # type: ignore[arg-type]

    split_items = {}
    for split in ["val", "test"] + (["ood"] if "ood" in loaders_obj["loaders"] else []):  # type: ignore[operator]
        split_items[split] = collect_items(
            split,
            loaders_obj["loaders"][split],  # type: ignore[index]
            base_model,
            posterior,
            phi_model,
            dp_model,
            args,
            device,
        )

    gate = select_gate(split_items["val"])
    (save_dir / "gate_selection.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")

    summary: Dict[str, object] = {
        "stage": "full_pip_rcpc_pilot",
        "seed": args.seed,
        "legal_single_frame": True,
        "base_ckpt": args.base_ckpt,
        "residual_ckpt": args.residual_ckpt,
        "base_args": base_args,
        "residual_args": residual_args,
        "split_counts": args.split_counts,
        "normalization": args.normalization,
        "gate": gate,
        "splits": {},
    }
    all_rows: List[Dict[str, object]] = []
    for split, items in split_items.items():
        rows_base = rows_for_items(items, "base", "D_b_base")
        rows_dp = rows_for_items(items, "dp", "D_p_phase_evidence")
        rows_dd = rows_for_items(items, "dd", "D_d_diffusion_posterior")
        rows_rcpc, acc = gate_rows(items, gate, mode="RCPC_final")
        split_rows = rows_base + rows_dp + rows_dd + rows_rcpc
        save_rows_csv(split_rows, save_dir / f"{split}_per_sample_metrics.csv")
        all_rows.extend([{**r, "split": split} for r in split_rows])
        summary["splits"][split] = {  # type: ignore[index]
            "D_b_base": summarize_rows(rows_base),
            "D_p_phase_evidence": summarize_rows(rows_dp),
            "D_d_diffusion_posterior": summarize_rows(rows_dd),
            "RCPC_final": summarize_rows(rows_rcpc),
            "rcpc_accepted_fraction": acc,
        }
    with (save_dir / "full_pip_rcpc_pilot_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (save_dir / "per_sample_metrics_all.csv").open("w", encoding="utf-8", newline="") as f:
        keys = ["split", "sample_id", "object_id", "pose_id", "config", "mode", "legal_single_frame"]
        for roi in ("object", "valid"):
            for metric in ("rmse", "mae", "edge_rmse", "normal_deg", "ssim"):
                keys.append(f"{roi}_{metric}")
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in keys})
    (save_dir / "full_pip_rcpc_pilot_report.md").write_text(make_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def make_report(summary: Dict[str, object]) -> str:
    lines = [
        "# Full PIP/RCPC Pilot Report",
        "",
        f"Seed: {summary['seed']}",
        "",
        "Formal test-time input is single-frame only. Teacher/PMP fields are used as train-time supervision or diagnostics.",
        "",
        "## Metrics",
        "",
        "| split | branch | object RMSE | valid RMSE | RCPC accept |",
        "|---|---|---:|---:|---:|",
    ]
    for split, data in summary["splits"].items():  # type: ignore[union-attr]
        accept = data.get("rcpc_accepted_fraction", "")  # type: ignore[attr-defined]
        for branch in ["D_b_base", "D_p_phase_evidence", "D_d_diffusion_posterior", "RCPC_final"]:
            s = data[branch]  # type: ignore[index]
            obj = s["object"]["rmse"]["mean"]
            valid = s["valid"]["rmse"]["mean"]
            lines.append(f"| {split} | {branch} | {obj:.4f} | {valid:.4f} | {accept if branch == 'RCPC_final' else ''} |")
    gate = summary["gate"]
    lines += [
        "",
        "## Gate",
        "",
        "```json",
        json.dumps(gate, indent=2, ensure_ascii=False),
        "```",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
