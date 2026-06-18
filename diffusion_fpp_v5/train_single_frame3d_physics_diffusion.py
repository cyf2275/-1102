"""Physics-input and residual-posterior validation on single_frame_3d data.

This is intentionally separate from the older my_fpp_dataset_v1 scripts.  The
new uploaded dataset is manifest driven, uses `depth_z` as the target, and must
not be mixed with the older `wall_normal_height` result tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
from tqdm import tqdm

from models.unet import ConditionalUNet
from physics_features_pip import FEATURE_ORDER, build_pip_features
from utils.metrics import compute_metrics


LEGAL_CONFIGS = {"raw", "raw_xy", "raw_single_phys", "teacher_aux"}
DIRECT_CONFIGS = ["raw", "raw_xy", "raw_single_phys", "teacher_aux"]
RESIDUAL_CONFIGS = ["raw", "raw_single_phys", "teacher_aux"]
METRIC_KEYS = ["rmse", "mae", "edge_rmse", "normal_deg", "ssim"]
ROI_PREFIXES = ["object", "valid"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_config(config: str) -> str:
    text = str(config).strip().lower().replace("-", "_").replace("+", "_")
    aliases = {
        "rawxy": "raw_xy",
        "single_phys": "raw_single_phys",
        "raw_singlephys": "raw_single_phys",
        "teacheraux": "teacher_aux",
    }
    text = aliases.get(text, text)
    if text not in LEGAL_CONFIGS:
        raise ValueError(f"unknown config: {config}")
    return text


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing manifest: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def robust_unit(x: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    vals = x[np.isfinite(x)]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(vals, [lo_q, hi_q])
    return np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)


def xy_maps(h: int, w: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None].repeat(w, axis=1)
    return np.stack([x, y], axis=0)


def finite_float(x: np.ndarray, fill: float = 0.0) -> np.ndarray:
    return np.nan_to_num(x.astype(np.float32, copy=False), nan=fill, posinf=fill, neginf=fill)


def normalization_from_stats(data_root: str | Path) -> Dict[str, float]:
    path = Path(data_root) / "normalization_stats.json"
    stats = load_json(path)
    depth = stats.get("depth_z", {})
    if not isinstance(depth, dict):
        raise KeyError(f"{path} does not contain depth_z stats")
    center = float(depth["mean"])
    p1 = float(depth["p1"])
    p995 = float(depth["p99_5"])
    scale = max(abs(p1 - center), abs(p995 - center), 1.0)
    return {
        "target": "depth_z",
        "source": str(path),
        "center_mm": center,
        "scale_mm": scale,
        "p1_mm": p1,
        "p99_5_mm": p995,
    }


def sample_paths(root: Path, row: Dict[str, str]) -> Dict[str, Path]:
    domain = str(row["domain"])
    object_id = int(row["object_id"])
    pose_id = int(row["pose_id"])
    sample_dir = root / "samples" / domain / f"obj{object_id:03d}" / f"pose{pose_id:02d}"
    return {
        "sample_dir": sample_dir,
        "primary": sample_dir / "input_vertical_0120.bmp",
        "ablation": sample_dir / "ablation_horizontal_0048.bmp",
        "labels": sample_dir / "labels.npz",
    }


class SingleFrame3DDataset(Dataset):
    """Manifest-based real-capture single-frame FPP dataset.

    Legal model inputs are image-derived only.  Phase and modulation arrays are
    returned as labels/weights for auxiliary losses and diagnostics; they are
    never concatenated into legal input tensors.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        config: str,
        norm: Optional[Dict[str, float]] = None,
        cache_features: bool = False,
        feature_cache_dir: str | Path | None = None,
        write_feature_cache: bool = False,
    ) -> None:
        self.root = Path(data_root)
        self.split = str(split)
        self.config = canonical_config(config)
        self.norm = norm or normalization_from_stats(self.root)
        self.cache_features = bool(cache_features)
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        self.write_feature_cache = bool(write_feature_cache)
        self.feature_cache: Dict[str, Tuple[np.ndarray, Dict[str, float]]] = {}
        self.rows = read_manifest(self.root / f"{self.split}_manifest.csv")
        self.center_mm = float(self.norm["center_mm"])
        self.scale_mm = float(self.norm["scale_mm"])
        self.channel_names = self._channel_names()

    def __len__(self) -> int:
        return len(self.rows)

    def _channel_names(self) -> List[str]:
        if self.config == "raw":
            return ["input_vertical_0120"]
        if self.config == "raw_xy":
            return ["input_vertical_0120", "x", "y"]
        return ["input_vertical_0120"] + [name for name in FEATURE_ORDER[1:]]

    def _load_image(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"missing BMP: {path}")
        img = Image.open(path).convert("L")
        return (np.asarray(img).astype(np.float32) / 255.0).astype(np.float32)

    def _single_phys(self, sample_id: str, raw_01: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
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
        features, carrier = build_pip_features(raw_01[None, :, :].astype(np.float32))
        if cache_path is not None and self.write_feature_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp.npz")
            np.savez(
                tmp_path,
                features=features.astype(np.float16),
                carrier_dx=np.array(float(carrier.get("dx", math.nan)), dtype=np.float32),
                carrier_dy=np.array(float(carrier.get("dy", math.nan)), dtype=np.float32),
                carrier_conf=np.array(float(carrier.get("spectral_confidence", math.nan)), dtype=np.float32),
            )
            Path(tmp_path).replace(cache_path)
        if self.cache_features:
            self.feature_cache[sample_id] = (features, carrier)
        return features, carrier

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[int(idx)]
        sample_id = str(row["sample_id"])
        object_id = int(row["object_id"])
        pose_id = int(row["pose_id"])
        paths = sample_paths(self.root, row)
        raw = self._load_image(paths["primary"])
        if not paths["labels"].exists():
            raise FileNotFoundError(f"missing labels: {paths['labels']}")
        with np.load(paths["labels"]) as z:
            depth_z = finite_float(z["depth_z"])
            valid_mask = z["valid_mask"].astype(np.float32)
            object_mask = z["object_mask"].astype(np.float32)
            phase_y = finite_float(z["phase_y"])
            phase_x = finite_float(z["phase_x"])
            bc_y = robust_unit(finite_float(z["bc_y"]))
            bc_x = robust_unit(finite_float(z["bc_x"]))

        h, w = raw.shape
        xy = xy_maps(h, w)
        features, carrier = self._single_phys(sample_id, raw)
        if self.config == "raw":
            cond = raw[None, :, :]
        elif self.config == "raw_xy":
            cond = np.concatenate([raw[None, :, :], xy], axis=0)
        elif self.config in {"raw_single_phys", "teacher_aux"}:
            cond = np.concatenate([raw[None, :, :], features[1:]], axis=0)
        else:
            raise AssertionError(self.config)

        phase_target = np.stack([
            np.sin(phase_y),
            np.cos(phase_y),
            np.sin(phase_x),
            np.cos(phase_x),
        ], axis=0).astype(np.float32)
        phase_conf = np.stack([bc_y, bc_y, bc_x, bc_x], axis=0).astype(np.float32)
        depth_norm = np.clip((depth_z - self.center_mm) / self.scale_mm, -1.0, 1.0).astype(np.float32)
        return {
            "sample_id": sample_id,
            "object_id": torch.tensor(object_id, dtype=torch.long),
            "pose_id": torch.tensor(pose_id, dtype=torch.long),
            "cond": torch.from_numpy(cond.astype(np.float32, copy=False)),
            "fringe": torch.from_numpy(raw[None, :, :].astype(np.float32)),
            "depth": torch.from_numpy(depth_norm[None, :, :]),
            "depth_raw": torch.from_numpy(depth_z[None, :, :]),
            "valid_mask": torch.from_numpy(valid_mask[None, :, :].astype(np.float32)),
            "object_mask": torch.from_numpy(object_mask[None, :, :].astype(np.float32)),
            "phase_target": torch.from_numpy(phase_target),
            "phase_conf": torch.from_numpy(phase_conf),
            "xy": torch.from_numpy(xy.astype(np.float32)),
            "scale_mm": torch.tensor(self.scale_mm, dtype=torch.float32),
            "center_mm": torch.tensor(self.center_mm, dtype=torch.float32),
            "carrier_dx": torch.tensor(float(carrier.get("dx", math.nan)), dtype=torch.float32),
            "carrier_dy": torch.tensor(float(carrier.get("dy", math.nan)), dtype=torch.float32),
            "legal_single_frame": True,
        }


def _resize_batch(out: Dict[str, object], image_h: int = 0, image_w: int = 0) -> Dict[str, object]:
    if not image_h or not image_w:
        return out
    size = (int(image_h), int(image_w))
    bilinear_keys = ["cond", "fringe", "depth", "depth_raw", "phase_target", "phase_conf", "xy"]
    nearest_keys = ["valid_mask", "object_mask"]
    for key in bilinear_keys:
        value = out.get(key)
        if torch.is_tensor(value) and value.ndim == 4 and value.shape[-2:] != size:
            out[key] = F.interpolate(value, size=size, mode="bilinear", align_corners=False)
    for key in nearest_keys:
        value = out.get(key)
        if torch.is_tensor(value) and value.ndim == 4 and value.shape[-2:] != size:
            out[key] = (F.interpolate(value, size=size, mode="nearest") > 0.5).to(dtype=value.dtype)
    return out


def collate_single_frame(batch: List[Dict[str, object]], image_h: int = 0, image_w: int = 0) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key in batch[0].keys():
        first = batch[0][key]
        if torch.is_tensor(first):
            out[key] = torch.stack([b[key] for b in batch], dim=0)  # type: ignore[index]
        elif isinstance(first, (bool, np.bool_)):
            out[key] = bool(first)
        else:
            out[key] = [b[key] for b in batch]
    return _resize_batch(out, image_h=image_h, image_w=image_w)


def create_loaders(
    data_root: str | Path,
    config: str,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    image_h: int,
    image_w: int,
    train_epoch_repeats: int,
    train_subset: int = 0,
    cache_features: bool = False,
    feature_cache_dir: str | Path | None = None,
    write_feature_cache: bool = False,
) -> Dict[str, object]:
    norm = normalization_from_stats(data_root)
    common = dict(
        data_root=data_root,
        config=config,
        norm=norm,
        cache_features=cache_features,
        feature_cache_dir=feature_cache_dir,
        write_feature_cache=write_feature_cache,
    )
    train_ds: Dataset = SingleFrame3DDataset(split="train", **common)
    val_ds = SingleFrame3DDataset(split="val", **common)
    test_ds = SingleFrame3DDataset(split="test", **common)
    train_meta = train_ds
    if train_subset:
        train_ds = Subset(train_ds, list(range(min(int(train_subset), len(train_ds)))))

    def collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
        return collate_single_frame(batch, image_h=image_h, image_w=image_w)

    loader_common = {"num_workers": int(num_workers), "pin_memory": True, "collate_fn": collate}
    if num_workers > 0:
        loader_common["persistent_workers"] = True
        loader_common["prefetch_factor"] = 2
    repeats = max(1, int(train_epoch_repeats))
    if repeats > 1:
        sampler = RandomSampler(train_ds, replacement=True, num_samples=len(train_ds) * repeats)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True, **loader_common)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_common)

    return {
        "train": train_loader,
        "train_eval": DataLoader(train_ds, batch_size=eval_batch_size, shuffle=False, **loader_common),
        "val": DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, **loader_common),
        "test": DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, **loader_common),
        "stats": norm,
        "cond_channels": len(train_meta.channel_names),  # type: ignore[attr-defined]
        "channel_names": list(train_meta.channel_names),  # type: ignore[attr-defined]
        "input_mode": canonical_config(config),
        "dataset": "single_frame_3d_dataset_v1_upload_smalltest",
        "split_counts": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
    }


def smoke_summary(loaders: Dict[str, object]) -> Dict[str, object]:
    batch = next(iter(loaders["train"]))  # type: ignore[index]
    assert isinstance(batch, dict)
    tensors = {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)}
    nan_counts = {
        k: int(torch.isnan(v.float()).sum().item())
        for k, v in batch.items()
        if torch.is_tensor(v) and v.is_floating_point()
    }
    return {
        "dataset": loaders["dataset"],
        "input_mode": loaders["input_mode"],
        "legal_single_frame": True,
        "cond_channels": loaders["cond_channels"],
        "channel_names": loaders["channel_names"],
        "normalization": loaders["stats"],
        "split_counts": loaders["split_counts"],
        "batch_tensor_shapes": tensors,
        "batch_nan_counts": nan_counts,
        "batch_valid_pixels": float(batch["valid_mask"].sum().item()),  # type: ignore[index]
        "batch_object_pixels": float(batch["object_mask"].sum().item()),  # type: ignore[index]
        "teacher_or_qc_fields_in_cond": any(
            name in {"phase_y", "phase_x", "bc_y", "bc_x", "phase_y_sin", "phase_x_sin", "bc_y_conf", "bc_x_conf"}
            for name in loaders["channel_names"]  # type: ignore[index]
        ),
    }


def precompute_feature_cache(args: argparse.Namespace) -> None:
    args.config = "raw_single_phys"
    cache_dir = Path(args.feature_cache_dir or (Path(args.data_root) / "physics_feature_cache_pip"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    norm = normalization_from_stats(args.data_root)
    total = 0
    missing = 0
    for split in ("train", "val", "test"):
        ds = SingleFrame3DDataset(
            data_root=args.data_root,
            split=split,
            config="raw_single_phys",
            norm=norm,
            cache_features=False,
            feature_cache_dir=cache_dir,
            write_feature_cache=True,
        )
        for idx in tqdm(range(len(ds)), desc=f"precompute physics {split}"):
            sample_id = ds.rows[idx]["sample_id"]
            cache_path = cache_dir / f"{sample_id}.npz"
            if cache_path.exists():
                total += 1
                continue
            _ = ds[idx]
            total += 1
            if not cache_path.exists():
                missing += 1
    summary = {
        "feature_cache_dir": str(cache_dir),
        "total_samples_seen": total,
        "missing_after_precompute": missing,
        "dtype": "float16 features on disk, loaded as float32",
        "feature_order": FEATURE_ORDER,
    }
    out_path = cache_dir / "precompute_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def masked_mean(x: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weight is None:
        return x.mean()
    weight = torch.clamp(weight.to(device=x.device, dtype=x.dtype), min=0.0)
    return (x * weight).sum() / weight.sum().clamp(min=1.0)


def charbonnier(pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None, eps: float = 1e-3) -> torch.Tensor:
    return masked_mean(torch.sqrt((pred - target) ** 2 + eps * eps), weight=weight)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    return masked_mean((pred - target) ** 2, weight=weight)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    pdx = pred[..., :, 1:] - pred[..., :, :-1]
    tdx = target[..., :, 1:] - target[..., :, :-1]
    pdy = pred[..., 1:, :] - pred[..., :-1, :]
    tdy = target[..., 1:, :] - target[..., :-1, :]
    if weight is None:
        return torch.mean(torch.abs(pdx - tdx)) + torch.mean(torch.abs(pdy - tdy))
    wx = weight[..., :, 1:] * weight[..., :, :-1]
    wy = weight[..., 1:, :] * weight[..., :-1, :]
    return masked_mean(torch.abs(pdx - tdx), wx) + masked_mean(torch.abs(pdy - tdy), wy)


def train_weight(batch: Dict[str, object], device: torch.device, object_weight: float) -> torch.Tensor:
    valid = batch["valid_mask"].to(device, non_blocking=True).float()  # type: ignore[index]
    obj = batch["object_mask"].to(device, non_blocking=True).float()  # type: ignore[index]
    extra = max(float(object_weight) - 1.0, 0.0)
    return valid * (1.0 + extra * obj)


def build_model(cond_channels: int, out_channels: int, args: argparse.Namespace) -> ConditionalUNet:
    return ConditionalUNet(
        in_channels=1,
        cond_channels=cond_channels,
        out_channels=out_channels,
        base_ch=args.base_channels,
        ch_mult=tuple(args.ch_mult),
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        time_emb_dim=args.time_emb_dim,
    )


def forward_direct(model: torch.nn.Module, batch: Dict[str, object], device: torch.device) -> torch.Tensor:
    baseline_arch = getattr(model, "_single_frame_baseline_arch", None)
    if baseline_arch:
        fringe = batch["fringe"].to(device, non_blocking=True).float()  # type: ignore[index]
        if baseline_arch == "unet":
            zeros = torch.zeros((fringe.shape[0], 1, fringe.shape[-2], fringe.shape[-1]), device=device)
            t = torch.zeros((fringe.shape[0],), dtype=torch.long, device=device)
            return torch.tanh(model(zeros, t, fringe))
        output = model(fringe)
        depth = output["depth"] if isinstance(output, dict) else output
        if baseline_arch == "pix2pix":
            return torch.clamp(depth, 0.0, 1.0) * 2.0 - 1.0
        return torch.tanh(depth)
    cond = batch["cond"].to(device, non_blocking=True).float()  # type: ignore[index]
    zeros = torch.zeros((cond.shape[0], 1, cond.shape[-2], cond.shape[-1]), device=device)
    t = torch.zeros((cond.shape[0],), dtype=torch.long, device=device)
    return torch.tanh(model(zeros, t, cond))


def teacher_aux_loss(pred: torch.Tensor, batch: Dict[str, object], device: torch.device) -> torch.Tensor:
    if pred.shape[1] < 5:
        return pred.new_tensor(0.0)
    phase_pred = pred[:, 1:5]
    phase_target = batch["phase_target"].to(device, non_blocking=True).float()  # type: ignore[index]
    phase_conf = batch["phase_conf"].to(device, non_blocking=True).float()  # type: ignore[index]
    valid = batch["valid_mask"].to(device, non_blocking=True).float()  # type: ignore[index]
    return charbonnier(phase_pred, phase_target, weight=phase_conf * valid)


def compute_direct_loss(pred: torch.Tensor, batch: Dict[str, object], device: torch.device, args: argparse.Namespace) -> torch.Tensor:
    height_pred = pred[:, :1]
    target = batch["depth"].to(device, non_blocking=True).float()  # type: ignore[index]
    weight = train_weight(batch, device, args.object_mask_weight)
    loss = charbonnier(height_pred, target, weight=weight)
    loss = loss + args.lambda_mse * masked_mse(height_pred, target, weight=weight)
    if args.lambda_grad > 0:
        loss = loss + args.lambda_grad * gradient_loss(height_pred, target, weight=weight)
    if args.config == "teacher_aux" and args.lambda_teacher_phase > 0:
        loss = loss + args.lambda_teacher_phase * teacher_aux_loss(pred, batch, device)
    return loss


def pred_to_depth_mm(pred_norm: torch.Tensor, batch: Dict[str, object]) -> torch.Tensor:
    scale = batch["scale_mm"].to(pred_norm.device, non_blocking=True).view(-1, 1, 1, 1)  # type: ignore[index]
    center = batch["center_mm"].to(pred_norm.device, non_blocking=True).view(-1, 1, 1, 1)  # type: ignore[index]
    return torch.clamp(pred_norm, -1.0, 1.0) * scale + center


def metric_row(pred_mm: torch.Tensor, target_mm: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    return compute_metrics(pred_mm, target_mm, mask=mask)


def row_from_prediction(pred_norm: torch.Tensor, batch: Dict[str, object], j: int, config: str, mode: str) -> Dict[str, object]:
    pred_j = j if pred_norm.shape[0] > j else 0
    pred_mm = pred_to_depth_mm(pred_norm[pred_j:pred_j + 1], {
        "scale_mm": batch["scale_mm"][j:j + 1],  # type: ignore[index]
        "center_mm": batch["center_mm"][j:j + 1],  # type: ignore[index]
    })
    target_mm = batch["depth_raw"].to(pred_mm.device, non_blocking=True).float()[j:j + 1]  # type: ignore[index]
    object_mask = batch["object_mask"].to(pred_mm.device, non_blocking=True).float()[j:j + 1]  # type: ignore[index]
    valid_mask = batch["valid_mask"].to(pred_mm.device, non_blocking=True).float()[j:j + 1]  # type: ignore[index]
    object_metrics = metric_row(pred_mm, target_mm, object_mask)
    valid_metrics = metric_row(pred_mm, target_mm, valid_mask)
    row: Dict[str, object] = {
        "sample_id": batch["sample_id"][j],  # type: ignore[index]
        "object_id": int(batch["object_id"][j].item()),  # type: ignore[index]
        "pose_id": int(batch["pose_id"][j].item()),  # type: ignore[index]
        "config": config,
        "mode": mode,
        "legal_single_frame": True,
    }
    for key in METRIC_KEYS:
        row[f"object_{key}"] = object_metrics[key]
        row[f"valid_{key}"] = valid_metrics[key]
    return row


def mean_std(rows: List[Dict[str, object]], key: str) -> Tuple[float, float]:
    vals = np.array([float(r[key]) for r in rows], dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std(ddof=1) if vals.size > 1 else 0.0)


def summarize_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {"n": len(rows)}
    for roi in ROI_PREFIXES:
        roi_summary = {}
        for key in METRIC_KEYS:
            mean, std = mean_std(rows, f"{roi}_{key}")
            roi_summary[key] = {"mean": mean, "std": std}
        out[roi] = roi_summary
    per_object: Dict[str, Dict[str, float]] = {}
    for obj in sorted({int(r["object_id"]) for r in rows}):
        subset = [r for r in rows if int(r["object_id"]) == obj]
        per_object[f"obj{obj:04d}"] = {
            "n": len(subset),
            "object_rmse_mean": mean_std(subset, "object_rmse")[0],
            "valid_rmse_mean": mean_std(subset, "valid_rmse")[0],
        }
    out["per_object"] = per_object
    return out


def save_rows(rows: List[Dict[str, object]], path: Path) -> None:
    keys = ["sample_id", "object_id", "pose_id", "config", "mode", "legal_single_frame"]
    for roi in ROI_PREFIXES:
        keys.extend([f"{roi}_{key}" for key in METRIC_KEYS])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


@torch.no_grad()
def evaluate_direct(model: torch.nn.Module, loader: Iterable[Dict[str, object]], device: torch.device, args: argparse.Namespace, mode: str) -> List[Dict[str, object]]:
    model.eval()
    rows: List[Dict[str, object]] = []
    for batch in tqdm(loader, desc=f"eval direct {mode}", leave=False):
        pred = forward_direct(model, batch, device)[:, :1]
        for j in range(pred.shape[0]):
            rows.append(row_from_prediction(pred, batch, j, args.config, mode))
    return rows


def _batch_rmse_values(pred_norm: torch.Tensor, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
    pred_mm = pred_to_depth_mm(pred_norm, batch)
    target_mm = batch["depth_raw"].to(pred_mm.device, non_blocking=True).float()  # type: ignore[index]
    out: Dict[str, torch.Tensor] = {}
    for roi, mask_key in (("object", "object_mask"), ("valid", "valid_mask")):
        mask = batch[mask_key].to(pred_mm.device, non_blocking=True).float()  # type: ignore[index]
        count = mask.sum(dim=(1, 2, 3))
        sq = ((pred_mm - target_mm) ** 2 * mask).sum(dim=(1, 2, 3))
        keep = count > 0
        out[roi] = torch.sqrt(sq[keep] / count[keep].clamp_min(1.0)).detach()
    return out


@torch.no_grad()
def evaluate_direct_fast(model: torch.nn.Module, loader: Iterable[Dict[str, object]], device: torch.device, args: argparse.Namespace, mode: str) -> Dict[str, float]:
    model.eval()
    values: Dict[str, List[float]] = {"object": [], "valid": []}
    for batch in tqdm(loader, desc=f"fast eval direct {mode}", leave=False):
        pred = forward_direct(model, batch, device)[:, :1]
        batch_values = _batch_rmse_values(pred, batch)
        for roi in values:
            values[roi].extend(float(x) for x in batch_values[roi].cpu().tolist())
    return {
        "n": float(len(values["object"])),
        "object_rmse": float(np.mean(values["object"])) if values["object"] else float("nan"),
        "valid_rmse": float(np.mean(values["valid"])) if values["valid"] else float("nan"),
    }


def save_checkpoint(path: Path, ep: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: GradScaler, args: argparse.Namespace, best: float, history: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": ep,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": vars(args),
        "best_val_object_rmse": best,
        "history": history,
    }, path)


def train_direct(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.config = canonical_config(args.config)
    loaders = create_loaders(
        data_root=args.data_root,
        config=args.config,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        image_h=args.image_h,
        image_w=args.image_w,
        train_epoch_repeats=args.train_epoch_repeats,
        train_subset=args.train_subset,
        cache_features=args.cache_features,
        feature_cache_dir=args.feature_cache_dir or None,
        write_feature_cache=False,
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    args.channel_names = loaders["channel_names"]
    args.cond_channels = int(loaders["cond_channels"])
    args.normalization = loaders["stats"]
    args.split_counts = loaders["split_counts"]
    with (save_dir / "loader_smoke_summary.json").open("w", encoding="utf-8") as f:
        json.dump(smoke_summary(loaders), f, indent=2, ensure_ascii=False)
    if args.smoke_only:
        print((save_dir / "loader_smoke_summary.json").read_text(encoding="utf-8"))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    out_channels = 5 if args.config == "teacher_aux" else 1
    model = build_model(args.cond_channels, out_channels, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))
    best = float("inf")
    history: List[Dict[str, object]] = []
    print(f"Device: {device}")
    print(f"Stage: direct | Config: {args.config} | Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"direct {args.config} {ep}/{args.epochs}"):  # type: ignore[index]
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                pred = forward_direct(model, batch, device)
                loss = compute_direct_loss(pred, batch, device, args)
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
            if args.full_train_val:
                val_rows = evaluate_direct(model, loaders["val"], device, args, mode="val")  # type: ignore[arg-type]
                val_summary = summarize_rows(val_rows)
                val_rmse = float(val_summary["object"]["rmse"]["mean"])  # type: ignore[index]
                log["val_valid_rmse"] = val_summary["valid"]["rmse"]["mean"]  # type: ignore[index]
                log["val_mode"] = "full_metrics"
            else:
                val_fast = evaluate_direct_fast(model, loaders["val"], device, args, mode="val")  # type: ignore[arg-type]
                val_rmse = float(val_fast["object_rmse"])
                log["val_valid_rmse"] = float(val_fast["valid_rmse"])
                log["val_mode"] = "fast_gpu_rmse"
            log["val_object_rmse"] = val_rmse
            if val_rmse < best:
                best = val_rmse
                save_checkpoint(save_dir / "checkpoints" / "best.pt", ep, model, optimizer, scaler, args, best, history)
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        if args.save_every > 0 and ep % args.save_every == 0:
            save_checkpoint(save_dir / "checkpoints" / f"epoch_{ep:03d}.pt", ep, model, optimizer, scaler, args, best, history)

    best_ckpt = save_dir / "checkpoints" / "best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(str(best_ckpt), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    val_rows = evaluate_direct(model, loaders["val"], device, args, mode="val")  # type: ignore[arg-type]
    test_rows = evaluate_direct(model, loaders["test"], device, args, mode="test")  # type: ignore[arg-type]
    eval_dir = save_dir / "evaluation"
    save_rows(val_rows, eval_dir / "val_per_sample_metrics.csv")
    save_rows(test_rows, eval_dir / "per_sample_metrics.csv")
    summary = {
        "stage": "direct",
        "config": args.config,
        "seed": args.seed,
        "legal_single_frame": True,
        "target": "depth_z",
        "metric_scope": "single_frame_3d dataset only; do not compare directly to wall_normal_height or FPP-ML-Bench depth RMSE",
        "checkpoint": str(best_ckpt),
        "best_val_object_rmse": best,
        "checkpoint_selection_metric": "full_val_object_rmse" if args.full_train_val else "fast_gpu_val_object_rmse",
        "val": summarize_rows(val_rows),
        "test": summarize_rows(test_rows),
        "history": history,
        "args": vars(args),
    }
    with (eval_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    acp = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    acp = acp / acp[0]
    betas = 1 - (acp[1:] / acp[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def namespace_from_dict(data: Dict[str, object]) -> argparse.Namespace:
    defaults = {
        "base_channels": 32,
        "ch_mult": [1, 2, 4, 8],
        "num_res_blocks": 1,
        "dropout": 0.05,
        "time_emb_dim": 128,
    }
    merged = dict(defaults)
    merged.update(data)
    return argparse.Namespace(**merged)


def load_base_model(path: str | Path, cond_channels: int, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, object]]:
    ckpt = torch.load(str(path), map_location=device)
    saved_args = ckpt.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = {}
    baseline_arch = str(saved_args.get("arch", "")).lower()
    if baseline_arch:
        if baseline_arch == "unet":
            model_args = namespace_from_dict(saved_args)
            model = ConditionalUNet(
                in_channels=1,
                cond_channels=1,
                out_channels=1,
                base_ch=int(getattr(model_args, "unet_base_channels", 32)),
                ch_mult=tuple(getattr(model_args, "unet_ch_mult", [1, 2, 4])),
                num_res_blocks=int(getattr(model_args, "unet_num_res_blocks", 1)),
                dropout=float(getattr(model_args, "dropout", 0.0)),
                time_emb_dim=int(getattr(model_args, "unet_time_emb_dim", 128)),
            ).to(device)
        else:
            from models.single_frame_baselines import build_single_frame_baseline

            model = build_single_frame_baseline(
                baseline_arch,
                in_channels=1,
                out_channels=1,
                base_channels=int(saved_args.get("base_channels") or 0) or None,
                dropout_rate=float(saved_args.get("dropout", 0.0)),
            ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        setattr(model, "_single_frame_baseline_arch", baseline_arch)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model, saved_args
    base_config = canonical_config(str(saved_args.get("config", "raw")))
    out_channels = 5 if base_config == "teacher_aux" else 1
    model_args = namespace_from_dict(saved_args)
    model = build_model(cond_channels, out_channels, model_args).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, saved_args


@torch.no_grad()
def base_predict_norm(base_model: torch.nn.Module, batch: Dict[str, object], device: torch.device) -> torch.Tensor:
    return forward_direct(base_model, batch, device)[:, :1]


class ResidualPosterior:
    def __init__(self, model: ConditionalUNet, timesteps: int, residual_scale: float, device: torch.device) -> None:
        self.model = model
        self.timesteps = int(timesteps)
        self.residual_scale = float(residual_scale)
        self.device = device
        betas = cosine_beta_schedule(self.timesteps).to(device)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        self.sqrt_acp = torch.sqrt(acp)
        self.sqrt_om = torch.sqrt(1.0 - acp)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self.sqrt_acp[t].view(-1, 1, 1, 1)
        so = self.sqrt_om[t].view(-1, 1, 1, 1)
        return sa * x0 + so * noise

    def cond_with_base(self, batch: Dict[str, object], base_norm: torch.Tensor) -> torch.Tensor:
        cond = batch["cond"].to(self.device, non_blocking=True).float()  # type: ignore[index]
        return torch.cat([cond, base_norm.detach()], dim=1)

    def target_residual(self, batch: Dict[str, object], base_norm: torch.Tensor) -> torch.Tensor:
        target = batch["depth"].to(self.device, non_blocking=True).float()  # type: ignore[index]
        residual = (target - base_norm.detach()) / max(self.residual_scale, 1e-6)
        return torch.clamp(residual, -1.0, 1.0)

    def training_loss(self, batch: Dict[str, object], base_model: torch.nn.Module, args: argparse.Namespace) -> torch.Tensor:
        base_norm = base_predict_norm(base_model, batch, self.device)
        cond = self.cond_with_base(batch, base_norm)
        target_res = self.target_residual(batch, base_norm)
        t = torch.randint(0, self.timesteps, (target_res.shape[0],), device=self.device)
        noisy = self.q_sample(target_res, t)
        pred_res = torch.tanh(self.model(noisy, t, cond))
        weight = train_weight(batch, self.device, args.object_mask_weight)
        loss = charbonnier(pred_res, target_res, weight=weight)
        loss = loss + args.lambda_mse * masked_mse(pred_res, target_res, weight=weight)
        if args.lambda_grad > 0:
            final_pred = torch.clamp(base_norm + self.residual_scale * pred_res, -1.0, 1.0)
            target = batch["depth"].to(self.device, non_blocking=True).float()  # type: ignore[index]
            loss = loss + args.lambda_grad * gradient_loss(final_pred, target, weight=weight)
        if args.lambda_final > 0:
            final_pred = torch.clamp(base_norm + self.residual_scale * pred_res, -1.0, 1.0)
            target = batch["depth"].to(self.device, non_blocking=True).float()  # type: ignore[index]
            loss = loss + args.lambda_final * charbonnier(final_pred, target, weight=weight)
        return loss

    @torch.no_grad()
    def sample(self, batch: Dict[str, object], base_model: torch.nn.Module, steps: int, ensemble_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_norm = base_predict_norm(base_model, batch, self.device)
        cond = self.cond_with_base(batch, base_norm)
        b, _, h, w = base_norm.shape
        seq = torch.linspace(self.timesteps - 1, 0, int(steps), device=self.device).long()
        preds = []
        stride = max(1, self.timesteps // max(1, int(steps)))
        for _ in range(max(1, int(ensemble_size))):
            x = torch.randn((b, 1, h, w), device=self.device)
            for t_val in seq:
                t_int = int(t_val.item())
                t = torch.full((b,), t_int, device=self.device, dtype=torch.long)
                x0 = torch.tanh(self.model(x, t, cond))
                if t_int == 0:
                    x = x0
                    continue
                prev_t = max(t_int - stride, 0)
                eps = (x - self.sqrt_acp[t].view(-1, 1, 1, 1) * x0) / self.sqrt_om[t].view(-1, 1, 1, 1).clamp(min=1e-6)
                x = self.sqrt_acp[prev_t].view(1, 1, 1, 1) * x0 + self.sqrt_om[prev_t].view(1, 1, 1, 1) * eps
            preds.append(torch.clamp(base_norm + self.residual_scale * torch.clamp(x, -1.0, 1.0), -1.0, 1.0))
        stack = torch.stack(preds, dim=0)
        mean = stack.mean(dim=0)
        unc = stack.std(dim=0, unbiased=False) if stack.shape[0] > 1 else torch.zeros_like(mean)
        return base_norm, mean, unc


def sample_record(pred_norm: torch.Tensor, batch: Dict[str, object], j: int, config: str, mode: str) -> Dict[str, object]:
    return row_from_prediction(pred_norm, batch, j, config, mode)


def compact_sample(batch: Dict[str, object], j: int, base: torch.Tensor, mean: torch.Tensor, unc: torch.Tensor) -> Dict[str, object]:
    return {
        "sample_id": batch["sample_id"][j],  # type: ignore[index]
        "object_id": int(batch["object_id"][j].item()),  # type: ignore[index]
        "pose_id": int(batch["pose_id"][j].item()),  # type: ignore[index]
        "scale_mm": batch["scale_mm"][j:j + 1].detach().cpu(),  # type: ignore[index]
        "center_mm": batch["center_mm"][j:j + 1].detach().cpu(),  # type: ignore[index]
        "depth_raw": batch["depth_raw"][j:j + 1].detach().cpu(),  # type: ignore[index]
        "object_mask": batch["object_mask"][j:j + 1].detach().cpu(),  # type: ignore[index]
        "valid_mask": batch["valid_mask"][j:j + 1].detach().cpu(),  # type: ignore[index]
        "base_norm": base[j:j + 1].detach().cpu(),
        "mean_norm": mean[j:j + 1].detach().cpu(),
        "unc_norm": unc[j:j + 1].detach().cpu(),
    }


def row_from_compact(item: Dict[str, object], pred_norm: torch.Tensor, config: str, mode: str) -> Dict[str, object]:
    batch = {
        "sample_id": [item["sample_id"]],
        "object_id": torch.tensor([int(item["object_id"])]),
        "pose_id": torch.tensor([int(item["pose_id"])]),
        "scale_mm": item["scale_mm"],
        "center_mm": item["center_mm"],
        "depth_raw": item["depth_raw"],
        "object_mask": item["object_mask"],
        "valid_mask": item["valid_mask"],
    }
    return row_from_prediction(pred_norm, batch, 0, config, mode)


@torch.no_grad()
def collect_residual_samples(
    posterior: ResidualPosterior,
    base_model: torch.nn.Module,
    loader: Iterable[Dict[str, object]],
    device: torch.device,
    args: argparse.Namespace,
    split: str,
) -> Dict[str, object]:
    posterior.model.eval()
    samples: List[Dict[str, object]] = []
    rows_base: List[Dict[str, object]] = []
    rows_mean: List[Dict[str, object]] = []
    for batch in tqdm(loader, desc=f"eval residual {split}", leave=False):
        base_norm, mean_norm, unc_norm = posterior.sample(batch, base_model, steps=args.sample_steps, ensemble_size=args.ensemble_size)
        for j in range(base_norm.shape[0]):
            rows_base.append(sample_record(base_norm, batch, j, args.config, "base_unet"))
            rows_mean.append(sample_record(mean_norm, batch, j, args.config, "posterior_mean"))
            samples.append(compact_sample(batch, j, base_norm, mean_norm, unc_norm))
    return {"samples": samples, "base_unet": rows_base, "posterior_mean": rows_mean}


def parse_alpha_grid(text: str) -> List[float]:
    vals = []
    for part in str(text).replace(",", " ").split():
        try:
            vals.append(min(max(float(part), 0.0), 1.0))
        except ValueError:
            pass
    return sorted(set(vals)) or [0.0, 0.25, 0.5, 0.75, 1.0]


def gate_rows(samples: List[Dict[str, object]], tau: float, alpha: float, args: argparse.Namespace, mode: str = "posterior_gate") -> Tuple[List[Dict[str, object]], float]:
    rows: List[Dict[str, object]] = []
    accepted = 0.0
    total = 0.0
    for item in samples:
        base = item["base_norm"]  # type: ignore[assignment]
        mean = item["mean_norm"]  # type: ignore[assignment]
        unc = item["unc_norm"]  # type: ignore[assignment]
        assert torch.is_tensor(base) and torch.is_tensor(mean) and torch.is_tensor(unc)
        if tau < 0:
            use = torch.zeros_like(base, dtype=torch.bool)
        else:
            correction = torch.abs(mean - base)
            use = (unc <= float(tau)) & (correction <= float(args.max_gate_correction))
        pred = torch.where(use, torch.clamp(base + float(alpha) * (mean - base), -1.0, 1.0), base)
        accepted += float(use.float().mean().item())
        total += 1.0
        rows.append(row_from_compact(item, pred, args.config, mode))
    return rows, accepted / max(total, 1.0)


def choose_gate_from_val(samples: List[Dict[str, object]], args: argparse.Namespace) -> Dict[str, object]:
    unc_vals = []
    for item in samples:
        unc = item["unc_norm"]  # type: ignore[assignment]
        assert torch.is_tensor(unc)
        arr = unc.numpy().reshape(-1)
        if arr.size:
            unc_vals.append(arr)
    if unc_vals:
        vals = np.concatenate(unc_vals)
        thresholds = [-1.0, 0.0] + [float(x) for x in np.percentile(vals, [10, 20, 40, 60, 80, 90, 95, 99])] + [float(np.max(vals) + 1e-6)]
    else:
        thresholds = [-1.0]
    best: Dict[str, object] = {
        "threshold": -1.0,
        "alpha": 0.0,
        "val_object_rmse": float("inf"),
        "val_valid_rmse": float("inf"),
        "accepted_fraction": 0.0,
        "rows": [],
    }
    for tau in thresholds:
        for alpha in parse_alpha_grid(args.alpha_grid):
            rows, accepted = gate_rows(samples, float(tau), float(alpha), args)
            summary = summarize_rows(rows)
            obj = float(summary["object"]["rmse"]["mean"])  # type: ignore[index]
            valid = float(summary["valid"]["rmse"]["mean"])  # type: ignore[index]
            if obj < float(best["val_object_rmse"]) - 1e-12:
                best = {
                    "threshold": float(tau),
                    "alpha": float(alpha),
                    "val_object_rmse": obj,
                    "val_valid_rmse": valid,
                    "accepted_fraction": float(accepted),
                    "rows": rows,
                }
    return best


def evaluate_residual_split(
    posterior: ResidualPosterior,
    base_model: torch.nn.Module,
    loader: Iterable[Dict[str, object]],
    device: torch.device,
    args: argparse.Namespace,
    gate: Optional[Dict[str, object]],
    split: str,
) -> Dict[str, object]:
    collected = collect_residual_samples(posterior, base_model, loader, device, args, split=split)
    if gate is None:
        chosen = choose_gate_from_val(collected["samples"], args)  # type: ignore[arg-type]
        gate_rows_selected = chosen["rows"]
        gate_info = {k: v for k, v in chosen.items() if k != "rows"}
    else:
        gate_rows_selected, accepted = gate_rows(
            collected["samples"],  # type: ignore[arg-type]
            float(gate["threshold"]),
            float(gate["alpha"]),
            args,
        )
        gate_info = dict(gate)
        gate_info["accepted_fraction"] = float(accepted)
    return {
        "samples": collected["samples"],
        "base_unet": collected["base_unet"],
        "posterior_mean": collected["posterior_mean"],
        "posterior_gate": gate_rows_selected,
        "gate": gate_info,
    }


@torch.no_grad()
def evaluate_residual_fast(
    posterior: ResidualPosterior,
    base_model: torch.nn.Module,
    loader: Iterable[Dict[str, object]],
    device: torch.device,
    args: argparse.Namespace,
    split: str,
) -> Dict[str, float]:
    posterior.model.eval()
    base_model.eval()
    values: Dict[str, List[float]] = {
        "base_object": [],
        "base_valid": [],
        "mean_object": [],
        "mean_valid": [],
    }
    for batch in tqdm(loader, desc=f"fast eval residual {split}", leave=False):
        base_norm, mean_norm, _ = posterior.sample(batch, base_model, steps=args.sample_steps, ensemble_size=args.ensemble_size)
        base_vals = _batch_rmse_values(base_norm, batch)
        mean_vals = _batch_rmse_values(mean_norm, batch)
        values["base_object"].extend(float(x) for x in base_vals["object"].cpu().tolist())
        values["base_valid"].extend(float(x) for x in base_vals["valid"].cpu().tolist())
        values["mean_object"].extend(float(x) for x in mean_vals["object"].cpu().tolist())
        values["mean_valid"].extend(float(x) for x in mean_vals["valid"].cpu().tolist())
    return {
        "n": float(len(values["mean_object"])),
        "base_object_rmse": float(np.mean(values["base_object"])) if values["base_object"] else float("nan"),
        "base_valid_rmse": float(np.mean(values["base_valid"])) if values["base_valid"] else float("nan"),
        "posterior_mean_object_rmse": float(np.mean(values["mean_object"])) if values["mean_object"] else float("nan"),
        "posterior_mean_valid_rmse": float(np.mean(values["mean_valid"])) if values["mean_valid"] else float("nan"),
    }


def train_residual(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.config = canonical_config(args.config)
    loaders = create_loaders(
        data_root=args.data_root,
        config=args.config,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        image_h=args.image_h,
        image_w=args.image_w,
        train_epoch_repeats=args.train_epoch_repeats,
        train_subset=args.train_subset,
        cache_features=args.cache_features,
        feature_cache_dir=args.feature_cache_dir or None,
        write_feature_cache=False,
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    args.channel_names = loaders["channel_names"]
    args.cond_channels = int(loaders["cond_channels"])
    args.posterior_cond_channels = args.cond_channels + 1
    args.normalization = loaders["stats"]
    args.split_counts = loaders["split_counts"]
    with (save_dir / "loader_smoke_summary.json").open("w", encoding="utf-8") as f:
        json.dump(smoke_summary(loaders), f, indent=2, ensure_ascii=False)
    if args.smoke_only:
        print((save_dir / "loader_smoke_summary.json").read_text(encoding="utf-8"))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    base_model, base_args = load_base_model(args.base_ckpt, args.cond_channels, device)
    args.base_config = canonical_config(str(base_args.get("config", args.config)))
    model = ConditionalUNet(
        in_channels=1,
        cond_channels=args.posterior_cond_channels,
        out_channels=1,
        base_ch=args.base_channels,
        ch_mult=tuple(args.ch_mult),
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        time_emb_dim=args.time_emb_dim,
    ).to(device)
    posterior = ResidualPosterior(model, timesteps=args.timesteps, residual_scale=args.residual_scale, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))
    best = float("inf")
    history: List[Dict[str, object]] = []
    print(f"Device: {device}")
    print(f"Stage: residual | Config: {args.config} | Base: {args.base_ckpt}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        seen = 0
        for batch in tqdm(loaders["train"], desc=f"residual {args.config} {ep}/{args.epochs}"):  # type: ignore[index]
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                loss = posterior.training_loss(batch, base_model, args)
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
            if args.full_train_val:
                val_eval = evaluate_residual_split(posterior, base_model, loaders["val"], device, args, gate=None, split="val")  # type: ignore[arg-type]
                gate_rows_val = val_eval["posterior_gate"]  # type: ignore[assignment]
                gate_summary = summarize_rows(gate_rows_val)  # type: ignore[arg-type]
                val_rmse = float(gate_summary["object"]["rmse"]["mean"])  # type: ignore[index]
                log["val_gate_object_rmse"] = val_rmse
                log["val_base_object_rmse"] = summarize_rows(val_eval["base_unet"])["object"]["rmse"]["mean"]  # type: ignore[arg-type,index]
                log["val_posterior_mean_object_rmse"] = summarize_rows(val_eval["posterior_mean"])["object"]["rmse"]["mean"]  # type: ignore[arg-type,index]
                log["val_gate"] = val_eval["gate"]
                log["val_mode"] = "full_metrics_gate"
            else:
                val_fast = evaluate_residual_fast(posterior, base_model, loaders["val"], device, args, split="val")  # type: ignore[arg-type]
                val_rmse = float(val_fast["posterior_mean_object_rmse"])
                log["val_gate_object_rmse"] = val_rmse
                log["val_base_object_rmse"] = float(val_fast["base_object_rmse"])
                log["val_posterior_mean_object_rmse"] = val_rmse
                log["val_posterior_mean_valid_rmse"] = float(val_fast["posterior_mean_valid_rmse"])
                log["val_mode"] = "fast_gpu_posterior_mean_rmse"
            if val_rmse < best:
                best = val_rmse
                save_checkpoint(save_dir / "checkpoints" / "best.pt", ep, model, optimizer, scaler, args, best, history)
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        if args.save_every > 0 and ep % args.save_every == 0:
            save_checkpoint(save_dir / "checkpoints" / f"epoch_{ep:03d}.pt", ep, model, optimizer, scaler, args, best, history)

    best_ckpt = save_dir / "checkpoints" / "best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(str(best_ckpt), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    val_eval = evaluate_residual_split(posterior, base_model, loaders["val"], device, args, gate=None, split="val")  # type: ignore[arg-type]
    gate_for_test = {k: v for k, v in val_eval["gate"].items() if k in {"threshold", "alpha"}}
    test_eval = evaluate_residual_split(posterior, base_model, loaders["test"], device, args, gate=gate_for_test, split="test")  # type: ignore[arg-type]
    eval_dir = save_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_rows(val_eval["base_unet"], eval_dir / "val_base_unet_per_sample_metrics.csv")  # type: ignore[arg-type]
    save_rows(val_eval["posterior_mean"], eval_dir / "val_posterior_mean_per_sample_metrics.csv")  # type: ignore[arg-type]
    save_rows(val_eval["posterior_gate"], eval_dir / "val_posterior_gate_per_sample_metrics.csv")  # type: ignore[arg-type]
    save_rows(test_eval["base_unet"], eval_dir / "test_base_unet_per_sample_metrics.csv")  # type: ignore[arg-type]
    save_rows(test_eval["posterior_mean"], eval_dir / "test_posterior_mean_per_sample_metrics.csv")  # type: ignore[arg-type]
    save_rows(test_eval["posterior_gate"], eval_dir / "per_sample_metrics.csv")  # type: ignore[arg-type]
    summary = {
        "stage": "residual",
        "config": args.config,
        "seed": args.seed,
        "legal_single_frame": True,
        "target": "depth_z",
        "metric_scope": "single_frame_3d dataset only; direct comparison is against frozen direct UNet checkpoints on the same split",
        "checkpoint": str(best_ckpt),
        "base_checkpoint": str(args.base_ckpt),
        "best_val_gate_object_rmse": best,
        "best_val_checkpoint_object_rmse": best,
        "checkpoint_selection_metric": "full_val_gate_object_rmse" if args.full_train_val else "fast_gpu_val_posterior_mean_object_rmse",
        "gate_selected_on_val": val_eval["gate"],
        "gate_applied_to_test": test_eval["gate"],
        "val_comparison": {
            "base_unet": summarize_rows(val_eval["base_unet"]),  # type: ignore[arg-type]
            "posterior_mean": summarize_rows(val_eval["posterior_mean"]),  # type: ignore[arg-type]
            "posterior_gate": summarize_rows(val_eval["posterior_gate"]),  # type: ignore[arg-type]
        },
        "test_comparison": {
            "base_unet": summarize_rows(test_eval["base_unet"]),  # type: ignore[arg-type]
            "posterior_mean": summarize_rows(test_eval["posterior_mean"]),  # type: ignore[arg-type]
            "posterior_gate": summarize_rows(test_eval["posterior_gate"]),  # type: ignore[arg-type]
        },
        "history": history,
        "args": vars(args),
    }
    with (eval_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def find_summary(root: Path, pattern: str) -> List[Path]:
    return sorted(root.glob(pattern))


def metric_from_summary(summary: Dict[str, object], stage: str, mode: str, roi: str = "object") -> float:
    if stage == "direct":
        return float(summary["test"][roi]["rmse"]["mean"])  # type: ignore[index]
    return float(summary["test_comparison"][mode][roi]["rmse"]["mean"])  # type: ignore[index]


def aggregate_direct(root: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    run_rows: List[Dict[str, object]] = []
    agg_rows: List[Dict[str, object]] = []
    for path in find_summary(root, "runs/direct_*_seed*/evaluation/summary.json"):
        data = load_json(path)
        cfg = str(data["config"])
        row = {
            "config": cfg,
            "seed": int(data["seed"]),
            "summary_path": str(path),
            "object_rmse": metric_from_summary(data, "direct", "test", "object"),
            "valid_rmse": metric_from_summary(data, "direct", "test", "valid"),
        }
        run_rows.append(row)
    for cfg in DIRECT_CONFIGS:
        subset = [r for r in run_rows if r["config"] == cfg]
        if not subset:
            continue
        obj = np.array([float(r["object_rmse"]) for r in subset], dtype=np.float64)
        valid = np.array([float(r["valid_rmse"]) for r in subset], dtype=np.float64)
        agg_rows.append({
            "config": cfg,
            "seeds": ",".join(str(r["seed"]) for r in sorted(subset, key=lambda x: int(x["seed"]))),
            "n": len(subset),
            "object_rmse_mean": float(obj.mean()),
            "object_rmse_std": float(obj.std(ddof=1) if obj.size > 1 else 0.0),
            "valid_rmse_mean": float(valid.mean()),
            "valid_rmse_std": float(valid.std(ddof=1) if valid.size > 1 else 0.0),
        })
    return run_rows, agg_rows


def aggregate_residual(root: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    run_rows: List[Dict[str, object]] = []
    agg_rows: List[Dict[str, object]] = []
    for path in find_summary(root, "runs/residual_*_seed*/evaluation/summary.json"):
        data = load_json(path)
        cfg = str(data["config"])
        base = metric_from_summary(data, "residual", "base_unet", "object")
        mean = metric_from_summary(data, "residual", "posterior_mean", "object")
        gate = metric_from_summary(data, "residual", "posterior_gate", "object")
        row = {
            "config": cfg,
            "seed": int(data["seed"]),
            "summary_path": str(path),
            "base_object_rmse": base,
            "posterior_mean_object_rmse": mean,
            "posterior_gate_object_rmse": gate,
            "gate_gain_percent": (base - gate) / base * 100.0 if base else float("nan"),
            "valid_base_rmse": metric_from_summary(data, "residual", "base_unet", "valid"),
            "valid_gate_rmse": metric_from_summary(data, "residual", "posterior_gate", "valid"),
        }
        run_rows.append(row)
    for cfg in RESIDUAL_CONFIGS:
        subset = [r for r in run_rows if r["config"] == cfg]
        if not subset:
            continue
        base = np.array([float(r["base_object_rmse"]) for r in subset], dtype=np.float64)
        mean = np.array([float(r["posterior_mean_object_rmse"]) for r in subset], dtype=np.float64)
        gate = np.array([float(r["posterior_gate_object_rmse"]) for r in subset], dtype=np.float64)
        gain = (base.mean() - gate.mean()) / base.mean() * 100.0 if base.mean() else float("nan")
        agg_rows.append({
            "config": cfg,
            "seeds": ",".join(str(r["seed"]) for r in sorted(subset, key=lambda x: int(x["seed"]))),
            "n": len(subset),
            "base_object_rmse_mean": float(base.mean()),
            "posterior_mean_object_rmse_mean": float(mean.mean()),
            "posterior_gate_object_rmse_mean": float(gate.mean()),
            "posterior_gate_object_rmse_std": float(gate.std(ddof=1) if gate.size > 1 else 0.0),
            "gate_gain_percent": float(gain),
        })
    return run_rows, agg_rows


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def direct_decision(agg: List[Dict[str, object]]) -> Dict[str, object]:
    by_cfg = {str(r["config"]): r for r in agg}
    out: Dict[str, object] = {}
    raw = by_cfg.get("raw")
    phys = by_cfg.get("raw_single_phys")
    teacher = by_cfg.get("teacher_aux")
    if raw and phys:
        gain = (float(raw["object_rmse_mean"]) - float(phys["object_rmse_mean"])) / float(raw["object_rmse_mean"]) * 100.0
        out["raw_single_phys_gain_percent"] = gain
        out["physics_input_supported"] = bool(gain >= 3.0)
    if phys and teacher:
        gain = (float(phys["object_rmse_mean"]) - float(teacher["object_rmse_mean"])) / float(phys["object_rmse_mean"]) * 100.0
        out["teacher_aux_gain_percent"] = gain
        out["teacher_aux_supported"] = bool(gain >= 2.0)
    return out


def residual_decisions(agg: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for row in agg:
        gain = float(row["gate_gain_percent"])
        if gain >= 2.0:
            verdict = "supports residual diffusion posterior gain"
        elif gain >= -1.0:
            verdict = "pilot or neutral trend"
        else:
            verdict = "does not support residual diffusion over frozen base"
        out[str(row["config"])] = {"gate_gain_percent": gain, "verdict": verdict}
    return out


def markdown_table(rows: List[Dict[str, object]], columns: List[Tuple[str, str]], digits: int = 4) -> List[str]:
    lines = ["| " + " | ".join(title for title, _ in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        vals = []
        for _, key in columns:
            val = row.get(key, "")
            if isinstance(val, float):
                vals.append(f"{val:.{digits}f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def summarize_experiment(args: argparse.Namespace) -> None:
    root = Path(args.save_dir)
    root.mkdir(parents=True, exist_ok=True)
    direct_runs, direct_agg = aggregate_direct(root)
    residual_runs, residual_agg = aggregate_residual(root)
    write_csv(direct_agg, root / "direct_aggregated_results.csv")
    write_csv(residual_agg, root / "residual_aggregated_results.csv")
    decisions = {
        "direct": direct_decision(direct_agg),
        "residual": residual_decisions(residual_agg),
    }
    summary = {
        "result_root": str(root),
        "target": "depth_z",
        "metric_scope": "single_frame_3d dataset only; do not compare directly to wall_normal_height or FPP-ML-Bench depth RMSE",
        "direct_runs": direct_runs,
        "direct_aggregated": direct_agg,
        "residual_runs": residual_runs,
        "residual_aggregated": residual_agg,
        "decisions": decisions,
    }
    with (root / "single_frame3d_physics_diffusion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    report: List[str] = [
        "# Single-Frame3D Physics/Diffusion Validation",
        "",
        "## Scope",
        "",
        "- Dataset: `single_frame_3d_dataset_v1_upload_smalltest`.",
        "- Target: `depth_z`.",
        "- Metrics are not directly comparable with old `wall_normal_height` or FPP-ML-Bench depth RMSE.",
        "- `phase_y/phase_x/bc_y/bc_x` are used only for auxiliary supervision or weighting, not as legal test-time inputs.",
        "",
        "## Direct UNet Results",
        "",
    ]
    if direct_agg:
        report.extend(markdown_table(direct_agg, [
            ("Config", "config"),
            ("Seeds", "seeds"),
            ("N", "n"),
            ("Object RMSE mean", "object_rmse_mean"),
            ("Object RMSE std", "object_rmse_std"),
            ("Valid RMSE mean", "valid_rmse_mean"),
        ]))
    else:
        report.append("No completed direct runs found.")
    report.extend(["", "## Residual Posterior Results", ""])
    if residual_agg:
        report.extend(markdown_table(residual_agg, [
            ("Config", "config"),
            ("Seeds", "seeds"),
            ("N", "n"),
            ("Base object RMSE", "base_object_rmse_mean"),
            ("Posterior mean RMSE", "posterior_mean_object_rmse_mean"),
            ("Posterior gate RMSE", "posterior_gate_object_rmse_mean"),
            ("Gate gain %", "gate_gain_percent"),
        ]))
    else:
        report.append("No completed residual runs found.")
    report.extend(["", "## Decisions", "", "```json", json.dumps(decisions, indent=2, ensure_ascii=False), "```", ""])
    (root / "single_frame3d_physics_diffusion_report.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["precompute_features", "direct", "residual", "summarize"], required=True)
    parser.add_argument("--data_root", default="/root/autodl-tmp/single_frame_3d_dataset_v1_upload_smalltest")
    parser.add_argument("--save_dir", default="/root/autodl-tmp/diffusion_fpp_v5/results/A_20260614_single_frame3d_physics_diffusion/debug")
    parser.add_argument("--config", default="raw")
    parser.add_argument("--base_ckpt", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_epoch_repeats", type=int, default=1)
    parser.add_argument("--train_subset", type=int, default=0)
    parser.add_argument("--image_h", type=int, default=480)
    parser.add_argument("--image_w", type=int, default=640)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--ch_mult", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--time_emb_dim", type=int, default=128)
    parser.add_argument("--lambda_mse", type=float, default=0.5)
    parser.add_argument("--lambda_grad", type=float, default=0.10)
    parser.add_argument("--lambda_teacher_phase", type=float, default=0.05)
    parser.add_argument("--lambda_final", type=float, default=0.50)
    parser.add_argument("--object_mask_weight", type=float, default=3.0)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--sample_steps", type=int, default=12)
    parser.add_argument("--ensemble_size", type=int, default=3)
    parser.add_argument("--residual_scale", type=float, default=0.25)
    parser.add_argument("--max_gate_correction", type=float, default=0.25)
    parser.add_argument("--alpha_grid", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--cache_features", action="store_true")
    parser.add_argument("--feature_cache_dir", default="")
    parser.add_argument("--full_train_val", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--smoke_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "precompute_features":
        precompute_feature_cache(args)
        return
    if args.stage == "summarize":
        summarize_experiment(args)
        return
    args.config = canonical_config(args.config)
    if args.stage == "direct":
        train_direct(args)
    elif args.stage == "residual":
        if not args.base_ckpt:
            raise ValueError("--base_ckpt is required for residual stage")
        train_residual(args)
    else:
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
