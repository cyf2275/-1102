"""Export frozen UCPF candidate tensors for FPP-ML-Bench.

The cache produced by this script is intentionally model-agnostic:
  D_b: conservative base prior from an existing base prediction cache
  D_p: frozen PSP/phase-depth branch prediction
  D_d: frozen diffusion posterior candidate

All depth candidates are stored in the same normalized [-1, 1] depth scale.
The valid mask is stored for losses/metrics/visualization only; it is not a
model input for UCPF.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from data.dataset_fpp_phase import create_fpp_phase_loaders
from eval_hierarchical_phase_fusion import (
    aux_predict01,
    build_aux_depth_model,
    build_depth_diffusion,
)
from eval_hierarchical_physical_gate import masked_mean
from eval_pixel_adaptive_gate import make_gate
from train_fpp_official_style_unet import METRIC_KEYS, prediction_to_mm, summarize
from utils.metrics import compute_metrics


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def open_memmap(path: Path, dtype: np.dtype | type, shape: tuple[int, ...]):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def write_tensor(mem, pos: int, tensor: torch.Tensor, dtype: np.dtype | type) -> int:
    arr = tensor.detach().cpu().numpy().astype(dtype, copy=False)
    n = arr.shape[0]
    mem[pos:pos + n] = arr
    return n


def norm_to_mm(depth_norm: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return prediction_to_mm(torch.clamp((depth_norm + 1.0) * 0.5, 0.0, 1.0), batch)


def metric_rows(pred_norm: torch.Tensor, batch: dict[str, torch.Tensor]) -> list[dict[str, float]]:
    pred_mm = norm_to_mm(pred_norm, batch)
    target_mm = batch["height_raw"].to(pred_norm.device, non_blocking=True)
    mask = batch["mask"].to(pred_norm.device, non_blocking=True)
    rows = []
    for j in range(pred_norm.shape[0]):
        rows.append(compute_metrics(pred_mm[j:j + 1], target_mm[j:j + 1], mask=mask[j:j + 1]))
    return rows


def select_diffusion_candidate(
    base: torch.Tensor,
    diff: torch.Tensor,
    edge: torch.Tensor,
    conf: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    mode = str(args.diff_candidate_mode)
    if mode == "raw":
        return torch.clamp(diff, -1.0, 1.0)

    pixel_cfg = {
        "alpha": args.pixel_alpha,
        "sample_edge_th": args.pixel_sample_edge_th,
        "edge_th": args.pixel_edge_th,
        "delta_min": args.pixel_delta_min,
        "conf_min": args.pixel_conf_min,
    }
    delta = torch.abs(diff - base)
    pixel_gate = make_gate(base, diff, edge, conf, pixel_cfg, mask=mask)
    pixel_pred = torch.clamp(base + args.pixel_alpha * pixel_gate * (diff - base), -1.0, 1.0)
    if mode == "pixel":
        return pixel_pred
    if mode != "hierarchical":
        raise ValueError(f"unknown diff_candidate_mode: {mode}")

    edge_mean = masked_mean(edge, mask)
    delta_mean = masked_mean(delta, mask)
    conf_mean = masked_mean(conf, mask)
    high_override = (
        (edge_mean >= args.high_edge_min)
        & (edge_mean <= args.high_edge_max)
        & (delta_mean >= args.high_delta_min)
        & (delta_mean <= args.high_delta_max)
        & (conf_mean >= args.high_conf_min)
        & (conf_mean <= args.high_conf_max)
    )
    low_edge_sample = edge_mean <= args.pixel_sample_edge_th

    hierarchical = base.clone()
    for j in range(base.shape[0]):
        if bool(high_override[j].item()):
            hierarchical[j:j + 1] = diff[j:j + 1]
        elif bool(low_edge_sample[j].item()):
            hierarchical[j:j + 1] = pixel_pred[j:j + 1]
    return torch.clamp(hierarchical, -1.0, 1.0)


@torch.no_grad()
def export_split(
    args: argparse.Namespace,
    split_name: str,
    depth_loader,
    phase_loader,
    depth_diffusion,
    aux_model,
    aux_args,
    aux_kind,
    aux_mode,
    device: torch.device,
) -> dict[str, Any]:
    out_dir = Path(args.save_dir)
    n_total = len(depth_loader.dataset)
    probe = depth_loader.dataset[0]
    _, h, w = probe["height"].shape
    cond_channels = int(probe["cond"].shape[0])

    arrays = {
        "d_b": open_memmap(out_dir / f"d_b_{split_name}_float16.npy", np.float16, (n_total, 1, h, w)),
        "d_p": open_memmap(out_dir / f"d_p_{split_name}_float16.npy", np.float16, (n_total, 1, h, w)),
        "d_d": open_memmap(out_dir / f"d_d_{split_name}_float16.npy", np.float16, (n_total, 1, h, w)),
        "target": open_memmap(out_dir / f"target_{split_name}_float16.npy", np.float16, (n_total, 1, h, w)),
        "target_mm": open_memmap(out_dir / f"target_mm_{split_name}_float32.npy", np.float32, (n_total, 1, h, w)),
        "mask": open_memmap(out_dir / f"mask_{split_name}_uint8.npy", np.uint8, (n_total, 1, h, w)),
        "edge": open_memmap(out_dir / f"edge_{split_name}_float16.npy", np.float16, (n_total, 1, h, w)),
        "phase_conf": open_memmap(out_dir / f"phase_conf_{split_name}_float16.npy", np.float16, (n_total, 1, h, w)),
        "fringe": open_memmap(out_dir / f"fringe_{split_name}_float16.npy", np.float16, (n_total, 1, h, w)),
        "physics_instr": open_memmap(
            out_dir / f"physics_instr_{split_name}_float16.npy",
            np.float16,
            (n_total, cond_channels, h, w),
        ),
        "depth_minmax": open_memmap(out_dir / f"depth_minmax_{split_name}_float32.npy", np.float32, (n_total, 2)),
        "sample_index": open_memmap(out_dir / f"sample_index_{split_name}_int32.npy", np.int32, (n_total,)),
        "object_index": open_memmap(out_dir / f"object_index_{split_name}_int32.npy", np.int32, (n_total,)),
    }

    rows_by_branch = {"d_b": [], "d_p": [], "d_d": []}
    write_pos = 0
    iterator = tqdm(
        zip(depth_loader, phase_loader),
        total=len(depth_loader),
        desc=f"export UCPF {split_name}",
    )
    for depth_batch, phase_batch in iterator:
        if not torch.equal(depth_batch["sample_index"], phase_batch["sample_index"]):
            raise RuntimeError(f"{split_name}: depth/phase loaders are not aligned")

        d_b = torch.clamp(depth_batch["base_height"].to(device, non_blocking=True), -1.0, 1.0)
        d_diff_raw = depth_diffusion.sample_ddim(
            depth_batch,
            steps=args.ddim_steps,
            ensemble_size=1,
            start_from_base=True,
            start_ratio=args.start_ratio,
            guidance=None,
            progress=False,
        )
        edge = torch.clamp(depth_batch["edge_score"].to(device, non_blocking=True), 0.0, 1.0)
        conf = torch.clamp(depth_batch["phase_conf"].to(device, non_blocking=True), 0.0, 1.0)
        mask = torch.clamp(depth_batch["mask"].to(device, non_blocking=True), 0.0, 1.0)
        d_d = select_diffusion_candidate(d_b, d_diff_raw, edge, conf, mask, args)
        d_p = aux_predict01(aux_model, phase_batch, device, aux_args, aux_kind, aux_mode) * 2.0 - 1.0
        d_p = torch.clamp(d_p, -1.0, 1.0)
        target = torch.clamp(depth_batch["height"].to(device, non_blocking=True), -1.0, 1.0)

        batch_size = d_b.shape[0]
        write_tensor(arrays["d_b"], write_pos, d_b, np.float16)
        write_tensor(arrays["d_p"], write_pos, d_p, np.float16)
        write_tensor(arrays["d_d"], write_pos, d_d, np.float16)
        write_tensor(arrays["target"], write_pos, target, np.float16)
        write_tensor(arrays["target_mm"], write_pos, depth_batch["height_raw"], np.float32)
        write_tensor(arrays["mask"], write_pos, (depth_batch["mask"] > 0.5).to(torch.uint8), np.uint8)
        write_tensor(arrays["edge"], write_pos, depth_batch["edge_score"], np.float16)
        write_tensor(arrays["phase_conf"], write_pos, depth_batch["phase_conf"], np.float16)
        write_tensor(arrays["fringe"], write_pos, depth_batch["fringe"], np.float16)
        write_tensor(arrays["physics_instr"], write_pos, depth_batch["cond"], np.float16)
        write_tensor(arrays["depth_minmax"], write_pos, depth_batch["depth_minmax"], np.float32)
        write_tensor(arrays["sample_index"], write_pos, depth_batch["sample_index"], np.int32)
        write_tensor(arrays["object_index"], write_pos, depth_batch["object_index"], np.int32)

        for name, pred in (("d_b", d_b), ("d_p", d_p), ("d_d", d_d)):
            rows_by_branch[name].extend(metric_rows(pred, depth_batch))
        write_pos += batch_size

    if write_pos != n_total:
        raise RuntimeError(f"{split_name}: wrote {write_pos} samples, expected {n_total}")

    for arr in arrays.values():
        arr.flush()

    return {
        "num_samples": n_total,
        "shape": [1, h, w],
        "cond_channels": cond_channels,
        "metrics": {name: summarize(rows) for name, rows in rows_by_branch.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export frozen UCPF candidate cache.")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_pspquad_cache_960")
    parser.add_argument("--save_dir", default="/root/autodl-tmp/fpp_ml_ucpf_cache_960")
    parser.add_argument("--base_prefix", default="base_c4_adapter")
    parser.add_argument("--depth_checkpoint", required=True)
    parser.add_argument("--phase_depth_checkpoint", required=True)
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--start_ratio", type=float, default=0.05)
    parser.add_argument("--diff_candidate_mode", choices=["raw", "pixel", "hierarchical"], default="hierarchical")
    parser.add_argument("--pixel_alpha", type=float, default=0.7)
    parser.add_argument("--pixel_sample_edge_th", type=float, default=0.47)
    parser.add_argument("--pixel_edge_th", type=float, default=1.0)
    parser.add_argument("--pixel_delta_min", type=float, default=0.12)
    parser.add_argument("--pixel_conf_min", type=float, default=0.0)
    parser.add_argument("--high_edge_min", type=float, default=0.58)
    parser.add_argument("--high_edge_max", type=float, default=0.62)
    parser.add_argument("--high_delta_min", type=float, default=0.09)
    parser.add_argument("--high_delta_max", type=float, default=0.105)
    parser.add_argument("--high_conf_min", type=float, default=0.76)
    parser.add_argument("--high_conf_max", type=float, default=0.80)
    parser.add_argument("--splits", default="train val test")
    parser.add_argument("--require_cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_batch_size != 1:
        print(
            "Warning: eval_batch_size != 1 can change deterministic DDIM noise grouping. "
            "Use batch size 1 for strict fixed-per-sample cache.",
            flush=True,
        )
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    depth_diffusion, include_ftp = build_depth_diffusion(args, device)
    aux_model, aux_args, aux_kind, aux_mode = build_aux_depth_model(args, device)
    phase_pred_prefix = getattr(aux_args, "phase_pred_prefix", None)

    loaders_depth = create_fpp_ml_bench_loaders(
        cache_dir=args.cache_dir,
        batch_size=1,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        include_ftp=include_ftp,
        image_h=args.image_size,
        image_w=args.image_size,
        require_cache=args.require_cache,
        base_prefix=args.base_prefix,
    )
    loaders_phase = create_fpp_phase_loaders(
        base_cache_dir=args.cache_dir,
        phase_cache_dir=args.phase_cache_dir,
        batch_size=1,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        image_h=args.image_size,
        image_w=args.image_size,
        phase_pred_prefix=phase_pred_prefix,
        require_cache=args.require_cache,
        preload_ram=bool(getattr(aux_args, "preload_ram", False)),
        train_minimal=bool(getattr(aux_args, "train_minimal", False)),
    )

    selected_splits = [s for s in str(args.splits).replace(",", " ").split() if s]
    if not selected_splits:
        raise ValueError("--splits must contain at least one split")

    manifest: dict[str, Any] = {
        "cache_type": "ucpf_frozen_candidates",
        "depth_scale": "normalized_minus1_to_1",
        "metric_unit": "mm",
        "mask_policy": "stored for losses/metrics/visualization only; not a UCPF input",
        "candidate_source": {
            "d_b_base_prefix": args.base_prefix,
            "d_p_checkpoint": args.phase_depth_checkpoint,
            "d_d_checkpoint": args.depth_checkpoint,
            "phase_pred_prefix": phase_pred_prefix,
            "aux_kind": aux_kind,
            "aux_mode": aux_mode,
            "ddim_steps": args.ddim_steps,
            "start_ratio": args.start_ratio,
            "diff_candidate_mode": args.diff_candidate_mode,
            "pixel_alpha": args.pixel_alpha,
            "pixel_sample_edge_th": args.pixel_sample_edge_th,
            "pixel_edge_th": args.pixel_edge_th,
            "pixel_delta_min": args.pixel_delta_min,
            "pixel_conf_min": args.pixel_conf_min,
            "high_edge_min": args.high_edge_min,
            "high_edge_max": args.high_edge_max,
            "high_delta_min": args.high_delta_min,
            "high_delta_max": args.high_delta_max,
            "high_conf_min": args.high_conf_min,
            "high_conf_max": args.high_conf_max,
            "eta": 0,
            "ensemble": 1,
            "sampling_seed": "PIPDiffusion fixed seed=0; export default eval_batch_size=1",
            "image_size": args.image_size,
        },
        "paths": {
            "base_cache_dir": args.cache_dir,
            "phase_cache_dir": args.phase_cache_dir,
            "save_dir": args.save_dir,
        },
        "splits": {},
        "metric_keys": METRIC_KEYS,
    }

    for split in selected_splits:
        loader_key = "train_eval" if split == "train" else split
        manifest["splits"][split] = export_split(
            args,
            split,
            loaders_depth[loader_key],
            loaders_phase[loader_key],
            depth_diffusion,
            aux_model,
            aux_args,
            aux_kind,
            aux_mode,
            device,
        )

    with (out_dir / "ucpf_candidate_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(manifest), f, indent=2, ensure_ascii=False)
    print(json.dumps(json_safe({"save_dir": str(out_dir), "splits": list(manifest["splits"])}), ensure_ascii=False))


if __name__ == "__main__":
    main()
