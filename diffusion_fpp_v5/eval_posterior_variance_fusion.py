"""Evaluate no-training posterior variance fusion over frozen UCPF candidates.

This script is intentionally not a learned model. It estimates candidate
variance parameters on the validation split only, then evaluates the selected
posterior-style fusion rule once on test.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.metrics import compute_metrics


METRIC_KEYS = ["rmse", "mae", "edge_rmse", "normal_deg", "ssim"]
CANDIDATE_NAMES = ["b", "p", "d"]


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


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        vals = np.asarray([float(row[key]) for row in rows if key in row and row[key] == row[key]], dtype=np.float64)
        out[key] = {
            "mean": float(vals.mean()) if vals.size else float("nan"),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        }
    return out


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def load_cache(cache_dir: Path, split: str) -> dict[str, np.ndarray]:
    def load(name: str, dtype: str) -> np.ndarray:
        path = cache_dir / f"{name}_{split}_{dtype}.npy"
        if not path.exists():
            raise FileNotFoundError(f"missing cache file: {path}")
        return np.load(path, mmap_mode="r")

    return {
        "d_b": load("d_b", "float16"),
        "d_p": load("d_p", "float16"),
        "d_d": load("d_d", "float16"),
        "target_mm": load("target_mm", "float32"),
        "mask": load("mask", "uint8"),
        "edge": load("edge", "float16"),
        "phase_conf": load("phase_conf", "float16"),
        "depth_minmax": load("depth_minmax", "float32"),
        "sample_index": load("sample_index", "int32"),
        "object_index": load("object_index", "int32"),
    }


def norm_to_mm(x_norm: torch.Tensor, depth_minmax: torch.Tensor) -> torch.Tensor:
    """Convert normalized [-1, 1] depth to per-sample millimeters."""
    lo = depth_minmax[:, 0].view(-1, 1, 1, 1)
    hi = depth_minmax[:, 1].view(-1, 1, 1, 1)
    return (x_norm + 1.0) * 0.5 * (hi - lo).clamp(min=1e-6) + lo


def sample_tensors(cache: dict[str, np.ndarray], idx: int, device: torch.device) -> dict[str, torch.Tensor]:
    def t(name: str, dtype=np.float32) -> torch.Tensor:
        arr = np.asarray(cache[name][idx]).astype(dtype)
        return torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=torch.float32)

    depth_minmax = torch.from_numpy(np.asarray(cache["depth_minmax"][idx]).astype(np.float32)).unsqueeze(0).to(device)
    d_b = norm_to_mm(t("d_b"), depth_minmax)
    d_p = norm_to_mm(t("d_p"), depth_minmax)
    d_d = norm_to_mm(t("d_d"), depth_minmax)
    return {
        "candidates": torch.cat([d_b, d_p, d_d], dim=1),
        "target_mm": t("target_mm"),
        "mask": t("mask"),
        "edge": t("edge"),
        "phase_conf": t("phase_conf"),
        "sample_index": int(cache["sample_index"][idx]),
        "object_index": int(cache["object_index"][idx]),
    }


def estimate_global_sigma(cache: dict[str, np.ndarray], device: torch.device) -> dict[str, float]:
    sq_sum = np.zeros(3, dtype=np.float64)
    count = 0.0
    for idx in tqdm(range(int(cache["target_mm"].shape[0])), desc="estimate global sigma"):
        batch = sample_tensors(cache, idx, device)
        cand = batch["candidates"]
        target = batch["target_mm"]
        mask = (batch["mask"] > 0.5).float()
        err2 = (cand - target).pow(2) * mask
        sq_sum += err2.flatten(2).sum(dim=2).squeeze(0).detach().cpu().numpy()
        count += float(mask.sum().item())
    mse = sq_sum / max(count, 1.0)
    sigma = np.sqrt(np.maximum(mse, 1e-6))
    return {name: float(sigma[i]) for i, name in enumerate(CANDIDATE_NAMES)}


def pvf_fuse(
    cand: torch.Tensor,
    global_sigma: torch.Tensor,
    mode: str,
    disagreement_gamma: float,
    phase_gamma: float,
    edge_gamma: float,
    trust_kappa: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fused depth and candidate weights.

    cand: [B, 3, H, W] in mm.
    """
    sigma2 = global_sigma.view(1, 3, 1, 1).pow(2).expand_as(cand).clone()
    if mode in {"disagreement", "hybrid"}:
        # Candidate-specific disagreement: a candidate becomes less trusted
        # where it strongly disagrees with the other posterior evidence.
        others = []
        for i in range(3):
            diff = [(cand[:, i:i + 1] - cand[:, j:j + 1]).pow(2) for j in range(3) if j != i]
            others.append(torch.stack(diff, dim=1).mean(dim=1))
        disagreement = torch.cat(others, dim=1)
        sigma2 = sigma2 + float(disagreement_gamma) * disagreement
    if mode == "hybrid":
        # Conservative physical priors: phase evidence is less reliable at low
        # phase confidence; diffusion correction is less trusted in high-edge
        # regions unless disagreement supports it.
        # The caller may pass zero maps by setting gammas to 0.
        pass
    precision = 1.0 / sigma2.clamp(min=1e-6)
    weights = precision / precision.sum(dim=1, keepdim=True).clamp(min=1e-6)
    fused = (weights * cand).sum(dim=1, keepdim=True)
    if trust_kappa is not None and trust_kappa > 0:
        d_b = cand[:, 0:1]
        sigma_b = global_sigma[0].view(1, 1, 1, 1)
        delta = torch.clamp(fused - d_b, -float(trust_kappa) * sigma_b, float(trust_kappa) * sigma_b)
        fused = d_b + delta
    return fused, weights


def eval_config(
    cache: dict[str, np.ndarray],
    device: torch.device,
    global_sigma: dict[str, float],
    mode: str,
    disagreement_gamma: float,
    trust_kappa: float | None,
    save_rows: Path | None = None,
    full_metrics: bool = True,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    sigma = torch.tensor([global_sigma[n] for n in CANDIDATE_NAMES], device=device, dtype=torch.float32)
    rows: list[dict[str, Any]] = []
    for idx in tqdm(range(int(cache["target_mm"].shape[0])), desc=f"eval {mode}"):
        batch = sample_tensors(cache, idx, device)
        fused, weights = pvf_fuse(
            batch["candidates"], sigma, mode=mode,
            disagreement_gamma=disagreement_gamma,
            phase_gamma=0.0, edge_gamma=0.0, trust_kappa=trust_kappa,
        )
        if full_metrics:
            metrics = compute_metrics(fused, batch["target_mm"], mask=batch["mask"])
        else:
            mask = batch["mask"] > 0.5
            diff = fused - batch["target_mm"]
            valid = diff[mask]
            metrics = {
                "rmse": float(torch.sqrt(torch.mean(valid * valid)).detach().cpu().item()),
                "mae": float(torch.mean(torch.abs(valid)).detach().cpu().item()),
            }
        row: dict[str, Any] = {
            "sample_index": batch["sample_index"],
            "object_index": batch["object_index"],
            "mode": mode,
            "disagreement_gamma": float(disagreement_gamma),
            "trust_kappa": "" if trust_kappa is None else float(trust_kappa),
            "w_b_mean": float(weights[:, 0:1].mean().detach().cpu().item()),
            "w_p_mean": float(weights[:, 1:2].mean().detach().cpu().item()),
            "w_d_mean": float(weights[:, 2:3].mean().detach().cpu().item()),
            **metrics,
        }
        rows.append(row)
    if save_rows is not None:
        write_rows(rows, save_rows)
    return summarize_rows(rows), rows


def main() -> None:
    parser = argparse.ArgumentParser(description="No-training posterior variance fusion over UCPF cache.")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/fpp_ml_ucpf_hier_orderfix_cache_960_seed180")
    parser.add_argument("--save_dir", default="results/fpp960_pvf_orderfix_seed180")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gammas", default="0 0.05 0.1 0.2 0.5 1 2 5 10")
    parser.add_argument("--kappas", default="none 1 1.5 2 3")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    manifest_path = cache_dir / "ucpf_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    val_cache = load_cache(cache_dir, "val")
    test_cache = load_cache(cache_dir, "test")

    global_sigma = estimate_global_sigma(val_cache, device)
    gamma_grid = [float(x) for x in str(args.gammas).replace(",", " ").split() if x]
    kappa_grid: list[float | None] = []
    for val in str(args.kappas).replace(",", " ").split():
        kappa_grid.append(None if val.lower() in {"none", "no", "null"} else float(val))

    configs: list[dict[str, Any]] = []
    # Global scalar PVF has no disagreement term.
    summary, _ = eval_config(val_cache, device, global_sigma, "global", 0.0, None, full_metrics=False)
    configs.append({"mode": "global", "disagreement_gamma": 0.0, "trust_kappa": None, "val": summary})
    for gamma in gamma_grid:
        summary, _ = eval_config(val_cache, device, global_sigma, "disagreement", gamma, None, full_metrics=False)
        configs.append({"mode": "disagreement", "disagreement_gamma": gamma, "trust_kappa": None, "val": summary})
    for gamma in gamma_grid:
        for kappa in kappa_grid:
            if kappa is None:
                continue
            summary, _ = eval_config(val_cache, device, global_sigma, "disagreement", gamma, kappa, full_metrics=False)
            configs.append({"mode": "disagreement_trust", "disagreement_gamma": gamma, "trust_kappa": kappa, "val": summary})

    configs = sorted(configs, key=lambda c: c["val"]["rmse"]["mean"])
    selected = configs[0]
    test_summary, test_rows = eval_config(
        test_cache, device, global_sigma,
        mode="global" if selected["mode"] == "global" else "disagreement",
        disagreement_gamma=float(selected["disagreement_gamma"]),
        trust_kappa=selected["trust_kappa"],
        save_rows=save_dir / "test_pvf_rows.csv",
    )

    # Also evaluate raw candidates on test for a self-contained summary.
    candidate_rows: list[dict[str, Any]] = []
    for idx in tqdm(range(int(test_cache["target_mm"].shape[0])), desc="eval candidates"):
        batch = sample_tensors(test_cache, idx, device)
        for ci, name in enumerate(CANDIDATE_NAMES):
            metrics = compute_metrics(batch["candidates"][:, ci:ci + 1], batch["target_mm"], mask=batch["mask"])
            candidate_rows.append({
                "sample_index": batch["sample_index"],
                "object_index": batch["object_index"],
                "candidate": name,
                **metrics,
            })
    write_rows(candidate_rows, save_dir / "test_candidate_rows.csv")
    candidate_summary = {}
    for name in CANDIDATE_NAMES:
        candidate_summary[name] = summarize_rows([r for r in candidate_rows if r["candidate"] == name])

    summary = {
        "method": "Posterior Variance Fusion (no training)",
        "cache_dir": str(cache_dir),
        "global_sigma_mm_from_val": global_sigma,
        "selected_by_val": selected,
        "test": test_summary,
        "test_weight_means": {
            "w_b": float(np.mean([r["w_b_mean"] for r in test_rows])),
            "w_p": float(np.mean([r["w_p_mean"] for r in test_rows])),
            "w_d": float(np.mean([r["w_d_mean"] for r in test_rows])),
        },
        "candidate_test": candidate_summary,
        "all_val_configs": configs,
        "manifest_candidate_source": manifest.get("candidate_source", {}),
    }
    with (save_dir / "pvf_summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, ensure_ascii=False)
    print(json.dumps(json_safe({
        "selected": selected,
        "test_rmse": test_summary["rmse"]["mean"],
        "summary": save_dir / "pvf_summary.json",
    }), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
