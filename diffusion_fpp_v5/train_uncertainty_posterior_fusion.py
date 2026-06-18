"""Train lightweight UCPF over frozen depth candidates.

UCPF learns reliability weights over frozen candidates only. It never receives
the benchmark mask as an input and never backpropagates into the candidate
generators. Validation selects checkpoint/T/kappa; test is evaluated only once
with the validation-selected configuration.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from diffusion_pip import grad_xy_padded, masked_mean, normal_loss
from utils.metrics import compute_metrics


METRIC_KEYS = ["rmse", "mae", "edge_rmse", "normal_deg", "ssim"]
CANDIDATE_FIELDS = {"b": "d_b", "p": "d_p", "d": "d_d"}


def parse_float_list(text: str) -> list[float]:
    vals = [float(x) for x in str(text).replace(",", " ").split() if x]
    return vals or [1.0]


def parse_candidates(text: str) -> list[str]:
    vals = [x.strip().lower() for x in str(text).replace(",", " ").split() if x.strip()]
    if not vals:
        vals = ["b", "p", "d"]
    for val in vals:
        if val not in CANDIDATE_FIELDS:
            raise ValueError(f"unknown candidate '{val}', expected subset of b,p,d")
    if "b" not in vals:
        raise ValueError("candidate list must include conservative base 'b'")
    out = []
    for val in vals:
        if val not in out:
            out.append(val)
    return out


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


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for key in METRIC_KEYS:
        vals = np.asarray([row[key] for row in rows], dtype=np.float64)
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


def write_history(rows: list[dict[str, Any]], path: Path) -> None:
    if rows:
        write_rows(rows, path)


class UCPFCandidateDataset(Dataset):
    def __init__(self, cache_dir: str | Path, split: str):
        self.cache_dir = Path(cache_dir)
        self.split = split
        manifest_path = self.cache_dir / "ucpf_candidate_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing UCPF manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        if split not in self.manifest.get("splits", {}):
            raise ValueError(f"split '{split}' not found in {manifest_path}")

        def load(name: str, dtype: str):
            path = self.cache_dir / f"{name}_{split}_{dtype}.npy"
            if not path.exists():
                raise FileNotFoundError(f"missing cache file: {path}")
            return np.load(path, mmap_mode="r")

        self.d_b = load("d_b", "float16")
        self.d_p = load("d_p", "float16")
        self.d_d = load("d_d", "float16")
        self.target = load("target", "float16")
        self.target_mm = load("target_mm", "float32")
        self.mask = load("mask", "uint8")
        self.edge = load("edge", "float16")
        self.phase_conf = load("phase_conf", "float16")
        self.fringe = load("fringe", "float16")
        self.physics_instr = load("physics_instr", "float16")
        self.depth_minmax = load("depth_minmax", "float32")
        self.sample_index = load("sample_index", "int32")
        self.object_index = load("object_index", "int32")

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        idx = int(idx)
        item = {
            "d_b": torch.from_numpy(np.asarray(self.d_b[idx]).astype(np.float32)).float(),
            "d_p": torch.from_numpy(np.asarray(self.d_p[idx]).astype(np.float32)).float(),
            "d_d": torch.from_numpy(np.asarray(self.d_d[idx]).astype(np.float32)).float(),
            "target": torch.from_numpy(np.asarray(self.target[idx]).astype(np.float32)).float(),
            "target_mm": torch.from_numpy(np.asarray(self.target_mm[idx]).astype(np.float32)).float(),
            "mask": torch.from_numpy(np.asarray(self.mask[idx]).astype(np.float32)).float(),
            "edge": torch.from_numpy(np.asarray(self.edge[idx]).astype(np.float32)).float(),
            "phase_conf": torch.from_numpy(np.asarray(self.phase_conf[idx]).astype(np.float32)).float(),
            "fringe": torch.from_numpy(np.asarray(self.fringe[idx]).astype(np.float32)).float(),
            "physics_instr": torch.from_numpy(np.asarray(self.physics_instr[idx]).astype(np.float32)).float(),
            "depth_minmax": torch.from_numpy(np.asarray(self.depth_minmax[idx]).astype(np.float32)).float(),
            "sample_index": torch.tensor(int(self.sample_index[idx]), dtype=torch.long),
            "object_index": torch.tensor(int(self.object_index[idx]), dtype=torch.long),
        }
        return item


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in batch[0]}


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = max(1, min(8, channels // 4))
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.net(x))


class LightweightUCPF(nn.Module):
    def __init__(self, in_channels: int, num_candidates: int, hidden: int = 32, blocks: int = 2,
                 use_uncertainty: bool = False):
        super().__init__()
        hidden = int(hidden)
        if hidden < 8:
            raise ValueError("hidden must be >= 8")
        self.num_candidates = int(num_candidates)
        self.use_uncertainty = bool(use_uncertainty)
        out_channels = self.num_candidates * (2 if self.use_uncertainty else 1)
        groups = max(1, min(8, hidden // 4))
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
        ]
        for _ in range(int(blocks)):
            layers.append(ResidualBlock(hidden))
        layers.append(nn.Conv2d(hidden, out_channels, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self.net(x)
        logits = out[:, :self.num_candidates]
        if not self.use_uncertainty:
            return logits, None
        raw_sigma = out[:, self.num_candidates:]
        return logits, raw_sigma


def selected_candidates(batch: dict[str, torch.Tensor], candidates: list[str], device: torch.device) -> torch.Tensor:
    return torch.cat([batch[CANDIDATE_FIELDS[c]].to(device, non_blocking=True) for c in candidates], dim=1)


def build_input(batch: dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    candidates = parse_candidates(args.candidates)
    cand = selected_candidates(batch, candidates, device)
    parts = [cand]
    pairwise = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            pairwise.append(torch.abs(cand[:, i:i + 1] - cand[:, j:j + 1]))
    if pairwise:
        parts.append(torch.cat(pairwise, dim=1))
    if not args.drop_edge_conf:
        parts.append(torch.clamp(batch["edge"].to(device, non_blocking=True), 0.0, 1.0))
        parts.append(torch.clamp(batch["phase_conf"].to(device, non_blocking=True), 0.0, 1.0))
    if args.input_mode == "x1":
        parts.append(batch["fringe"].to(device, non_blocking=True))
        if not args.drop_physics_instr:
            parts.append(batch["physics_instr"].to(device, non_blocking=True))
    return torch.cat(parts, dim=1)


def infer_input_channels(dataset: UCPFCandidateDataset, args: argparse.Namespace) -> int:
    candidates = parse_candidates(args.candidates)
    n = len(candidates)
    channels = n + (n * (n - 1)) // 2
    if not args.drop_edge_conf:
        channels += 2
    if args.input_mode == "x1":
        channels += 1
        if not args.drop_physics_instr:
            channels += int(dataset.physics_instr.shape[1])
    return channels


def norm_to_mm(depth_norm: torch.Tensor, depth_minmax: torch.Tensor) -> torch.Tensor:
    depth01 = torch.clamp((depth_norm + 1.0) * 0.5, 0.0, 1.0)
    dmin = depth_minmax[:, 0].view(-1, 1, 1, 1).to(depth_norm.device)
    dmax = depth_minmax[:, 1].view(-1, 1, 1, 1).to(depth_norm.device)
    return depth01 * (dmax - dmin).clamp(min=1e-6) + dmin


def sigma_to_norm_delta(sigma_mm: torch.Tensor, depth_minmax: torch.Tensor) -> torch.Tensor:
    dmin = depth_minmax[:, 0].view(-1, 1, 1, 1).to(sigma_mm.device)
    dmax = depth_minmax[:, 1].view(-1, 1, 1, 1).to(sigma_mm.device)
    return 2.0 * sigma_mm / (dmax - dmin).clamp(min=1e-6)


def bounded_sigma(raw_sigma: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    sigma_min = float(args.sigma_min_mm)
    sigma_max = float(args.sigma_max_mm)
    if sigma_max <= sigma_min:
        raise ValueError("sigma_max_mm must be greater than sigma_min_mm")
    return sigma_min + (sigma_max - sigma_min) * torch.sigmoid(raw_sigma)


def masked_mean_channels(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = torch.clamp(mask.to(device=x.device, dtype=x.dtype), 0.0, 1.0)
    denom = mask.sum().clamp(min=1.0) * max(1, x.shape[1])
    return (x * mask).sum() / denom


def gradient_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pdx, pdy = grad_xy_padded(pred)
    tdx, tdy = grad_xy_padded(target)
    return masked_mean(torch.abs(pdx - tdx), mask=mask) + masked_mean(torch.abs(pdy - tdy), mask=mask)


def forward_ucpf(
    model: LightweightUCPF,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    temperature: float,
    kappa: float,
    apply_clip: bool,
) -> dict[str, torch.Tensor]:
    candidates = parse_candidates(args.candidates)
    cand = selected_candidates(batch, candidates, device)
    x = build_input(batch, args, device)
    logits, raw_sigma = model(x)
    if args.use_uncertainty:
        if raw_sigma is None:
            raise RuntimeError("model did not return uncertainty channels")
        sigma_mm = bounded_sigma(raw_sigma, args)
        score = logits - torch.log(sigma_mm * sigma_mm + float(args.eps)) / max(float(temperature), 1e-6)
    else:
        sigma_mm = torch.ones_like(logits)
        score = logits
    weights = torch.softmax(score, dim=1)
    d_f = (weights * cand).sum(dim=1, keepdim=True)
    base_idx = candidates.index("b")
    d_b = cand[:, base_idx:base_idx + 1]
    if apply_clip:
        sigma_b_norm = sigma_to_norm_delta(sigma_mm[:, base_idx:base_idx + 1], batch["depth_minmax"])
        delta = torch.clamp(d_f - d_b, -float(kappa) * sigma_b_norm, float(kappa) * sigma_b_norm)
        final = torch.clamp(d_b + delta, -1.0, 1.0)
    else:
        final = torch.clamp(d_f, -1.0, 1.0)
    return {
        "candidates": cand,
        "logits": logits,
        "sigma_mm": sigma_mm,
        "weights": weights,
        "d_f": d_f,
        "final": final,
    }


def compute_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    kappa: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["target"].to(device, non_blocking=True)
    mask = torch.clamp(batch["mask"].to(device, non_blocking=True), 0.0, 1.0)
    final = output["final"]
    d_f = output["d_f"]
    cand = output["candidates"]
    candidates = parse_candidates(args.candidates)
    base_idx = candidates.index("b")

    depth = masked_mean(torch.abs(final - target), mask=mask)
    loss = depth
    parts = {"depth": float(depth.detach().cpu())}

    if args.lambda_edge > 0:
        edge = gradient_l1(final, target, mask)
        loss = loss + float(args.lambda_edge) * edge
        parts["edge"] = float(edge.detach().cpu())
    if args.lambda_normal > 0:
        nloss = normal_loss(final, target, mask=mask)
        loss = loss + float(args.lambda_normal) * nloss
        parts["normal"] = float(nloss.detach().cpu())
    if args.use_uncertainty and args.lambda_cal > 0 and not args.disable_cal:
        depth_minmax = batch["depth_minmax"].to(device, non_blocking=True)
        target_mm = batch["target_mm"].to(device, non_blocking=True)
        cand_mm = norm_to_mm(cand, depth_minmax)
        err_mm = torch.abs(cand_mm - target_mm).clamp(min=float(args.eps))
        sigma = output["sigma_mm"].clamp(min=float(args.eps))
        cal = masked_mean_channels(torch.abs(torch.log(sigma) - torch.log(err_mm)), mask)
        loss = loss + float(args.lambda_cal) * cal
        parts["cal"] = float(cal.detach().cpu())
    if args.trust_mode in {"soft", "clip"} and args.lambda_trust > 0:
        depth_minmax = batch["depth_minmax"].to(device, non_blocking=True)
        sigma_b_norm = sigma_to_norm_delta(output["sigma_mm"][:, base_idx:base_idx + 1], depth_minmax)
        d_b = cand[:, base_idx:base_idx + 1]
        trust = masked_mean(F.relu(torch.abs(d_f - d_b) - float(kappa) * sigma_b_norm), mask=mask)
        loss = loss + float(args.lambda_trust) * trust
        parts["trust"] = float(trust.detach().cpu())
    parts["total"] = float(loss.detach().cpu())
    return loss, parts


def train_one_epoch(
    model: LightweightUCPF,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    steps = 0
    pbar = tqdm(loader, desc="train UCPF", leave=False)
    apply_clip = args.trust_mode == "clip"
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp and device.type == "cuda"):
            output = forward_ucpf(
                model,
                batch,
                args,
                device,
                temperature=args.temperature,
                kappa=args.kappa,
                apply_clip=apply_clip,
            )
            loss, parts = compute_loss(output, batch, args, device, args.kappa)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        steps += 1
        pbar.set_postfix(loss=f"{parts['total']:.4f}")
    return {key: value / max(1, steps) for key, value in totals.items()}


def save_ucpf_visual(
    batch: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    save_path: Path,
    sample_offset: int,
) -> None:
    candidates = parse_candidates(args.candidates)
    target_mm = batch["target_mm"].to(device)
    depth_minmax = batch["depth_minmax"].to(device)
    final_mm = norm_to_mm(output["final"], depth_minmax)
    cand_mm = norm_to_mm(output["candidates"], depth_minmax)
    weights = output["weights"]
    sigma = output["sigma_mm"]
    mask = batch["mask"].to(device) > 0.5
    fringe = batch["fringe"].to(device)
    err = torch.abs(final_mm - target_mm)

    j = sample_offset
    valid = mask[j, 0].detach().cpu().numpy()
    target_np = target_mm[j, 0].detach().cpu().numpy()
    final_np = final_mm[j, 0].detach().cpu().numpy()
    err_np = err[j, 0].detach().cpu().numpy()
    valid_target = target_np[valid]
    vmin = float(valid_target.min()) if valid_target.size else float(target_np.min())
    vmax = float(valid_target.max()) if valid_target.size else float(target_np.max())

    panels: list[tuple[str, np.ndarray, str, float | None, float | None]] = [
        ("Fringe", fringe[j, 0].detach().cpu().numpy(), "gray", None, None),
        ("GT", np.ma.masked_where(~valid, target_np), "viridis", vmin, vmax),
    ]
    for idx, name in enumerate(candidates):
        panels.append((
            f"D_{name}",
            np.ma.masked_where(~valid, cand_mm[j, idx].detach().cpu().numpy()),
            "viridis",
            vmin,
            vmax,
        ))
    panels.append(("D_final", np.ma.masked_where(~valid, final_np), "viridis", vmin, vmax))
    panels.append(("Abs error", np.ma.masked_where(~valid, err_np), "hot", None, None))
    for idx, name in enumerate(candidates):
        panels.append((f"pi_{name}", np.ma.masked_where(~valid, weights[j, idx].detach().cpu().numpy()), "magma", 0.0, 1.0))
    if args.use_uncertainty:
        for idx, name in enumerate(candidates):
            panels.append((f"sigma_{name} mm", np.ma.masked_where(~valid, sigma[j, idx].detach().cpu().numpy()), "plasma", args.sigma_min_mm, args.sigma_max_mm))

    cols = 4
    rows = int(math.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.8 * rows))
    axes_flat = np.asarray(axes).reshape(-1)
    for ax, (title, image, cmap, lo, hi) in zip(axes_flat, panels):
        im = ax.imshow(image, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.axis("off")
        if title.startswith("pi_") or title.startswith("sigma") or title == "Abs error":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for ax in axes_flat[len(panels):]:
        ax.axis("off")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate(
    model: LightweightUCPF,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    temperature: float,
    kappa: float,
    split: str,
    out_dir: Path | None = None,
    save_visuals: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    apply_clip = args.trust_mode == "clip"
    visual_count = 0
    pbar = tqdm(loader, desc=f"eval {split}", leave=False)
    for batch in pbar:
        output = forward_ucpf(model, batch, args, device, temperature, kappa, apply_clip)
        final_mm = norm_to_mm(output["final"], batch["depth_minmax"].to(device, non_blocking=True))
        target_mm = batch["target_mm"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        weights = output["weights"]
        sigma = output["sigma_mm"]
        bs = final_mm.shape[0]
        for j in range(bs):
            metrics = compute_metrics(final_mm[j:j + 1], target_mm[j:j + 1], mask=mask[j:j + 1])
            row: dict[str, Any] = {
                "sample": int(batch["sample_index"][j].item()),
                "object_index": int(batch["object_index"][j].item()),
                **metrics,
            }
            for idx, name in enumerate(parse_candidates(args.candidates)):
                valid = mask[j:j + 1] > 0.5
                wv = weights[j:j + 1, idx:idx + 1]
                sv = sigma[j:j + 1, idx:idx + 1]
                row[f"pi_{name}_mean"] = float(wv[valid].mean().item()) if valid.any() else float(wv.mean().item())
                row[f"sigma_{name}_mean"] = float(sv[valid].mean().item()) if valid.any() else float(sv.mean().item())
            rows.append(row)
            if save_visuals and out_dir is not None and visual_count < args.save_vis_count:
                save_ucpf_visual(
                    batch,
                    output,
                    args,
                    device,
                    out_dir / "visuals" / f"{split}_sample_{len(rows)-1:03d}.png",
                    j,
                )
                visual_count += 1
    summary = summarize_rows(rows)
    summary["n"] = len(rows)
    summary["temperature"] = float(temperature)
    summary["kappa"] = float(kappa)
    return summary, rows


def select_by_validation(
    model: LightweightUCPF,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    temperatures = [1.0] if not args.use_uncertainty else parse_float_list(args.temperatures)
    kappas = parse_float_list(args.kappas) if args.trust_mode == "clip" else [float(args.kappa)]
    grid = []
    for temp in temperatures:
        for kap in kappas:
            summary, _ = evaluate(model, val_loader, args, device, temp, kap, split="val_grid")
            grid.append({
                "temperature": float(temp),
                "kappa": float(kap),
                "rmse": summary["rmse"]["mean"],
                "mae": summary["mae"]["mean"],
                "edge_rmse": summary["edge_rmse"]["mean"],
                "normal_deg": summary["normal_deg"]["mean"],
            })
    best = min(grid, key=lambda row: row["rmse"])
    return {"grid": grid, "selected": best}


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def set_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_preset(args: argparse.Namespace) -> None:
    preset = str(args.preset).lower()
    if preset == "custom":
        return
    if preset == "u0_x0":
        args.input_mode = "x0"
        args.use_uncertainty = False
        args.trust_mode = "none"
        args.lambda_edge = 0.0
        args.lambda_normal = 0.0
        args.lambda_cal = 0.0
        args.lambda_trust = 0.0
    elif preset == "u1_x0":
        args.input_mode = "x0"
        args.use_uncertainty = True
        args.trust_mode = "none"
        args.lambda_edge = 0.0
        args.lambda_normal = 0.0
        args.lambda_cal = 0.05
        args.lambda_trust = 0.0
    elif preset == "u2_soft_x0":
        args.input_mode = "x0"
        args.use_uncertainty = True
        args.trust_mode = "soft"
        args.lambda_edge = 0.03
        args.lambda_normal = 0.01
        args.lambda_cal = 0.05
        args.lambda_trust = 0.02
    elif preset == "u2_clip_x0":
        args.input_mode = "x0"
        args.use_uncertainty = True
        args.trust_mode = "clip"
        args.lambda_edge = 0.03
        args.lambda_normal = 0.01
        args.lambda_cal = 0.05
        args.lambda_trust = 0.02
    elif preset == "ucpf_full_x1":
        args.input_mode = "x1"
        args.use_uncertainty = True
        args.trust_mode = "clip"
        args.lambda_edge = 0.03
        args.lambda_normal = 0.01
        args.lambda_cal = 0.05
        args.lambda_trust = 0.02
    else:
        raise ValueError(f"unknown preset: {args.preset}")


def make_loaders(args: argparse.Namespace):
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    if args.num_workers > 0:
        common["persistent_workers"] = True
        common["prefetch_factor"] = 4
    train_ds = UCPFCandidateDataset(args.cache_dir, "train")
    val_ds = UCPFCandidateDataset(args.cache_dir, "val")
    test_ds = UCPFCandidateDataset(args.cache_dir, "test")
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    eval_common = dict(common)
    eval_common["batch_size"] = args.eval_batch_size
    return (
        train_ds,
        DataLoader(train_ds, shuffle=False, drop_last=False, **eval_common),
        train_loader,
        DataLoader(val_ds, shuffle=False, drop_last=False, **eval_common),
        DataLoader(test_ds, shuffle=False, drop_last=False, **eval_common),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UCPF over frozen FPP depth candidates.")
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/fpp_ml_ucpf_cache_960")
    parser.add_argument("--save_dir", default="results/fpp960_ucpf")
    parser.add_argument("--preset", choices=["custom", "u0_x0", "u1_x0", "u2_soft_x0", "u2_clip_x0", "ucpf_full_x1"], default="custom")
    parser.add_argument("--input_mode", choices=["x0", "x1"], default="x0")
    parser.add_argument("--candidates", default="b,p,d")
    parser.add_argument("--drop_edge_conf", action="store_true")
    parser.add_argument("--drop_physics_instr", action="store_true")
    parser.add_argument("--use_uncertainty", action="store_true")
    parser.add_argument("--disable_cal", action="store_true")
    parser.add_argument("--trust_mode", choices=["none", "soft", "clip"], default="none")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=180)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=1.5)
    parser.add_argument("--temperatures", default="0.5 1.0 2.0")
    parser.add_argument("--kappas", default="1.0 1.5 2.0 3.0")
    parser.add_argument("--sigma_min_mm", type=float, default=0.05)
    parser.add_argument("--sigma_max_mm", type=float, default=30.0)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--lambda_edge", type=float, default=0.0)
    parser.add_argument("--lambda_normal", type=float, default=0.0)
    parser.add_argument("--lambda_cal", type=float, default=0.0)
    parser.add_argument("--lambda_trust", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save_vis_count", type=int, default=8)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_preset(args)
    args.candidates = ",".join(parse_candidates(args.candidates))
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    train_ds, train_eval_loader, train_loader, val_loader, test_loader = make_loaders(args)
    in_channels = infer_input_channels(train_ds, args)
    num_candidates = len(parse_candidates(args.candidates))
    model = LightweightUCPF(
        in_channels=in_channels,
        num_candidates=num_candidates,
        hidden=args.hidden,
        blocks=args.blocks,
        use_uncertainty=args.use_uncertainty,
    ).to(device)

    print(
        json.dumps(
            {
                "device": str(device),
                "preset": args.preset,
                "input_channels": in_channels,
                "candidates": parse_candidates(args.candidates),
                "params": parameter_count(model),
                "save_dir": str(save_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")
    best_path = save_dir / "best_ucpf.pt"
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--eval_only requires --checkpoint")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        selection = {"selected": {"temperature": args.temperature, "kappa": args.kappa}, "grid": []}
    else:
        for epoch in range(1, int(args.epochs) + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, args, device)
            val_summary, _ = evaluate(
                model,
                val_loader,
                args,
                device,
                temperature=args.temperature,
                kappa=args.kappa,
                split="val",
            )
            val_rmse = float(val_summary["rmse"]["mean"])
            row = {
                "epoch": epoch,
                "val_rmse": val_rmse,
                "val_mae": val_summary["mae"]["mean"],
                "val_edge_rmse": val_summary["edge_rmse"]["mean"],
                "val_normal_deg": val_summary["normal_deg"]["mean"],
                **{f"train_{k}": v for k, v in train_loss.items()},
            }
            history.append(row)
            write_history(history, save_dir / "train_history.csv")
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if val_rmse < best_val:
                best_val = val_rmse
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "epoch": epoch,
                        "best_val_rmse": best_val,
                        "input_channels": in_channels,
                        "num_candidates": num_candidates,
                        "params": parameter_count(model),
                    },
                    best_path,
                )
            if epoch - best_epoch >= int(args.early_stop_patience):
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
                break
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        selection = select_by_validation(model, val_loader, args, device)

    selected = selection["selected"]
    selected_t = float(selected["temperature"])
    selected_k = float(selected["kappa"])
    train_eval_summary, train_rows = evaluate(
        model,
        train_eval_loader,
        args,
        device,
        selected_t,
        selected_k,
        split="train_eval",
    )
    val_summary, val_rows = evaluate(
        model,
        val_loader,
        args,
        device,
        selected_t,
        selected_k,
        split="val_selected",
    )
    test_summary, test_rows = evaluate(
        model,
        test_loader,
        args,
        device,
        selected_t,
        selected_k,
        split="test_selected",
        out_dir=save_dir,
        save_visuals=args.save_vis_count > 0,
    )

    write_rows(train_rows, save_dir / "train_eval_per_sample_metrics.csv")
    write_rows(val_rows, save_dir / "val_per_sample_metrics.csv")
    write_rows(test_rows, save_dir / "test_per_sample_metrics.csv")

    summary = {
        "method": "UCPF",
        "args": vars(args),
        "input_channels": in_channels,
        "num_candidates": num_candidates,
        "params": parameter_count(model),
        "checkpoint": str(best_path if not args.eval_only else args.checkpoint),
        "best_epoch": best_epoch,
        "val_selection": selection,
        "selected_temperature": selected_t,
        "selected_kappa": selected_k,
        "train_eval": train_eval_summary,
        "val": val_summary,
        "test": test_summary,
        "constraints": {
            "mask_input": False,
            "frozen_candidates": True,
            "test_selection": False,
            "sigma_min_mm": args.sigma_min_mm,
            "sigma_max_mm": args.sigma_max_mm,
        },
    }
    with (save_dir / "ucpf_summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, ensure_ascii=False)
    print(json.dumps(json_safe({"test_rmse": test_summary["rmse"]["mean"], "summary": str(save_dir / "ucpf_summary.json")}), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
