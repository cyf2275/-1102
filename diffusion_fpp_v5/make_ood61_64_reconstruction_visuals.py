"""Make reconstruction figures for the 61-64 OOD SingleFrame3D set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_single_frame3d_physics_diffusion import (
    ConditionalUNet,
    ResidualPosterior,
    SingleFrame3DDataset,
    build_model,
    collate_single_frame,
    forward_direct,
    load_base_model,
    pred_to_depth_mm,
)


NAMES = {
    "raw": "Single",
    "raw_single_phys": "Physics",
    "teacher_aux": "Aux",
}


def ns(data: Dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(**data)


def make_loader(args: argparse.Namespace, config: str) -> DataLoader:
    ds = SingleFrame3DDataset(
        data_root=args.data_root,
        split="test",
        config=config,
        cache_features=True,
        feature_cache_dir=args.feature_cache_dir or None,
    )

    def collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
        return collate_single_frame(batch, image_h=args.image_h, image_w=args.image_w)

    return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)


def load_direct_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, SimpleNamespace]:
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = dict(ckpt.get("args", {}))
    model_args = ns(saved_args)
    out_channels = 5 if saved_args.get("config") == "teacher_aux" else 1
    model = build_model(int(saved_args["cond_channels"]), out_channels, model_args).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model_args


def rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool) & np.isfinite(pred) & np.isfinite(target)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean((pred[m] - target[m]) ** 2)))


def collect_direct(args: argparse.Namespace, device: torch.device, samples: set[str]) -> tuple[dict, dict]:
    meta: Dict[str, Dict[str, np.ndarray | int | str]] = {}
    preds: Dict[str, Dict[str, List[np.ndarray]]] = {cfg: {} for cfg in NAMES}
    for cfg in NAMES:
        loader = make_loader(args, cfg)
        for seed in args.seeds:
            ckpt = args.result_root / "runs" / f"direct_{cfg}_seed{seed}" / "checkpoints" / "best.pt"
            model, _ = load_direct_model(ckpt, device)
            with torch.no_grad():
                for batch in loader:
                    pred_norm = forward_direct(model, batch, device)[:, :1]
                    pred_mm = pred_to_depth_mm(pred_norm, batch).detach().cpu().numpy()[:, 0]
                    ids = list(batch["sample_id"])  # type: ignore[arg-type]
                    for j, sid in enumerate(ids):
                        if sid not in samples:
                            continue
                        preds[cfg].setdefault(sid, []).append(pred_mm[j].copy())
                        if sid not in meta:
                            meta[sid] = {
                                "sample_id": sid,
                                "object_id": int(batch["object_id"][j].item()),  # type: ignore[index]
                                "pose_id": int(batch["pose_id"][j].item()),  # type: ignore[index]
                                "input": batch["fringe"][j, 0].cpu().numpy().copy(),  # type: ignore[index]
                                "target": batch["depth_raw"][j, 0].cpu().numpy().copy(),  # type: ignore[index]
                                "object_mask": batch["object_mask"][j, 0].cpu().numpy().copy(),  # type: ignore[index]
                                "valid_mask": batch["valid_mask"][j, 0].cpu().numpy().copy(),  # type: ignore[index]
                            }
    avg = {cfg: {sid: np.mean(stack, axis=0) for sid, stack in by_sample.items()} for cfg, by_sample in preds.items()}
    return meta, avg


def load_residual_model(ckpt_path: Path, device: torch.device) -> tuple[ResidualPosterior, torch.nn.Module, SimpleNamespace]:
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = dict(ckpt.get("args", {}))
    model_args = ns(saved_args)
    base_model, _ = load_base_model(saved_args["base_ckpt"], int(saved_args["cond_channels"]), device)
    model = ConditionalUNet(
        in_channels=1,
        cond_channels=int(saved_args.get("posterior_cond_channels", int(saved_args["cond_channels"]) + 1)),
        out_channels=1,
        base_ch=int(saved_args["base_channels"]),
        ch_mult=tuple(saved_args["ch_mult"]),
        num_res_blocks=int(saved_args["num_res_blocks"]),
        dropout=float(saved_args["dropout"]),
        time_emb_dim=int(saved_args["time_emb_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    posterior = ResidualPosterior(
        model,
        timesteps=int(saved_args["timesteps"]),
        residual_scale=float(saved_args["residual_scale"]),
        device=device,
    )
    return posterior, base_model, model_args


def collect_residual(args: argparse.Namespace, device: torch.device, samples: set[str]) -> dict:
    cfg = "raw_single_phys"
    loader = make_loader(args, cfg)
    out: Dict[str, Dict[str, List[np.ndarray]]] = {"base": {}, "mean": {}, "gate": {}}
    for seed in args.seeds:
        ckpt = args.result_root / "runs" / f"residual_{cfg}_seed{seed}" / "checkpoints" / "best.pt"
        posterior, base_model, model_args = load_residual_model(ckpt, device)
        summary_path = args.ood_eval_root / "runs" / f"residual_{cfg}_seed{seed}" / "summary.json"
        gate = json.loads(summary_path.read_text(encoding="utf-8")).get("gate", {})
        tau = float(gate.get("threshold", -1.0))
        alpha = float(gate.get("alpha", 0.0))
        max_gate = float(getattr(model_args, "max_gate_correction", 0.25))
        with torch.no_grad():
            for batch in loader:
                base_norm, mean_norm, unc_norm = posterior.sample(
                    batch,
                    base_model,
                    steps=args.sample_steps,
                    ensemble_size=args.ensemble_size,
                )
                correction = torch.abs(mean_norm - base_norm)
                if tau < 0:
                    use = torch.zeros_like(base_norm, dtype=torch.bool)
                else:
                    use = (unc_norm <= tau) & (correction <= max_gate)
                gate_norm = torch.where(use, torch.clamp(base_norm + alpha * (mean_norm - base_norm), -1.0, 1.0), base_norm)
                base_mm = pred_to_depth_mm(base_norm, batch).detach().cpu().numpy()[:, 0]
                mean_mm = pred_to_depth_mm(mean_norm, batch).detach().cpu().numpy()[:, 0]
                gate_mm = pred_to_depth_mm(gate_norm, batch).detach().cpu().numpy()[:, 0]
                ids = list(batch["sample_id"])  # type: ignore[arg-type]
                for j, sid in enumerate(ids):
                    if sid not in samples:
                        continue
                    out["base"].setdefault(sid, []).append(base_mm[j].copy())
                    out["mean"].setdefault(sid, []).append(mean_mm[j].copy())
                    out["gate"].setdefault(sid, []).append(gate_mm[j].copy())
    return {mode: {sid: np.mean(stack, axis=0) for sid, stack in by_sample.items()} for mode, by_sample in out.items()}


def masked(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask.astype(bool), arr, np.nan)


def limits(arrays: Iterable[np.ndarray], masks: Iterable[np.ndarray], lo: float, hi: float) -> tuple[float, float]:
    vals = []
    for arr, mask in zip(arrays, masks):
        m = mask.astype(bool) & np.isfinite(arr)
        if np.any(m):
            vals.append(arr[m])
    if not vals:
        return 0.0, 1.0
    all_vals = np.concatenate(vals)
    a, b = np.percentile(all_vals, [lo, hi])
    if not np.isfinite(a) or not np.isfinite(b) or abs(float(b - a)) < 1e-6:
        a, b = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    if abs(float(b - a)) < 1e-6:
        b = a + 1.0
    return float(a), float(b)


def save_direct_figure(args: argparse.Namespace, meta: dict, preds: dict, sample_ids: list[str]) -> None:
    cols = ["Input", "GT", "Single", "Physics", "Aux", "|Physics-GT|", "|Aux-GT|"]
    fig, axes = plt.subplots(len(sample_ids), len(cols), figsize=(17, 3.1 * len(sample_ids)), constrained_layout=True)
    if len(sample_ids) == 1:
        axes = axes[None, :]
    for r, sid in enumerate(sample_ids):
        item = meta[sid]
        target = item["target"]
        mask = item["object_mask"]
        single = preds["raw"][sid]
        phys = preds["raw_single_phys"][sid]
        aux = preds["teacher_aux"][sid]
        vmin, vmax = limits([target, single, phys, aux], [mask] * 4, 1, 99)
        emax = max(0.5, limits([np.abs(phys - target), np.abs(aux - target)], [mask, mask], 50, 98)[1])
        maps = [
            (item["input"], "gray", None, None),
            (masked(target, mask), "viridis", vmin, vmax),
            (masked(single, mask), "viridis", vmin, vmax),
            (masked(phys, mask), "viridis", vmin, vmax),
            (masked(aux, mask), "viridis", vmin, vmax),
            (masked(np.abs(phys - target), mask), "magma", 0, emax),
            (masked(np.abs(aux - target), mask), "magma", 0, emax),
        ]
        for c, (arr, cmap, lo, hi) in enumerate(maps):
            ax = axes[r, c]
            ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
            ax.axis("off")
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
        title = (
            f"obj{item['object_id']:03d}/pose{item['pose_id']:02d}  "
            f"RMSE single {rmse(single, target, mask):.2f}, physics {rmse(phys, target, mask):.2f}, aux {rmse(aux, target, mask):.2f}"
        )
        axes[r, 0].set_ylabel(title, fontsize=10, rotation=0, labelpad=78, va="center")
    fig.suptitle("OOD 61-64 direct reconstruction, 3-seed averaged", fontsize=14)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / "ood61_64_reconstruction_direct_contact.png", dpi=170)
    plt.close(fig)


def save_diffusion_figure(args: argparse.Namespace, meta: dict, direct_preds: dict, residual_preds: dict, sample_ids: list[str]) -> None:
    cols = ["Input", "GT", "Physics base", "Diff mean", "Diff gate", "Base err", "Gate err", "Gate-Base err"]
    fig, axes = plt.subplots(len(sample_ids), len(cols), figsize=(19, 3.1 * len(sample_ids)), constrained_layout=True)
    if len(sample_ids) == 1:
        axes = axes[None, :]
    for r, sid in enumerate(sample_ids):
        item = meta[sid]
        target = item["target"]
        mask = item["object_mask"]
        base = residual_preds["base"][sid]
        mean = residual_preds["mean"][sid]
        gate = residual_preds["gate"][sid]
        vmin, vmax = limits([target, base, mean, gate], [mask] * 4, 1, 99)
        base_err = np.abs(base - target)
        gate_err = np.abs(gate - target)
        delta = gate_err - base_err
        emax = max(0.5, limits([base_err, gate_err], [mask, mask], 50, 98)[1])
        dmax = max(0.1, float(np.nanpercentile(np.abs(delta[mask.astype(bool)]), 98)) if np.any(mask) else 0.1)
        maps = [
            (item["input"], "gray", None, None),
            (masked(target, mask), "viridis", vmin, vmax),
            (masked(base, mask), "viridis", vmin, vmax),
            (masked(mean, mask), "viridis", vmin, vmax),
            (masked(gate, mask), "viridis", vmin, vmax),
            (masked(base_err, mask), "magma", 0, emax),
            (masked(gate_err, mask), "magma", 0, emax),
            (masked(delta, mask), "coolwarm", -dmax, dmax),
        ]
        for c, (arr, cmap, lo, hi) in enumerate(maps):
            ax = axes[r, c]
            ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
            ax.axis("off")
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
        title = (
            f"obj{item['object_id']:03d}/pose{item['pose_id']:02d}  "
            f"base {rmse(base, target, mask):.2f}, mean {rmse(mean, target, mask):.2f}, gate {rmse(gate, target, mask):.2f}"
        )
        axes[r, 0].set_ylabel(title, fontsize=10, rotation=0, labelpad=84, va="center")
    fig.suptitle("OOD 61-64 residual diffusion on physics branch, 3-seed averaged", fontsize=14)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / "ood61_64_reconstruction_diffusion_physics_contact.png", dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--result_root", type=Path, required=True)
    parser.add_argument("--ood_eval_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", default=[
        "new0612_obj061_pose01",
        "new0612_obj062_pose05",
        "new0612_obj063_pose01",
        "new0612_obj064_pose02",
    ])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_h", type=int, default=480)
    parser.add_argument("--image_w", type=int, default=640)
    parser.add_argument("--sample_steps", type=int, default=12)
    parser.add_argument("--ensemble_size", type=int, default=3)
    parser.add_argument("--feature_cache_dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_set = set(args.samples)
    meta, direct_preds = collect_direct(args, device, sample_set)
    missing = [sid for sid in args.samples if sid not in meta]
    if missing:
        raise RuntimeError(f"missing selected samples: {missing}")
    residual_preds = collect_residual(args, device, sample_set)
    save_direct_figure(args, meta, direct_preds, args.samples)
    save_diffusion_figure(args, meta, direct_preds, residual_preds, args.samples)
    manifest = {
        "samples": args.samples,
        "figures": [
            str(args.out_dir / "ood61_64_reconstruction_direct_contact.png"),
            str(args.out_dir / "ood61_64_reconstruction_diffusion_physics_contact.png"),
        ],
    }
    (args.out_dir / "ood61_64_reconstruction_visual_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
