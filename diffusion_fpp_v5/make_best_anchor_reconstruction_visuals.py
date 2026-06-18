from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_refined_xphase_reliability_selector import (
    ReliabilityMLP,
    eval_entries,
    forward_pack,
    load_all_models,
    rmse_np,
)


def to_mm(norm: np.ndarray, scale: float, center: float) -> np.ndarray:
    return norm.astype(np.float32) * float(scale) + float(center)


def final_norm(
    anchor: np.ndarray,
    refined: np.ndarray,
    prob: np.ndarray,
    unc: np.ndarray,
    delta_x: np.ndarray,
    gate: Dict[str, object],
) -> tuple[np.ndarray, np.ndarray | None, float]:
    alpha = float(gate["alpha"])
    if gate["kind"] == "rule":
        use = np.ones_like(anchor, dtype=bool)
        if "unc_max" in gate:
            use &= unc <= float(gate["unc_max"])
        if "delta_max" in gate:
            use &= delta_x <= float(gate["delta_max"])
        final = np.clip(anchor + alpha * (refined - anchor) * use.astype(np.float32), -1.0, 1.0)
        return final, use.astype(np.float32), alpha
    if gate["kind"] == "mlp_hard":
        use = prob >= float(gate["threshold"])
        final = np.clip(anchor + alpha * (refined - anchor) * use.astype(np.float32), -1.0, 1.0)
        return final, use.astype(np.float32), alpha
    if gate["kind"] == "mlp_soft":
        final = np.clip(anchor + alpha * (refined - anchor) * prob.astype(np.float32), -1.0, 1.0)
        return final, prob.astype(np.float32), alpha
    raise ValueError(str(gate["kind"]))


def masked(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = a.astype(np.float32).copy()
    out[~mask.astype(bool)] = np.nan
    return out


def figure_for_sample(sample: Dict[str, object], path: Path) -> None:
    mask = sample["mask"].astype(bool)  # type: ignore[union-attr]
    target = sample["target"]  # type: ignore[assignment]
    depth_maps = [
        ("GT depth", target),
        ("Base", sample["base"]),
        ("X phase", sample["x"]),
        ("Anchor", sample["anchor"]),
        ("Diffusion candidate", sample["refined"]),
        ("MLP final", sample["mlp"]),
        ("Rule final", sample["rule"]),
        ("True-x oracle", sample["oracle"]),
    ]
    vals = []
    for _, arr in depth_maps:
        vals.append(np.asarray(arr)[mask])
    vals_cat = np.concatenate([v[np.isfinite(v)] for v in vals if v.size])
    vmin = float(np.percentile(vals_cat, 1)) if vals_cat.size else 0.0
    vmax = float(np.percentile(vals_cat, 99)) if vals_cat.size else 1.0

    anchor_err = np.abs(sample["anchor"] - target)  # type: ignore[operator]
    mlp_err = np.abs(sample["mlp"] - target)  # type: ignore[operator]
    rule_err = np.abs(sample["rule"] - target)  # type: ignore[operator]
    err_vals = anchor_err[mask & np.isfinite(anchor_err)]
    err_vmax = float(np.percentile(err_vals, 95)) if err_vals.size else 1.0
    err_vmax = max(err_vmax, 0.1)

    panels = [
        (sample["fringe"], "Input vertical 0120", "gray", None, None),
        (target, "GT depth", "viridis", vmin, vmax),
        (sample["base"], f"Base\nRMSE {sample['base_rmse']:.3f}", "viridis", vmin, vmax),
        (sample["x"], f"X phase\nRMSE {sample['x_rmse']:.3f}", "viridis", vmin, vmax),
        (sample["anchor"], f"Anchor\nRMSE {sample['anchor_rmse']:.3f}", "viridis", vmin, vmax),
        (sample["refined"], f"Diffusion candidate\nRMSE {sample['refined_rmse']:.3f}", "viridis", vmin, vmax),
        (sample["mlp"], f"MLP final\nRMSE {sample['mlp_rmse']:.3f}", "viridis", vmin, vmax),
        (sample["rule"], f"Rule final\nRMSE {sample['rule_rmse']:.3f}", "viridis", vmin, vmax),
        (sample["mlp_weight"], f"MLP weight\nmean {sample['mlp_accept']:.2f}", "magma", 0.0, 1.0),
        (sample["rule_weight"], f"Rule mask\nmean {sample['rule_accept']:.2f}", "magma", 0.0, 1.0),
        (anchor_err, "Anchor abs error", "magma", 0.0, err_vmax),
        (mlp_err, "MLP abs error", "magma", 0.0, err_vmax),
        (rule_err, "Rule abs error", "magma", 0.0, err_vmax),
        (np.abs(sample["refined"] - target), "Candidate abs error", "magma", 0.0, err_vmax),
        (sample["unc"], "Posterior uncertainty", "magma", None, None),
        (sample["prob"], "MLP probability", "magma", 0.0, 1.0),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(16, 12), constrained_layout=True)
    for ax, (arr, title, cmap, lo, hi) in zip(axes.flat, panels):
        arr_np = np.asarray(arr)
        show = arr_np if title.startswith("Input") else masked(arr_np, mask)
        im = ax.imshow(show, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"{sample['split']} | {sample['sample_id']} | "
        f"MLP gain {sample['mlp_gain']:.3f} mm | Rule gain {sample['rule_gain']:.3f} mm",
        fontsize=11,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def overview(samples: List[Dict[str, object]], path: Path) -> None:
    if not samples:
        return
    cols = [
        ("fringe", "Input", "gray"),
        ("target", "GT", "viridis"),
        ("anchor", "Anchor", "viridis"),
        ("mlp", "MLP final", "viridis"),
        ("rule", "Rule final", "viridis"),
        ("mlp_error", "MLP error", "magma"),
        ("rule_error", "Rule error", "magma"),
    ]
    n = len(samples)
    fig, axes = plt.subplots(n, len(cols), figsize=(2.7 * len(cols), 2.25 * n), constrained_layout=True)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
    for r, sample in enumerate(samples):
        mask = sample["mask"].astype(bool)  # type: ignore[union-attr]
        target = sample["target"]  # type: ignore[assignment]
        vals = np.concatenate([
            np.asarray(sample["target"])[mask],
            np.asarray(sample["anchor"])[mask],
            np.asarray(sample["mlp"])[mask],
            np.asarray(sample["rule"])[mask],
        ])
        vals = vals[np.isfinite(vals)]
        vmin = float(np.percentile(vals, 1)) if vals.size else 0.0
        vmax = float(np.percentile(vals, 99)) if vals.size else 1.0
        err = np.abs(np.asarray(sample["anchor"]) - target)
        err_vals = err[mask & np.isfinite(err)]
        err_vmax = max(float(np.percentile(err_vals, 95)) if err_vals.size else 1.0, 0.1)
        derived = {
            "mlp_error": np.abs(np.asarray(sample["mlp"]) - target),
            "rule_error": np.abs(np.asarray(sample["rule"]) - target),
        }
        for c, (key, title, cmap) in enumerate(cols):
            ax = axes[r, c]
            arr = derived[key] if key in derived else sample[key]
            if key == "fringe":
                show = arr
                lo = hi = None
            elif "error" in key:
                show = masked(np.asarray(arr), mask)
                lo, hi = 0.0, err_vmax
            else:
                show = masked(np.asarray(arr), mask)
                lo, hi = vmin, vmax
            ax.imshow(show, cmap=cmap, vmin=lo, vmax=hi)
            ax.axis("off")
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.text(
                    0.02,
                    0.98,
                    f"{sample['split']} {sample['sample_id']}\n"
                    f"A {sample['anchor_rmse']:.2f} M {sample['mlp_rmse']:.2f} R {sample['rule_rmse']:.2f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
                )
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector_ckpt", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["test", "ood"])
    parser.add_argument("--max_per_split", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--phase_sample_steps", type=int, default=12)
    parser.add_argument("--phase_ensemble_size", type=int, default=3)
    args_cli = parser.parse_args()

    ckpt = torch.load(args_cli.selector_ckpt, map_location="cpu")
    run_args = SimpleNamespace(**ckpt["args"])
    run_args.num_workers = int(args_cli.num_workers)
    run_args.eval_batch_size = 1
    run_args.batch_size = 1
    run_args.phase_sample_steps = int(args_cli.phase_sample_steps)
    run_args.phase_ensemble_size = int(args_cli.phase_ensemble_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    models = load_all_models(run_args, device)
    feature_names = ckpt["feature_names"]
    selector = ReliabilityMLP(len(feature_names)).to(device)
    selector.load_state_dict(ckpt["model_state_dict"])
    selector.eval()
    mean_t = torch.from_numpy(ckpt["mean"].astype(np.float32)).to(device)
    std_t = torch.from_numpy(ckpt["std"].astype(np.float32)).to(device)
    rule_gate = ckpt["summary"]["rule_gate"]
    mlp_gate = ckpt["summary"]["mlp_gate"]

    out_dir = Path(args_cli.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_samples: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []

    for split in args_cli.splits:
        loader = models["loaders_obj"]["loaders"][split]  # type: ignore[index]
        split_samples: List[Dict[str, object]] = []
        for batch in loader:
            pack = forward_pack(batch, models, run_args, device)
            feats = pack["features"]
            b, c, h, w = feats.shape
            flat = feats.permute(0, 2, 3, 1).reshape(-1, c)
            prob = torch.sigmoid(selector((flat - mean_t) / std_t)).reshape(b, h, w).detach().cpu().numpy().astype(np.float32)
            base_n = pack["base_norm"].detach().cpu().numpy()[:, 0].astype(np.float32)
            x_n = pack["x_norm"].detach().cpu().numpy()[:, 0].astype(np.float32)
            anchor_n = pack["anchor_norm"].detach().cpu().numpy()[:, 0].astype(np.float32)
            refined_n = pack["refined_norm"].detach().cpu().numpy()[:, 0].astype(np.float32)
            oracle_n = pack["oracle_norm"].detach().cpu().numpy()[:, 0].astype(np.float32)
            unc = pack["phi_unc"].mean(dim=1).detach().cpu().numpy().astype(np.float32)
            delta_x = np.abs(refined_n - x_n).astype(np.float32)
            target = batch["depth_raw"].detach().cpu().numpy()[:, 0].astype(np.float32)  # type: ignore[index]
            mask = batch["object_mask"].detach().cpu().numpy()[:, 0].astype(bool)  # type: ignore[index]
            fringe = batch["fringe"].detach().cpu().numpy()[:, 0].astype(np.float32)  # type: ignore[index]
            scale = batch["scale_mm"].detach().cpu().numpy().astype(np.float32)  # type: ignore[index]
            center = batch["center_mm"].detach().cpu().numpy().astype(np.float32)  # type: ignore[index]

            for j, sample_id in enumerate(list(batch["sample_id"])):  # type: ignore[arg-type]
                mlp_n, mlp_w, _ = final_norm(anchor_n[j], refined_n[j], prob[j], unc[j], delta_x[j], mlp_gate)
                rule_n, rule_w, _ = final_norm(anchor_n[j], refined_n[j], prob[j], unc[j], delta_x[j], rule_gate)
                sample = {
                    "split": split,
                    "sample_id": str(sample_id),
                    "object_id": int(batch["object_id"][j].item()),  # type: ignore[index]
                    "pose_id": int(batch["pose_id"][j].item()),  # type: ignore[index]
                    "fringe": fringe[j],
                    "target": target[j],
                    "mask": mask[j],
                    "prob": prob[j],
                    "unc": unc[j],
                    "base": to_mm(base_n[j], scale[j], center[j]),
                    "x": to_mm(x_n[j], scale[j], center[j]),
                    "anchor": to_mm(anchor_n[j], scale[j], center[j]),
                    "refined": to_mm(refined_n[j], scale[j], center[j]),
                    "oracle": to_mm(oracle_n[j], scale[j], center[j]),
                    "mlp": to_mm(mlp_n, scale[j], center[j]),
                    "rule": to_mm(rule_n, scale[j], center[j]),
                    "mlp_weight": mlp_w if mlp_w is not None else prob[j],
                    "rule_weight": rule_w if rule_w is not None else np.ones_like(prob[j]),
                }
                for key in ["base", "x", "anchor", "refined", "oracle", "mlp", "rule"]:
                    sample[f"{key}_rmse"] = rmse_np(sample[key], sample["target"], sample["mask"])  # type: ignore[arg-type]
                sample["mlp_gain"] = float(sample["anchor_rmse"] - sample["mlp_rmse"])  # type: ignore[operator]
                sample["rule_gain"] = float(sample["anchor_rmse"] - sample["rule_rmse"])  # type: ignore[operator]
                sample["mlp_accept"] = float(np.mean(sample["mlp_weight"][mask[j]]))  # type: ignore[index]
                sample["rule_accept"] = float(np.mean(sample["rule_weight"][mask[j]]))  # type: ignore[index]
                split_samples.append(sample)
                rows.append({
                    k: sample[k]
                    for k in [
                        "split",
                        "sample_id",
                        "object_id",
                        "pose_id",
                        "base_rmse",
                        "x_rmse",
                        "anchor_rmse",
                        "refined_rmse",
                        "mlp_rmse",
                        "rule_rmse",
                        "oracle_rmse",
                        "mlp_gain",
                        "rule_gain",
                        "mlp_accept",
                        "rule_accept",
                    ]
                })

        selected: List[Dict[str, object]] = []
        if split_samples:
            for key, reverse in [("mlp_gain", True), ("rule_gain", True), ("mlp_gain", False), ("rule_gain", False)]:
                ordered = sorted(split_samples, key=lambda x: float(x[key]), reverse=reverse)
                for s in ordered:
                    if all(s["sample_id"] != t["sample_id"] for t in selected):
                        selected.append(s)
                        break
            by_anchor = sorted(split_samples, key=lambda x: float(x["anchor_rmse"]), reverse=True)
            for s in by_anchor:
                if len(selected) >= int(args_cli.max_per_split):
                    break
                if all(s["sample_id"] != t["sample_id"] for t in selected):
                    selected.append(s)
            for idx, s in enumerate(selected[: int(args_cli.max_per_split)]):
                tag = f"{split}_{idx:02d}_{s['sample_id']}"
                figure_for_sample(s, out_dir / split / f"{tag}.png")
                all_samples.append(s)

    overview(all_samples, out_dir / "best_method_reconstruction_overview.png")
    with (out_dir / "best_method_visual_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "selector_ckpt": args_cli.selector_ckpt,
        "anchor_mode": getattr(run_args, "anchor_mode", None),
        "rule_gate": rule_gate,
        "mlp_gate": mlp_gate,
        "splits": args_cli.splits,
        "selected": [
            {
                "split": s["split"],
                "sample_id": s["sample_id"],
                "anchor_rmse": s["anchor_rmse"],
                "mlp_rmse": s["mlp_rmse"],
                "rule_rmse": s["rule_rmse"],
                "mlp_gain": s["mlp_gain"],
                "rule_gain": s["rule_gain"],
            }
            for s in all_samples
        ],
        "files": {
            "overview": str(out_dir / "best_method_reconstruction_overview.png"),
            "metrics": str(out_dir / "best_method_visual_metrics.csv"),
        },
    }
    (out_dir / "best_method_visual_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
