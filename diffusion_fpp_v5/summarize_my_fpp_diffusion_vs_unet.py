from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


DIRECT_CONFIGS = ["raw", "raw_xy", "raw_single_phys", "teacher_aux"]


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def metric(summary: Dict[str, object], mode: str, roi: str = "object") -> float:
    if mode == "posterior_gate":
        block = summary
    else:
        comp = summary.get("comparison", {})
        block = comp.get(mode, {}) if isinstance(comp, dict) else {}
    if not isinstance(block, dict):
        return float("nan")
    roi_block = block.get(roi, {})
    if not isinstance(roi_block, dict):
        return float("nan")
    rmse = roi_block.get("rmse", {})
    if not isinstance(rmse, dict):
        return float("nan")
    return as_float(rmse.get("mean"))


def per_object(summary: Dict[str, object], mode: str, obj: str) -> float:
    if mode == "posterior_gate":
        block = summary
    else:
        comp = summary.get("comparison", {})
        block = comp.get(mode, {}) if isinstance(comp, dict) else {}
    if not isinstance(block, dict):
        return float("nan")
    po = block.get("per_object", {})
    if not isinstance(po, dict):
        return float("nan")
    item = po.get(obj, {})
    if not isinstance(item, dict):
        return float("nan")
    return as_float(item.get("object_rmse_mean"))


def discover_residual_runs(root: Path) -> List[Dict[str, object]]:
    records = []
    for path in sorted(root.glob("**/evaluation/summary.json")):
        summary = read_json(path)
        args = summary.get("args", {})
        if not isinstance(args, dict):
            args = {}
        if str(summary.get("experiment_role")) != "constrained residual diffusion posterior":
            continue
        gate = summary.get("gate", {})
        if not isinstance(gate, dict):
            gate = {}
        record = {
            "run_dir": str(path.parent.parent),
            "summary_path": str(path),
            "config": str(args.get("config", summary.get("input_mode", ""))),
            "base_config": str(args.get("base_config", "")),
            "seed": int(args.get("seed", summary.get("seed", -1))),
            "legal_single_frame": bool(summary.get("legal_single_frame", True)),
            "base_object_rmse": metric(summary, "base_unet", "object"),
            "posterior_mean_object_rmse": metric(summary, "posterior_mean", "object"),
            "posterior_gate_object_rmse": metric(summary, "posterior_gate", "object"),
            "base_valid_rmse": metric(summary, "base_unet", "valid"),
            "posterior_mean_valid_rmse": metric(summary, "posterior_mean", "valid"),
            "posterior_gate_valid_rmse": metric(summary, "posterior_gate", "valid"),
            "obj0011_gate_rmse": per_object(summary, "posterior_gate", "obj0011"),
            "obj0012_gate_rmse": per_object(summary, "posterior_gate", "obj0012"),
            "obj0011_base_rmse": per_object(summary, "base_unet", "obj0011"),
            "obj0012_base_rmse": per_object(summary, "base_unet", "obj0012"),
            "gate_threshold": as_float(gate.get("threshold")),
            "gate_accept_val": as_float(gate.get("accepted_fraction")),
            "gate_accept_test": as_float(gate.get("test_accepted_fraction")),
        }
        records.append(record)
    return records


def discover_direct_runs(root: Path) -> List[Dict[str, object]]:
    records = []
    for path in sorted(root.glob("**/evaluation/summary.json")):
        summary = read_json(path)
        args = summary.get("args", {})
        if not isinstance(args, dict):
            args = {}
        if int(args.get("train_subset", 0) or 0) > 0:
            continue
        role = str(summary.get("experiment_role", "input ablation"))
        if role != "input ablation":
            continue
        config = str(summary.get("config", args.get("config", "")))
        if config not in DIRECT_CONFIGS:
            continue
        mw = as_float(args.get("object_mask_weight"))
        if abs(mw - 3.0) > 1e-6:
            continue
        records.append({
            "config": config,
            "seed": int(args.get("seed", summary.get("seed", -1))),
            "object_rmse": metric(summary, "posterior_gate", "object"),
            "valid_rmse": metric(summary, "posterior_gate", "valid"),
            "obj0011_rmse": per_object(summary, "posterior_gate", "obj0011"),
            "obj0012_rmse": per_object(summary, "posterior_gate", "obj0012"),
        })
    return records


def median_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.median(arr)), float(np.std(arr, ddof=1) if arr.size > 1 else 0.0)


def aggregate_residual(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for r in records:
        grouped[(str(r["config"]), str(r["base_config"]))].append(r)
    rows = []
    for (config, base_config), group in sorted(grouped.items()):
        base_med, base_std = median_std([as_float(r["base_object_rmse"]) for r in group])
        mean_med, mean_std_v = median_std([as_float(r["posterior_mean_object_rmse"]) for r in group])
        gate_med, gate_std = median_std([as_float(r["posterior_gate_object_rmse"]) for r in group])
        valid_med, _ = median_std([as_float(r["posterior_gate_valid_rmse"]) for r in group])
        obj11_med, _ = median_std([as_float(r["obj0011_gate_rmse"]) for r in group])
        obj12_med, _ = median_std([as_float(r["obj0012_gate_rmse"]) for r in group])
        gain_vs_base = 100.0 * (base_med - gate_med) / base_med if np.isfinite(base_med) and base_med > 0 else float("nan")
        rows.append({
            "config": config,
            "base_config": base_config,
            "n_seeds": len(group),
            "seeds": ",".join(str(r["seed"]) for r in sorted(group, key=lambda x: int(x["seed"]))),
            "base_object_rmse_median": base_med,
            "base_object_rmse_std": base_std,
            "posterior_mean_object_rmse_median": mean_med,
            "posterior_mean_object_rmse_std": mean_std_v,
            "posterior_gate_object_rmse_median": gate_med,
            "posterior_gate_object_rmse_std": gate_std,
            "posterior_gate_valid_rmse_median": valid_med,
            "obj0011_gate_rmse_median": obj11_med,
            "obj0012_gate_rmse_median": obj12_med,
            "gain_vs_base_percent": gain_vs_base,
            "legal_single_frame": all(bool(r["legal_single_frame"]) for r in group),
        })
    return rows


def aggregate_direct(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in records:
        grouped[str(r["config"])].append(r)
    rows = []
    for config, group in sorted(grouped.items()):
        obj_med, obj_std = median_std([as_float(r["object_rmse"]) for r in group])
        valid_med, _ = median_std([as_float(r["valid_rmse"]) for r in group])
        obj11_med, _ = median_std([as_float(r["obj0011_rmse"]) for r in group])
        obj12_med, _ = median_std([as_float(r["obj0012_rmse"]) for r in group])
        rows.append({
            "config": config,
            "n_seeds": len(group),
            "seeds": ",".join(str(r["seed"]) for r in sorted(group, key=lambda x: int(x["seed"]))),
            "object_rmse_median": obj_med,
            "object_rmse_std": obj_std,
            "valid_rmse_median": valid_med,
            "obj0011_rmse_median": obj11_med,
            "obj0012_rmse_median": obj12_med,
        })
    return rows


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(x: object, digits: int = 4) -> str:
    value = as_float(x)
    if not np.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def decision(residual_rows: List[Dict[str, object]], direct_rows: List[Dict[str, object]]) -> str:
    if not residual_rows:
        return "No constrained residual diffusion posterior runs found yet."
    best_direct = min((as_float(r["object_rmse_median"]) for r in direct_rows), default=float("nan"))
    best_res = min((as_float(r["posterior_gate_object_rmse_median"]) for r in residual_rows), default=float("nan"))
    best_gain = max((as_float(r["gain_vs_base_percent"]) for r in residual_rows), default=float("nan"))
    if np.isfinite(best_direct) and np.isfinite(best_res) and best_res < best_direct and np.isfinite(best_gain) and best_gain > 1.0:
        return "On this self-built validation set, the constrained residual diffusion posterior improves over its frozen UNet base and beats the best direct UNet-style baseline."
    if np.isfinite(best_gain) and best_gain > 0.0:
        return "The residual posterior shows only marginal or inconsistent gains over its frozen UNet base; it is not strong enough to claim diffusion superiority on this self-built set."
    return "The residual posterior does not improve over its frozen UNet bases on this self-built set; keep the diffusion superiority claim grounded in the larger FPP-ML-Bench D47/RCPC evidence chain."


def report_markdown(direct_rows: List[Dict[str, object]], residual_rows: List[Dict[str, object]]) -> str:
    lines = [
        "# My FPP Diffusion-vs-UNet Check",
        "",
        "## Claim Boundary",
        "",
        "- The paper-level claim should be: structured physical information plus constrained diffusion posterior plus RCPC is better than a reproduced raw-fringe UNet.",
        "- This is not a claim that naive physics stacking or unconstrained full-depth diffusion is always better.",
        "- The self-built real-capture set is small; it can support a validation/pilot statement, not replace the FPP-ML-Bench main line.",
        "",
        "## FPP-ML-Bench Evidence Chain",
        "",
        "| Method | Role | RMSE mm |",
        "|---|---|---:|",
        "| Raw-fringe UNet | reproduced direct baseline | 19.6610 |",
        "| Physics-instruction adapter | structured physical input | 18.8993 |",
        "| PSP-like phase branch | structured phase branch | 18.5235 |",
        "| D47 constrained diffusion posterior | physics + constrained posterior | 18.0680 |",
        "| RCPC/E84 final | posterior + validation-frozen risk control | 17.9021 +/- 0.0433; best 17.8475 |",
        "",
        "## Self-Built Direct UNet Baselines",
        "",
        "| Config | Seeds | Object RMSE median | Object RMSE std | Valid RMSE median | obj11 | obj12 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in direct_rows:
        lines.append(
            f"| {row['config']} | {row['n_seeds']} | {fmt(row['object_rmse_median'])} | {fmt(row['object_rmse_std'])} | "
            f"{fmt(row['valid_rmse_median'])} | {fmt(row['obj0011_rmse_median'])} | {fmt(row['obj0012_rmse_median'])} |"
        )
    lines.extend([
        "",
        "## Self-Built Constrained Residual Diffusion Posterior",
        "",
        "| Config | Base | Legal | Seeds | Base RMSE | Posterior mean RMSE | Gate RMSE | Gain vs base | Valid RMSE | obj11 | obj12 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in residual_rows:
        lines.append(
            f"| {row['config']} | {row['base_config']} | {'yes' if row['legal_single_frame'] else 'no'} | {row['n_seeds']} | "
            f"{fmt(row['base_object_rmse_median'])} | {fmt(row['posterior_mean_object_rmse_median'])} | "
            f"{fmt(row['posterior_gate_object_rmse_median'])} | {fmt(row['gain_vs_base_percent'], 2)}% | "
            f"{fmt(row['posterior_gate_valid_rmse_median'])} | {fmt(row['obj0011_gate_rmse_median'])} | {fmt(row['obj0012_gate_rmse_median'])} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        decision(residual_rows, direct_rows),
        "",
        "## Wording",
        "",
        "Use this wording if the self-built residual posterior is positive: the real-capture pilot is consistent with the main FPP-ML-Bench claim.",
        "Use this wording if it is negative: the real-capture set confirms physical input/teacher supervision, while diffusion superiority remains supported by the larger FPP-ML-Bench constrained posterior and RCPC chain.",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct_results_root", default="cloud_results/A_20260611_my_fpp_physics_validation_gpuopt_full")
    parser.add_argument("--residual_results_root", default="cloud_results/B_20260611_my_fpp_diffusion_vs_unet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    direct_root = Path(args.direct_results_root)
    residual_root = Path(args.residual_results_root)
    residual_root.mkdir(parents=True, exist_ok=True)
    direct_records = discover_direct_runs(direct_root)
    residual_records = discover_residual_runs(residual_root)
    direct_rows = aggregate_direct(direct_records)
    residual_rows = aggregate_residual(residual_records)
    write_csv(direct_records, residual_root / "direct_run_results.csv")
    write_csv(direct_rows, residual_root / "direct_aggregated_results.csv")
    write_csv(residual_records, residual_root / "residual_run_results.csv")
    write_csv(residual_rows, residual_root / "residual_aggregated_results.csv")
    summary = {
        "direct_results_root": str(direct_root),
        "residual_results_root": str(residual_root),
        "direct_aggregated": direct_rows,
        "residual_aggregated": residual_rows,
        "decision": decision(residual_rows, direct_rows),
    }
    with (residual_root / "diffusion_vs_unet_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    (residual_root / "diffusion_vs_unet_report.md").write_text(report_markdown(direct_rows, residual_rows), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
