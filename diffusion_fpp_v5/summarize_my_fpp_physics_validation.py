from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Tuple

import numpy as np


PRIMARY_CONFIGS = ["raw", "raw_xy", "raw_single_phys", "teacher_aux", "teacher_oracle"]


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def run_record(summary_path: Path) -> Dict[str, object]:
    summary = read_json(summary_path)
    args = summary.get("args", {})
    if not isinstance(args, dict):
        args = {}
    eval_dir = summary_path.parent
    run_dir = eval_dir.parent
    role = str(summary.get("experiment_role", "input ablation"))
    if int(args.get("train_subset", 0) or 0) > 0:
        role = "overfit smoke"
    record = {
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "per_sample_path": str(eval_dir / "per_sample_metrics.csv"),
        "config": str(summary.get("config", args.get("config", run_dir.name))),
        "seed": int(args.get("seed", summary.get("seed", -1))),
        "legal_single_frame": bool(summary.get("legal_single_frame", args.get("legal_single_frame", False))),
        "object_rmse": as_float(summary.get("object", {}).get("rmse", {}).get("mean") if isinstance(summary.get("object"), dict) else np.nan),
        "valid_rmse": as_float(summary.get("valid", {}).get("rmse", {}).get("mean") if isinstance(summary.get("valid"), dict) else np.nan),
        "object_mae": as_float(summary.get("object", {}).get("mae", {}).get("mean") if isinstance(summary.get("object"), dict) else np.nan),
        "valid_mae": as_float(summary.get("valid", {}).get("mae", {}).get("mean") if isinstance(summary.get("valid"), dict) else np.nan),
        "mask_weight": as_float(args.get("object_mask_weight", np.nan)),
        "experiment_role": role,
        "summary": summary,
    }
    per_object = summary.get("per_object", {})
    if isinstance(per_object, dict):
        for obj in ("obj0011", "obj0012"):
            obj_data = per_object.get(obj, {})
            if isinstance(obj_data, dict):
                record[f"{obj}_object_rmse"] = as_float(obj_data.get("object_rmse_mean"))
                record[f"{obj}_valid_rmse"] = as_float(obj_data.get("valid_rmse_mean"))
    return record


def discover_runs(results_root: Path) -> List[Dict[str, object]]:
    records = []
    for path in sorted(results_root.glob("**/evaluation/summary.json")):
        try:
            records.append(run_record(path))
        except Exception as exc:
            print(f"skip {path}: {exc}")
    return records


def group_key(record: Dict[str, object]) -> Tuple[str, float, str]:
    return (str(record["config"]), float(record.get("mask_weight", np.nan)), str(record.get("experiment_role", "input ablation")))


def aggregate(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, float, str], List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[group_key(record)].append(record)
    rows = []
    for (config, mask_weight, role), group in sorted(grouped.items()):
        obj = np.array([float(r["object_rmse"]) for r in group], dtype=np.float64)
        valid = np.array([float(r["valid_rmse"]) for r in group], dtype=np.float64)
        row = {
            "config": config,
            "mask_weight": mask_weight,
            "experiment_role": role,
            "legal_single_frame": all(bool(r["legal_single_frame"]) for r in group),
            "n_seeds": len(group),
            "seeds": ",".join(str(r["seed"]) for r in sorted(group, key=lambda x: int(x["seed"]))),
            "object_rmse_median": float(np.nanmedian(obj)),
            "object_rmse_mean": float(np.nanmean(obj)),
            "object_rmse_std": float(np.nanstd(obj, ddof=1)) if len(obj) > 1 else 0.0,
            "valid_rmse_median": float(np.nanmedian(valid)),
            "valid_rmse_mean": float(np.nanmean(valid)),
            "valid_rmse_std": float(np.nanstd(valid, ddof=1)) if len(valid) > 1 else 0.0,
        }
        for obj_name in ("obj0011", "obj0012"):
            vals = np.array([as_float(r.get(f"{obj_name}_object_rmse")) for r in group], dtype=np.float64)
            row[f"{obj_name}_object_rmse_median"] = float(np.nanmedian(vals)) if vals.size else float("nan")
        rows.append(row)
    return rows


def select_baseline(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    raw = [r for r in records if r["config"] == "raw" and str(r.get("experiment_role")) == "input ablation"]
    if not raw:
        return []
    raw_mw3 = [r for r in raw if abs(float(r.get("mask_weight", np.nan)) - 3.0) < 1e-6]
    return raw_mw3 or raw


def paired_wins(candidate: List[Dict[str, object]], baseline: List[Dict[str, object]]) -> Dict[str, object]:
    if not candidate or not baseline:
        return {"paired_samples": 0, "wins": 0, "losses": 0, "ties": 0}
    base_by_seed = {int(r["seed"]): r for r in baseline}
    cand_by_seed = {int(r["seed"]): r for r in candidate}
    common_seeds = sorted(set(base_by_seed) & set(cand_by_seed))
    wins = losses = ties = paired = 0
    for seed in common_seeds:
        base_path = Path(str(base_by_seed[seed]["per_sample_path"]))
        cand_path = Path(str(cand_by_seed[seed]["per_sample_path"]))
        if not base_path.exists() or not cand_path.exists():
            continue
        base_rows = {str(r["sample_id"]): as_float(r["object_rmse"]) for r in read_rows(base_path)}
        cand_rows = {str(r["sample_id"]): as_float(r["object_rmse"]) for r in read_rows(cand_path)}
        for sample_id in sorted(set(base_rows) & set(cand_rows)):
            paired += 1
            if cand_rows[sample_id] < base_rows[sample_id] - 1e-9:
                wins += 1
            elif cand_rows[sample_id] > base_rows[sample_id] + 1e-9:
                losses += 1
            else:
                ties += 1
    return {"paired_samples": paired, "wins": wins, "losses": losses, "ties": ties}


def improvement(candidate_median: float, baseline_median: float) -> float:
    if not np.isfinite(candidate_median) or not np.isfinite(baseline_median) or baseline_median <= 0:
        return float("nan")
    return 100.0 * (baseline_median - candidate_median) / baseline_median


def decision(agg_rows: List[Dict[str, object]]) -> str:
    primary = [r for r in agg_rows if str(r["experiment_role"]) == "input ablation"]
    raw_rows = [r for r in primary if r["config"] == "raw" and abs(float(r["mask_weight"]) - 3.0) < 1e-6]
    if not raw_rows:
        raw_rows = [r for r in primary if r["config"] == "raw"]
    if not raw_rows:
        return "No raw baseline found yet; decision deferred."
    raw = raw_rows[0]
    raw_med = float(raw["object_rmse_median"])
    comparable = [
        r for r in primary
        if r["config"] != "raw" and abs(float(r["mask_weight"]) - float(raw["mask_weight"])) < 1e-6
    ]
    if not comparable:
        return "Only raw baseline is available so far; physics/teacher comparison is not decided yet."

    def find(config: str) -> Dict[str, object] | None:
        rows = [r for r in primary if r["config"] == config and abs(float(r["mask_weight"]) - float(raw["mask_weight"])) < 1e-6]
        return rows[0] if rows else None

    phys = find("raw_single_phys")
    teacher_aux = find("teacher_aux")
    oracle = find("teacher_oracle")
    phys_gain = improvement(float(phys["object_rmse_median"]), raw_med) if phys else float("nan")
    aux_gain = improvement(float(teacher_aux["object_rmse_median"]), raw_med) if teacher_aux else float("nan")
    oracle_gain = improvement(float(oracle["object_rmse_median"]), raw_med) if oracle else float("nan")

    phys_both_objects = False
    if phys:
        phys_both_objects = (
            float(phys.get("obj0011_object_rmse_median", np.inf)) < float(raw.get("obj0011_object_rmse_median", -np.inf))
            and float(phys.get("obj0012_object_rmse_median", np.inf)) < float(raw.get("obj0012_object_rmse_median", -np.inf))
        )
    if phys and phys_gain >= 5.0 and phys_both_objects:
        return (
            "single_phys improves >=5% over raw and improves both obj0011/obj0012: "
            "single-frame derived physical channels appear effective on the real-capture validation set."
        )
    if teacher_aux and aux_gain >= 5.0:
        return (
            "teacher_aux improves >=5% but single_phys is not stably positive: teacher phase is useful as "
            "train-time supervision, but this does not prove test-time single-frame physics input is sufficient."
        )
    if oracle and oracle_gain >= 5.0:
        return (
            "teacher_oracle is strong while legal inputs are not: multi-frame phase information is valuable, "
            "but the current single-frame model does not capture it reliably."
        )
    return "No legal or teacher-assisted config clearly beats raw: keep D47+RCPC/E84 as the main line and treat this as real-capture negative validation."


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


def report_markdown(records: List[Dict[str, object]], agg_rows: List[Dict[str, object]], wins: Dict[str, Dict[str, object]]) -> str:
    lines = [
        "# My FPP Real-Capture Physics Validation",
        "",
        "## Scope",
        "",
        "- Main paper line remains `D47 diffusion posterior + validation-frozen RCPC/E84`.",
        "- This report is a small real-capture validation, not a replacement for the FPP-ML-Bench main result.",
        "- Target is `wall_normal_height`; metrics are height RMSE on this dataset only and must not be directly compared with FPP-ML-Bench absolute depth RMSE.",
        "- Legal single-frame inputs are `raw`, `raw_xy`, `raw_single_phys`, and `teacher_aux` where teacher phase is used only as train-time supervision.",
        "- `teacher_oracle` is an illegal single-frame upper-bound diagnostic.",
        "",
        "## Leakage Boundaries",
        "",
        "- `phase_y_capture` and `phase_x_capture` are not legal test-time inputs.",
        "- `bc_y` and `bc_x` are not legal test-time inputs; they may be used only for QC, teacher loss weighting, or oracle diagnostics.",
        "- `object_mask_clean_v1` is not a model input; it is used only for loss weighting and evaluation ROI.",
        "",
        "## Aggregated Results",
        "",
        "| Config | Role | Legal | Mask W | Seeds | Object RMSE median | Object RMSE std | Valid RMSE median | obj11 RMSE | obj12 RMSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg_rows:
        lines.append(
            "| {config} | {role} | {legal} | {mask} | {seeds} | {obj_med} | {obj_std} | {valid_med} | {obj11} | {obj12} |".format(
                config=row["config"],
                role=row["experiment_role"],
                legal="yes" if row["legal_single_frame"] else "no",
                mask=fmt(row["mask_weight"], 1),
                seeds=row["n_seeds"],
                obj_med=fmt(row["object_rmse_median"]),
                obj_std=fmt(row["object_rmse_std"]),
                valid_med=fmt(row["valid_rmse_median"]),
                obj11=fmt(row.get("obj0011_object_rmse_median")),
                obj12=fmt(row.get("obj0012_object_rmse_median")),
            )
        )
    lines.extend([
        "",
        "## Paired Win/Loss Against Raw",
        "",
        "| Config | Paired samples | Wins | Losses | Ties |",
        "|---|---:|---:|---:|---:|",
    ])
    for config, item in wins.items():
        lines.append(f"| {config} | {item['paired_samples']} | {item['wins']} | {item['losses']} | {item['ties']} |")
    lines.extend([
        "",
        "## Decision",
        "",
        decision(agg_rows),
        "",
        "## Run Count",
        "",
        f"- Discovered completed runs: {len(records)}",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="cloud_results/A_20260611_my_fpp_physics_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)
    records = discover_runs(root)
    agg_rows = aggregate(records)
    baseline = select_baseline(records)
    baseline_mask = float(baseline[0].get("mask_weight", np.nan)) if baseline else np.nan
    wins = {}
    for config in PRIMARY_CONFIGS:
        candidate = [
            r for r in records
            if r["config"] == config
            and str(r.get("experiment_role")) == "input ablation"
            and (not np.isfinite(baseline_mask) or abs(float(r.get("mask_weight", np.nan)) - baseline_mask) < 1e-6)
        ]
        if config == "raw" or not candidate:
            continue
        wins[config] = paired_wins(candidate, baseline)
    write_csv(records, root / "all_run_results.csv")
    write_csv(agg_rows, root / "aggregated_results.csv")
    summary = {
        "results_root": str(root),
        "n_runs": len(records),
        "aggregated": agg_rows,
        "paired_win_loss": wins,
        "decision": decision(agg_rows),
    }
    with (root / "physics_validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    (root / "physics_validation_report.md").write_text(report_markdown(records, agg_rows, wins), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
