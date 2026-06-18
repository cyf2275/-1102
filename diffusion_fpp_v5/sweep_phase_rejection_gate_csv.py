from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ("rmse", "mae", "edge_rmse", "normal_deg", "ssim")
FEATURES = (
    ("edge_mean", "edge"),
    ("delta_mean", "delta"),
    ("phase_conf_mean", "phase_conf"),
    ("pixel_selected_frac", "pixel_frac"),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def fmt_weight(weight: float) -> str:
    text = f"{weight:.8f}".rstrip("0").rstrip(".")
    return text if text else "0"


def available_weights(rows: list[dict[str, str]]) -> list[float]:
    return sorted({f(row["phase_weight"]) for row in rows})


def build_weight_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        out[(str(row["sample"]), fmt_weight(f(row["phase_weight"])))] = row
    return out


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "num_samples": len(rows),
        "selected": int(sum(1 for row in rows if f(row["selected_phase_weight"]) > 0.0)),
    }
    for metric in METRICS:
        out[metric] = mean([f(row[metric]) for row in rows])
    return out


def fixed_weight_summary(rows: list[dict[str, str]], weight: float) -> dict[str, Any]:
    picked = []
    for row in rows:
        out = {"selected_phase_weight": weight}
        for metric in METRICS:
            out[metric] = f(row[metric])
        picked.append(out)
    return summarize(picked)


def branch_summary(rows: list[dict[str, str]], branch: str) -> dict[str, Any]:
    picked = []
    for row in rows:
        out = {"selected_phase_weight": 0.0}
        for metric in METRICS:
            out[metric] = f(row[f"{branch}_{metric}"])
        picked.append(out)
    return summarize(picked)


def threshold_values(rows: list[dict[str, str]], key: str, max_values: int) -> list[float]:
    vals = sorted({f(row.get(key)) for row in rows})
    if not vals:
        return [0.0]
    if len(vals) <= max_values:
        picked = vals
    else:
        picked = []
        for i in range(max_values):
            idx = round(i * (len(vals) - 1) / max(1, max_values - 1))
            picked.append(vals[idx])
    extras = {
        "edge_mean": [0.35, 0.40, 0.42, 0.45, 0.47, 0.50, 0.55, 0.58, 0.60, 0.62],
        "delta_mean": [0.07, 0.09, 0.10, 0.105, 0.11, 0.12, 0.14],
        "phase_conf_mean": [0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80],
        "pixel_selected_frac": [0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30],
    }.get(key, [])
    lo, hi = vals[0], vals[-1]
    picked.extend(x for x in extras if lo <= x <= hi)
    return sorted({round(x, 8) for x in picked})


def make_conditions(
    rows: list[dict[str, str]],
    max_active_features: int,
    max_thresholds_per_feature: int,
) -> list[dict[str, Any]]:
    grids = {
        key: threshold_values(rows, key, max_thresholds_per_feature)
        for key, _label in FEATURES
    }
    conditions: list[dict[str, Any]] = []
    for n_active in range(1, max_active_features + 1):
        for feature_subset in combinations(FEATURES, n_active):
            op_lists = []
            value_lists = []
            for key, label in feature_subset:
                op_lists.append([">=", "<="])
                value_lists.append(grids[key])
            for ops in product(*op_lists):
                for values in product(*value_lists):
                    clauses = []
                    for (key, label), op, value in zip(feature_subset, ops, values):
                        clauses.append({"key": key, "label": label, "op": op, "value": value})
                    conditions.append({"clauses": clauses})
    return conditions


def select_flag(row: dict[str, str], cfg: dict[str, Any]) -> bool:
    for clause in cfg["clauses"]:
        value = f(row.get(clause["key"]))
        if clause["op"] == ">=":
            if value < float(clause["value"]):
                return False
        elif clause["op"] == "<=":
            if value > float(clause["value"]):
                return False
        else:
            raise ValueError(f"unsupported op: {clause['op']}")
    return True


def apply_rule(
    hier_rows: list[dict[str, str]],
    fused_rows: list[dict[str, str]],
    cfg: dict[str, Any],
    low_weight: float,
    high_weight: float,
) -> list[dict[str, Any]]:
    fused_by_key = build_weight_index(fused_rows)
    low_key = fmt_weight(low_weight)
    high_key = fmt_weight(high_weight)
    selected: list[dict[str, Any]] = []
    for hrow in sorted(hier_rows, key=lambda row: int(float(row["sample"]))):
        use_phase = select_flag(hrow, cfg)
        chosen_weight = high_weight if use_phase else low_weight
        chosen_key = high_key if use_phase else low_key
        key = (str(hrow["sample"]), chosen_key)
        if key not in fused_by_key:
            raise KeyError(f"missing fused row for sample={key[0]}, phase_weight={chosen_key}")
        frow = fused_by_key[key]
        out: dict[str, Any] = {
            "sample": str(hrow["sample"]),
            "selected_phase_weight": chosen_weight,
            "rule": "phase" if use_phase else "hierarchical",
        }
        for key_name, _label in FEATURES:
            out[key_name] = f(hrow.get(key_name))
        for metric in METRICS:
            out[metric] = f(frow[metric])
        selected.append(out)
    return selected


def cfg_complexity(cfg: dict[str, Any]) -> tuple[int, str]:
    clauses = cfg["clauses"]
    text = " & ".join(
        f"{c['label']}{c['op']}{float(c['value']):.5g}" for c in clauses
    )
    return len(clauses), text


def search(
    val_hier: list[dict[str, str]],
    val_fused: list[dict[str, str]],
    low_weight: float,
    high_weights: list[float],
    max_active_features: int,
    max_thresholds_per_feature: int,
    min_selected: int,
    max_selected: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    conditions = make_conditions(val_hier, max_active_features, max_thresholds_per_feature)
    ordered_hier = sorted(val_hier, key=lambda row: int(float(row["sample"])))
    samples = [str(row["sample"]) for row in ordered_hier]
    feature_arrays = {
        key: np.asarray([f(row.get(key)) for row in ordered_hier], dtype=np.float64)
        for key, _label in FEATURES
    }
    fused_by_key = build_weight_index(val_fused)
    low_key = fmt_weight(low_weight)
    low_metrics = {
        metric: np.asarray([f(fused_by_key[(sample, low_key)][metric]) for sample in samples], dtype=np.float64)
        for metric in METRICS
    }
    high_metrics_by_weight = {}
    for high_weight in high_weights:
        high_key = fmt_weight(high_weight)
        high_metrics_by_weight[high_weight] = {
            metric: np.asarray([f(fused_by_key[(sample, high_key)][metric]) for sample in samples], dtype=np.float64)
            for metric in METRICS
        }

    candidates: list[dict[str, Any]] = []
    n_samples = len(samples)
    for cfg in conditions:
        selected_mask = np.ones((n_samples,), dtype=bool)
        for clause in cfg["clauses"]:
            values = feature_arrays[clause["key"]]
            threshold = float(clause["value"])
            if clause["op"] == ">=":
                selected_mask &= values >= threshold
            elif clause["op"] == "<=":
                selected_mask &= values <= threshold
            else:
                raise ValueError(f"unsupported op: {clause['op']}")
        selected = int(selected_mask.sum())
        if selected < min_selected:
            continue
        if max_selected is not None and selected > max_selected:
            continue
        n_clauses, rule_text = cfg_complexity(cfg)
        for high_weight in high_weights:
            if fmt_weight(high_weight) == fmt_weight(low_weight):
                continue
            high_metrics = high_metrics_by_weight[high_weight]
            summary: dict[str, Any] = {
                "num_samples": n_samples,
                "selected": selected,
            }
            for metric in METRICS:
                vals = np.where(selected_mask, high_metrics[metric], low_metrics[metric])
                summary[metric] = float(vals.mean())
            candidates.append(
                {
                    "cfg": cfg,
                    "rule_text": rule_text,
                    "low_weight": low_weight,
                    "high_weight": high_weight,
                    "val": summary,
                    "sort_key": (
                        float(summary["rmse"]),
                        n_clauses,
                        abs(selected - n_samples / 2.0),
                        high_weight,
                    ),
                }
            )
    if not candidates:
        raise RuntimeError("no phase-rejection candidates met the constraints")
    candidates.sort(key=lambda row: row["sort_key"])
    return candidates[0], candidates


def write_selected(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample",
        "selected_phase_weight",
        "rule",
        *[key for key, _label in FEATURES],
        *METRICS,
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_top_csv(rows: list[dict[str, Any]], path: Path, limit: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "rule_text",
        "low_weight",
        "high_weight",
        "val_rmse",
        "val_mae",
        "val_edge_rmse",
        "val_normal_deg",
        "val_ssim",
        "val_selected",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows[:limit], start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "rule_text": row["rule_text"],
                    "low_weight": row["low_weight"],
                    "high_weight": row["high_weight"],
                    "val_rmse": row["val"]["rmse"],
                    "val_mae": row["val"]["mae"],
                    "val_edge_rmse": row["val"]["edge_rmse"],
                    "val_normal_deg": row["val"]["normal_deg"],
                    "val_ssim": row["val"]["ssim"],
                    "val_selected": row["val"]["selected"],
                }
            )


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in str(text).replace(",", " ").split() if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only sweep for a sample-level phase rejection gate. "
            "The selected rule is applied unchanged to test."
        )
    )
    parser.add_argument("--val_hier_csv", type=Path, required=True)
    parser.add_argument("--test_hier_csv", type=Path, required=True)
    parser.add_argument("--val_fused_csv", type=Path, required=True)
    parser.add_argument("--test_fused_csv", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--low_weight", type=float, default=0.0)
    parser.add_argument("--high_weights", default="auto")
    parser.add_argument("--max_active_features", type=int, default=3)
    parser.add_argument("--max_thresholds_per_feature", type=int, default=14)
    parser.add_argument("--min_selected", type=int, default=1)
    parser.add_argument("--max_selected", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    val_hier = read_rows(args.val_hier_csv)
    test_hier = read_rows(args.test_hier_csv)
    val_fused = read_rows(args.val_fused_csv)
    test_fused = read_rows(args.test_fused_csv)

    val_weights = available_weights(val_fused)
    test_weights = set(fmt_weight(x) for x in available_weights(test_fused))
    if args.high_weights == "auto":
        high_weights = [w for w in val_weights if w > 0.0 and fmt_weight(w) in test_weights]
    else:
        high_weights = [
            w for w in parse_float_list(args.high_weights)
            if fmt_weight(w) in test_weights and fmt_weight(w) in {fmt_weight(v) for v in val_weights}
        ]
    if not high_weights:
        raise RuntimeError("no valid high phase weights are available in both val/test fused CSVs")

    selected, candidates = search(
        val_hier=val_hier,
        val_fused=val_fused,
        low_weight=args.low_weight,
        high_weights=high_weights,
        max_active_features=args.max_active_features,
        max_thresholds_per_feature=args.max_thresholds_per_feature,
        min_selected=args.min_selected,
        max_selected=args.max_selected,
    )
    test_rows = apply_rule(
        test_hier,
        test_fused,
        selected["cfg"],
        selected["low_weight"],
        selected["high_weight"],
    )
    val_rows = apply_rule(
        val_hier,
        val_fused,
        selected["cfg"],
        selected["low_weight"],
        selected["high_weight"],
    )
    test_summary = summarize(test_rows)
    val_summary = summarize(val_rows)

    write_selected(val_rows, args.save_dir / "val_selected_rows.csv")
    write_selected(test_rows, args.save_dir / "test_selected_rows.csv")
    write_top_csv(candidates, args.save_dir / "top_val_candidates.csv", args.top_k)

    summary = {
        "method": "validation-frozen phase rejection gate",
        "selection_policy": "All thresholds, high weight, and rule family are selected only by validation RMSE; test is eval-only.",
        "selected_rule": {
            "rule_text": selected["rule_text"],
            "clauses": selected["cfg"]["clauses"],
            "low_weight": selected["low_weight"],
            "high_weight": selected["high_weight"],
        },
        "inputs": {
            "val_hier_csv": str(args.val_hier_csv),
            "test_hier_csv": str(args.test_hier_csv),
            "val_fused_csv": str(args.val_fused_csv),
            "test_fused_csv": str(args.test_fused_csv),
        },
        "available_high_weights": high_weights,
        "val": {
            "selected_gate": val_summary,
            "hierarchical": branch_summary(val_hier, "hierarchical"),
            "base": branch_summary(val_hier, "base"),
            "diff": branch_summary(val_hier, "diff"),
            "pixel_gated": branch_summary(val_hier, "pixel_gated"),
            "phase_branch": branch_summary(val_hier, "phase_branch"),
            "fixed_phase_weights": {
                fmt_weight(w): fixed_weight_summary(
                    [row for row in val_fused if fmt_weight(f(row["phase_weight"])) == fmt_weight(w)],
                    w,
                )
                for w in [args.low_weight, *high_weights]
            },
        },
        "test": {
            "selected_gate": test_summary,
            "hierarchical": branch_summary(test_hier, "hierarchical"),
            "base": branch_summary(test_hier, "base"),
            "diff": branch_summary(test_hier, "diff"),
            "pixel_gated": branch_summary(test_hier, "pixel_gated"),
            "phase_branch": branch_summary(test_hier, "phase_branch"),
            "fixed_phase_weights": {
                fmt_weight(w): fixed_weight_summary(
                    [row for row in test_fused if fmt_weight(f(row["phase_weight"])) == fmt_weight(w)],
                    w,
                )
                for w in [args.low_weight, *high_weights]
            },
        },
        "top_by_val": [
            {
                "rank": idx + 1,
                "rule_text": row["rule_text"],
                "low_weight": row["low_weight"],
                "high_weight": row["high_weight"],
                "val": row["val"],
            }
            for idx, row in enumerate(candidates[: min(args.top_k, 20)])
        ],
    }
    with (args.save_dir / "phase_rejection_gate_sweep_summary.json").open("w", encoding="utf-8") as fobj:
        json.dump(summary, fobj, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "selected_rule": summary["selected_rule"],
                "val_rmse": val_summary["rmse"],
                "test_rmse": test_summary["rmse"],
                "test_selected": test_summary["selected"],
                "hier_test_rmse": summary["test"]["hierarchical"]["rmse"],
                "phase_branch_test_rmse": summary["test"]["phase_branch"]["rmse"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
