from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRIC_KEYS = ["rmse", "mae", "edge_rmse", "normal_deg", "ssim"]
FEATURE_RULES = [
    ("edge_mean", "le"),
    ("delta_mean", "ge"),
    ("delta_edge_mean", "ge"),
    ("delta_lowconf_mean", "ge"),
    ("phase_conf_mean", "ge"),
]


def read_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def threshold_candidates(rows, feature):
    values = sorted({float(row[feature]) for row in rows})
    if not values:
        return []
    mids = [(a + b) * 0.5 for a, b in zip(values[:-1], values[1:])]
    return [values[0] - 1e-6, *values, *mids, values[-1] + 1e-6]


def passes(row, clauses):
    for feature, rule, threshold in clauses:
        value = float(row[feature])
        if rule == "le" and value > threshold:
            return False
        if rule == "ge" and value < threshold:
            return False
    return True


def summarize(rows, clauses, candidate_prefix):
    selected = [passes(row, clauses) for row in rows]
    out = {"n": len(rows), "selected": int(sum(selected))}
    for metric in METRIC_KEYS:
        vals = []
        for row, use_candidate in zip(rows, selected):
            prefix = candidate_prefix if use_candidate else "base"
            vals.append(float(row[f"{prefix}_{metric}"]))
        arr = np.asarray(vals, dtype=np.float64)
        out[metric] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        }
    return out


def direct_summary(rows, prefix):
    out = {"n": len(rows)}
    for metric in METRIC_KEYS:
        arr = np.asarray([float(row[f"{prefix}_{metric}"]) for row in rows], dtype=np.float64)
        out[metric] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        }
    return out


def candidate_search(rows, candidate_prefix, min_selected, min_selected_frac):
    effective_min = max(int(min_selected), int(np.ceil(len(rows) * float(min_selected_frac))))
    candidates = []

    for feature, rule in FEATURE_RULES:
        for threshold in threshold_candidates(rows, feature):
            clauses = [(feature, rule, threshold)]
            summary = summarize(rows, clauses, candidate_prefix)
            if summary["selected"] >= effective_min:
                candidates.append({"clauses": clauses, "summary": summary})

    edge_thresholds = threshold_candidates(rows, "edge_mean")
    for feature, rule in FEATURE_RULES:
        if feature == "edge_mean":
            continue
        for edge_threshold in edge_thresholds:
            for threshold in threshold_candidates(rows, feature):
                clauses = [("edge_mean", "le", edge_threshold), (feature, rule, threshold)]
                summary = summarize(rows, clauses, candidate_prefix)
                if summary["selected"] >= effective_min:
                    candidates.append({"clauses": clauses, "summary": summary})

    if not candidates:
        raise RuntimeError("no gate candidate satisfies min_selected/min_selected_frac")
    return sorted(candidates, key=lambda c: c["summary"]["rmse"]["mean"])


def save_selected_rows(rows, clauses, candidate_prefix, path):
    keys = [
        "sample",
        "selected",
        "clauses",
        "delta_mean",
        "delta_edge_mean",
        "delta_lowconf_mean",
        "phase_conf_mean",
        "edge_mean",
    ]
    for prefix in ("base", candidate_prefix, "final"):
        keys.extend(f"{prefix}_{metric}" for metric in METRIC_KEYS)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            use_candidate = passes(row, clauses)
            final_prefix = candidate_prefix if use_candidate else "base"
            out = {
                "sample": row["sample"],
                "selected": int(use_candidate),
                "clauses": json.dumps(clauses, ensure_ascii=False),
                "delta_mean": row["delta_mean"],
                "delta_edge_mean": row["delta_edge_mean"],
                "delta_lowconf_mean": row["delta_lowconf_mean"],
                "phase_conf_mean": row["phase_conf_mean"],
                "edge_mean": row["edge_mean"],
            }
            for metric in METRIC_KEYS:
                out[f"base_{metric}"] = row[f"base_{metric}"]
                out[f"{candidate_prefix}_{metric}"] = row[f"{candidate_prefix}_{metric}"]
                out[f"final_{metric}"] = row[f"{final_prefix}_{metric}"]
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--candidate_prefix", default="blend")
    parser.add_argument("--min_selected", type=int, default=3)
    parser.add_argument("--min_selected_frac", type=float, default=0.25)
    args = parser.parse_args()

    val_rows = read_rows(Path(args.val_csv))
    test_rows = read_rows(Path(args.test_csv))
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = candidate_search(
        val_rows,
        args.candidate_prefix,
        args.min_selected,
        args.min_selected_frac,
    )
    best = candidates[0]
    clauses = best["clauses"]
    result = {
        "gate_search": {
            "type": "single_or_edge_and_feature",
            "min_selected": args.min_selected,
            "min_selected_frac": args.min_selected_frac,
            "candidate_prefix": args.candidate_prefix,
        },
        "selected": {
            "clauses": clauses,
            "val_rmse": best["summary"]["rmse"]["mean"],
            "val_selected": best["summary"]["selected"],
        },
        "val": {
            "base": direct_summary(val_rows, "base"),
            args.candidate_prefix: direct_summary(val_rows, args.candidate_prefix),
            "selected": best["summary"],
        },
        "test": {
            "base": direct_summary(test_rows, "base"),
            args.candidate_prefix: direct_summary(test_rows, args.candidate_prefix),
            "selected": summarize(test_rows, clauses, args.candidate_prefix),
        },
        "top_val_candidates": [
            {
                "clauses": candidate["clauses"],
                "val": candidate["summary"],
                "test": summarize(test_rows, candidate["clauses"], args.candidate_prefix),
            }
            for candidate in candidates[:20]
        ],
    }
    (out_dir / "conjunctive_gate_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_selected_rows(test_rows, clauses, args.candidate_prefix, out_dir / "test_selected_rows.csv")
    print(json.dumps({
        "selected": result["selected"],
        "val_base": result["val"]["base"]["rmse"]["mean"],
        "val_selected": result["val"]["selected"]["rmse"]["mean"],
        "test_base": result["test"]["base"]["rmse"]["mean"],
        "test_selected": result["test"]["selected"]["rmse"]["mean"],
        "test_selected_n": result["test"]["selected"]["selected"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
