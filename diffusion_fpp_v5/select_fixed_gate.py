from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRIC_KEYS = ["rmse", "mae", "edge_rmse", "normal_deg", "ssim"]


def read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            row[key] = int(value) if key == "sample" else float(value)
    return rows


def make_selector(feature, rule, threshold):
    if rule == "le":
        return lambda row: float(row[feature]) <= threshold
    if rule == "ge":
        return lambda row: float(row[feature]) >= threshold
    raise ValueError(f"unknown rule: {rule}")


def direct_summary(rows, prefix):
    out = {"n": len(rows)}
    for metric in METRIC_KEYS:
        arr = np.asarray([row[f"{prefix}_{metric}"] for row in rows], dtype=np.float64)
        out[metric] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        }
    return out


def gated_summary(rows, selector):
    out = {"n": len(rows), "selected": int(sum(selector(row) for row in rows))}
    for metric in METRIC_KEYS:
        vals = []
        for row in rows:
            prefix = "blend" if selector(row) else "base"
            vals.append(row[f"{prefix}_{metric}"])
        arr = np.asarray(vals, dtype=np.float64)
        out[metric] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        }
    return out


def selected_subset_summary(rows, selector):
    selected = [row for row in rows if selector(row)]
    if not selected:
        return {"n": 0}
    out = {"n": len(selected)}
    for prefix in ("base", "blend", "diff"):
        out[prefix] = direct_summary(selected, prefix)
    out["blend_wins"] = int(sum(row["blend_rmse"] < row["base_rmse"] for row in selected))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--save_json", required=True)
    parser.add_argument("--feature", default="edge_mean")
    parser.add_argument("--rule", choices=["le", "ge"], default="le")
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()

    val_rows = read_rows(args.val_csv)
    test_rows = read_rows(args.test_csv)
    selector = make_selector(args.feature, args.rule, args.threshold)
    result = {
        "fixed_gate": {
            "feature": args.feature,
            "rule": args.rule,
            "threshold": float(args.threshold),
        },
        "val": {
            "base": direct_summary(val_rows, "base"),
            "blend": direct_summary(val_rows, "blend"),
            "gated": gated_summary(val_rows, selector),
            "selected_subset": selected_subset_summary(val_rows, selector),
        },
        "test": {
            "base": direct_summary(test_rows, "base"),
            "blend": direct_summary(test_rows, "blend"),
            "gated": gated_summary(test_rows, selector),
            "selected_subset": selected_subset_summary(test_rows, selector),
        },
    }
    out = Path(args.save_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "gate": result["fixed_gate"],
        "val_base": result["val"]["base"]["rmse"]["mean"],
        "val_blend": result["val"]["blend"]["rmse"]["mean"],
        "val_gated": result["val"]["gated"]["rmse"]["mean"],
        "test_base": result["test"]["base"]["rmse"]["mean"],
        "test_blend": result["test"]["blend"]["rmse"]["mean"],
        "test_gated": result["test"]["gated"]["rmse"]["mean"],
        "test_selected": result["test"]["gated"]["selected"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
