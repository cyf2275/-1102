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
            if key != "sample":
                row[key] = float(value)
            else:
                row[key] = int(value)
    return rows


def summarize(rows, selector):
    out = {"n": len(rows), "selected": int(sum(selector(row) for row in rows))}
    for metric in METRIC_KEYS:
        vals = []
        for row in rows:
            prefix = "blend" if selector(row) else "base"
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


def threshold_candidates(rows, feature):
    vals = sorted({float(row[feature]) for row in rows})
    if not vals:
        return [0.0]
    mids = [(a + b) * 0.5 for a, b in zip(vals[:-1], vals[1:])]
    return [vals[0] - 1e-6, *vals, *mids, vals[-1] + 1e-6]


def make_selector(rule, feature, threshold):
    if rule == "le":
        return lambda row: float(row[feature]) <= threshold
    if rule == "ge":
        return lambda row: float(row[feature]) >= threshold
    raise ValueError(f"unknown rule: {rule}")


def best_single_feature_gate(rows, feature, rule, min_selected=1):
    best = None
    for threshold in threshold_candidates(rows, feature):
        selector = make_selector(rule, feature, threshold)
        summary = summarize(rows, selector)
        if summary["selected"] < min_selected:
            continue
        candidate = {
            "feature": feature,
            "rule": rule,
            "threshold": float(threshold),
            "summary": summary,
        }
        if best is None or summary["rmse"]["mean"] < best["summary"]["rmse"]["mean"]:
            best = candidate
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--save_json", required=True)
    parser.add_argument("--min_selected", type=int, default=1)
    args = parser.parse_args()

    val_rows = read_rows(args.val_csv)
    test_rows = read_rows(args.test_csv)
    search_space = [
        ("edge_mean", "le"),
        ("delta_mean", "ge"),
        ("delta_edge_mean", "ge"),
        ("delta_lowconf_mean", "ge"),
        ("phase_conf_mean", "ge"),
    ]
    candidates = [
        best_single_feature_gate(val_rows, feature, rule, min_selected=args.min_selected)
        for feature, rule in search_space
    ]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        raise RuntimeError("no valid gate candidates")
    best = min(candidates, key=lambda c: c["summary"]["rmse"]["mean"])
    selector = make_selector(best["rule"], best["feature"], best["threshold"])
    result = {
        "selected_gate": {
            "feature": best["feature"],
            "rule": best["rule"],
            "threshold": best["threshold"],
            "val_rmse": best["summary"]["rmse"]["mean"],
        },
        "val": {
            "base": direct_summary(val_rows, "base"),
            "blend": direct_summary(val_rows, "blend"),
            "gated": summarize(val_rows, selector),
        },
        "test": {
            "base": direct_summary(test_rows, "base"),
            "blend": direct_summary(test_rows, "blend"),
            "gated": summarize(test_rows, selector),
        },
        "all_val_candidates": candidates,
    }
    out = Path(args.save_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "gate": result["selected_gate"],
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
