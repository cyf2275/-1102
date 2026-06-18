from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRIC_KEYS = ["rmse", "mae", "edge_rmse", "normal_deg", "ssim"]
BASE_FEATURES = [
    "edge_mean",
    "phase_conf_mean",
    "delta_mean",
    "delta_edge_mean",
    "delta_lowconf_mean",
]


def read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            row[key] = int(value) if key == "sample" else float(value)
    return rows


def feature_matrix(rows, mean=None, std=None, interactions=True):
    x = np.asarray([[row[name] for name in BASE_FEATURES] for row in rows], dtype=np.float64)
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0)
    z = (x - mean) / np.maximum(std, 1e-6)
    feats = [z]
    if interactions:
        pairs = []
        for i in range(z.shape[1]):
            for j in range(i, z.shape[1]):
                pairs.append((z[:, i] * z[:, j])[:, None])
        feats.append(np.concatenate(pairs, axis=1))
    feats.append(np.ones((z.shape[0], 1), dtype=np.float64))
    return np.concatenate(feats, axis=1), mean, std


def metric_values(rows, selector, candidate_prefix):
    out = []
    for row in rows:
        prefix = candidate_prefix if selector(row) else "base"
        out.append({key: float(row[f"{prefix}_{key}"]) for key in METRIC_KEYS})
    return out


def direct_summary(rows, prefix):
    out = {"n": len(rows)}
    for key in METRIC_KEYS:
        arr = np.asarray([float(row[f"{prefix}_{key}"]) for row in rows], dtype=np.float64)
        out[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        }
    return out


def summarize(rows, selector, candidate_prefix):
    selected = int(sum(bool(selector(row)) for row in rows))
    vals = metric_values(rows, selector, candidate_prefix)
    out = {"n": len(rows), "selected": selected}
    for key in METRIC_KEYS:
        arr = np.asarray([row[key] for row in vals], dtype=np.float64)
        out[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        }
    return out


def oracle_summary(rows, candidate_prefix):
    return summarize(
        rows,
        lambda row: float(row[f"{candidate_prefix}_rmse"]) < float(row["base_rmse"]),
        candidate_prefix,
    )


def threshold_candidates(scores):
    vals = sorted({float(v) for v in scores})
    if not vals:
        return [0.0]
    mids = [(a + b) * 0.5 for a, b in zip(vals[:-1], vals[1:])]
    return [vals[0] - 1e-6, *vals, *mids, vals[-1] + 1e-6]


def best_threshold_on_val(val_rows, val_scores, candidate_prefix, min_selected=1):
    best = None
    for threshold in threshold_candidates(val_scores):
        selector = lambda row, t=threshold, scores=dict(zip([r["sample"] for r in val_rows], val_scores)): scores[row["sample"]] > t
        summary = summarize(val_rows, selector, candidate_prefix)
        if summary["selected"] < min_selected:
            continue
        if best is None or summary["rmse"]["mean"] < best["summary"]["rmse"]["mean"]:
            best = {"threshold": float(threshold), "summary": summary}
    if best is None:
        raise RuntimeError("no valid threshold")
    return best


def train_ridge(train_rows, candidate_prefix, alpha=1.0, interactions=True):
    x_train, mean, std = feature_matrix(train_rows, interactions=interactions)
    y = np.asarray([
        float(row["base_rmse"]) - float(row[f"{candidate_prefix}_rmse"])
        for row in train_rows
    ], dtype=np.float64)
    reg = np.eye(x_train.shape[1], dtype=np.float64) * float(alpha)
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(x_train.T @ x_train + reg, x_train.T @ y)
    return {"coef": coef, "mean": mean, "std": std, "interactions": interactions}


def predict(model, rows):
    x, _, _ = feature_matrix(rows, mean=model["mean"], std=model["std"], interactions=model["interactions"])
    return x @ model["coef"]


def score_selector(rows, scores, threshold):
    score_map = {row["sample"]: float(score) for row, score in zip(rows, scores)}
    return lambda row: score_map[row["sample"]] > threshold


def single_feature_candidates(train_rows, val_rows, candidate_prefix, min_selected=1):
    candidates = []
    for feature in BASE_FEATURES:
        for rule in ("le", "ge"):
            train_vals = [row[feature] for row in train_rows]
            for threshold in threshold_candidates(train_vals):
                if rule == "le":
                    selector = lambda row, f=feature, t=threshold: row[f] <= t
                else:
                    selector = lambda row, f=feature, t=threshold: row[f] >= t
                train_summary = summarize(train_rows, selector, candidate_prefix)
                val_summary = summarize(val_rows, selector, candidate_prefix)
                if train_summary["selected"] < min_selected or val_summary["selected"] < min_selected:
                    continue
                candidates.append({
                    "type": "single_feature",
                    "feature": feature,
                    "rule": rule,
                    "threshold": float(threshold),
                    "train": train_summary,
                    "val": val_summary,
                    "selector": selector,
                })
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--save_json", required=True)
    parser.add_argument("--min_selected", type=int, default=3)
    parser.add_argument("--ridge_alphas", default="0.01 0.1 1.0 10.0")
    parser.add_argument(
        "--candidate_prefix",
        default="blend",
        help="CSV metric prefix for the diffusion-corrected candidate, e.g. blend or ensemble.",
    )
    args = parser.parse_args()

    train_rows = read_rows(args.train_csv)
    val_rows = read_rows(args.val_csv)
    test_rows = read_rows(args.test_csv)

    candidate_prefix = str(args.candidate_prefix)
    candidates = single_feature_candidates(
        train_rows,
        val_rows,
        candidate_prefix,
        min_selected=args.min_selected,
    )
    for alpha in [float(x) for x in args.ridge_alphas.replace(",", " ").split() if x]:
        model = train_ridge(train_rows, candidate_prefix, alpha=alpha, interactions=True)
        val_scores = predict(model, val_rows)
        best = best_threshold_on_val(
            val_rows,
            val_scores,
            candidate_prefix,
            min_selected=args.min_selected,
        )
        train_scores = predict(model, train_rows)
        selector_train = score_selector(train_rows, train_scores, best["threshold"])
        selector_val = score_selector(val_rows, val_scores, best["threshold"])
        candidates.append({
            "type": "ridge_improvement",
            "alpha": alpha,
            "threshold": best["threshold"],
            "coef": [float(x) for x in model["coef"]],
            "feature_mean": [float(x) for x in model["mean"]],
            "feature_std": [float(x) for x in model["std"]],
            "train": summarize(train_rows, selector_train, candidate_prefix),
            "val": summarize(val_rows, selector_val, candidate_prefix),
            "selector": ("ridge", model, best["threshold"]),
        })

    best = min(candidates, key=lambda c: c["val"]["rmse"]["mean"])
    if best["type"] == "single_feature":
        selector_test = best["selector"]
    else:
        _, model, threshold = best["selector"]
        selector_test = score_selector(test_rows, predict(model, test_rows), threshold)
    result = {
        "selected_gate": {k: v for k, v in best.items() if k not in {"selector", "train", "val"}},
        "candidate_prefix": candidate_prefix,
        "train": {
            "base": direct_summary(train_rows, "base"),
            "candidate": direct_summary(train_rows, candidate_prefix),
            "oracle": oracle_summary(train_rows, candidate_prefix),
            "gated": best["train"],
        },
        "val": {
            "base": direct_summary(val_rows, "base"),
            "candidate": direct_summary(val_rows, candidate_prefix),
            "oracle": oracle_summary(val_rows, candidate_prefix),
            "gated": best["val"],
        },
        "test": {
            "base": direct_summary(test_rows, "base"),
            "candidate": direct_summary(test_rows, candidate_prefix),
            "oracle": oracle_summary(test_rows, candidate_prefix),
            "gated": summarize(test_rows, selector_test, candidate_prefix),
        },
        "top_val_candidates": [
            {k: v for k, v in c.items() if k not in {"selector"}}
            for c in sorted(candidates, key=lambda c: c["val"]["rmse"]["mean"])[:20]
        ],
    }
    out = Path(args.save_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "gate": result["selected_gate"],
        "val_base": result["val"]["base"]["rmse"]["mean"],
        "val_candidate": result["val"]["candidate"]["rmse"]["mean"],
        "val_gated": result["val"]["gated"]["rmse"]["mean"],
        "val_oracle": result["val"]["oracle"]["rmse"]["mean"],
        "test_base": result["test"]["base"]["rmse"]["mean"],
        "test_candidate": result["test"]["candidate"]["rmse"]["mean"],
        "test_gated": result["test"]["gated"]["rmse"]["mean"],
        "test_oracle": result["test"]["oracle"]["rmse"]["mean"],
        "test_selected": result["test"]["gated"]["selected"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
