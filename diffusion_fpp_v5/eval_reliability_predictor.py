"""Evaluate lightweight posterior reliability predictors on frozen candidates.

This script is intentionally small: it does not train a depth network.  It fits
sample-level risk scores from train split candidate statistics, selects only the
threshold/correction weight on validation split, and evaluates the selected
configuration once on test.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ("rmse", "mae")
BASE_FEATURES = (
    "edge_mean",
    "phase_conf_mean",
    "delta_pd_mean",
    "delta_pd_edge_mean",
    "delta_pd_lowconf_mean",
    "delta_bd_mean",
    "delta_bp_mean",
)


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in str(text).replace(",", " ").split() if x]


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_array(cache_dir: Path, name: str, split: str, dtype_suffix: str):
    path = cache_dir / f"{name}_{split}_{dtype_suffix}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r")


def norm_to_mm(depth_norm: np.ndarray, depth_minmax: np.ndarray) -> np.ndarray:
    depth01 = np.clip((depth_norm.astype(np.float32) + 1.0) * 0.5, 0.0, 1.0)
    dmin = float(depth_minmax[0])
    dmax = float(depth_minmax[1])
    return depth01 * max(dmax - dmin, 1e-6) + dmin


def masked_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if not np.any(m):
        return 0.0
    return float(arr[m].astype(np.float64).mean())


def metric_pair(pred_mm: np.ndarray, target_mm: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    m = mask.astype(bool)
    if not np.any(m):
        return {"rmse": float("nan"), "mae": float("nan")}
    err = pred_mm.astype(np.float64)[m] - target_mm.astype(np.float64)[m]
    return {
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
    }


def summarize(rows: list[dict[str, Any]], prefix: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    if prefix is not None:
        out["selected"] = int(sum(int(row.get("selected", 0)) for row in rows))
    for key in METRICS:
        vals = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[key] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0,
        }
    return out


def feature_matrix(rows: list[dict[str, Any]], mean=None, std=None, interactions: bool = True):
    raw = np.asarray([[float(row[name]) for name in BASE_FEATURES] for row in rows], dtype=np.float64)
    if mean is None:
        mean = raw.mean(axis=0)
    if std is None:
        std = raw.std(axis=0)
    z = (raw - mean) / np.maximum(std, 1e-6)
    pieces = [z]
    if interactions:
        inter = []
        for i in range(z.shape[1]):
            inter.append((z[:, i] * z[:, i])[:, None])
            for j in range(i + 1, z.shape[1]):
                inter.append((z[:, i] * z[:, j])[:, None])
        pieces.append(np.concatenate(inter, axis=1))
    pieces.append(np.ones((z.shape[0], 1), dtype=np.float64))
    return np.concatenate(pieces, axis=1), mean, std


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    reg = np.eye(x.shape[1], dtype=np.float64) * float(alpha)
    reg[-1, -1] = 0.0
    return np.linalg.solve(x.T @ x + reg, x.T @ y)


def logistic_fit(x: np.ndarray, y: np.ndarray, alpha: float, steps: int = 3000, lr: float = 0.05) -> np.ndarray:
    w = np.zeros((x.shape[1],), dtype=np.float64)
    y = y.astype(np.float64)
    reg_mask = np.ones_like(w)
    reg_mask[-1] = 0.0
    for _ in range(int(steps)):
        logits = np.clip(x @ w, -40.0, 40.0)
        prob = 1.0 / (1.0 + np.exp(-logits))
        grad = (x.T @ (prob - y)) / max(len(y), 1) + float(alpha) * reg_mask * w
        w -= float(lr) * grad
    return w


def candidate_thresholds(scores: np.ndarray) -> list[float]:
    vals = [float(np.quantile(scores, q)) for q in np.linspace(0.0, 1.0, 61)]
    vals.extend([float(scores.min() - 1e-6), float(scores.max() + 1e-6)])
    return sorted(set(vals))


def extract_split(cache_dir: Path, split: str, weights: list[float]) -> list[dict[str, Any]]:
    d_b = load_array(cache_dir, "d_b", split, "float16")
    d_p = load_array(cache_dir, "d_p", split, "float16")
    d_d = load_array(cache_dir, "d_d", split, "float16")
    target_mm = load_array(cache_dir, "target_mm", split, "float32")
    mask = load_array(cache_dir, "mask", split, "uint8")
    edge = load_array(cache_dir, "edge", split, "float16")
    phase_conf = load_array(cache_dir, "phase_conf", split, "float16")
    depth_minmax = load_array(cache_dir, "depth_minmax", split, "float32")
    sample_index = load_array(cache_dir, "sample_index", split, "int32")
    object_index = load_array(cache_dir, "object_index", split, "int32")

    rows = []
    for idx in range(int(d_d.shape[0])):
        m = mask[idx, 0].astype(bool)
        db = d_b[idx, 0].astype(np.float32)
        dp = d_p[idx, 0].astype(np.float32)
        dd = d_d[idx, 0].astype(np.float32)
        e = np.clip(edge[idx, 0].astype(np.float32), 0.0, 1.0)
        c = np.clip(phase_conf[idx, 0].astype(np.float32), 0.0, 1.0)
        tgt = target_mm[idx, 0].astype(np.float32)
        mm = depth_minmax[idx]
        db_mm = norm_to_mm(db, mm)
        dp_mm = norm_to_mm(dp, mm)
        dd_mm = norm_to_mm(dd, mm)

        delta_pd = np.abs(dp - dd)
        row: dict[str, Any] = {
            "split": split,
            "idx": idx,
            "sample_index": int(sample_index[idx]),
            "object_index": int(object_index[idx]),
            "edge_mean": masked_mean(e, m),
            "phase_conf_mean": masked_mean(c, m),
            "delta_pd_mean": masked_mean(delta_pd, m),
            "delta_pd_edge_mean": masked_mean(delta_pd * e, m),
            "delta_pd_lowconf_mean": masked_mean(delta_pd * (1.0 - c), m),
            "delta_bd_mean": masked_mean(np.abs(db - dd), m),
            "delta_bp_mean": masked_mean(np.abs(db - dp), m),
        }
        for name, pred in (("b", db_mm), ("p", dp_mm), ("d", dd_mm)):
            vals = metric_pair(pred, tgt, m)
            row.update({f"{name}_{k}": v for k, v in vals.items()})
        for weight in weights:
            blend = dd + float(weight) * (dp - dd)
            vals = metric_pair(norm_to_mm(blend, mm), tgt, m)
            key = f"w{weight:g}"
            row.update({f"{key}_{k}": v for k, v in vals.items()})
            row[f"{key}_gain_rmse"] = row["d_rmse"] - vals["rmse"]
            row[f"{key}_useful"] = int(vals["rmse"] < row["d_rmse"])
        rows.append(row)
    return rows


def direct_rows(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    return [{key: row[f"{prefix}_{key}"] for key in METRICS} for row in rows]


def apply_selector(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float, weight: float) -> list[dict[str, Any]]:
    out = []
    key = f"w{weight:g}"
    for row, score in zip(rows, scores):
        selected = bool(float(score) >= float(threshold))
        src = key if selected else "d"
        out.append({
            "sample_index": row["sample_index"],
            "object_index": row["object_index"],
            "score": float(score),
            "threshold": float(threshold),
            "selected": int(selected),
            "selected_weight": float(weight) if selected else 0.0,
            "edge_mean": row["edge_mean"],
            "phase_conf_mean": row["phase_conf_mean"],
            "delta_pd_mean": row["delta_pd_mean"],
            **{metric: float(row[f"{src}_{metric}"]) for metric in METRICS},
        })
    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/fpp_ml_ucpf_hier_orderfix_cache_960_seed180")
    parser.add_argument("--save_dir", default="/root/autodl-tmp/diffusion_fpp_v5/results/fpp960_reliability_predictor_orderfix_seed180")
    parser.add_argument("--weights", default="0.3 0.45 0.55 0.6 0.65 0.75 1.0")
    parser.add_argument("--ridge_alphas", default="0.001 0.01 0.1 1.0 10.0 100.0")
    parser.add_argument("--logistic_alphas", default="0.0 0.0001 0.001 0.01 0.1")
    parser.add_argument("--min_selected", type=int, default=3)
    parser.add_argument("--interactions", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    weights = parse_float_list(args.weights)

    rows_by_split = {split: extract_split(cache_dir, split, weights) for split in ("train", "val", "test")}
    x_train, mean, std = feature_matrix(rows_by_split["train"], interactions=bool(args.interactions))
    x_val, _, _ = feature_matrix(rows_by_split["val"], mean=mean, std=std, interactions=bool(args.interactions))
    x_test, _, _ = feature_matrix(rows_by_split["test"], mean=mean, std=std, interactions=bool(args.interactions))

    candidates = []
    for weight in weights:
        key = f"w{weight:g}"
        y_gain = np.asarray([row[f"{key}_gain_rmse"] for row in rows_by_split["train"]], dtype=np.float64)
        y_cls = np.asarray([row[f"{key}_useful"] for row in rows_by_split["train"]], dtype=np.float64)

        for alpha in parse_float_list(args.ridge_alphas):
            coef = ridge_fit(x_train, y_gain, alpha)
            val_score = x_val @ coef
            test_score = x_test @ coef
            for threshold in candidate_thresholds(val_score):
                val_rows = apply_selector(rows_by_split["val"], val_score, threshold, weight)
                if sum(row["selected"] for row in val_rows) < int(args.min_selected):
                    continue
                test_rows = apply_selector(rows_by_split["test"], test_score, threshold, weight)
                candidates.append({
                    "type": "ridge_gain",
                    "weight": float(weight),
                    "alpha": float(alpha),
                    "threshold": float(threshold),
                    "val": summarize(val_rows, prefix="selected"),
                    "test": summarize(test_rows, prefix="selected"),
                    "coef": coef,
                    "test_rows": test_rows,
                })

        # If one class is absent, logistic is not defined for that weight.
        if 0.0 < y_cls.mean() < 1.0:
            for alpha in parse_float_list(args.logistic_alphas):
                coef = logistic_fit(x_train, y_cls, alpha)
                val_score = 1.0 / (1.0 + np.exp(-np.clip(x_val @ coef, -40.0, 40.0)))
                test_score = 1.0 / (1.0 + np.exp(-np.clip(x_test @ coef, -40.0, 40.0)))
                for threshold in candidate_thresholds(val_score):
                    val_rows = apply_selector(rows_by_split["val"], val_score, threshold, weight)
                    if sum(row["selected"] for row in val_rows) < int(args.min_selected):
                        continue
                    test_rows = apply_selector(rows_by_split["test"], test_score, threshold, weight)
                    candidates.append({
                        "type": "logistic_useful",
                        "weight": float(weight),
                        "alpha": float(alpha),
                        "threshold": float(threshold),
                        "val": summarize(val_rows, prefix="selected"),
                        "test": summarize(test_rows, prefix="selected"),
                        "coef": coef,
                        "test_rows": test_rows,
                    })

    if not candidates:
        raise RuntimeError("no reliability predictor candidates were generated")
    best = min(candidates, key=lambda item: item["val"]["rmse"]["mean"])
    selected_rows = best.pop("test_rows")
    write_csv(selected_rows, save_dir / "test_selected_rows.csv")
    write_csv(rows_by_split["val"], save_dir / "val_features_metrics.csv")
    write_csv(rows_by_split["test"], save_dir / "test_features_metrics.csv")

    baselines = {}
    for split, rows in rows_by_split.items():
        baselines[split] = {
            "d_b": summarize(direct_rows(rows, "b")),
            "d_p": summarize(direct_rows(rows, "p")),
            "d_d": summarize(direct_rows(rows, "d")),
        }
        for weight in weights:
            baselines[split][f"blend_w{weight:g}"] = summarize(direct_rows(rows, f"w{weight:g}"))

    result = {
        "method": "Lightweight Reliability Predictor for Selective Posterior Correction",
        "cache_dir": str(cache_dir),
        "features": list(BASE_FEATURES),
        "interactions": bool(args.interactions),
        "selection": "fit score on train; select threshold and correction weight by validation RMSE; evaluate test once",
        "selected_by_val": {k: json_safe(v) for k, v in best.items() if k != "coef"},
        "selected_coef": json_safe(best["coef"]),
        "feature_mean": json_safe(mean),
        "feature_std": json_safe(std),
        "baselines": baselines,
        "top_by_val": [
            {k: json_safe(v) for k, v in item.items() if k not in {"coef", "test_rows"}}
            for item in sorted(candidates, key=lambda item: item["val"]["rmse"]["mean"])[:20]
        ],
        "top_by_test_oracle": [
            {k: json_safe(v) for k, v in item.items() if k not in {"coef", "test_rows"}}
            for item in sorted(candidates, key=lambda item: item["test"]["rmse"]["mean"])[:20]
        ],
    }
    with (save_dir / "reliability_predictor_summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(result), f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "selected": result["selected_by_val"],
        "test_rmse": result["selected_by_val"]["test"]["rmse"]["mean"],
        "test_selected": result["selected_by_val"]["test"]["selected"],
        "summary": str(save_dir / "reliability_predictor_summary.json"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
