from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ["rmse", "mae", "edge_rmse", "normal_deg", "ssim"]
BASE_FEATURES = ["edge_mean", "delta_mean", "phase_conf_mean", "pixel_selected_frac"]


def fmt_weight(weight: float) -> str:
    text = f"{weight:.8f}".rstrip("0").rstrip(".")
    return text if text else "0"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def weight_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out = {}
    for row in rows:
        out[(str(row["sample"]), fmt_weight(float(row["phase_weight"])))] = row
    return out


def feature_matrix(rows: list[dict[str, str]], mean=None, std=None):
    raw = []
    for row in rows:
        vals = []
        for name in BASE_FEATURES:
            vals.append(float(row.get(name, 0.0)))
        raw.append(vals)
    x = np.asarray(raw, dtype=np.float64)
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0)
    z = (x - mean) / np.maximum(std, 1e-6)
    pieces = [z, z * z]
    inter = []
    for i in range(z.shape[1]):
        for j in range(i + 1, z.shape[1]):
            inter.append((z[:, i] * z[:, j])[:, None])
    if inter:
        pieces.append(np.concatenate(inter, axis=1))
    pieces.append(np.ones((z.shape[0], 1), dtype=np.float64))
    return np.concatenate(pieces, axis=1), mean, std


def summarize(selected_rows: list[dict[str, object]]) -> dict[str, object]:
    out = {
        "num_samples": len(selected_rows),
        "selected": int(sum(1 for row in selected_rows if row["selected_phase_weight"] > 0)),
    }
    for key in METRICS:
        vals = np.asarray([float(row[key]) for row in selected_rows], dtype=np.float64)
        out[key] = float(vals.mean())
    return out


def build_rows(
    hier_rows: list[dict[str, str]],
    fused_rows: list[dict[str, str]],
    score: np.ndarray,
    threshold: float,
    high_weight: float,
    low_weight: float = 0.0,
) -> list[dict[str, object]]:
    fused = weight_index(fused_rows)
    high_key = fmt_weight(high_weight)
    low_key = fmt_weight(low_weight)
    out = []
    for idx, hrow in enumerate(hier_rows):
        sample = str(hrow["sample"])
        use_phase = bool(score[idx] >= threshold)
        chosen_weight = high_weight if use_phase else low_weight
        chosen_key = high_key if use_phase else low_key
        frow = fused[(sample, chosen_key)]
        row = {
            "sample": sample,
            "score": float(score[idx]),
            "threshold": float(threshold),
            "selected_phase_weight": float(chosen_weight),
            "edge_mean": float(hrow.get("edge_mean", 0.0)),
            "delta_mean": float(hrow.get("delta_mean", 0.0)),
            "phase_conf_mean": float(hrow.get("phase_conf_mean", 0.0)),
        }
        for metric in METRICS:
            row[metric] = float(frow[metric])
        out.append(row)
    return out


def write_selected(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample",
        "score",
        "threshold",
        "selected_phase_weight",
        "edge_mean",
        "delta_mean",
        "phase_conf_mean",
        *METRICS,
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fit_scores(
    train_hier: list[dict[str, str]],
    train_fused: list[dict[str, str]],
    high_weight: float,
    ridge_alpha: float,
):
    fused = weight_index(train_fused)
    low_key = fmt_weight(0.0)
    high_key = fmt_weight(high_weight)
    y = []
    for row in train_hier:
        sample = str(row["sample"])
        base_rmse = float(fused[(sample, low_key)]["rmse"])
        high_rmse = float(fused[(sample, high_key)]["rmse"])
        y.append(base_rmse - high_rmse)
    y_arr = np.asarray(y, dtype=np.float64)
    x, mean, std = feature_matrix(train_hier)
    xtx = x.T @ x
    beta = np.linalg.solve(xtx + ridge_alpha * np.eye(xtx.shape[0]), x.T @ y_arr)
    train_score = x @ beta
    return beta, mean, std, train_score


def score_rows(rows: list[dict[str, str]], beta, mean, std):
    x, _, _ = feature_matrix(rows, mean=mean, std=std)
    return x @ beta


def candidate_thresholds(scores: np.ndarray) -> list[float]:
    qs = np.linspace(0.0, 1.0, 41)
    vals = [float(np.quantile(scores, q)) for q in qs]
    vals.extend([float(scores.min() - 1e-6), float(scores.max() + 1e-6)])
    return sorted(set(vals))


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_hier_csv", type=Path, required=True)
    parser.add_argument("--val_hier_csv", type=Path, required=True)
    parser.add_argument("--test_hier_csv", type=Path, required=True)
    parser.add_argument("--train_fused_csv", type=Path, required=True)
    parser.add_argument("--val_fused_csv", type=Path, required=True)
    parser.add_argument("--test_fused_csv", type=Path, required=True)
    parser.add_argument("--high_weights", default="0.45,0.5,0.55,0.6,0.65,0.7")
    parser.add_argument("--ridge_alphas", default="0.001,0.01,0.1,1.0,10.0")
    parser.add_argument("--min_selected", type=int, default=3)
    parser.add_argument("--save_dir", type=Path, required=True)
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    train_h = read_csv(args.train_hier_csv)
    val_h = read_csv(args.val_hier_csv)
    test_h = read_csv(args.test_hier_csv)
    train_f = read_csv(args.train_fused_csv)
    val_f = read_csv(args.val_fused_csv)
    test_f = read_csv(args.test_fused_csv)

    candidates = []
    for high_weight in parse_float_list(args.high_weights):
        for ridge_alpha in parse_float_list(args.ridge_alphas):
            beta, mean, std, _ = fit_scores(train_h, train_f, high_weight, ridge_alpha)
            val_score = score_rows(val_h, beta, mean, std)
            test_score = score_rows(test_h, beta, mean, std)
            for threshold in candidate_thresholds(val_score):
                val_rows = build_rows(val_h, val_f, val_score, threshold, high_weight)
                if sum(1 for row in val_rows if row["selected_phase_weight"] > 0) < args.min_selected:
                    continue
                test_rows = build_rows(test_h, test_f, test_score, threshold, high_weight)
                candidates.append(
                    {
                        "high_weight": high_weight,
                        "ridge_alpha": ridge_alpha,
                        "threshold": threshold,
                        "val": summarize(val_rows),
                        "test": summarize(test_rows),
                    }
                )
    if not candidates:
        raise RuntimeError("no candidates met min_selected")

    selected = min(candidates, key=lambda row: row["val"]["rmse"])
    beta, mean, std, _ = fit_scores(train_h, train_f, selected["high_weight"], selected["ridge_alpha"])
    val_score = score_rows(val_h, beta, mean, std)
    test_score = score_rows(test_h, beta, mean, std)
    val_rows = build_rows(val_h, val_f, val_score, selected["threshold"], selected["high_weight"])
    test_rows = build_rows(test_h, test_f, test_score, selected["threshold"], selected["high_weight"])
    write_selected(val_rows, args.save_dir / "val_selected_rows.csv")
    write_selected(test_rows, args.save_dir / "test_selected_rows.csv")

    summary = {
        "method": "learned physical phase gate",
        "features": BASE_FEATURES,
        "selection": "ridge model fit on train phase-gain; threshold and high_weight selected by val RMSE",
        "selected_by_val": selected,
        "top_by_val": sorted(candidates, key=lambda row: row["val"]["rmse"])[:10],
        "top_by_test_oracle": sorted(candidates, key=lambda row: row["test"]["rmse"])[:10],
    }
    with (args.save_dir / "learned_phase_gate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
