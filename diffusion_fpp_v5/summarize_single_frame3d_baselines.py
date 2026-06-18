"""Summarize quick single_frame3d backbone baselines.

The script reads baseline run folders produced by
`train_single_frame3d_backbone_baselines.py` and optionally appends existing
phase-posterior/RCPC anchor-ablation results for side-by-side quick screening.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_from_summary(summary: Dict[str, object], split: str, roi: str = "object", metric: str = "rmse") -> float:
    splits = summary.get("splits", {})
    if isinstance(splits, dict):
        data = splits.get(split)
        if isinstance(data, dict):
            roi_data = data.get(roi)
            if isinstance(roi_data, dict):
                metric_data = roi_data.get(metric)
                if isinstance(metric_data, dict) and "mean" in metric_data:
                    return float(metric_data["mean"])
    data = summary.get(split)
    if isinstance(data, dict):
        roi_data = data.get(roi)
        if isinstance(roi_data, dict):
            metric_data = roi_data.get(metric)
            if isinstance(metric_data, dict) and "mean" in metric_data:
                return float(metric_data["mean"])
    return float("nan")


def collect_baselines(result_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary_path in sorted(result_dir.glob("*/evaluation/summary.json")):
        summary = read_json(summary_path)
        method = str(summary.get("arch") or summary_path.parents[1].name)
        rows.append({
            "method": method,
            "source": str(summary_path),
            "kind": "quick_1seed_baseline",
            "seed": summary.get("seed", ""),
            "epochs": summary.get("epochs", ""),
            "test_object_rmse": metric_from_summary(summary, "test", "object", "rmse"),
            "test_valid_rmse": metric_from_summary(summary, "test", "valid", "rmse"),
            "ood_object_rmse": metric_from_summary(summary, "ood", "object", "rmse"),
            "ood_valid_rmse": metric_from_summary(summary, "ood", "valid", "rmse"),
            "best_val_object_rmse": float(summary.get("best_val_object_rmse", float("nan"))),
            "paper_final": bool(summary.get("paper_final", False)),
        })
    return rows


def mean_for_ours(rows: Iterable[Dict[str, object]], anchor_mode: str, split: str, field: str) -> float:
    vals = [
        float(r[field])
        for r in rows
        if str(r.get("anchor_mode")) == anchor_mode and str(r.get("split")) == split and field in r
    ]
    return float(np.mean(vals)) if vals else float("nan")


def append_ours(rows: List[Dict[str, object]], path: Path, label_prefix: str, anchor_mode: str = "base_x_mean") -> None:
    if not path.exists():
        return
    data = read_json(path)
    raw_rows = data.get("rows", [])
    if not isinstance(raw_rows, list):
        return
    for field, label in [
        ("anchor", f"{label_prefix} anchor base+x mean"),
        ("mlp", f"{label_prefix} full MLP"),
        ("rule", f"{label_prefix} full rule"),
        ("refined", f"{label_prefix} diffusion candidate"),
    ]:
        rows.append({
            "method": label,
            "source": str(path),
            "kind": "existing_ours_reference",
            "seed": "0/1/2",
            "epochs": "",
            "test_object_rmse": mean_for_ours(raw_rows, anchor_mode, "test", field),
            "test_valid_rmse": float("nan"),
            "ood_object_rmse": mean_for_ours(raw_rows, anchor_mode, "ood", field),
            "ood_valid_rmse": float("nan"),
            "best_val_object_rmse": mean_for_ours(raw_rows, anchor_mode, "val", field),
            "paper_final": False,
        })


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    fields = [
        "method",
        "kind",
        "seed",
        "epochs",
        "best_val_object_rmse",
        "test_object_rmse",
        "test_valid_rmse",
        "ood_object_rmse",
        "ood_valid_rmse",
        "paper_final",
        "source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def fmt(x: object) -> str:
    try:
        v = float(x)
    except Exception:
        return ""
    return "" if not np.isfinite(v) else f"{v:.4f}"


def write_report(rows: List[Dict[str, object]], path: Path) -> None:
    ordered = sorted(rows, key=lambda r: float(r.get("test_object_rmse", float("inf"))))
    lines = [
        "# Single-frame3D Backbone Baseline Quick Screening",
        "",
        "This is a 1-seed quick screening table, not the final paper table.",
        "All baseline rows use `input_vertical_0120.bmp` as legal test-time input and target `depth_z`.",
        "",
        "| method | kind | seed | epochs | val object RMSE | test object RMSE | OOD object RMSE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['method']} | {row['kind']} | {row['seed']} | {row['epochs']} | "
            f"{fmt(row['best_val_object_rmse'])} | {fmt(row['test_object_rmse'])} | {fmt(row['ood_object_rmse'])} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- Use this table to decide which direct backbones deserve 3-seed, longer training.",
        "- Existing ours rows may come from prior 3-seed anchor-ablation summaries and are included only as reference.",
        "- If a 40-epoch baseline is still improving on validation, it must be retrained to convergence before paper use.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_rows(rows: List[Dict[str, object]], path: Path) -> None:
    keep = [r for r in rows if np.isfinite(float(r.get("test_object_rmse", float("nan"))))]
    if not keep:
        return
    keep = sorted(keep, key=lambda r: float(r["test_object_rmse"]))
    labels = [str(r["method"]) for r in keep]
    test = [float(r["test_object_rmse"]) for r in keep]
    ood = [float(r["ood_object_rmse"]) if np.isfinite(float(r.get("ood_object_rmse", float("nan")))) else np.nan for r in keep]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8.0, 0.62 * len(labels)), 4.6), constrained_layout=True)
    ax.bar(x - width / 2, test, width, label="test")
    ax.bar(x + width / 2, ood, width, label="OOD 61-64")
    ax.set_ylabel("Object RMSE")
    ax.set_title("Single-frame3D quick baseline screening")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--ours_fullchain_json", default="")
    parser.add_argument("--ours_fixed_json", default="")
    parser.add_argument("--save_prefix", default="baseline_comparison_quick1seed")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    rows = collect_baselines(result_dir)
    if args.ours_fullchain_json:
        append_ours(rows, Path(args.ours_fullchain_json), "ours fullchain")
    if args.ours_fixed_json:
        append_ours(rows, Path(args.ours_fixed_json), "ours fixed posterior")
    rows = sorted(rows, key=lambda r: (str(r["kind"]), float(r.get("test_object_rmse", float("inf")))))

    write_csv(rows, result_dir / f"{args.save_prefix}_summary.csv")
    write_report(rows, result_dir / f"{args.save_prefix}_report.md")
    plot_rows(rows, result_dir / f"{args.save_prefix}_rmse.png")
    (result_dir / f"{args.save_prefix}_summary.json").write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
