"""Create compact comparison tables from available result summaries."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_summary(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def metric(summary, key):
    val = summary.get(key)
    if isinstance(val, dict):
        return val.get("mean"), val.get("std")
    return val, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="/root/diffusion_fpp_v5/results")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    rows = [
        {"name": "UNet_baseline_reported", "summary": None, "rmse": 7.49, "note": "reported baseline"},
        {"name": "v3_existing_eval", "summary": out_dir / "unified_eval" / "v3_existing" / "summary.json", "note": "existing checkpoint, unified eval"},
        {"name": "v5_sweep_best", "summary": out_dir / "fringe_physics" / "sampling_sweep" / "test_best" / "summary.json", "note": "full-val selected sampling"},
        {"name": "v35_hilbert_dwt", "summary": out_dir / "v35_hilbert_dwt" / "evaluation" / "summary.json", "note": "phase/edge condition"},
    ]
    table = []
    for row in rows:
        record = {"name": row["name"], "note": row["note"]}
        if row.get("summary") is None:
            record.update({"rmse": row["rmse"], "rmse_std": "", "mae": "", "edge_rmse": "", "normal_deg": "", "ssim": ""})
        else:
            summary = load_summary(row["summary"])
            if summary is None:
                record.update({"rmse": "", "rmse_std": "", "mae": "", "edge_rmse": "", "normal_deg": "", "ssim": ""})
            else:
                rmse, rmse_std = metric(summary, "rmse")
                mae, _ = metric(summary, "mae")
                edge, _ = metric(summary, "edge_rmse")
                normal, _ = metric(summary, "normal_deg")
                ssim, _ = metric(summary, "ssim")
                record.update({"rmse": rmse, "rmse_std": rmse_std, "mae": mae,
                               "edge_rmse": edge, "normal_deg": normal, "ssim": ssim})
        table.append(record)
    csv_path = out_dir / "comparison_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "rmse", "rmse_std", "mae", "edge_rmse", "normal_deg", "ssim", "note"])
        writer.writeheader()
        writer.writerows(table)
    md = ["| model | RMSE | MAE | edge RMSE | normal | SSIM | note |",
          "|---|---:|---:|---:|---:|---:|---|"]
    for r in table:
        def fmt(v):
            return "" if v == "" or v is None else f"{float(v):.4f}"
        md.append(f"| {r['name']} | {fmt(r['rmse'])} | {fmt(r['mae'])} | {fmt(r['edge_rmse'])} | {fmt(r['normal_deg'])} | {fmt(r['ssim'])} | {r['note']} |")
    (out_dir / "comparison_table.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
