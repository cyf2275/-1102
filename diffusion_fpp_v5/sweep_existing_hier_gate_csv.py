from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path


METRICS = ("rmse", "mae", "edge_rmse", "normal_deg", "ssim")
BRANCHES = ("base", "diff", "pixel_gated", "hierarchical")


def read_rows(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            if key == "branch":
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def mean(values):
    return sum(values) / max(1, len(values))


def summarize(rows, selector):
    picked = []
    counts = {"base": 0, "diff": 0, "pixel_gated": 0, "hierarchical": 0}
    for row in rows:
        branch = selector(row)
        counts.setdefault(branch, 0)
        counts[branch] += 1
        picked.append(row)
    out = {"n": len(rows), "counts": counts}
    for metric in METRICS:
        vals = [row[f"{selector(row)}_{metric}"] for row in rows]
        out[metric] = {"mean": mean(vals)}
    return out


def summarize_fixed_branch(rows, branch):
    return summarize(rows, lambda _row: branch)


def summarize_existing_hier(rows):
    return summarize(rows, lambda _row: "hierarchical")


def summarize_oracle(rows):
    def selector(row):
        return min(("base", "diff", "pixel_gated"), key=lambda b: row[f"{b}_rmse"])

    return summarize(rows, selector)


def thresholds(rows, key, extra=()):
    vals = sorted(float(r[key]) for r in rows)
    if not vals:
        return sorted(set(extra))
    qs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    picked = []
    n = len(vals)
    for q in qs:
        picked.append(vals[min(n - 1, max(0, round(q * (n - 1))))])
    return sorted(set(round(x, 6) for x in [*picked, *extra]))


def build_selector(cfg):
    def selector(row):
        if (
            cfg["diff_edge_low"] <= row["edge_mean"] <= cfg["diff_edge_high"]
            and cfg["diff_delta_low"] <= row["delta_mean"] <= cfg["diff_delta_high"]
            and row["phase_conf_mean"] >= cfg["diff_conf_low"]
        ):
            return "diff"
        if (
            row["edge_mean"] <= cfg["pixel_edge_high"]
            and row["pixel_selected_frac"] >= cfg["pixel_frac_low"]
            and row["delta_mean"] >= cfg["pixel_delta_low"]
        ):
            return "pixel_gated"
        return "base"

    return selector


def search(val_rows):
    edge_th = thresholds(val_rows, "edge_mean", extra=(0.47, 0.58, 0.62))
    delta_th = thresholds(val_rows, "delta_mean", extra=(0.09, 0.105, 0.12))
    conf_th = thresholds(val_rows, "phase_conf_mean", extra=(0.70, 0.76, 0.80))
    frac_th = thresholds(val_rows, "pixel_selected_frac", extra=(0.0, 0.05, 0.10, 0.20, 0.30))

    diff_ranked = []
    # Stage 1: D47-style high-detail diffusion override, with pixel branch disabled.
    for e0, e1 in product(edge_th, edge_th):
        if e0 > e1:
            continue
        for d0, d1 in product(delta_th, delta_th):
            if d0 > d1:
                continue
            for c0 in conf_th:
                cfg = {
                    "diff_edge_low": e0,
                    "diff_edge_high": e1,
                    "diff_delta_low": d0,
                    "diff_delta_high": d1,
                    "diff_conf_low": c0,
                    "pixel_edge_high": -1.0,
                    "pixel_frac_low": 1.1,
                    "pixel_delta_low": 1.1,
                }
                summary = summarize(val_rows, build_selector(cfg))
                complexity = summary["counts"]["diff"]
                diff_ranked.append((summary["rmse"]["mean"], complexity, cfg, summary))

    diff_ranked.sort(key=lambda item: (item[0], item[1]))
    diff_ranked = diff_ranked[:50]

    # Stage 2: add a bounded low-edge pixel branch on top of the best diffusion gates.
    pixel_cfgs = []
    pixel_cfgs.append(
        {
            "pixel_edge_high": -1.0,
            "pixel_frac_low": 1.1,
            "pixel_delta_low": 1.1,
        }
    )
    for pe in edge_th:
        for pf in frac_th:
            for pd in delta_th:
                pixel_cfgs.append(
                    {
                        "pixel_edge_high": pe,
                        "pixel_frac_low": pf,
                        "pixel_delta_low": pd,
                    }
                )

    best = None
    for _score, _complexity, diff_cfg, _summary in diff_ranked:
        for pix_cfg in pixel_cfgs:
            cfg = {**diff_cfg, **pix_cfg}
            selector = build_selector(cfg)
            summary = summarize(val_rows, selector)
            score = summary["rmse"]["mean"]
            # Prefer simpler gates when RMSE ties nearly exactly.
            complexity = summary["counts"]["diff"] + summary["counts"]["pixel_gated"]
            key = (score, complexity)
            if best is None or key < best["key"]:
                best = {"key": key, "cfg": cfg, "val": summary}
    return best


def write_selected_rows(rows, selector, path):
    keys = [
        "sample",
        "selected_branch",
        "original_branch",
        "edge_mean",
        "delta_mean",
        "phase_conf_mean",
        "pixel_selected_frac",
    ]
    for metric in METRICS:
        keys.append(metric)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            branch = selector(row)
            out = {
                "sample": int(row["sample"]),
                "selected_branch": branch,
                "original_branch": row["branch"],
                "edge_mean": row["edge_mean"],
                "delta_mean": row["delta_mean"],
                "phase_conf_mean": row["phase_conf_mean"],
                "pixel_selected_frac": row["pixel_selected_frac"],
            }
            for metric in METRICS:
                out[metric] = row[f"{branch}_{metric}"]
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()

    val_rows = read_rows(args.val_csv)
    test_rows = read_rows(args.test_csv)
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best = search(val_rows)
    selector = build_selector(best["cfg"])
    result = {
        "purpose": "Post-hoc sample-level gate sweep on saved D47 per-sample metrics. The gate is selected by validation RMSE and then applied unchanged to test.",
        "selected_gate": best["cfg"],
        "val": {
            "base": summarize_fixed_branch(val_rows, "base"),
            "diff": summarize_fixed_branch(val_rows, "diff"),
            "pixel_gated": summarize_fixed_branch(val_rows, "pixel_gated"),
            "original_hierarchical": summarize_existing_hier(val_rows),
            "selected_gate": best["val"],
            "oracle_best_of_base_diff_pixel": summarize_oracle(val_rows),
        },
        "test": {
            "base": summarize_fixed_branch(test_rows, "base"),
            "diff": summarize_fixed_branch(test_rows, "diff"),
            "pixel_gated": summarize_fixed_branch(test_rows, "pixel_gated"),
            "original_hierarchical": summarize_existing_hier(test_rows),
            "selected_gate": summarize(test_rows, selector),
            "oracle_best_of_base_diff_pixel": summarize_oracle(test_rows),
        },
    }

    with open(out_dir / "d47_existing_gate_sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    write_selected_rows(val_rows, selector, out_dir / "val_selected_gate_rows.csv")
    write_selected_rows(test_rows, selector, out_dir / "test_selected_gate_rows.csv")
    print(json.dumps({
        "selected_gate": result["selected_gate"],
        "val_rmse": result["val"]["selected_gate"]["rmse"]["mean"],
        "test_rmse": result["test"]["selected_gate"]["rmse"]["mean"],
        "test_counts": result["test"]["selected_gate"]["counts"],
        "original_test_rmse": result["test"]["original_hierarchical"]["rmse"]["mean"],
        "oracle_test_rmse": result["test"]["oracle_best_of_base_diff_pixel"]["rmse"]["mean"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
