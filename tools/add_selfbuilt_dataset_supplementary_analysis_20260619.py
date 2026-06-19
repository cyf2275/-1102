from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path.cwd() / "cloud_results" / "A_20260619_selfbuilt_dataset_paper_experiments"
DIRECT = ROOT / "A_20260619_formal_strong_backbone_direct_seed012"
SELECTOR = ROOT / "A_20260619_formal_attention_unet_ours_selector_seed012"
ASSET_DIR = ROOT / "paper_summary_assets"
OUT_JSON = ROOT / "paper_per_sample_significance.json"
OUT_CSV = ROOT / "paper_per_sample_significance.csv"
PER_SAMPLE_MEAN_CSV = ROOT / "paper_per_sample_mean_rmse.csv"
PLOT_PATH = ASSET_DIR / "per_sample_paired_improvement.png"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mean(xs: list[float]) -> float:
    return float(statistics.mean(xs))


def sample_std(xs: list[float]) -> float:
    return float(statistics.stdev(xs)) if len(xs) > 1 else 0.0


def exact_sign_pvalue(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * cdf))


def bootstrap_ci(values: np.ndarray, n_boot: int = 20000, seed: int = 20260619) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = values[idx].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def load_direct_means() -> dict[str, dict[str, dict[str, float]]]:
    # output: split -> method -> sample_id -> 3-seed mean RMSE
    raw: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for arch in ["attention_unet", "unetpp"]:
        for seed in [0, 1, 2]:
            base = DIRECT / f"{arch}_seed{seed}" / "evaluation"
            for split, filename in [("test", "per_sample_metrics.csv"), ("ood", "ood_per_sample_metrics.csv")]:
                for row in read_csv(base / filename):
                    raw[split][arch][row["sample_id"]].append(float(row["object_rmse"]))
    out: dict[str, dict[str, dict[str, float]]] = {}
    for split, methods in raw.items():
        out[split] = {}
        for method, samples in methods.items():
            out[split][method] = {sid: mean(vals) for sid, vals in samples.items()}
    return out


def load_selector_means() -> dict[str, dict[str, dict[str, float]]]:
    raw: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    columns = {
        "anchor": "anchor_rmse",
        "rule": "rule_rmse",
        "mlp": "mlp_rmse",
        "x_phase": "x_rmse",
        "refined": "refined_rmse",
        "sample_rcpc": "sample_rcpc_rmse",
    }
    for seed in [0, 1, 2]:
        path = SELECTOR / f"seed{seed}" / f"reliability_selector_seed{seed}" / "reliability_selector_per_sample.csv"
        for row in read_csv(path):
            split = row["split"]
            if split not in {"test", "ood"}:
                continue
            sid = row["sample_id"]
            for method, col in columns.items():
                raw[split][method][sid].append(float(row[col]))
    out: dict[str, dict[str, dict[str, float]]] = {}
    for split, methods in raw.items():
        out[split] = {}
        for method, samples in methods.items():
            out[split][method] = {sid: mean(vals) for sid, vals in samples.items()}
    return out


def compare(
    split: str,
    baseline_name: str,
    candidate_name: str,
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, Any]:
    ids = sorted(set(baseline) & set(candidate))
    base = np.array([baseline[i] for i in ids], dtype=float)
    cand = np.array([candidate[i] for i in ids], dtype=float)
    diff = base - cand
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    ties = int((diff == 0).sum())
    ci_lo, ci_hi = bootstrap_ci(diff)
    return {
        "split": split,
        "baseline": baseline_name,
        "candidate": candidate_name,
        "n": len(ids),
        "baseline_mean": float(base.mean()),
        "candidate_mean": float(cand.mean()),
        "mean_improvement_mm": float(diff.mean()),
        "median_improvement_mm": float(np.median(diff)),
        "relative_improvement_percent": float(diff.mean() / base.mean() * 100.0),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": float(wins / len(ids)) if ids else float("nan"),
        "sign_test_p_two_sided": exact_sign_pvalue(wins, losses),
        "bootstrap_95ci_mean_improvement_mm": [ci_lo, ci_hi],
    }


def write_per_sample_means(direct: dict[str, Any], selector: dict[str, Any]) -> None:
    rows = []
    for split in ["test", "ood"]:
        sample_ids = sorted(set().union(*[set(v) for v in direct[split].values()], *[set(v) for v in selector[split].values()]))
        for sid in sample_ids:
            row: dict[str, Any] = {"split": split, "sample_id": sid}
            for method in ["attention_unet", "unetpp"]:
                row[method] = direct[split].get(method, {}).get(sid, "")
            for method in ["anchor", "rule", "mlp", "x_phase", "refined", "sample_rcpc"]:
                row[method] = selector[split].get(method, {}).get(sid, "")
            rows.append(row)
    with PER_SAMPLE_MEAN_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "split",
            "sample_id",
            "attention_unet",
            "unetpp",
            "anchor",
            "rule",
            "mlp",
            "x_phase",
            "refined",
            "sample_rcpc",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(results: list[dict[str, Any]]) -> None:
    keep = [
        r
        for r in results
        if r["baseline"] in {"attention_unet", "unetpp"} and r["candidate"] in {"anchor", "rule", "mlp"}
    ]
    labels = [f"{r['split']}\\n{r['baseline']}→{r['candidate']}" for r in keep]
    means = [r["mean_improvement_mm"] for r in keep]
    lo = [r["mean_improvement_mm"] - r["bootstrap_95ci_mean_improvement_mm"][0] for r in keep]
    hi = [r["bootstrap_95ci_mean_improvement_mm"][1] - r["mean_improvement_mm"] for r in keep]
    colors = ["#4C78A8" if r["split"] == "test" else "#F58518" for r in keep]
    x = np.arange(len(keep))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axhline(0, color="black", linewidth=1)
    ax.bar(x, means, color=colors)
    ax.errorbar(x, means, yerr=[lo, hi], fmt="none", ecolor="black", capsize=3, linewidth=1)
    ax.set_ylabel("Paired mean RMSE improvement (mm)")
    ax.set_title("Per-sample paired improvement after averaging seeds")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=220)
    plt.close(fig)


def main() -> None:
    direct = load_direct_means()
    selector = load_selector_means()
    write_per_sample_means(direct, selector)
    comparisons = []
    for split in ["test", "ood"]:
        for baseline in ["attention_unet", "unetpp"]:
            for candidate in ["anchor", "rule", "mlp"]:
                comparisons.append(compare(split, baseline, candidate, direct[split][baseline], selector[split][candidate]))
    OUT_JSON.write_text(json.dumps(comparisons, ensure_ascii=False, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "split",
            "baseline",
            "candidate",
            "n",
            "baseline_mean",
            "candidate_mean",
            "mean_improvement_mm",
            "median_improvement_mm",
            "relative_improvement_percent",
            "wins",
            "losses",
            "ties",
            "win_rate",
            "sign_test_p_two_sided",
            "bootstrap_95ci_mean_improvement_mm",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in comparisons:
            row = dict(r)
            row["bootstrap_95ci_mean_improvement_mm"] = json.dumps(row["bootstrap_95ci_mean_improvement_mm"])
            writer.writerow(row)
    make_plot(comparisons)
    print(json.dumps({
        "significance_json": str(OUT_JSON),
        "significance_csv": str(OUT_CSV),
        "per_sample_mean_csv": str(PER_SAMPLE_MEAN_CSV),
        "plot": str(PLOT_PATH),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
