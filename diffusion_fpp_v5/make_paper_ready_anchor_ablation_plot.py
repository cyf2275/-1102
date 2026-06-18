from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_aggregate(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["aggregate"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir",
        type=Path,
        default=Path("cloud_results") / "A_20260618_paper_ready_anchor_ablation",
    )
    parser.add_argument("--out_name", default="paper_ready_anchor_ablation_rmse")
    args = parser.parse_args()

    result_dir = args.result_dir
    fixed = load_aggregate(result_dir / "anchor_ablation_fixed_posterior_summary.json")
    full = load_aggregate(result_dir / "anchor_ablation_fullchain_summary.json")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    configs = [
        ("Fixed posterior", fixed, ["x_phase", "base", "base_x_mean"]),
        ("Full chain", full, ["x_phase", "base_x_mean"]),
    ]
    colors = {
        "anchor": "#6b7280",
        "rule": "#0f766e",
        "mlp": "#b45309",
        "local_oracle_anchor_refined": "#7c3aed",
    }
    labels = {
        "anchor": "Anchor",
        "rule": "Rule RCPC",
        "mlp": "MLP RCPC",
        "local_oracle_anchor_refined": "Local oracle",
    }

    for ax, (title, data, anchors) in zip(axes, configs):
        groups = [(anchor, split) for anchor in anchors for split in ["test", "ood"]]
        x = np.arange(len(groups))
        width = 0.18
        keys = ["anchor", "rule", "mlp", "local_oracle_anchor_refined"]
        for i, key in enumerate(keys):
            means = [data[a][s][key]["mean"] for a, s in groups]
            stds = [data[a][s][key]["std"] for a, s in groups]
            ax.bar(
                x + (i - 1.5) * width,
                means,
                width,
                yerr=stds,
                capsize=2,
                color=colors[key],
                label=labels[key],
                linewidth=0.4,
                edgecolor="black",
            )
        xt = [f"{a.replace('_', '-')}\n{s.upper()}" for a, s in groups]
        ax.set_xticks(x)
        ax.set_xticklabels(xt, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_ylim(1.1, 1.9)

    axes[0].set_ylabel("Object RMSE (mm)")
    handles, labs = axes[1].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    for ext in ["png", "svg", "pdf"]:
        fig.savefig(result_dir / f"{args.out_name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(result_dir / f"{args.out_name}.png")


if __name__ == "__main__":
    main()
