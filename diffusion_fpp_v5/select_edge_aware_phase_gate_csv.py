from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRIC_KEYS = ["rmse", "mae", "edge_rmse", "normal_deg", "ssim"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(value: str | float) -> float:
    return float(value)


def summarize(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        raise ValueError("no rows to summarize")
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in METRIC_KEYS
    }


def format_weight(weight: float) -> str:
    text = f"{weight:.8f}".rstrip("0").rstrip(".")
    return text if text else "0"


def build_weight_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        sample = str(row["sample"])
        weight = format_weight(f(row["phase_weight"]))
        out[(sample, weight)] = row
    return out


def select_split(
    hier_rows: list[dict[str, str]],
    fused_rows: list[dict[str, str]],
    edge_tau: float,
    low_weight: float,
    high_weight: float,
    edge_op: str = ">=",
    delta_min: float | None = None,
    delta_max: float | None = None,
    phase_conf_min: float | None = None,
    phase_conf_max: float | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    fused_by_key = build_weight_index(fused_rows)
    selected: list[dict[str, object]] = []
    counts: dict[str, int] = {}

    low_weight_key = format_weight(low_weight)
    high_weight_key = format_weight(high_weight)

    for hrow in sorted(hier_rows, key=lambda row: int(float(row["sample"]))):
        sample = str(hrow["sample"])
        edge_mean = f(hrow["edge_mean"])
        delta_mean = f(hrow["delta_mean"])
        phase_conf_mean = f(hrow["phase_conf_mean"])
        if edge_op == ">=":
            selected_flag = edge_mean >= edge_tau
        elif edge_op == "<=":
            selected_flag = edge_mean <= edge_tau
        else:
            raise ValueError(f"unsupported edge_op: {edge_op}")
        if delta_min is not None:
            selected_flag = selected_flag and delta_mean >= delta_min
        if delta_max is not None:
            selected_flag = selected_flag and delta_mean <= delta_max
        if phase_conf_min is not None:
            selected_flag = selected_flag and phase_conf_mean >= phase_conf_min
        if phase_conf_max is not None:
            selected_flag = selected_flag and phase_conf_mean <= phase_conf_max

        chosen_weight = high_weight if selected_flag else low_weight
        chosen_key = high_weight_key if selected_flag else low_weight_key
        key = (sample, chosen_key)
        if key not in fused_by_key:
            available = sorted(weight for s, weight in fused_by_key if s == sample)
            raise KeyError(
                f"missing fused row for sample={sample}, weight={chosen_key}; "
                f"available weights={available}"
            )
        fused = fused_by_key[key]
        counts[chosen_key] = counts.get(chosen_key, 0) + 1
        out = {
            "sample": sample,
            "edge_mean": edge_mean,
            "phase_conf_mean": phase_conf_mean,
            "delta_mean": delta_mean,
            "selected_phase_weight": chosen_weight,
            "rule": "selected_phase" if selected_flag else "kept_depth",
        }
        for metric in METRIC_KEYS:
            out[metric] = f(fused[metric])
        selected.append(out)

    summary = summarize(selected)
    summary["num_samples"] = len(selected)
    return selected, {"metrics": summary, "weight_counts": counts}


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "sample",
        "edge_mean",
        "phase_conf_mean",
        "delta_mean",
        "selected_phase_weight",
        "rule",
        *METRIC_KEYS,
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in keys})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select fixed phase-fusion rows using a physics-motivated edge-aware sample gate."
    )
    parser.add_argument("--val_hier_csv", type=Path, required=True)
    parser.add_argument("--test_hier_csv", type=Path, required=True)
    parser.add_argument("--val_fused_csv", type=Path, required=True)
    parser.add_argument("--test_fused_csv", type=Path, required=True)
    parser.add_argument("--edge_tau", type=float, default=0.45)
    parser.add_argument("--edge_op", choices=[">=", "<="], default=">=")
    parser.add_argument("--delta_min", type=float, default=None)
    parser.add_argument("--delta_max", type=float, default=None)
    parser.add_argument("--phase_conf_min", type=float, default=None)
    parser.add_argument("--phase_conf_max", type=float, default=None)
    parser.add_argument("--low_weight", type=float, default=0.0)
    parser.add_argument("--high_weight", type=float, default=0.45)
    parser.add_argument("--save_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    val_rows, val_summary = select_split(
        read_rows(args.val_hier_csv),
        read_rows(args.val_fused_csv),
        args.edge_tau,
        args.low_weight,
        args.high_weight,
        args.edge_op,
        args.delta_min,
        args.delta_max,
        args.phase_conf_min,
        args.phase_conf_max,
    )
    test_rows, test_summary = select_split(
        read_rows(args.test_hier_csv),
        read_rows(args.test_fused_csv),
        args.edge_tau,
        args.low_weight,
        args.high_weight,
        args.edge_op,
        args.delta_min,
        args.delta_max,
        args.phase_conf_min,
        args.phase_conf_max,
    )

    write_rows(val_rows, args.save_dir / "val_selected_rows.csv")
    write_rows(test_rows, args.save_dir / "test_selected_rows.csv")

    summary = {
        "method": "E79 edge-aware adaptive phase gate",
        "rule": {
            "description": "Use high phase-fusion weight only for high-edge samples; keep depth posterior for low-edge samples.",
            "edge_tau": args.edge_tau,
            "edge_op": args.edge_op,
            "delta_min": args.delta_min,
            "delta_max": args.delta_max,
            "phase_conf_min": args.phase_conf_min,
            "phase_conf_max": args.phase_conf_max,
            "low_weight": args.low_weight,
            "high_weight": args.high_weight,
        },
        "inputs": {
            "val_hier_csv": str(args.val_hier_csv),
            "test_hier_csv": str(args.test_hier_csv),
            "val_fused_csv": str(args.val_fused_csv),
            "test_fused_csv": str(args.test_fused_csv),
        },
        "val": val_summary,
        "test": test_summary,
    }
    with (args.save_dir / "edge_aware_phase_gate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
