from __future__ import annotations

import json
from pathlib import Path


def metric_mean(value):
    if isinstance(value, dict):
        value = value.get("mean")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def add_metric(rows, path, label, block):
    if not isinstance(block, dict):
        return
    rmse = metric_mean(block.get("rmse"))
    if rmse is None:
        return
    rows.append(
        {
            "rmse": rmse,
            "path": str(path),
            "label": label,
            "mae": metric_mean(block.get("mae")),
            "edge_rmse": metric_mean(block.get("edge_rmse")),
            "normal_deg": metric_mean(block.get("normal_deg")),
            "ssim": metric_mean(block.get("ssim")),
        }
    )


def scan_path(path, rows):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(data, dict):
        return

    add_metric(rows, path, "root", data)

    selected = data.get("selected_by_val")
    if isinstance(selected, dict) and isinstance(selected.get("test_rmse"), (int, float)):
        rows.append(
            {
                "rmse": float(selected["test_rmse"]),
                "path": str(path),
                "label": "selected_by_val",
                "mae": None,
                "edge_rmse": selected.get("test_edge_rmse"),
                "normal_deg": selected.get("test_normal_deg"),
                "ssim": None,
            }
        )

    methods = data.get("methods")
    if isinstance(methods, dict):
        for name, block in methods.items():
            add_metric(rows, path, f"methods.{name}", block)

    for split in ("test", "val"):
        split_block = data.get(split)
        if not isinstance(split_block, dict):
            continue
        add_metric(rows, path, split, split_block)
        for key, value in split_block.items():
            if isinstance(value, dict):
                add_metric(rows, path, f"{split}.{key}", value)
                if key in {"weights", "ridge"}:
                    for sub_key, sub_value in value.items():
                        add_metric(rows, path, f"{split}.{key}.{sub_key}", sub_value)
                else:
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            add_metric(rows, path, f"{split}.{key}.{sub_key}", sub_value)


def main():
    base = Path("results")
    rows = []
    for path in sorted(base.glob("**/*summary*.json")):
        scan_path(path, rows)

    rows.sort(key=lambda row: row["rmse"])
    print("rmse\tlabel\tmae\tedge_rmse\tnormal_deg\tssim\tpath")
    for row in rows[:120]:
        def fmt(value):
            return "" if value is None else f"{float(value):.9f}"

        print(
            f"{row['rmse']:.9f}\t{row['label']}\t"
            f"{fmt(row['mae'])}\t"
            f"{fmt(row['edge_rmse'])}\t"
            f"{fmt(row['normal_deg'])}\t"
            f"{fmt(row['ssim'])}\t"
            f"{row['path']}"
        )


if __name__ == "__main__":
    main()
