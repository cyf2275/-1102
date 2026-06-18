from __future__ import annotations

import json
from pathlib import Path


def metric(summary, key):
    value = summary.get(key)
    if isinstance(value, dict):
        return value.get("mean")
    if isinstance(value, (int, float)):
        return value
    return None


def main():
    base = Path("results")
    rows = []
    for path in list(base.glob("**/summary.json")) + list(base.glob("**/evaluation/summary.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(summary, dict):
            continue
        rmse = metric(summary, "rmse")
        if rmse is None:
            continue
        rows.append(
            (
                float(rmse),
                str(path),
                metric(summary, "mae"),
                metric(summary, "edge_rmse"),
                metric(summary, "normal_deg"),
                metric(summary, "ssim"),
            )
        )

    print("top depth summaries")
    for row in sorted(rows, key=lambda item: item[0])[:50]:
        print("\t".join(str(x) for x in row))

    print("\nphase related summaries")
    for path in sorted(base.glob("**/*phase*summary*.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(path, exc)
            continue
        if isinstance(summary, dict):
            print(f"{path}\t{json.dumps(summary, ensure_ascii=False)[:1200]}")


if __name__ == "__main__":
    main()
