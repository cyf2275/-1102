#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
BASE_PREFIX=base_c4_adapter

run_subset() {
  local name="$1"
  shift
  local seeds=("$@")
  local out="/root/autodl-tmp/diffusion_fpp_v5/results/${name}_adaptive_features_a050"
  local checkpoints=()
  for seed in "${seeds[@]}"; do
    checkpoints+=("/root/autodl-tmp/diffusion_fpp_v5/results/pip_d17_lowt015_seed${seed}_base_residual_e1_gate050_lr3e5/checkpoints/best.pt")
  done
  echo "===== D19 EVAL ${name} seeds=${seeds[*]} $(date '+%F %T') ====="
  /root/miniconda3/bin/python eval_seed_ensemble_adaptive_features.py \
    --checkpoints "${checkpoints[@]}" \
    --cache_dir "$CACHE" \
    --base_prefix "$BASE_PREFIX" \
    --save_dir "$out" \
    --splits val test \
    --image_h 960 \
    --image_w 960 \
    --ddim_steps 20 \
    --eval_batch_size 1 \
    --num_workers 0 \
    --start_ratio 0.05 \
    --alpha 0.5 \
    --require_cache

  /root/miniconda3/bin/python select_prefix_adaptive_gate.py \
    --val_csv "$out/val_adaptive_features.csv" \
    --test_csv "$out/test_adaptive_features.csv" \
    --save_json "$out/selected_gate_summary.json" \
    --candidate_prefix ensemble \
    --min_selected 3
}

echo "===== D19 LOW-T015 VAL-TOP SUBSET ENSEMBLES START $(date '+%F %T') ====="
run_subset pip_d19_lowt015_valtop2_seed2_0 2 0
run_subset pip_d19_lowt015_valtop3_seed2_0_42 2 0 42
run_subset pip_d19_lowt015_valtop4_seed2_0_42_1 2 0 42 1

/root/miniconda3/bin/python - <<'PY'
import csv
import json
from pathlib import Path

import numpy as np

METRIC_KEYS = ["rmse", "mae", "edge_rmse", "normal_deg", "ssim"]
base = Path("/root/autodl-tmp/diffusion_fpp_v5/results")
names = [
    "pip_d19_lowt015_valtop2_seed2_0",
    "pip_d19_lowt015_valtop3_seed2_0_42",
    "pip_d19_lowt015_valtop4_seed2_0_42_1",
]

def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            row[key] = int(value) if key == "sample" else float(value)
    return rows

def summarize_candidate(rows, feature, rule, threshold):
    def selected(row):
        return row[feature] <= threshold if rule == "le" else row[feature] >= threshold
    out = {"selected": sum(1 for row in rows if selected(row))}
    for metric in METRIC_KEYS:
        vals = [
            row[f"{'ensemble' if selected(row) else 'base'}_{metric}"]
            for row in rows
        ]
        out[metric] = float(np.mean(vals))
    return out

summary = []
for name in names:
    out = base / f"{name}_adaptive_features_a050"
    gate = json.loads((out / "selected_gate_summary.json").read_text(encoding="utf-8"))
    test_rows = read_rows(out / "test_adaptive_features.csv")
    candidate_tests = []
    for c in gate["all_val_candidates"]:
        candidate_tests.append({
            "feature": c["feature"],
            "rule": c["rule"],
            "threshold": c["threshold"],
            "val_rmse": c["summary"]["rmse"]["mean"],
            "test": summarize_candidate(test_rows, c["feature"], c["rule"], c["threshold"]),
        })
    summary.append({
        "name": name,
        "selected_gate": gate["selected_gate"],
        "val_base_rmse": gate["val"]["base"]["rmse"]["mean"],
        "val_ensemble_rmse": gate["val"]["ensemble"]["rmse"]["mean"],
        "val_gated_rmse": gate["val"]["gated"]["rmse"]["mean"],
        "test_base_rmse": gate["test"]["base"]["rmse"]["mean"],
        "test_ensemble_rmse": gate["test"]["ensemble"]["rmse"]["mean"],
        "test_gated_rmse": gate["test"]["gated"]["rmse"]["mean"],
        "test_selected": gate["test"]["gated"]["selected"],
        "candidate_tests": candidate_tests,
    })

out_path = base / "pip_d19_lowt015_valtop_subset_summary.json"
out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "===== D19 ALL DONE $(date '+%F %T') ====="
