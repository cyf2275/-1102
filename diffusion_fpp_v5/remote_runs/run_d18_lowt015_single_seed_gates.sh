#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
BASE_PREFIX=base_c4_adapter
seeds=(0 1 2 3 4 42 123 456)

echo "===== D18 LOW-T015 SINGLE-SEED GATE EVAL START $(date '+%F %T') ====="
for seed in "${seeds[@]}"; do
  run_dir="/root/autodl-tmp/diffusion_fpp_v5/results/pip_d17_lowt015_seed${seed}_base_residual_e1_gate050_lr3e5"
  feat_dir="/root/autodl-tmp/diffusion_fpp_v5/results/pip_d18_lowt015_seed${seed}_adaptive_features_a050"
  echo "===== D18 EVAL seed=${seed} $(date '+%F %T') ====="
  for split in val test; do
    /root/miniconda3/bin/python eval_adaptive_blend_features.py \
      --checkpoint "$run_dir/checkpoints/best.pt" \
      --cache_dir "$CACHE" \
      --base_prefix "$BASE_PREFIX" \
      --save_dir "$feat_dir" \
      --split "$split" \
      --image_h 960 \
      --image_w 960 \
      --ddim_steps 20 \
      --ensemble 1 \
      --eval_batch_size 1 \
      --num_workers 0 \
      --start_ratio 0.05 \
      --alpha 0.5 \
      --require_cache
  done
  /root/miniconda3/bin/python select_adaptive_gate.py \
    --val_csv "$feat_dir/val_adaptive_features.csv" \
    --test_csv "$feat_dir/test_adaptive_features.csv" \
    --save_json "$feat_dir/selected_gate_summary.json" \
    --min_selected 3
done

/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path

base = Path("/root/autodl-tmp/diffusion_fpp_v5/results")
seeds = [0, 1, 2, 3, 4, 42, 123, 456]
rows = []
for seed in seeds:
    path = base / f"pip_d18_lowt015_seed{seed}_adaptive_features_a050" / "selected_gate_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "seed": seed,
        "gate": data["selected_gate"],
        "val_base_rmse": data["val"]["base"]["rmse"]["mean"],
        "val_blend_rmse": data["val"]["blend"]["rmse"]["mean"],
        "val_gated_rmse": data["val"]["gated"]["rmse"]["mean"],
        "test_base_rmse": data["test"]["base"]["rmse"]["mean"],
        "test_blend_rmse": data["test"]["blend"]["rmse"]["mean"],
        "test_gated_rmse": data["test"]["gated"]["rmse"]["mean"],
        "test_selected": data["test"]["gated"]["selected"],
    })
summary = {
    "seeds": rows,
    "mean_test_blend_rmse": sum(r["test_blend_rmse"] for r in rows) / len(rows),
    "mean_test_gated_rmse": sum(r["test_gated_rmse"] for r in rows) / len(rows),
    "wins_vs_base": sum(r["test_gated_rmse"] < r["test_base_rmse"] for r in rows),
    "wins_vs_d8_ensemble_gate_18_603": sum(r["test_gated_rmse"] < 18.60298638343811 for r in rows),
    "best_seed": min(rows, key=lambda r: r["test_gated_rmse"]),
}
out = base / "pip_d18_lowt015_single_seed_gate_summary.json"
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "===== D18 ALL DONE $(date '+%F %T') ====="
