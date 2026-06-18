#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
BASE_PREFIX=base_c4_adapter
OUT=/root/autodl-tmp/diffusion_fpp_v5/results/pip_d15_seed_ensemble_adaptive_features_a050

checkpoints=()
for seed in 0 1 2 3 4 42 123 456; do
  checkpoints+=("/root/autodl-tmp/diffusion_fpp_v5/results/pip_d8_seed${seed}_base_residual_e1_gate050_lr3e5/checkpoints/best.pt")
done

echo "===== D15 FEATURE GENERATION START $(date '+%F %T') ====="
/root/miniconda3/bin/python eval_seed_ensemble_adaptive_features.py \
  --checkpoints "${checkpoints[@]}" \
  --cache_dir "$CACHE" \
  --base_prefix "$BASE_PREFIX" \
  --save_dir "$OUT" \
  --splits train val test \
  --image_h 960 \
  --image_w 960 \
  --ddim_steps 20 \
  --eval_batch_size 1 \
  --num_workers 0 \
  --start_ratio 0.05 \
  --alpha 0.5 \
  --require_cache

echo "===== D15 LEARNED GATE START $(date '+%F %T') ====="
/root/miniconda3/bin/python select_learned_acceptance_gate.py \
  --train_csv "$OUT/train_adaptive_features.csv" \
  --val_csv "$OUT/val_adaptive_features.csv" \
  --test_csv "$OUT/test_adaptive_features.csv" \
  --save_json "$OUT/learned_acceptance_summary.json" \
  --min_selected 3 \
  --ridge_alphas "0.001 0.01 0.1 1.0 10.0 100.0"

echo "===== D15 ALL DONE $(date '+%F %T') ====="
