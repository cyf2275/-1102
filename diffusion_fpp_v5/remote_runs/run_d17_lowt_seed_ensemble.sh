#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
BASE_PREFIX=base_c4_adapter
TMAX="${TMAX:-0.10}"
TAG="${TAG:-010}"
OUT="/root/autodl-tmp/diffusion_fpp_v5/results/pip_d17_lowt${TAG}_seed_ensemble_adaptive_features_a050"

seeds=(0 1 2 3 4 42 123 456)
checkpoints=()

echo "===== D17 LOW-T SEED ENSEMBLE START tag=${TAG} tmax=${TMAX} $(date '+%F %T') ====="
for seed in "${seeds[@]}"; do
  run_dir="/root/autodl-tmp/diffusion_fpp_v5/results/pip_d17_lowt${TAG}_seed${seed}_base_residual_e1_gate050_lr3e5"
  checkpoints+=("${run_dir}/checkpoints/best.pt")
  if [ -f "${run_dir}/checkpoints/best.pt" ]; then
    echo "===== D17 SKIP existing seed=${seed} ${run_dir} ====="
    continue
  fi
  echo "===== D17 TRAIN seed=${seed} tag=${TAG} tmax=${TMAX} $(date '+%F %T') ====="
  /root/miniconda3/bin/python train_pip_lite.py \
    --dataset fpp_ml_bench \
    --cache_dir "$CACHE" \
    --save_dir "$run_dir" \
    --epochs 1 \
    --eval_every 1 \
    --save_every 0 \
    --skip_final_test \
    --batch_size 1 \
    --eval_batch_size 1 \
    --num_workers 8 \
    --require_cache \
    --image_h 960 \
    --image_w 960 \
    --target_mode base_residual \
    --base_prefix "$BASE_PREFIX" \
    --condition_injection adapter \
    --physics_channels 0-8 \
    --adapter_hidden 32 \
    --base_channels 48 \
    --timesteps 200 \
    --ddim_steps 20 \
    --ensemble 1 \
    --sample_start_ratio 0.05 \
    --train_t_min_ratio 0.0 \
    --train_t_max_ratio "$TMAX" \
    --lr 3e-5 \
    --base_residual_gate 0.5 \
    --seed "$seed"
done

echo "===== D17 ENSEMBLE FEATURE EVAL tag=${TAG} $(date '+%F %T') ====="
/root/miniconda3/bin/python eval_seed_ensemble_adaptive_features.py \
  --checkpoints "${checkpoints[@]}" \
  --cache_dir "$CACHE" \
  --base_prefix "$BASE_PREFIX" \
  --save_dir "$OUT" \
  --splits val test \
  --image_h 960 \
  --image_w 960 \
  --ddim_steps 20 \
  --eval_batch_size 1 \
  --num_workers 0 \
  --start_ratio 0.05 \
  --alpha 0.5 \
  --require_cache

/root/miniconda3/bin/python select_adaptive_gate.py \
  --val_csv "$OUT/val_adaptive_features.csv" \
  --test_csv "$OUT/test_adaptive_features.csv" \
  --save_json "$OUT/selected_gate_summary.json" \
  --min_selected 3

echo "===== D17 ALL DONE tag=${TAG} tmax=${TMAX} $(date '+%F %T') ====="
