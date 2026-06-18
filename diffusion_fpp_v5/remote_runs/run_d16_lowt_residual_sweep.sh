#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
BASE_PREFIX=base_c4_adapter

run_one() {
  local name="$1"
  local tmax="$2"
  local run_dir="/root/autodl-tmp/diffusion_fpp_v5/results/${name}"
  local feat_dir="/root/autodl-tmp/diffusion_fpp_v5/results/${name}_adaptive_features_a050"

  echo "===== D16 TRAIN ${name} tmax=${tmax} $(date '+%F %T') ====="
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
    --train_t_max_ratio "$tmax" \
    --lr 3e-5 \
    --base_residual_gate 0.5 \
    --seed 0

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
}

run_one pip_d16_lowt005_base_residual_e1_seed0 0.05
run_one pip_d16_lowt010_base_residual_e1_seed0 0.10
run_one pip_d16_lowt015_base_residual_e1_seed0 0.15

echo "===== D16 ALL DONE $(date '+%F %T') ====="
