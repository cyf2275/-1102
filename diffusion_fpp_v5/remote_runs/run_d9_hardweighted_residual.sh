#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
BASE_PREFIX=base_c4_adapter
EDGE_TH=0.4674050956964493

run_one() {
  local name="$1"
  local tmax="$2"
  local base_err_weight="$3"
  local base_err_gamma="$4"
  local low_edge_weight="$5"
  local seed="$6"

  local run_dir="/root/autodl-tmp/diffusion_fpp_v5/results/${name}"
  local feat_dir="/root/autodl-tmp/diffusion_fpp_v5/results/${name}_adaptive_features_a050"

  echo "===== D9 START ${name} $(date '+%F %T') ====="
  /root/miniconda3/bin/python train_pip_lite.py \
    --dataset fpp_ml_bench \
    --cache_dir "$CACHE" \
    --save_dir "$run_dir" \
    --epochs 5 \
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
    --train_t_max_ratio "$tmax" \
    --lr 3e-5 \
    --base_residual_gate 0.5 \
    --base_error_loss_weight "$base_err_weight" \
    --base_error_loss_gamma "$base_err_gamma" \
    --low_edge_loss_weight "$low_edge_weight" \
    --low_edge_threshold "$EDGE_TH" \
    --seed "$seed"

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
      --num_workers 8 \
      --start_ratio 0.05 \
      --alpha 0.5 \
      --require_cache
  done

  /root/miniconda3/bin/python select_adaptive_gate.py \
    --val_csv "$feat_dir/val_adaptive_features.csv" \
    --test_csv "$feat_dir/test_adaptive_features.csv" \
    --save_json "$feat_dir/selected_gate_summary.json" \
    --min_selected 3

  echo "===== D9 END ${name} $(date '+%F %T') ====="
}

run_one pip_d9a_harderr6_lowedge1_t025_e5_seed0 0.25 6.0 1.0 1.0 0
run_one pip_d9b_harderr8_lowedge2_t035_e5_seed0 0.35 8.0 0.7 2.0 0
run_one pip_d9c_harderr12_t025_e5_seed0 0.25 12.0 0.5 0.0 0

echo "===== D9 ALL DONE $(date '+%F %T') ====="
