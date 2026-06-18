#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
BASE_PREFIX=base_c4_adapter

run_pixel_gate() {
  local name="$1"
  local ckpt="$2"
  local out_dir="/root/autodl-tmp/diffusion_fpp_v5/results/${name}"

  echo "===== D10 START ${name} $(date '+%F %T') ====="
  /root/miniconda3/bin/python eval_pixel_adaptive_gate.py \
    --checkpoint "$ckpt" \
    --cache_dir "$CACHE" \
    --base_prefix "$BASE_PREFIX" \
    --save_dir "$out_dir" \
    --image_h 960 \
    --image_w 960 \
    --ddim_steps 20 \
    --ensemble 1 \
    --eval_batch_size 1 \
    --num_workers 8 \
    --start_ratio 0.05 \
    --alphas "0.25 0.35 0.50" \
    --edge_thresholds "0.25 0.35 0.4674050956964493 0.60 0.80 1.00" \
    --delta_mins "0.0 0.02 0.05 0.08 0.12 0.16" \
    --conf_mins "0.0 0.2 0.4" \
    --require_cache
  echo "===== D10 END ${name} $(date '+%F %T') ====="
}

run_pixel_gate \
  pip_d10_pixel_gate_d8_seed0 \
  /root/autodl-tmp/diffusion_fpp_v5/results/pip_d8_seed0_base_residual_e1_gate050_lr3e5/checkpoints/best.pt

run_pixel_gate \
  pip_d10_pixel_gate_d8_seed123 \
  /root/autodl-tmp/diffusion_fpp_v5/results/pip_d8_seed123_base_residual_e1_gate050_lr3e5/checkpoints/best.pt

echo "===== D10 PROBE DONE $(date '+%F %T') ====="
