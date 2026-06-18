#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
OUT=/root/autodl-tmp/diffusion_fpp_v5/results/pip_d28_phase_residual_projection_480

echo "===== D28 depth-to-detrended-phase-residual projection $(date '+%F %T') ====="
/root/miniconda3/bin/python train_phase_residual_projection_pip.py \
  --dataset fpp_ml_bench \
  --cache_dir "$CACHE" \
  --save_dir "$OUT" \
  --epochs 40 \
  --batch_size 4 \
  --num_workers 8 \
  --lr 2e-4 \
  --hidden_dim 64 \
  --num_layers 4 \
  --target_channel 3 \
  --depth_input raw_mm \
  --eval_every 5 \
  --image_h 480 \
  --image_w 480 \
  --require_cache

echo "===== D28 DONE $(date '+%F %T') ====="
