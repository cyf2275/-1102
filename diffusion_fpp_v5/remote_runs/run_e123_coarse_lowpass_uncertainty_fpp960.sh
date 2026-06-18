#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
SAVE=results/fpp960_e123_coarse_lowpass_uncertainty_ch32_e60
MASTER=results/remote_logs/e123_coarse_lowpass_uncertainty_master.log

echo "START E123 FPP-ML low-pass CoarseNet + uncertainty $(date '+%F %T')" | tee "$MASTER"

"$PY" -m py_compile train_coarse_lowpass_pip.py

"$PY" train_coarse_lowpass_pip.py \
  --dataset fpp_ml_bench \
  --cache_dir "$BASE_CACHE" \
  --save_dir "$SAVE" \
  --epochs 60 \
  --batch_size 4 \
  --eval_batch_size 2 \
  --num_workers 12 \
  --lr 2e-4 \
  --base_channels 32 \
  --eval_every 5 \
  --lambda_grad 0.1 \
  --lambda_unc 0.05 \
  --lowpass_factor 8 \
  --image_h 960 \
  --image_w 960 \
  --require_cache \
  2>&1 | tee results/remote_logs/e123_coarse_lowpass_uncertainty_train.log

echo "DONE E123 FPP-ML low-pass CoarseNet + uncertainty $(date '+%F %T')" | tee -a "$MASTER"
