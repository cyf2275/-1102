#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY=/root/miniconda3/bin/python
CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
SAVE=results/fpp960_e180_official_fringe_plus_physics_ftp_e120
MASTER=results/remote_logs/e180_official_fringe_plus_physics_ftp_master.log

echo "START E180 official fringe_plus_physics + FTP $(date '+%F %T')" | tee "$MASTER"

"$PY" -m py_compile train_fpp_official_style_unet.py

"$PY" train_fpp_official_style_unet.py \
  --cache_dir "$CACHE" \
  --save_dir "$SAVE" \
  --input_mode fringe_plus_physics \
  --include_ftp \
  --physics_channels 0-10 \
  --epochs 120 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --num_workers 12 \
  --image_size 960 \
  --lr 1e-4 \
  --alpha 0.7 \
  --eval_metrics_every 5 \
  --save_every 10 \
  --require_cache \
  --final_checkpoint best_rmse \
  2>&1 | tee results/remote_logs/e180_official_fringe_plus_physics_ftp_train.log

echo "DONE E180 official fringe_plus_physics + FTP $(date '+%F %T')" | tee -a "$MASTER"
