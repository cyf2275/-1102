#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY=/root/miniconda3/bin/python
SAVE=results/e65_phase_projection_rawmm_960_e40

echo "START E65 phase projection rawmm 960 $(date '+%F %T')" | tee results/remote_logs/e65_phase_projection_rawmm_960_master.log

"$PY" train_phase_projection_pip.py \
  --dataset fpp_ml_bench \
  --cache_dir /root/autodl-tmp/fpp_ml_bench_cache_960_fgfix \
  --save_dir "$SAVE" \
  --epochs 40 \
  --batch_size 2 \
  --num_workers 8 \
  --lr 2e-4 \
  --hidden_dim 64 \
  --num_layers 4 \
  --depth_input raw_mm \
  --eval_every 5 \
  --image_h 960 \
  --image_w 960 \
  --require_cache \
  2>&1 | tee results/remote_logs/e65_phase_projection_rawmm_960_e40.log

echo "DONE E65 phase projection rawmm 960 $(date '+%F %T')" | tee -a results/remote_logs/e65_phase_projection_rawmm_960_master.log
