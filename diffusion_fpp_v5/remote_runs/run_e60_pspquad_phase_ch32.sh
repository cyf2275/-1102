#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY=/root/miniconda3/bin/python

echo "START E60 PSP/quadrature phase diffusion ch32 $(date '+%F %T')" | tee results/remote_logs/e60_pspquad_phase_ch32_master.log

"$PY" train_fpp_phase_diffusion.py \
  --base_cache_dir /root/autodl-tmp/fpp_ml_bench_cache_960_fgfix \
  --phase_cache_dir /root/autodl-tmp/fpp_ml_pspquad_cache_960 \
  --save_dir results/fpp960_e60_pspquad_phase_diffusion_ch32_e30 \
  --phase_channels 0-12 \
  --epochs 30 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --num_workers 8 \
  --image_size 960 \
  --base_channels 32 \
  --ch_mult 1,2,4,8,8 \
  --adapter_hidden 32 \
  --dropout 0.05 \
  --target_channels 3 \
  --timesteps 200 \
  --ddim_steps 20 \
  --ensemble 3 \
  --sample_start_from ftp \
  --sample_start_ratio 0.7 \
  --train_start_from target \
  --train_t_min_ratio 0.0 \
  --train_t_max_ratio 1.0 \
  --phase_weight 1.0 \
  --grad_weight 0.05 \
  --unit_weight 0.02 \
  --uph_norm sample \
  --uph_start_from half \
  --uph_weight 0.5 \
  --uph_grad_weight 0.02 \
  --selection_metric phase_aligned_mae_rad \
  --lr 8e-5 \
  --weight_decay 1e-5 \
  --grad_clip 1.0 \
  --eval_every 1 \
  --save_every 1 \
  --seed 60 \
  2>&1 | tee results/remote_logs/fpp960_e60_pspquad_phase_diffusion_ch32_e30.log

echo "DONE E60 PSP/quadrature phase diffusion ch32 $(date '+%F %T')" | tee -a results/remote_logs/e60_pspquad_phase_ch32_master.log
