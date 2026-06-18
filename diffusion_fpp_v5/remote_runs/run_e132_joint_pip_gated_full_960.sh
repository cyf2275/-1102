#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

/root/miniconda3/bin/python train_joint_pip_diffusion.py \
  --cache_dir /root/autodl-tmp/fpp_ml_bench_cache_960_fgfix \
  --phase_cache_dir /root/autodl-tmp/fpp_ml_pspquad_cache_960 \
  --save_dir /root/autodl-tmp/diffusion_fpp_v5/results/e132_joint_pip_gated_full_960 \
  --base_prefix base_c4_adapter \
  --joint_mode full \
  --learned_residual_gate \
  --gate_init 0.05 \
  --epochs 60 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --num_workers 12 \
  --lr 3e-5 \
  --base_channels 24 \
  --adapter_hidden 24 \
  --coarse_channels 24 \
  --timesteps 200 \
  --ddim_steps 20 \
  --ensemble 1 \
  --eval_every 2 \
  --save_every 4 \
  --image_h 960 \
  --image_w 1280 \
  --lowpass_factor 8 \
  --train_t_min_ratio 0.0 \
  --train_t_max_ratio 0.15 \
  --sample_start_ratio 0.05 \
  --base_error_loss_weight 2.0 \
  --base_error_loss_gamma 1.0 \
  --lambda_depth 1.0 \
  --lambda_residual 0.2 \
  --lambda_coarse 0.2 \
  --lambda_coarse_grad 0.1 \
  --lambda_uncertainty 0.05 \
  --lambda_oriented 0.08 \
  --lambda_edge 0.03 \
  --lambda_normal 0.01 \
  --lambda_gate_l1 0.001 \
  --require_cache \
  2>&1 | tee results/remote_logs/e132_joint_pip_gated_full_960.log
