#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PSP_CACHE=/root/autodl-tmp/fpp_ml_pspquad_cache_960
BASE_PREFIX=base_c4_adapter
PHASE_PREFIX=phase_pred_e63_ens_e28_e62_w055
COARSE_CKPT=results/fpp960_e123_coarse_lowpass_uncertainty_ch32_e60/checkpoints/best.pt
FULL_PREFIX=phase_pred_e63_ens_e28_e62_w055_e123_du
SAVE=results/fpp960_e125_pipfull_e63_e123_du_depthdiff_ch24_e16
MASTER=results/remote_logs/e125_pipfull_du_master.log

echo "START E125 PIP-Full E63 + E123 D_c/U_c diffusion $(date '+%F %T')" | tee "$MASTER"

"$PY" -m py_compile precompute_coarse_condition_cache.py train_pip_lite.py

"$PY" precompute_coarse_condition_cache.py \
  --coarse_checkpoint "$COARSE_CKPT" \
  --cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PSP_CACHE" \
  --phase_pred_prefix "$PHASE_PREFIX" \
  --output_prefix "$FULL_PREFIX" \
  --coarse_mode depth_unc \
  --image_h 960 \
  --image_w 960 \
  --lowpass_factor 8 \
  --eval_batch_size 2 \
  --num_workers 8 \
  --require_cache \
  2>&1 | tee results/remote_logs/e125_precompute_du.log

"$PY" train_pip_lite.py \
  --dataset fpp_ml_bench \
  --cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PSP_CACHE" \
  --phase_pred_prefix "$FULL_PREFIX" \
  --append_phase_pred_to_cond \
  --save_dir "$SAVE" \
  --epochs 16 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --num_workers 12 \
  --lr 3e-5 \
  --seed 125 \
  --base_channels 24 \
  --condition_injection adapter \
  --adapter_hidden 24 \
  --target_mode base_residual \
  --base_prefix "$BASE_PREFIX" \
  --base_residual_gate 0.5 \
  --timesteps 200 \
  --ddim_steps 20 \
  --ensemble 1 \
  --eval_every 2 \
  --save_every 2 \
  --save_epoch_checkpoints \
  --lambda_oriented 0.08 \
  --lambda_edge 0.03 \
  --lambda_normal 0.01 \
  --lambda_phase 0.0 \
  --image_h 960 \
  --image_w 960 \
  --require_cache \
  --sample_start_ratio 0.05 \
  --train_t_min_ratio 0.0 \
  --train_t_max_ratio 0.15 \
  --base_error_loss_weight 2.5 \
  --base_error_loss_gamma 1.0 \
  --low_edge_loss_weight 1.0 \
  --low_edge_threshold 0.467 \
  --blend_loss_alpha 0.5 \
  --skip_final_test \
  2>&1 | tee results/remote_logs/e125_pipfull_du_train.log

echo "DONE E125 PIP-Full E63 + E123 D_c/U_c diffusion $(date '+%F %T')" | tee -a "$MASTER"
