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
PHASE_PREFIX=phase_pred_e63_ens_e28_e62_w055_e123_du
SAVE=results/fpp960_e127_pipfull_du_fullx0_basestart_ch24_e32
MASTER=results/remote_logs/e127_pipfull_du_fullx0_master.log

echo "START E127 PIP-Full D_c/U_c full_x0 base-start diffusion $(date '+%F %T')" | tee "$MASTER"

"$PY" -m py_compile train_pip_lite.py eval_hierarchical_physical_gate.py

"$PY" train_pip_lite.py \
  --dataset fpp_ml_bench \
  --cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PSP_CACHE" \
  --phase_pred_prefix "$PHASE_PREFIX" \
  --append_phase_pred_to_cond \
  --save_dir "$SAVE" \
  --epochs 32 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --num_workers 12 \
  --lr 3e-5 \
  --seed 127 \
  --base_channels 24 \
  --condition_injection adapter \
  --adapter_hidden 24 \
  --target_mode full_x0 \
  --base_prefix "$BASE_PREFIX" \
  --timesteps 200 \
  --ddim_steps 20 \
  --ensemble 1 \
  --eval_every 4 \
  --save_every 4 \
  --save_epoch_checkpoints \
  --lambda_oriented 0.08 \
  --lambda_edge 0.03 \
  --lambda_normal 0.01 \
  --lambda_phase 0.0 \
  --image_h 960 \
  --image_w 960 \
  --require_cache \
  --train_start_from_base \
  --sample_start_from_base \
  --sample_start_ratio 0.05 \
  --train_t_min_ratio 0.0 \
  --train_t_max_ratio 0.15 \
  --base_error_loss_weight 2.5 \
  --base_error_loss_gamma 1.0 \
  --low_edge_loss_weight 1.0 \
  --low_edge_threshold 0.467 \
  2>&1 | tee results/remote_logs/e127_pipfull_du_fullx0_train.log

"$PY" eval_hierarchical_physical_gate.py \
  --checkpoint "$SAVE/checkpoints/best.pt" \
  --cache_dir "$BASE_CACHE" \
  --base_prefix "$BASE_PREFIX" \
  --save_dir results/e127_pipfull_du_fullx0_best_gate_eval \
  --image_h 960 \
  --image_w 960 \
  --ddim_steps 20 \
  --ensemble 1 \
  --eval_batch_size 1 \
  --num_workers 8 \
  --start_ratio 0.05 \
  --pixel_alpha 0.7 \
  --pixel_sample_edge_th 0.47 \
  --pixel_edge_th 1.0 \
  --pixel_delta_min 0.12 \
  --pixel_conf_min 0.0 \
  --require_cache \
  2>&1 | tee results/remote_logs/e127_pipfull_du_fullx0_gate_eval.log

echo "DONE E127 PIP-Full D_c/U_c full_x0 base-start diffusion $(date '+%F %T')" | tee -a "$MASTER"
