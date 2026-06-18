#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs remote_runs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PSP_CACHE=/root/autodl-tmp/fpp_ml_pspquad_cache_960
BASE_PREFIX=base_c4_adapter
PHASE_PREFIX=phase_pred_e63_ens_e28_e62_w055
SAVE=results/fpp960_e66_e63phase_fullx0_basestart_depthdiff_ch24_e30
GATE=results/e66_epoch001_hierarchical_physical_gate
MASTER=results/remote_logs/e66_e63phase_fullx0_basestart_depthdiff_master.log

echo "START E66 E63 phase-conditioned full-x0 base-start diffusion $(date '+%F %T')" | tee "$MASTER"

"$PY" train_pip_lite.py \
  --dataset fpp_ml_bench \
  --cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PSP_CACHE" \
  --phase_pred_prefix "$PHASE_PREFIX" \
  --append_phase_pred_to_cond \
  --save_dir "$SAVE" \
  --epochs 30 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --num_workers 12 \
  --lr 3e-5 \
  --seed 66 \
  --base_channels 24 \
  --condition_injection adapter \
  --adapter_hidden 24 \
  --target_mode full_x0 \
  --base_prefix "$BASE_PREFIX" \
  --timesteps 200 \
  --ddim_steps 20 \
  --ensemble 1 \
  --eval_every 1 \
  --save_every 1 \
  --save_epoch_checkpoints \
  --lambda_oriented 0.08 \
  --lambda_edge 0.03 \
  --lambda_normal 0.01 \
  --lambda_phase 0.0 \
  --image_h 960 \
  --image_w 960 \
  --require_cache \
  --sample_start_from_base \
  --sample_start_ratio 0.08 \
  --train_start_from_base \
  --train_t_min_ratio 0.0 \
  --train_t_max_ratio 0.18 \
  --base_error_loss_weight 2.0 \
  --base_error_loss_gamma 1.0 \
  --low_edge_loss_weight 1.0 \
  --low_edge_threshold 0.467 \
  --skip_final_test \
  2>&1 | tee results/remote_logs/e66_e63phase_fullx0_basestart_depthdiff_train.log

echo "EVAL E66 best checkpoint with hierarchical physical gate $(date '+%F %T')" | tee -a "$MASTER"
"$PY" eval_hierarchical_physical_gate.py \
  --checkpoint "$SAVE/checkpoints/best.pt" \
  --cache_dir "$BASE_CACHE" \
  --base_prefix "$BASE_PREFIX" \
  --save_dir "$GATE" \
  --image_h 960 \
  --image_w 960 \
  --ddim_steps 20 \
  --ensemble 1 \
  --eval_batch_size 1 \
  --num_workers 8 \
  --start_ratio 0.08 \
  --pixel_alpha 0.7 \
  --pixel_sample_edge_th 0.47 \
  --pixel_edge_th 1.0 \
  --pixel_delta_min 0.12 \
  --pixel_conf_min 0.0 \
  --high_edge_min 0.58 \
  --high_edge_max 0.62 \
  --high_delta_min 0.09 \
  --high_delta_max 0.105 \
  --high_conf_min 0.76 \
  --high_conf_max 0.80 \
  --require_cache \
  2>&1 | tee results/remote_logs/e66_hierarchical_gate_eval.log

echo "DONE E66 $(date '+%F %T')" | tee -a "$MASTER"
