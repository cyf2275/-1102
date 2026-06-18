#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY=/root/miniconda3/bin/python
CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
SAVE=results/d59_ch24_baseerror_lowedge_depthdiff_e3
EVAL=results/d59_epoch001_hierarchical_physical_gate

echo "START D59 weighted depth diffusion $(date '+%F %T')" | tee results/remote_logs/d59_weighted_depthdiff_master.log

"$PY" train_pip_lite.py \
  --dataset fpp_ml_bench \
  --cache_dir "$CACHE" \
  --save_dir "$SAVE" \
  --epochs 3 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --num_workers 8 \
  --lr 3e-05 \
  --seed 59 \
  --base_channels 24 \
  --condition_injection adapter \
  --adapter_hidden 16 \
  --physics_channels 0-8 \
  --target_mode base_residual \
  --base_prefix base_c4_adapter \
  --base_residual_gate 0.5 \
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
  --sample_start_ratio 0.05 \
  --train_t_min_ratio 0.0 \
  --train_t_max_ratio 0.15 \
  --base_error_loss_weight 3.0 \
  --base_error_loss_gamma 1.0 \
  --low_edge_loss_weight 1.5 \
  --low_edge_threshold 0.467 \
  --skip_final_test \
  2>&1 | tee results/remote_logs/d59_weighted_depthdiff_train.log

echo "EVAL D59 epoch001 hierarchical gate $(date '+%F %T')" | tee -a results/remote_logs/d59_weighted_depthdiff_master.log
"$PY" eval_hierarchical_physical_gate.py \
  --checkpoint "$SAVE/checkpoints/epoch_001.pt" \
  --cache_dir "$CACHE" \
  --base_prefix base_c4_adapter \
  --save_dir "$EVAL" \
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
  --high_edge_min 0.58 \
  --high_edge_max 0.62 \
  --high_delta_min 0.09 \
  --high_delta_max 0.105 \
  --high_conf_min 0.76 \
  --high_conf_max 0.80 \
  --require_cache \
  2>&1 | tee results/remote_logs/d59_weighted_depthdiff_hier_eval.log

echo "DONE D59 weighted depth diffusion $(date '+%F %T')" | tee -a results/remote_logs/d59_weighted_depthdiff_master.log
