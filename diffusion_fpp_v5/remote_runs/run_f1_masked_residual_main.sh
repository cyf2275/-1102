#!/usr/bin/env bash
set -uo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

PY=/root/miniconda3/bin/python
CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PHASE_CACHE=/root/autodl-tmp/fpp_ml_pspquad_cache_960
BASE_PREFIX=base_c4_adapter
LOGDIR=/root/autodl-tmp/diffusion_fpp_v5/results/_logs
mkdir -p "$LOGDIR"

echo "===== F1 masked residual main started $(date '+%F %T') ====="
nvidia-smi || true

common_args=(
  --cache_dir "$CACHE"
  --phase_cache_dir "$PHASE_CACHE"
  --base_prefix "$BASE_PREFIX"
  --require_cache
  --include_ftp
  --joint_mode no_unc
  --learned_residual_gate
  --gate_init 0.08
  --hard_mask_mode mixed
  --hard_error_weight 1.0
  --hard_physics_weight 0.7
  --hard_conf_power 1.0
  --hard_mask_threshold 0.35
  --hard_mask_sharpness 8.0
  --hard_mask_focus_weight 2.0
  --lambda_gate_supervision 0.12
  --lambda_gate_l1 0.002
  --base_error_loss_weight 0.5
  --base_error_loss_gamma 1.0
  --lambda_depth 1.0
  --lambda_residual 0.3
  --lambda_coarse 0.05
  --lambda_coarse_grad 0.05
  --lambda_uncertainty 0.0
  --lambda_oriented 0.05
  --lambda_edge 0.02
  --lambda_normal 0.005
  --base_channels 24
  --adapter_hidden 24
  --coarse_channels 24
  --timesteps 200
  --ddim_steps 20
  --ensemble 1
  --train_t_min_ratio 0.0
  --train_t_max_ratio 0.15
  --sample_start_ratio 0.05
  --image_h 960
  --image_w 1280
  --train_crop_h 640
  --train_crop_w 896
  --train_epoch_repeats 2
  --eval_batch_size 1
  --num_workers 12
  --lr 3e-5
  --weight_decay 1e-5
  --eval_every 5
  --save_every 5
)

run_train() {
  local seed="$1"
  local epochs="$2"
  local save_dir="$3"
  local log="$LOGDIR/$(basename "$save_dir").log"
  echo "===== run seed=${seed} epochs=${epochs} save=${save_dir} $(date '+%F %T') =====" | tee -a "$log"
  "$PY" train_joint_pip_diffusion.py \
    "${common_args[@]}" \
    --save_dir "$save_dir" \
    --epochs "$epochs" \
    --seed "$seed" \
    --batch_size 4 2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  if [[ "$status" -ne 0 ]]; then
    echo "===== batch_size=4 failed with status=${status}; retry batch_size=2 $(date '+%F %T') =====" | tee -a "$log"
    "$PY" train_joint_pip_diffusion.py \
      "${common_args[@]}" \
      --save_dir "${save_dir}_bs2_retry" \
      --epochs "$epochs" \
      --seed "$seed" \
      --batch_size 2 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
  fi
  if [[ "$status" -ne 0 ]]; then
    echo "===== batch_size=2 failed with status=${status}; retry batch_size=1 $(date '+%F %T') =====" | tee -a "$log"
    "$PY" train_joint_pip_diffusion.py \
      "${common_args[@]}" \
      --save_dir "${save_dir}_bs1_retry" \
      --epochs "$epochs" \
      --seed "$seed" \
      --batch_size 1 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
  fi
  return "$status"
}

run_smoke() {
  local save_dir="/root/autodl-tmp/diffusion_fpp_v5/results/_smoke_f1_masked_residual_b4"
  local log="$LOGDIR/_smoke_f1_masked_residual_b4.log"
  echo "===== smoke run $(date '+%F %T') =====" | tee -a "$log"
  "$PY" train_joint_pip_diffusion.py \
    "${common_args[@]}" \
    --save_dir "$save_dir" \
    --epochs 1 \
    --seed 131 \
    --batch_size 4 \
    --max_train_batches 6 \
    --eval_every 1 \
    --save_every 1 \
    --skip_final_test 2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  if [[ "$status" -ne 0 ]]; then
    echo "===== smoke batch_size=4 failed; retry batch_size=2 $(date '+%F %T') =====" | tee -a "$log"
    "$PY" train_joint_pip_diffusion.py \
      "${common_args[@]}" \
      --save_dir "${save_dir}_bs2_retry" \
      --epochs 1 \
      --seed 131 \
      --batch_size 2 \
      --max_train_batches 6 \
      --eval_every 1 \
      --save_every 1 \
      --skip_final_test 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
  fi
  if [[ "$status" -ne 0 ]]; then
    echo "===== smoke batch_size=2 failed; retry batch_size=1 $(date '+%F %T') =====" | tee -a "$log"
    "$PY" train_joint_pip_diffusion.py \
      "${common_args[@]}" \
      --save_dir "${save_dir}_bs1_retry" \
      --epochs 1 \
      --seed 131 \
      --batch_size 1 \
      --max_train_batches 6 \
      --eval_every 1 \
      --save_every 1 \
      --skip_final_test 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
  fi
  return "$status"
}

run_smoke || exit 1

run_train 131 180 /root/autodl-tmp/diffusion_fpp_v5/results/fpp960_f1_masked_residual_b4_seed131
run_train 132 180 /root/autodl-tmp/diffusion_fpp_v5/results/fpp960_f1_masked_residual_b4_seed132
run_train 133 180 /root/autodl-tmp/diffusion_fpp_v5/results/fpp960_f1_masked_residual_b4_seed133

echo "===== F1 masked residual main finished $(date '+%F %T') ====="
