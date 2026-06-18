#!/usr/bin/env bash
set -uo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5 || exit 1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

PY=/root/miniconda3/bin/python
CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PHASE_CACHE=/root/autodl-tmp/fpp_ml_pspquad_cache_960
BASE_PREFIX=base_c4_adapter
LOGDIR=/root/autodl-tmp/diffusion_fpp_v5/results/_logs
mkdir -p "$LOGDIR"

wait_for_current_queue() {
  echo "===== postqueue waiting for F2-F5 summaries $(date '+%F %T') ====="
  while true; do
    local pending=0
    for d in \
      results/fpp960_f2_conservative_masked_residual_seed131 \
      results/fpp960_f2_conservative_masked_residual_seed132 \
      results/fpp960_f3_global_residual_control_seed131 \
      results/fpp960_f4_masked_residual_no_coarse_seed131 \
      results/fpp960_f5_physics_only_gate_seed131; do
      [[ -f "$d/evaluation/summary.json" ]] || pending=1
    done
    if [[ "$pending" -eq 0 ]]; then
      break
    fi
    nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader,nounits || true
    sleep 300
  done
  echo "===== postqueue detected F2-F5 complete $(date '+%F %T') ====="
}

run_alpha_gate() {
  local out_dir=results/fpp960_post_f2_joint_alpha_gate_seed131_132
  if [[ -f "$out_dir/joint_alpha_gate_summary.json" ]]; then
    echo "===== skip existing joint alpha/gate sweep $(date '+%F %T') ====="
    return 0
  fi
  echo "===== run joint alpha/gate sweep on F2 seeds $(date '+%F %T') ====="
  "$PY" eval_joint_alpha_sweep_gate.py \
    --checkpoints \
      results/fpp960_f2_conservative_masked_residual_seed131/checkpoints/best.pt \
      results/fpp960_f2_conservative_masked_residual_seed132/checkpoints/best.pt \
    --cache_dir "$CACHE" \
    --phase_cache_dir "$PHASE_CACHE" \
    --base_prefix "$BASE_PREFIX" \
    --save_dir "$out_dir" \
    --splits val test \
    --gate_select_split val \
    --image_h 960 \
    --image_w 1280 \
    --ddim_steps 20 \
    --start_ratio 0.05 \
    --eval_batch_size 1 \
    --num_workers 8 \
    --require_cache \
    --save_long_csv \
    --alphas 0.0 0.03 0.05 0.08 0.10 0.15 0.20 0.25 0.35 0.50 0.75 1.0 \
    --gate_features all \
    --min_selected 3
}

base_args=(
  --cache_dir "$CACHE"
  --phase_cache_dir "$PHASE_CACHE"
  --base_prefix "$BASE_PREFIX"
  --require_cache
  --include_ftp
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
  --batch_size 2
  --eval_batch_size 1
  --num_workers 12
  --lr 3e-5
  --weight_decay 1e-5
  --eval_every 5
  --save_every 5
  --lambda_depth 1.0
  --lambda_residual 0.2
  --lambda_oriented 0.05
  --lambda_edge 0.02
  --lambda_normal 0.005
)

run_f6() {
  local name=fpp960_f6_ultra_conservative_masked_residual
  local seed=133
  local epochs=140
  local save_dir="/root/autodl-tmp/diffusion_fpp_v5/results/${name}_seed${seed}"
  local log="$LOGDIR/${name}_seed${seed}.log"
  if [[ -f "$save_dir/evaluation/summary.json" ]]; then
    echo "===== skip existing ${name}_seed${seed} $(date '+%F %T') =====" | tee -a "$log"
    return 0
  fi
  local resume_args=()
  if [[ -f "$save_dir/checkpoints/latest.pt" ]]; then
    resume_args=(--resume "$save_dir/checkpoints/latest.pt")
    echo "===== resume ${name}_seed${seed} from latest.pt $(date '+%F %T') =====" | tee -a "$log"
  fi
  echo "===== run ${name} seed=${seed} epochs=${epochs} $(date '+%F %T') =====" | tee -a "$log"
  "$PY" train_joint_pip_diffusion.py \
    "${base_args[@]}" \
    --save_dir "$save_dir" \
    --epochs "$epochs" \
    --seed "$seed" \
    "${resume_args[@]}" \
    --joint_mode no_unc \
    --learned_residual_gate \
    --gate_init 0.03 \
    --hard_mask_mode mixed \
    --hard_error_weight 1.0 \
    --hard_physics_weight 1.0 \
    --hard_conf_power 1.5 \
    --hard_mask_threshold 0.50 \
    --hard_mask_sharpness 12.0 \
    --hard_mask_focus_weight 2.0 \
    --lambda_gate_supervision 0.12 \
    --lambda_gate_l1 0.04 \
    --base_error_loss_weight 0.10 \
    --base_error_loss_gamma 1.0 \
    --base_residual_gate 0.20 \
    --lambda_coarse 0.05 \
    --lambda_coarse_grad 0.05 \
    --lambda_uncertainty 0.0 2>&1 | tee -a "$log"
  return ${PIPESTATUS[0]}
}

wait_for_current_queue
run_alpha_gate 2>&1 | tee -a "$LOGDIR/postqueue_joint_alpha_gate.log"
run_f6

echo "===== postqueue finished $(date '+%F %T') ====="
