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

wait_for_f1() {
  echo "===== F2 queue waiting for F1 continuation $(date '+%F %T') ====="
  while pgrep -f 'train_joint_pip_diffusion.py.*fpp960_f1_masked_residual_seed13[23]' >/dev/null 2>&1 || \
        pgrep -f 'run_f1_continue_seed132_133.sh' >/dev/null 2>&1; do
    nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader,nounits || true
    sleep 300
  done
  echo "===== F1 continuation no longer running $(date '+%F %T') ====="
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
  --lambda_residual 0.3
  --lambda_oriented 0.05
  --lambda_edge 0.02
  --lambda_normal 0.005
)

run_job() {
  local name="$1"
  local seed="$2"
  local epochs="$3"
  shift 3
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
    "$@" 2>&1 | tee -a "$log"
  return ${PIPESTATUS[0]}
}

wait_for_f1

# F2: conservative main variant. Reduces residual amplitude and strengthens sparsity
# to test whether F1 over-corrected test samples.
run_job fpp960_f2_conservative_masked_residual 131 160 \
  --joint_mode no_unc \
  --learned_residual_gate \
  --gate_init 0.05 \
  --hard_mask_mode mixed \
  --hard_error_weight 1.0 \
  --hard_physics_weight 0.7 \
  --hard_conf_power 1.0 \
  --hard_mask_threshold 0.38 \
  --hard_mask_sharpness 10.0 \
  --hard_mask_focus_weight 1.2 \
  --lambda_gate_supervision 0.08 \
  --lambda_gate_l1 0.01 \
  --base_error_loss_weight 0.25 \
  --base_error_loss_gamma 1.0 \
  --base_residual_gate 0.5 \
  --lambda_coarse 0.05 \
  --lambda_coarse_grad 0.05 \
  --lambda_uncertainty 0.0

run_job fpp960_f2_conservative_masked_residual 132 160 \
  --joint_mode no_unc \
  --learned_residual_gate \
  --gate_init 0.05 \
  --hard_mask_mode mixed \
  --hard_error_weight 1.0 \
  --hard_physics_weight 0.7 \
  --hard_conf_power 1.0 \
  --hard_mask_threshold 0.38 \
  --hard_mask_sharpness 10.0 \
  --hard_mask_focus_weight 1.2 \
  --lambda_gate_supervision 0.08 \
  --lambda_gate_l1 0.01 \
  --base_error_loss_weight 0.25 \
  --base_error_loss_gamma 1.0 \
  --base_residual_gate 0.5 \
  --lambda_coarse 0.05 \
  --lambda_coarse_grad 0.05 \
  --lambda_uncertainty 0.0

# F3: global residual negative control. This tells us whether hard masking is
# genuinely useful or the residual diffusion only learns a global correction.
run_job fpp960_f3_global_residual_control 131 120 \
  --joint_mode no_unc \
  --hard_mask_mode none \
  --base_error_loss_weight 0.5 \
  --base_error_loss_gamma 1.0 \
  --lambda_coarse 0.05 \
  --lambda_coarse_grad 0.05 \
  --lambda_uncertainty 0.0

# F4: no-coarse masked residual. Separates the effect of hard-region diffusion
# from the low-pass CoarseNet branch.
run_job fpp960_f4_masked_residual_no_coarse 131 120 \
  --joint_mode no_coarse \
  --learned_residual_gate \
  --gate_init 0.08 \
  --hard_mask_mode mixed \
  --hard_error_weight 1.0 \
  --hard_physics_weight 0.7 \
  --hard_conf_power 1.0 \
  --hard_mask_threshold 0.35 \
  --hard_mask_sharpness 8.0 \
  --hard_mask_focus_weight 2.0 \
  --lambda_gate_supervision 0.12 \
  --lambda_gate_l1 0.002 \
  --base_error_loss_weight 0.5 \
  --base_error_loss_gamma 1.0 \
  --lambda_coarse 0.0 \
  --lambda_coarse_grad 0.0 \
  --lambda_uncertainty 0.0

# F5: physics-only gate. This is stricter for paper because it removes the
# training-only oracle residual target from the hard-mask target.
run_job fpp960_f5_physics_only_gate 131 120 \
  --joint_mode no_unc \
  --learned_residual_gate \
  --gate_init 0.08 \
  --hard_mask_mode physics \
  --hard_physics_weight 1.0 \
  --hard_conf_power 1.0 \
  --hard_mask_threshold 0.35 \
  --hard_mask_sharpness 8.0 \
  --hard_mask_focus_weight 1.5 \
  --lambda_gate_supervision 0.10 \
  --lambda_gate_l1 0.004 \
  --base_error_loss_weight 0.25 \
  --base_error_loss_gamma 1.0 \
  --lambda_coarse 0.05 \
  --lambda_coarse_grad 0.05 \
  --lambda_uncertainty 0.0

echo "===== F2 queue finished all planned jobs $(date '+%F %T') ====="
