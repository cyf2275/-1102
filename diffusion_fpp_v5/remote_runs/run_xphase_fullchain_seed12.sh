#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5

ROOT=/root/autodl-tmp/diffusion_fpp_v5/results/A_20260618_refined_xphase_depth
DATA=/root/autodl-tmp/single_frame_3d_dataset_v1_upload_smalltest
EXTRA=/root/autodl-tmp/single_frame_3d_dataset_v1_teacher_extra
OOD=/root/autodl-tmp/single_frame_3d_dataset_v1_upload_smalltest_ood61_64
CACHE=/root/autodl-tmp/single_frame_3d_dataset_v1_upload_smalltest/physics_feature_cache_pip
BASE=/root/autodl-tmp/diffusion_fpp_v5/results/A_20260614_single_frame3d_physics_diffusion/runs/direct_teacher_aux_seed0/checkpoints/best.pt
XDIAG=/root/autodl-tmp/diffusion_fpp_v5/results/A_20260617_single_frame3d_phase_residual_diagnosis_xonly

mkdir -p "$ROOT/logs"

for seed in 1 2; do
  RUN="$ROOT/fullchain_seed${seed}"
  mkdir -p "$RUN"
  echo "START fullchain seed${seed} $(date '+%F %T')"

  echo "START xphase posterior seed${seed} $(date '+%F %T')"
  /root/miniconda3/bin/python -u train_single_frame3d_xphase_diffusion_rcpc.py \
    --data_root "$DATA" \
    --teacher_extra_root "$EXTRA" \
    --ood_root "$OOD" \
    --save_dir "$RUN/xphase_diffusion_rcpc" \
    --base_ckpt "$BASE" \
    --x_diag_dir "$XDIAG" \
    --seed "$seed" \
    --batch_size 4 \
    --eval_batch_size 2 \
    --num_workers 2 \
    --phase_epochs 30 \
    --cache_features \
    --feature_cache_dir "$CACHE" \
    > "$ROOT/logs/fullchain_seed${seed}_xphase.log" 2>&1
  echo "DONE xphase posterior seed${seed} $(date '+%F %T')"

  PHASE_CKPT="$RUN/xphase_diffusion_rcpc/x_phase_diffusion/checkpoints/best.pt"
  echo "START refined depth seed${seed} $(date '+%F %T')"
  /root/miniconda3/bin/python -u train_single_frame3d_refined_xphase_depth.py \
    --data_root "$DATA" \
    --teacher_extra_root "$EXTRA" \
    --ood_root "$OOD" \
    --save_dir "$RUN/refined_xphase_depth" \
    --base_ckpt "$BASE" \
    --x_diag_dir "$XDIAG" \
    --phase_posterior_ckpt "$PHASE_CKPT" \
    --seed "$seed" \
    --batch_size 2 \
    --eval_batch_size 1 \
    --num_workers 2 \
    --depth_epochs 30 \
    --cache_features \
    --feature_cache_dir "$CACHE" \
    > "$ROOT/logs/fullchain_seed${seed}_refined_depth.log" 2>&1
  echo "DONE refined depth seed${seed} $(date '+%F %T')"

  REFINED_CKPT="$RUN/refined_xphase_depth/refined_xphase_depth/checkpoints/best.pt"
  REFINED_SUMMARY="$RUN/refined_xphase_depth/refined_xphase_depth_summary.json"
  echo "START reliability selector fullchain seed${seed} $(date '+%F %T')"
  /root/miniconda3/bin/python -u train_refined_xphase_reliability_selector.py \
    --data_root "$DATA" \
    --teacher_extra_root "$EXTRA" \
    --ood_root "$OOD" \
    --save_dir "$RUN/reliability_selector" \
    --base_ckpt "$BASE" \
    --x_diag_dir "$XDIAG" \
    --phase_posterior_ckpt "$PHASE_CKPT" \
    --refined_depth_ckpt "$REFINED_CKPT" \
    --summary_path "$REFINED_SUMMARY" \
    --seed "$seed" \
    --batch_size 1 \
    --eval_batch_size 1 \
    --num_workers 2 \
    --cache_features \
    --feature_cache_dir "$CACHE" \
    --train_pixels_per_sample 2048 \
    --max_train_pixels 700000 \
    --selector_epochs 25 \
    > "$ROOT/logs/fullchain_seed${seed}_reliability_selector.log" 2>&1
  echo "DONE reliability selector fullchain seed${seed} $(date '+%F %T')"

  echo "DONE fullchain seed${seed} $(date '+%F %T')"
done
