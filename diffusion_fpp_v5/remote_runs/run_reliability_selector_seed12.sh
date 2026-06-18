#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5

OUT=/root/autodl-tmp/diffusion_fpp_v5/results/A_20260618_refined_xphase_depth
mkdir -p "$OUT/logs"

for seed in 1 2; do
  echo "START reliability selector seed${seed} $(date '+%F %T')"
  /root/miniconda3/bin/python -u train_refined_xphase_reliability_selector.py \
    --data_root /root/autodl-tmp/single_frame_3d_dataset_v1_upload_smalltest \
    --teacher_extra_root /root/autodl-tmp/single_frame_3d_dataset_v1_teacher_extra \
    --ood_root /root/autodl-tmp/single_frame_3d_dataset_v1_upload_smalltest_ood61_64 \
    --save_dir "$OUT" \
    --base_ckpt /root/autodl-tmp/diffusion_fpp_v5/results/A_20260614_single_frame3d_physics_diffusion/runs/direct_teacher_aux_seed0/checkpoints/best.pt \
    --x_diag_dir /root/autodl-tmp/diffusion_fpp_v5/results/A_20260617_single_frame3d_phase_residual_diagnosis_xonly \
    --phase_posterior_ckpt /root/autodl-tmp/diffusion_fpp_v5/results/A_20260617_single_frame3d_xphase_diffusion_rcpc/x_phase_diffusion/checkpoints/best.pt \
    --refined_depth_ckpt /root/autodl-tmp/diffusion_fpp_v5/results/A_20260618_refined_xphase_depth/refined_xphase_depth/checkpoints/best.pt \
    --summary_path /root/autodl-tmp/diffusion_fpp_v5/results/A_20260618_refined_xphase_depth/refined_xphase_depth_summary.json \
    --batch_size 1 \
    --eval_batch_size 1 \
    --num_workers 2 \
    --cache_features \
    --feature_cache_dir /root/autodl-tmp/single_frame_3d_dataset_v1_upload_smalltest/physics_feature_cache_pip \
    --train_pixels_per_sample 2048 \
    --max_train_pixels 700000 \
    --selector_epochs 25 \
    --seed "$seed" \
    > "$OUT/logs/reliability_selector_seed${seed}.log" 2>&1
  echo "DONE reliability selector seed${seed} $(date '+%F %T')"
done
