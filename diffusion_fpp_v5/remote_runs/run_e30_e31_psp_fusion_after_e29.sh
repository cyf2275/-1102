#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PSP_CACHE=/root/autodl-tmp/fpp_ml_pspquad_cache_960
PRED_PREFIX=phase_pred_e28_pspquad_ddim20_e3
INIT_RAW=results/fpp960_a_fringe_unet_control/checkpoints/best.pt

E30_DIR=results/fpp960_e30_pspquad_pred_plus_fringe_from_rawA_e60
E31_DIR=results/fpp960_e31_pspquad_gt_plus_fringe_from_rawA_e40

echo "START E30 PSP-pred plus fringe depth $(date '+%F %T')" \
  | tee results/remote_logs/e30_e31_psp_fusion_master.log
"$PY" train_fpp_phase2depth_unet.py \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PSP_CACHE" \
  --phase_pred_prefix "$PRED_PREFIX" \
  --save_dir "$E30_DIR" \
  --input_mode phase_pred_plus_fringe \
  --init_checkpoint "$INIT_RAW" \
  --epochs 60 \
  --batch_size 2 \
  --eval_batch_size 2 \
  --num_workers 8 \
  --image_size 960 \
  --lr 2e-5 \
  --weight_decay 1e-5 \
  --alpha 0.7 \
  --eval_metrics_every 1 \
  --save_every 5 \
  --eval_initial \
  --seed 42 \
  2>&1 | tee results/remote_logs/fpp960_e30_pspquad_pred_plus_fringe_from_rawA_e60.log

echo "START E31 GT PSP plus fringe upper bound $(date '+%F %T')" \
  | tee -a results/remote_logs/e30_e31_psp_fusion_master.log
"$PY" train_fpp_phase2depth_unet.py \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PSP_CACHE" \
  --save_dir "$E31_DIR" \
  --input_mode gt_phase_plus_fringe \
  --init_checkpoint "$INIT_RAW" \
  --epochs 40 \
  --batch_size 2 \
  --eval_batch_size 2 \
  --num_workers 8 \
  --image_size 960 \
  --lr 2e-5 \
  --weight_decay 1e-5 \
  --alpha 0.7 \
  --eval_metrics_every 1 \
  --save_every 5 \
  --eval_initial \
  --seed 42 \
  2>&1 | tee results/remote_logs/fpp960_e31_pspquad_gt_plus_fringe_from_rawA_e40.log

echo "DONE E30/E31 PSP fusion $(date '+%F %T')" \
  | tee -a results/remote_logs/e30_e31_psp_fusion_master.log
