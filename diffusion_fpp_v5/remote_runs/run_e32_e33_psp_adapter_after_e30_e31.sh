#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PSP_CACHE=/root/autodl-tmp/fpp_ml_pspquad_cache_960
PRED_PREFIX=phase_pred_e28_pspquad_ddim20_e3
INIT_RAW=results/fpp960_a_fringe_unet_control/checkpoints/best.pt

E32_DIR=results/fpp960_e32d_pspquad_pred_xy_adapter_freeze_rawA_bs6_mintrain_e80
E33_DIR=results/fpp960_e33d_pspquad_gt_xy_adapter_freeze_rawA_bs6_mintrain_e60

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "WAIT E30/E31 before E32/E33 $(date '+%F %T')" \
  | tee results/remote_logs/e32_e33_psp_adapter_master.log
while pgrep -f "fpp960_e30_pspquad_pred_plus_fringe|fpp960_e31_pspquad_gt_plus_fringe" >/dev/null 2>&1; do
  sleep 60
done

echo "START E32 predicted PSP adapter $(date '+%F %T')" \
  | tee -a results/remote_logs/e32_e33_psp_adapter_master.log
"$PY" train_fpp_psp_adapter_unet.py \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PSP_CACHE" \
  --phase_pred_prefix "$PRED_PREFIX" \
  --save_dir "$E32_DIR" \
  --base_checkpoint "$INIT_RAW" \
  --cond_mode phase_pred_xy \
  --freeze_backbone \
  --epochs 80 \
  --batch_size 6 \
  --eval_batch_size 6 \
  --num_workers 16 \
  --image_size 960 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --adapter_hidden 32 \
  --alpha 0.7 \
  --eval_every 5 \
  --eval_metrics_every 5 \
  --save_every 5 \
  --eval_initial \
  --train_minimal \
  --seed 42 \
  2>&1 | tee results/remote_logs/fpp960_e32d_pspquad_pred_xy_adapter_freeze_rawA_bs6_mintrain_e80.log

echo "START E33 GT PSP adapter upper bound $(date '+%F %T')" \
  | tee -a results/remote_logs/e32_e33_psp_adapter_master.log
"$PY" train_fpp_psp_adapter_unet.py \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PSP_CACHE" \
  --save_dir "$E33_DIR" \
  --base_checkpoint "$INIT_RAW" \
  --cond_mode gt_psp_xy \
  --freeze_backbone \
  --epochs 60 \
  --batch_size 6 \
  --eval_batch_size 6 \
  --num_workers 16 \
  --image_size 960 \
  --lr 1e-3 \
  --weight_decay 1e-5 \
  --adapter_hidden 32 \
  --alpha 0.7 \
  --eval_every 5 \
  --eval_metrics_every 5 \
  --save_every 5 \
  --eval_initial \
  --train_minimal \
  --seed 42 \
  2>&1 | tee results/remote_logs/fpp960_e33d_pspquad_gt_xy_adapter_freeze_rawA_bs6_mintrain_e60.log

echo "DONE E32/E33 PSP adapter $(date '+%F %T')" \
  | tee -a results/remote_logs/e32_e33_psp_adapter_master.log
