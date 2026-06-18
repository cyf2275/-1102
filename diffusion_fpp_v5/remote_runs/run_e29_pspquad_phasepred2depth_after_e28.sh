#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PSP_CACHE=/root/autodl-tmp/fpp_ml_pspquad_cache_960
PRED_PREFIX=phase_pred_e28_pspquad_ddim20_e3
INIT=results/fpp960_e11b_gtphase_phase2depth_full_from_e11a_lr3e5/checkpoints/best_rmse.pt
SAVE_DIR=results/fpp960_e29_pspquad_phasepred2depth_from_e11b_e30

echo "WAIT E28 before E29 $(date '+%F %T')" | tee results/remote_logs/e29_pspquad_phasepred2depth_master.log
while true; do
  if [[ -f "$PSP_CACHE/${PRED_PREFIX}_test_float16.npy" ]]; then
    if ! pgrep -f "fpp960_e28_pspquad_phase_diffusion_ch24_e20|${PRED_PREFIX}" >/dev/null 2>&1; then
      break
    fi
  fi
  sleep 30
done

echo "START E29 PSP quadrature phase-pred-to-depth $(date '+%F %T')" \
  | tee -a results/remote_logs/e29_pspquad_phasepred2depth_master.log
"$PY" train_fpp_phase2depth_unet.py \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PSP_CACHE" \
  --phase_pred_prefix "$PRED_PREFIX" \
  --save_dir "$SAVE_DIR" \
  --input_mode phase_pred \
  --init_checkpoint "$INIT" \
  --epochs 30 \
  --batch_size 2 \
  --eval_batch_size 2 \
  --num_workers 8 \
  --image_size 960 \
  --lr 3e-5 \
  --weight_decay 1e-5 \
  --alpha 0.7 \
  --eval_metrics_every 1 \
  --save_every 1 \
  --eval_initial \
  --seed 42 \
  2>&1 | tee results/remote_logs/fpp960_e29_pspquad_phasepred2depth_from_e11b_e30.log

echo "DONE E29 PSP quadrature phase-pred-to-depth $(date '+%F %T')" \
  | tee -a results/remote_logs/e29_pspquad_phasepred2depth_master.log
