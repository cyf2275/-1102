#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PSP_CACHE=/root/autodl-tmp/fpp_ml_pspquad_cache_960
SAVE_DIR=results/fpp960_e28_pspquad_phase_diffusion_ch24_e20
PRED_PREFIX=phase_pred_e28_pspquad_ddim20_e3

echo "WAIT E27 before E28 $(date '+%F %T')" | tee results/remote_logs/e28_pspquad_master.log
while pgrep -f "fpp960_e27_globaluph_calibdepth_w002_clip300_ch24_e20|phase_pred_e27_globaluph_calibdepth|fpp960_e27_globaluph_calibdepth_depth_eval" >/dev/null 2>&1; do
  sleep 30
done

if [[ ! -f "$PSP_CACHE/phase_cache_manifest.json" ]]; then
  while [[ -f "$PSP_CACHE/.building" ]]; do
    echo "WAIT PSP quadrature cache build $(date '+%F %T')" | tee -a results/remote_logs/e28_pspquad_master.log
    sleep 30
  done
fi

if [[ ! -f "$PSP_CACHE/phase_cache_manifest.json" ]]; then
  mkdir -p "$PSP_CACHE"
  touch "$PSP_CACHE/.building"
  trap 'rm -f "$PSP_CACHE/.building"' EXIT
  echo "BUILD PSP quadrature cache $(date '+%F %T')" | tee -a results/remote_logs/e28_pspquad_master.log
  "$PY" prepare_fpp_psp_quad_cache.py \
    --base_cache_dir "$BASE_CACHE" \
    --source_phase_cache_dir /root/autodl-tmp/fpp_ml_phase_cache_960 \
    --output_phase_cache_dir "$PSP_CACHE" \
    --raw_root /root/autodl-tmp/datasets/fpp-ml-bench/fpp_synthetic_dataset \
    --steps 18 \
    2>&1 | tee results/remote_logs/prepare_fpp_pspquad_cache_960.log
  rm -f "$PSP_CACHE/.building"
  trap - EXIT
fi

echo "START E28 PSP quadrature diffusion $(date '+%F %T')" | tee -a results/remote_logs/e28_pspquad_master.log
"$PY" train_fpp_phase_diffusion.py \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PSP_CACHE" \
  --save_dir "$SAVE_DIR" \
  --phase_channels 0-12 \
  --epochs 20 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --num_workers 8 \
  --image_size 960 \
  --base_channels 24 \
  --ch_mult 1,2,4,8,8 \
  --adapter_hidden 24 \
  --dropout 0.05 \
  --target_channels 3 \
  --timesteps 200 \
  --ddim_steps 20 \
  --ensemble 3 \
  --sample_start_from ftp \
  --sample_start_ratio 0.7 \
  --train_start_from target \
  --grad_weight 0.05 \
  --unit_weight 0.02 \
  --uph_norm sample \
  --uph_weight 0.5 \
  --uph_grad_weight 0.02 \
  --uph_start_from half \
  --selection_metric phase_aligned_mae_rad \
  --seed 42 \
  --save_every 1 \
  2>&1 | tee results/remote_logs/fpp960_e28_pspquad_phase_diffusion_ch24_e20.log

echo "PRECOMPUTE E28 predictions $(date '+%F %T')" | tee -a results/remote_logs/e28_pspquad_master.log
"$PY" precompute_fpp_phase_diffusion_predictions.py \
  --checkpoint "$SAVE_DIR/checkpoints/best_phase.pt" \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PSP_CACHE" \
  --output_prefix "$PRED_PREFIX" \
  --splits train,val,test \
  --image_size 960 \
  --batch_size 1 \
  --num_workers 8 \
  --ddim_steps 20 \
  --ensemble 3 \
  --sample_start_from ftp \
  --sample_start_ratio 0.7 \
  2>&1 | tee results/remote_logs/fpp960_e28_pspquad_precompute.log

echo "DONE E28 PSP quadrature diffusion $(date '+%F %T')" | tee -a results/remote_logs/e28_pspquad_master.log
