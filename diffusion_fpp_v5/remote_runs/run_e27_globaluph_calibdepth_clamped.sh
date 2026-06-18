#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PHASE_CACHE=/root/autodl-tmp/fpp_ml_phase_cache_960
CALIB_JSON=results/fpp960_e24_phase_calibrated_depth_e5pred/phase_calibrated_depth_summary.json
SAVE_DIR=results/fpp960_e27_globaluph_calibdepth_w002_clip300_ch24_e20
PRED_PREFIX=phase_pred_e27_globaluph_calibdepth_w002_clip300_ddim20_e3

echo "START E27 clamped global-uph calibrated-depth diffusion $(date '+%F %T')" \
  | tee results/remote_logs/e27_globaluph_calibdepth_clamped_master.log

"$PY" train_fpp_phase_diffusion.py \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PHASE_CACHE" \
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
  --uph_norm global \
  --uph_weight 2.0 \
  --uph_grad_weight 0.05 \
  --uph_start_from coord_auto \
  --selection_metric phase_uph_score \
  --uph_select_weight 4.0 \
  --calib_depth_weight 0.02 \
  --calib_depth_summary "$CALIB_JSON" \
  --calib_depth_source gt_raw \
  --calib_depth_scale 200.0 \
  --calib_depth_clip 300.0 \
  --calib_depth_start_epoch 1 \
  --seed 42 \
  --save_every 1 \
  2>&1 | tee results/remote_logs/fpp960_e27_globaluph_calibdepth_w002_clip300_ch24_e20.log

echo "PRECOMPUTE E27 predictions $(date '+%F %T')" \
  | tee -a results/remote_logs/e27_globaluph_calibdepth_clamped_master.log
"$PY" precompute_fpp_phase_diffusion_predictions.py \
  --checkpoint "$SAVE_DIR/checkpoints/best_phase.pt" \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PHASE_CACHE" \
  --output_prefix "$PRED_PREFIX" \
  --splits train,val,test \
  --image_size 960 \
  --batch_size 1 \
  --num_workers 8 \
  --ddim_steps 20 \
  --ensemble 3 \
  --sample_start_from ftp \
  --sample_start_ratio 0.7 \
  2>&1 | tee results/remote_logs/fpp960_e27_globaluph_calibdepth_precompute.log

read GMIN GMAX < <("$PY" - <<'PY'
import torch
ckpt = torch.load("results/fpp960_e27_globaluph_calibdepth_w002_clip300_ch24_e20/checkpoints/best_phase.pt", map_location="cpu")
args = ckpt.get("args", {})
print(args.get("uph_global_min", 0.0), args.get("uph_global_max", 1.0))
PY
)

echo "EVAL E27 calibrated depth gmin=$GMIN gmax=$GMAX $(date '+%F %T')" \
  | tee -a results/remote_logs/e27_globaluph_calibdepth_clamped_master.log
"$PY" eval_phase_calibrated_depth.py \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PHASE_CACHE" \
  --phase_pred_prefix "$PRED_PREFIX" \
  --save_dir results/fpp960_e27_globaluph_calibdepth_depth_eval \
  --degree 2 \
  --fit_step 8 \
  --eval_step 2 \
  --max_train_pixels 300000 \
  --pred_global_min "$GMIN" \
  --pred_global_max "$GMAX" \
  2>&1 | tee results/remote_logs/fpp960_e27_globaluph_calibdepth_depth_eval.log

echo "DONE E27 clamped global-uph calibrated-depth diffusion $(date '+%F %T')" \
  | tee -a results/remote_logs/e27_globaluph_calibdepth_clamped_master.log
