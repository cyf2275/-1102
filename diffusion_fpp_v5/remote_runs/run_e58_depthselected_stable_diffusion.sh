#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PHASE_CACHE=/root/autodl-tmp/fpp_ml_phase_cache_960
PRIOR=results/e54_uph_prior_fit_xy2_phase/uph_prior_summary.json
CALIB=results/fpp960_e24_phase_calibrated_depth_e5pred/phase_calibrated_depth_summary.json
SAVE=results/fpp960_e58_depthselected_stable_diffusion_ch24_e30
PREFIX=phase_pred_e58_depthselected_stable_ddim20_ftp01
DEPTH_OUT=results/fpp960_e58_depthselected_stable_calibrated_depth

echo "START E58 stable depth-selected diffusion $(date '+%F %T')" | tee results/remote_logs/e58_depthselected_stable_master.log

"$PY" train_fpp_phase_diffusion.py \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PHASE_CACHE" \
  --save_dir "$SAVE" \
  --phase_channels 0-12 \
  --epochs 30 \
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
  --ensemble 1 \
  --sample_start_from ftp \
  --sample_start_ratio 0.1 \
  --train_start_from ftp \
  --train_t_min_ratio 0.0 \
  --train_t_max_ratio 0.2 \
  --phase_weight 0.05 \
  --grad_weight 0.005 \
  --unit_weight 0.005 \
  --uph_norm global \
  --uph_representation prior_residual \
  --uph_prior_summary "$PRIOR" \
  --uph_start_from half \
  --uph_weight 5.0 \
  --uph_grad_weight 0.05 \
  --selection_metric calib_depth_rmse_mm \
  --calib_depth_weight 0.20 \
  --calib_depth_summary "$CALIB" \
  --calib_depth_source gt_raw \
  --calib_depth_scale 100.0 \
  --calib_depth_clip 200.0 \
  --calib_depth_start_epoch 1 \
  --lr 5e-5 \
  --weight_decay 1e-5 \
  --grad_clip 0.5 \
  --eval_every 1 \
  --save_every 1 \
  --seed 46 \
  --no_amp \
  2>&1 | tee results/remote_logs/fpp960_e58_depthselected_stable_diffusion_ch24_e30.log

echo "PRECOMPUTE E58 $(date '+%F %T')" | tee -a results/remote_logs/e58_depthselected_stable_master.log
"$PY" precompute_fpp_phase_diffusion_predictions.py \
  --checkpoint "$SAVE/checkpoints/best_phase.pt" \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PHASE_CACHE" \
  --output_prefix "$PREFIX" \
  --splits train,val,test \
  --image_size 960 \
  --batch_size 1 \
  --num_workers 8 \
  --ddim_steps 20 \
  --ensemble 1 \
  --sample_start_from ftp \
  --sample_start_ratio 0.1 \
  2>&1 | tee results/remote_logs/e58_depthselected_stable_precompute.log

echo "CALIBRATED DEPTH E58 $(date '+%F %T')" | tee -a results/remote_logs/e58_depthselected_stable_master.log
"$PY" eval_phase_calibrated_depth.py \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PHASE_CACHE" \
  --phase_pred_prefix "$PREFIX" \
  --save_dir "$DEPTH_OUT" \
  --degree 2 \
  --fit_step 8 \
  --eval_step 2 \
  --max_train_pixels 300000 \
  --pred_uph_representation prior_residual \
  --uph_prior_summary "$PRIOR" \
  2>&1 | tee results/remote_logs/e58_depthselected_stable_calibrated_depth.log

echo "DONE E58 stable depth-selected diffusion $(date '+%F %T')" | tee -a results/remote_logs/e58_depthselected_stable_master.log
