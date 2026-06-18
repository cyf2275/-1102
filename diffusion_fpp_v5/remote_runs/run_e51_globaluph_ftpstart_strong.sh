#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs remote_runs

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PHASE_CACHE=/root/autodl-tmp/fpp_ml_phase_cache_960
CALIB_JSON=results/fpp960_e24_phase_calibrated_depth_e5pred/phase_calibrated_depth_summary.json
SAVE_DIR=results/fpp960_e51_globaluph_ftpstart_strong_ch24_e30
MASTER=results/remote_logs/e51_globaluph_ftpstart_strong_master.log

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "START E51 global-UPH FTP-start strong diffusion $(date '+%F %T')" | tee "$MASTER"
"$PY" train_fpp_phase_diffusion.py \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PHASE_CACHE" \
  --save_dir "$SAVE_DIR" \
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
  --sample_start_ratio 0.7 \
  --train_start_from ftp \
  --train_t_min_ratio 0.0 \
  --train_t_max_ratio 1.0 \
  --grad_weight 0.02 \
  --unit_weight 0.02 \
  --uph_norm global \
  --uph_weight 8.0 \
  --uph_grad_weight 0.20 \
  --uph_start_from coord_auto \
  --selection_metric phase_uph_score \
  --uph_select_weight 12.0 \
  --calib_depth_weight 0.05 \
  --calib_depth_summary "$CALIB_JSON" \
  --calib_depth_source gt_raw \
  --calib_depth_scale 100.0 \
  --calib_depth_clip 200.0 \
  --calib_depth_start_epoch 1 \
  --seed 42 \
  --save_every 1 \
  2>&1 | tee results/remote_logs/fpp960_e51_globaluph_ftpstart_strong_ch24_e30.log

read GMIN GMAX < <("$PY" - <<'PY'
import torch
ckpt = torch.load("results/fpp960_e51_globaluph_ftpstart_strong_ch24_e30/checkpoints/best_phase.pt", map_location="cpu")
args = ckpt.get("args", {})
print(args.get("uph_global_min", 0.0), args.get("uph_global_max", 1.0))
PY
)

for RATIO in 0.3 0.5 0.7; do
  TAG="${RATIO/./p}"
  PREFIX="phase_pred_e51_globaluph_ftpstart_s${TAG}_ddim20_e1"
  echo "PRECOMPUTE E51 ratio=${RATIO} prefix=${PREFIX} $(date '+%F %T')" | tee -a "$MASTER"
  "$PY" precompute_fpp_phase_diffusion_predictions.py \
    --checkpoint "$SAVE_DIR/checkpoints/best_phase.pt" \
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
    --sample_start_ratio "$RATIO" \
    2>&1 | tee "results/remote_logs/fpp960_e51_precompute_s${TAG}.log"

  echo "EVAL E51 calibrated depth ratio=${RATIO} gmin=$GMIN gmax=$GMAX $(date '+%F %T')" | tee -a "$MASTER"
  "$PY" eval_phase_calibrated_depth.py \
    --base_cache_dir "$BASE_CACHE" \
    --phase_cache_dir "$PHASE_CACHE" \
    --phase_pred_prefix "$PREFIX" \
    --save_dir "results/fpp960_e51_globaluph_calibrated_depth_s${TAG}_e1" \
    --degree 2 \
    --fit_step 8 \
    --eval_step 2 \
    --max_train_pixels 300000 \
    --pred_global_min "$GMIN" \
    --pred_global_max "$GMAX" \
    2>&1 | tee "results/remote_logs/fpp960_e51_calibrated_depth_s${TAG}_e1.log"
done

echo "DONE E51 global-UPH FTP-start strong diffusion $(date '+%F %T')" | tee -a "$MASTER"
