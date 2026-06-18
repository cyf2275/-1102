#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PHASE_CACHE=/root/autodl-tmp/fpp_ml_phase_cache_960
SAVE_DIR=results/fpp960_e27_globaluph_calibdepth_w002_clip300_ch24_e20
PRED_PREFIX=phase_pred_e27_globaluph_calibdepth_w002_clip300_ddim20_e3

echo "POST E27 precompute $(date '+%F %T')" | tee results/remote_logs/e27_post_then_e28.log
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

echo "POST E27 calibrated depth gmin=$GMIN gmax=$GMAX $(date '+%F %T')" \
  | tee -a results/remote_logs/e27_post_then_e28.log
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

echo "START E28 after E27 post $(date '+%F %T')" | tee -a results/remote_logs/e27_post_then_e28.log
bash remote_runs/run_e28_pspquad_after_e27.sh
