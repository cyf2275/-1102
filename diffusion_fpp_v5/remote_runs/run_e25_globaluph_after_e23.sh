#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PHASE_CACHE=/root/autodl-tmp/fpp_ml_phase_cache_960
E23_PID_FILE=results/remote_logs/e23_phase_ablation_queue.pid
MASTER_LOG=results/remote_logs/e25_globaluph_after_e23_master.log

echo "WATCH e23 queue before E25 $(date '+%F %T')" | tee -a "$MASTER_LOG"
while true; do
  if [[ -f "$E23_PID_FILE" ]]; then
    E23_PID="$(cat "$E23_PID_FILE" || true)"
    if [[ -n "$E23_PID" ]] && kill -0 "$E23_PID" 2>/dev/null; then
      sleep 20
      continue
    fi
  fi
  if pgrep -f "train_fpp_phase_diffusion.py" >/dev/null 2>&1; then
    sleep 20
    continue
  fi
  break
done

echo "START E25 global absolute uph diffusion $(date '+%F %T')" | tee -a "$MASTER_LOG"
"$PY" train_fpp_phase_diffusion.py \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PHASE_CACHE" \
  --save_dir results/fpp960_e25_globaluph_phase_diffusion_ch24_e16 \
  --phase_channels 0-12 \
  --epochs 16 \
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
  --seed 42 \
  --save_every 1 \
  2>&1 | tee results/remote_logs/fpp960_e25_globaluph_phase_diffusion_ch24_e16.log

echo "PRECOMPUTE E25 predictions $(date '+%F %T')" | tee -a "$MASTER_LOG"
"$PY" precompute_fpp_phase_diffusion_predictions.py \
  --checkpoint results/fpp960_e25_globaluph_phase_diffusion_ch24_e16/checkpoints/best_phase.pt \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PHASE_CACHE" \
  --output_prefix phase_pred_e25_globaluph_ddim20_e3 \
  --splits train,val,test \
  --image_size 960 \
  --batch_size 1 \
  --num_workers 8 \
  --ddim_steps 20 \
  --ensemble 3 \
  --sample_start_from ftp \
  --sample_start_ratio 0.7 \
  2>&1 | tee results/remote_logs/fpp960_e25_globaluph_precompute.log

read GMIN GMAX < <("$PY" - <<'PY'
import torch
ckpt = torch.load("results/fpp960_e25_globaluph_phase_diffusion_ch24_e16/checkpoints/best_phase.pt", map_location="cpu")
args = ckpt.get("args", {})
print(args.get("uph_global_min", 0.0), args.get("uph_global_max", 1.0))
PY
)

echo "EVAL E25 calibrated depth gmin=$GMIN gmax=$GMAX $(date '+%F %T')" | tee -a "$MASTER_LOG"
"$PY" eval_phase_calibrated_depth.py \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PHASE_CACHE" \
  --phase_pred_prefix phase_pred_e25_globaluph_ddim20_e3 \
  --save_dir results/fpp960_e25_globaluph_calibrated_depth \
  --degree 2 \
  --fit_step 8 \
  --eval_step 2 \
  --max_train_pixels 300000 \
  --pred_global_min "$GMIN" \
  --pred_global_max "$GMAX" \
  2>&1 | tee results/remote_logs/fpp960_e25_globaluph_calibrated_depth.log

echo "DONE E25 global uph route $(date '+%F %T')" | tee -a "$MASTER_LOG"
