#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs remote_runs

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PHASE_CACHE=/root/autodl-tmp/fpp_ml_phase_cache_960
MASTER=results/remote_logs/e53_existing_globaluph_sampling_sweep_master.log

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_case() {
  local name="$1"
  local ckpt="$2"
  local start="$3"
  local ratio="$4"
  local steps="$5"
  local prefix="phase_pred_${name}_${start}_s${ratio/./p}_ddim${steps}_e1"
  local out="results/${name}_${start}_s${ratio/./p}_calibrated_depth"

  read GMIN GMAX < <("$PY" - "$ckpt" <<'PY'
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu")
args = ckpt.get("args", {})
print(args.get("uph_global_min", 0.0), args.get("uph_global_max", 1.0))
PY
)
  echo "START ${name} start=${start} ratio=${ratio} steps=${steps} gmin=${GMIN} gmax=${GMAX} $(date '+%F %T')" | tee -a "$MASTER"
  "$PY" precompute_fpp_phase_diffusion_predictions.py \
    --checkpoint "$ckpt" \
    --base_cache_dir "$BASE_CACHE" \
    --phase_cache_dir "$PHASE_CACHE" \
    --output_prefix "$prefix" \
    --splits train,val,test \
    --image_size 960 \
    --batch_size 1 \
    --num_workers 8 \
    --ddim_steps "$steps" \
    --ensemble 1 \
    --sample_start_from "$start" \
    --sample_start_ratio "$ratio" \
    2>&1 | tee "results/remote_logs/${prefix}_precompute.log"

  "$PY" eval_phase_calibrated_depth.py \
    --base_cache_dir "$BASE_CACHE" \
    --phase_cache_dir "$PHASE_CACHE" \
    --phase_pred_prefix "$prefix" \
    --save_dir "$out" \
    --degree 2 \
    --fit_step 8 \
    --eval_step 2 \
    --max_train_pixels 300000 \
    --pred_global_min "$GMIN" \
    --pred_global_max "$GMAX" \
    2>&1 | tee "results/remote_logs/${prefix}_calibrated_depth.log"
  echo "DONE ${name} start=${start} ratio=${ratio} steps=${steps} $(date '+%F %T')" | tee -a "$MASTER"
}

echo "START E53 existing global-UPH sampling sweep $(date '+%F %T')" | tee "$MASTER"
E25=results/fpp960_e25_globaluph_phase_diffusion_ch24_e16/checkpoints/best_phase.pt
E27=results/fpp960_e27_globaluph_calibdepth_w002_clip300_ch24_e20/checkpoints/best_phase.pt

run_case e53_e25 "$E25" noise 1.0 50
run_case e53_e25 "$E25" ftp 0.3 20
run_case e53_e25 "$E25" ftp 0.5 20
run_case e53_e27 "$E27" noise 1.0 50
run_case e53_e27 "$E27" ftp 0.3 20
run_case e53_e27 "$E27" ftp 0.5 20

echo "DONE E53 existing global-UPH sampling sweep $(date '+%F %T')" | tee -a "$MASTER"
