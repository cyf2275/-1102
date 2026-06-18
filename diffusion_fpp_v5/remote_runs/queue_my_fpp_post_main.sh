#!/usr/bin/env bash
set -euo pipefail

# Post-main queue for the real-capture my_fpp validation.
# It waits for the currently running input-ablation batch, summarizes it,
# selects the best legal physics input from validation histories, then
# runs the small-scale residual posterior pilot against raw.

PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
CODE_ROOT="${CODE_ROOT:-/root/autodl-tmp/diffusion_fpp_v5}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
PROCESSED_DIR="${PROCESSED_DIR:-/root/autodl-tmp/orderfix_0610_cleanmask_v1}"
SPLIT_DIR="${SPLIT_DIR:-/root/autodl-tmp/splits}"
RESULT_ROOT="${RESULT_ROOT:-/root/autodl-tmp/diffusion_fpp_v5/results/A_20260611_my_fpp_physics_validation_gpuopt_full}"
IMAGE_H="${IMAGE_H:-480}"
IMAGE_W="${IMAGE_W:-640}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BASE_CHANNELS="${BASE_CHANNELS:-32}"
EPOCHS="${EPOCHS:-40}"
TRAIN_EPOCH_REPEATS="${TRAIN_EPOCH_REPEATS:-4}"
EVAL_EVERY="${EVAL_EVERY:-5}"
WAIT_PID="${WAIT_PID:-}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"
}

wait_for_main() {
  if [[ -z "${WAIT_PID}" && -f "${RESULT_ROOT}/full_run.pid" ]]; then
    WAIT_PID="$(tr -dc '0-9' < "${RESULT_ROOT}/full_run.pid" || true)"
  fi

  if [[ -n "${WAIT_PID}" ]] && kill -0 "${WAIT_PID}" 2>/dev/null; then
    log "Waiting for main validation process PID=${WAIT_PID}"
    while kill -0 "${WAIT_PID}" 2>/dev/null; do
      sleep "${WAIT_POLL_SECONDS}"
    done
    log "Main validation process PID=${WAIT_PID} has exited"
  else
    log "No live WAIT_PID found; checking for active train/run processes under RESULT_ROOT"
  fi

  while pgrep -af "train_my_fpp_.*${RESULT_ROOT}" >/dev/null 2>&1; do
    log "A training process for this result root is still active; waiting"
    sleep "${WAIT_POLL_SECONDS}"
  done
}

pick_best_legal_physics_config() {
  "${PYTHON}" - "${RESULT_ROOT}" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = []
for config in ("raw_xy", "raw_single_phys"):
    scores = []
    for history_path in sorted((root / "runs").glob(f"{config}_mw3_seed*/history.json")):
        eval_summary = history_path.parent / "evaluation" / "summary.json"
        if not eval_summary.exists():
            continue
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        vals = []
        if isinstance(history, list):
            for item in history:
                if not isinstance(item, dict) or "val_object_rmse" not in item:
                    continue
                try:
                    value = float(item["val_object_rmse"])
                except Exception:
                    continue
                if math.isfinite(value):
                    vals.append(value)
        if vals:
            scores.append(min(vals))
    if scores:
        candidates.append((statistics.median(scores), config, len(scores)))

if not candidates:
    raise SystemExit("No completed legal physics validation history found for diffusion pilot selection.")

candidates.sort()
print(candidates[0][1])
PY
}

run_diffusion_pilot() {
  local config="$1"
  local seed="$2"
  local safe_name="${config//+/_}"
  safe_name="${safe_name//raw_xy/raw_xy}"
  safe_name="${safe_name//raw_single_phys/raw_single_phys}"
  local save_dir="${RESULT_ROOT}/runs/diffusion_pilot_${safe_name}_seed${seed}"

  if [[ -f "${save_dir}/evaluation/summary.json" ]]; then
    log "Skip existing diffusion pilot: ${config} seed ${seed}"
    return
  fi

  local extra=()
  if [[ "${config}" == "raw_single_phys" || "${config}" == "raw+single_phys" ]]; then
    extra+=(--cache_features)
  fi

  log "Run diffusion pilot: ${config} seed ${seed}"
  "${PYTHON}" train_my_fpp_diffusion_pilot.py \
    --data_root "${DATA_ROOT}" \
    --processed_dir "${PROCESSED_DIR}" \
    --split_dir "${SPLIT_DIR}" \
    --config "${config}" \
    --seed "${seed}" \
    --image_h "${IMAGE_H}" \
    --image_w "${IMAGE_W}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --base_channels "${BASE_CHANNELS}" \
    --time_emb_dim 128 \
    --epochs "${EPOCHS}" \
    --train_epoch_repeats "${TRAIN_EPOCH_REPEATS}" \
    --eval_every "${EVAL_EVERY}" \
    --save_every 10 \
    --save_dir "${save_dir}" \
    "${extra[@]}"
}

package_outputs() {
  local cloud_root="/root/autodl-tmp/cloud_results"
  local cloud_link="${cloud_root}/$(basename "${RESULT_ROOT}")"
  mkdir -p "${cloud_root}"
  ln -sfn "${RESULT_ROOT}" "${cloud_link}"

  log "Packaging final report bundle"
  tar -czf "${RESULT_ROOT}/final_report_bundle.tar.gz" \
    -C "${RESULT_ROOT}" \
    physics_validation_report.md \
    physics_validation_summary.json \
    aggregated_results.csv \
    all_run_results.csv \
    runs/*/evaluation/summary.json \
    runs/*/evaluation/per_sample_metrics.csv \
    2>/dev/null || true
  log "Cloud-results link: ${cloud_link}"
}

main() {
  cd "${CODE_ROOT}"
  mkdir -p "${RESULT_ROOT}/runs"

  wait_for_main

  log "Summarizing completed main validation runs"
  "${PYTHON}" summarize_my_fpp_physics_validation.py --results_root "${RESULT_ROOT}"

  local best_config
  best_config="$(pick_best_legal_physics_config)"
  log "Best legal physics config selected for pilot: ${best_config}"

  for config in raw "${best_config}"; do
    for seed in 0 1 2; do
      run_diffusion_pilot "${config}" "${seed}"
    done
  done

  log "Final summary after diffusion pilot"
  "${PYTHON}" summarize_my_fpp_physics_validation.py --results_root "${RESULT_ROOT}"
  package_outputs
  log "Post-main queue done: ${RESULT_ROOT}/physics_validation_report.md"
}

main "$@"
