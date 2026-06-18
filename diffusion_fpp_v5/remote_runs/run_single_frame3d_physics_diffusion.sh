#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
CODE_ROOT="${CODE_ROOT:-/root/autodl-tmp/diffusion_fpp_v5}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/single_frame_3d_dataset_v1_upload_smalltest}"
RESULT_ROOT="${RESULT_ROOT:-/root/autodl-tmp/diffusion_fpp_v5/results/A_20260614_single_frame3d_physics_diffusion}"
IMAGE_H="${IMAGE_H:-480}"
IMAGE_W="${IMAGE_W:-640}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DIRECT_EPOCHS="${DIRECT_EPOCHS:-40}"
RESIDUAL_EPOCHS="${RESIDUAL_EPOCHS:-40}"
TRAIN_EPOCH_REPEATS="${TRAIN_EPOCH_REPEATS:-1}"
EVAL_EVERY="${EVAL_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-0}"
SAMPLE_STEPS="${SAMPLE_STEPS:-12}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-3}"
BASE_CHANNELS="${BASE_CHANNELS:-32}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-${DATA_ROOT}/physics_feature_cache_pip}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${CODE_ROOT}"
mkdir -p "${RESULT_ROOT}/runs" "${RESULT_ROOT}/logs"

echo "[0/3] precompute physics feature cache -> ${FEATURE_CACHE_DIR}"
"${PYTHON}" train_single_frame3d_physics_diffusion.py \
  --stage precompute_features \
  --data_root "${DATA_ROOT}" \
  --feature_cache_dir "${FEATURE_CACHE_DIR}" \
  2>&1 | tee "${RESULT_ROOT}/logs/precompute_features.log"

run_direct() {
  local config="$1"
  local seed="$2"
  local save_dir="${RESULT_ROOT}/runs/direct_${config}_seed${seed}"
  local log="${RESULT_ROOT}/logs/direct_${config}_seed${seed}.log"
  if [[ -f "${save_dir}/evaluation/summary.json" ]]; then
    echo "Skip existing direct ${config} seed ${seed}"
    return 0
  fi
  echo "Direct config=${config} seed=${seed}"
  "${PYTHON}" train_single_frame3d_physics_diffusion.py \
    --stage direct \
    --data_root "${DATA_ROOT}" \
    --save_dir "${save_dir}" \
    --config "${config}" \
    --seed "${seed}" \
    --epochs "${DIRECT_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --train_epoch_repeats "${TRAIN_EPOCH_REPEATS}" \
    --image_h "${IMAGE_H}" \
    --image_w "${IMAGE_W}" \
    --base_channels "${BASE_CHANNELS}" \
    --eval_every "${EVAL_EVERY}" \
    --save_every "${SAVE_EVERY}" \
    --feature_cache_dir "${FEATURE_CACHE_DIR}" \
    --object_mask_weight 3.0 \
    2>&1 | tee "${log}"
}

run_residual() {
  local config="$1"
  local seed="$2"
  local base_ckpt="${RESULT_ROOT}/runs/direct_${config}_seed${seed}/checkpoints/best.pt"
  local save_dir="${RESULT_ROOT}/runs/residual_${config}_seed${seed}"
  local log="${RESULT_ROOT}/logs/residual_${config}_seed${seed}.log"
  if [[ ! -f "${base_ckpt}" ]]; then
    echo "Missing direct base checkpoint: ${base_ckpt}" >&2
    return 1
  fi
  if [[ -f "${save_dir}/evaluation/summary.json" ]]; then
    echo "Skip existing residual ${config} seed ${seed}"
    return 0
  fi
  echo "Residual config=${config} seed=${seed}"
  "${PYTHON}" train_single_frame3d_physics_diffusion.py \
    --stage residual \
    --data_root "${DATA_ROOT}" \
    --save_dir "${save_dir}" \
    --config "${config}" \
    --base_ckpt "${base_ckpt}" \
    --seed "${seed}" \
    --epochs "${RESIDUAL_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --train_epoch_repeats "${TRAIN_EPOCH_REPEATS}" \
    --image_h "${IMAGE_H}" \
    --image_w "${IMAGE_W}" \
    --base_channels "${BASE_CHANNELS}" \
    --eval_every "${EVAL_EVERY}" \
    --save_every "${SAVE_EVERY}" \
    --feature_cache_dir "${FEATURE_CACHE_DIR}" \
    --object_mask_weight 3.0 \
    --sample_steps "${SAMPLE_STEPS}" \
    --ensemble_size "${ENSEMBLE_SIZE}" \
    2>&1 | tee "${log}"
}

for config in raw raw_xy raw_single_phys teacher_aux; do
  for seed in 0 1 2; do
    run_direct "${config}" "${seed}"
  done
done

for config in raw raw_single_phys teacher_aux; do
  for seed in 0 1 2; do
    run_residual "${config}" "${seed}"
  done
done

"${PYTHON}" train_single_frame3d_physics_diffusion.py \
  --stage summarize \
  --data_root "${DATA_ROOT}" \
  --save_dir "${RESULT_ROOT}"

tar -czf "${RESULT_ROOT}/single_frame3d_physics_diffusion_bundle.tar.gz" \
  -C "${RESULT_ROOT}" \
  single_frame3d_physics_diffusion_report.md \
  single_frame3d_physics_diffusion_summary.json \
  direct_aggregated_results.csv \
  residual_aggregated_results.csv \
  logs \
  runs/*/evaluation/summary.json \
  runs/*/evaluation/per_sample_metrics.csv \
  2>/dev/null || true

echo "Done: ${RESULT_ROOT}/single_frame3d_physics_diffusion_report.md"
