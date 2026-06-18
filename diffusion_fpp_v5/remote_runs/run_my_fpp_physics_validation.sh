#!/usr/bin/env bash
set -euo pipefail

# Real-capture physics-input validation for my_fpp_dataset_v1.
# This script assumes the server data layout prepared on 2026-06-11:
#   /root/autodl-tmp/orderfix_0610_cleanmask_v1
#   /root/autodl-tmp/splits
#
# It writes only experiment outputs. It does not modify the NPZ dataset.

PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
CODE_ROOT="${CODE_ROOT:-/root/autodl-tmp/diffusion_fpp_v5}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
PROCESSED_DIR="${PROCESSED_DIR:-/root/autodl-tmp/orderfix_0610_cleanmask_v1}"
SPLIT_DIR="${SPLIT_DIR:-/root/autodl-tmp/splits}"
RESULT_ROOT="${RESULT_ROOT:-/root/autodl-tmp/diffusion_fpp_v5/results/A_20260611_my_fpp_physics_validation}"
IMAGE_H="${IMAGE_H:-480}"
IMAGE_W="${IMAGE_W:-640}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BASE_CHANNELS="${BASE_CHANNELS:-32}"
EPOCHS="${EPOCHS:-40}"
TRAIN_EPOCH_REPEATS="${TRAIN_EPOCH_REPEATS:-4}"
EVAL_EVERY="${EVAL_EVERY:-5}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

COMMON=(
  --data_root "${DATA_ROOT}"
  --processed_dir "${PROCESSED_DIR}"
  --split_dir "${SPLIT_DIR}"
  --image_h "${IMAGE_H}"
  --image_w "${IMAGE_W}"
  --batch_size "${BATCH_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --base_channels "${BASE_CHANNELS}"
  --time_emb_dim 128
  --epochs "${EPOCHS}"
  --train_epoch_repeats "${TRAIN_EPOCH_REPEATS}"
  --eval_every "${EVAL_EVERY}"
  --save_every 10
)

cd "${CODE_ROOT}"
mkdir -p "${RESULT_ROOT}/runs"

echo "[1/8] Loader smoke"
"${PYTHON}" train_my_fpp_input_ablation.py "${COMMON[@]}" \
  --config raw \
  --save_dir "${RESULT_ROOT}/runs/00_loader_smoke_raw" \
  --smoke_only

echo "[2/8] QC visualization"
"${PYTHON}" train_my_fpp_input_ablation.py "${COMMON[@]}" \
  --config raw+single_phys \
  --save_dir "${RESULT_ROOT}/runs/01_qc_raw_single_phys" \
  --visualize_only \
  --visualize_count 8 \
  --cache_features

echo "[3/8] Two-sample overfit"
"${PYTHON}" train_my_fpp_input_ablation.py "${COMMON[@]}" \
  --config raw \
  --save_dir "${RESULT_ROOT}/runs/02_overfit_raw_tiny" \
  --train_subset 2 \
  --train_epoch_repeats 8 \
  --epochs 10 \
  --eval_every 1 \
  --base_channels 16 \
  --time_emb_dim 64

echo "[4/8] Raw mask-weight sensitivity"
for mw in 1 3; do
  for seed in 0 1 2; do
    "${PYTHON}" train_my_fpp_input_ablation.py "${COMMON[@]}" \
      --config raw \
      --seed "${seed}" \
      --object_mask_weight "${mw}" \
      --save_dir "${RESULT_ROOT}/runs/raw_mw${mw}_seed${seed}"
  done
done

echo "[5/8] Legal single-frame input ablations"
for config in raw+xy raw+single_phys; do
  safe_name="${config//+/_}"
  for seed in 0 1 2; do
    extra=()
    if [[ "${config}" == "raw+single_phys" ]]; then
      extra+=(--cache_features)
    fi
    "${PYTHON}" train_my_fpp_input_ablation.py "${COMMON[@]}" \
      --config "${config}" \
      --seed "${seed}" \
      --object_mask_weight 3 \
      --save_dir "${RESULT_ROOT}/runs/${safe_name}_mw3_seed${seed}" \
      "${extra[@]}"
  done
done

echo "[6/8] Teacher auxiliary supervision"
for seed in 0 1 2; do
  "${PYTHON}" train_my_fpp_input_ablation.py "${COMMON[@]}" \
    --config teacher_aux \
    --seed "${seed}" \
    --object_mask_weight 3 \
    --save_dir "${RESULT_ROOT}/runs/teacher_aux_mw3_seed${seed}" \
    --cache_features
done

echo "[7/8] Teacher oracle upper-bound diagnostic"
for seed in 0 1 2; do
  "${PYTHON}" train_my_fpp_input_ablation.py "${COMMON[@]}" \
    --config teacher_oracle \
    --seed "${seed}" \
    --object_mask_weight 3 \
    --save_dir "${RESULT_ROOT}/runs/teacher_oracle_mw3_seed${seed}" \
    --cache_features
done

if [[ "${RUN_DIFFUSION_PILOT:-0}" == "1" ]]; then
  echo "[8/8] Optional diffusion pilot"
  for config in raw raw+single_phys; do
    safe_name="${config//+/_}"
    for seed in 0 1 2; do
      extra=()
      if [[ "${config}" == "raw+single_phys" ]]; then
        extra+=(--cache_features)
      fi
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
        --epochs 40 \
        --train_epoch_repeats "${TRAIN_EPOCH_REPEATS}" \
        --eval_every "${EVAL_EVERY}" \
        --save_every 10 \
        --save_dir "${RESULT_ROOT}/runs/diffusion_pilot_${safe_name}_seed${seed}" \
        "${extra[@]}"
    done
  done
else
  echo "[8/8] Diffusion pilot skipped. Set RUN_DIFFUSION_PILOT=1 to run it."
fi

"${PYTHON}" summarize_my_fpp_physics_validation.py --results_root "${RESULT_ROOT}"
echo "Done: ${RESULT_ROOT}/physics_validation_report.md"
