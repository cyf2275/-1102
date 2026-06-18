#!/usr/bin/env bash
set -euo pipefail

# Fair diffusion-vs-UNet check for the self-built real-capture dataset.
# It freezes the already trained direct UNet checkpoints and trains only a
# bounded residual diffusion posterior on top of each base prediction.

PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
CODE_ROOT="${CODE_ROOT:-/root/autodl-tmp/diffusion_fpp_v5}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
PROCESSED_DIR="${PROCESSED_DIR:-/root/autodl-tmp/orderfix_0610_cleanmask_v1}"
SPLIT_DIR="${SPLIT_DIR:-/root/autodl-tmp/splits}"
DIRECT_ROOT="${DIRECT_ROOT:-/root/autodl-tmp/diffusion_fpp_v5/results/A_20260611_my_fpp_physics_validation_gpuopt_full}"
RESULT_ROOT="${RESULT_ROOT:-/root/autodl-tmp/diffusion_fpp_v5/results/B_20260611_my_fpp_diffusion_vs_unet}"
IMAGE_H="${IMAGE_H:-480}"
IMAGE_W="${IMAGE_W:-640}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BASE_CHANNELS="${BASE_CHANNELS:-32}"
EPOCHS="${EPOCHS:-40}"
TRAIN_EPOCH_REPEATS="${TRAIN_EPOCH_REPEATS:-4}"
EVAL_EVERY="${EVAL_EVERY:-5}"
SAMPLE_STEPS="${SAMPLE_STEPS:-12}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-3}"

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
  --sample_steps "${SAMPLE_STEPS}"
  --ensemble_size "${ENSEMBLE_SIZE}"
  --object_mask_weight 3
)

cd "${CODE_ROOT}"
mkdir -p "${RESULT_ROOT}/runs"

run_one() {
  local config="$1"
  local base_dir_prefix="$2"
  local seed="$3"
  local base_ckpt="${DIRECT_ROOT}/runs/${base_dir_prefix}_seed${seed}/checkpoints/best.pt"
  local save_dir="${RESULT_ROOT}/runs/${config}_residual_posterior_seed${seed}"
  if [[ ! -f "${base_ckpt}" ]]; then
    echo "Missing base checkpoint: ${base_ckpt}" >&2
    return 1
  fi
  if [[ -f "${save_dir}/evaluation/summary.json" ]]; then
    echo "Skip existing ${save_dir}"
    return 0
  fi
  local extra=()
  if [[ "${config}" == "raw_single_phys" || "${config}" == "teacher_aux" ]]; then
    extra+=(--cache_features)
  fi
  echo "Run residual posterior config=${config} seed=${seed} base=${base_ckpt}"
  "${PYTHON}" train_my_fpp_residual_posterior.py "${COMMON[@]}" \
    --config "${config}" \
    --seed "${seed}" \
    --base_ckpt "${base_ckpt}" \
    --save_dir "${save_dir}" \
    "${extra[@]}"
}

echo "[1/4] raw residual posterior over raw UNet"
for seed in 0 1 2; do
  run_one raw raw_mw3 "${seed}"
done

echo "[2/4] physics residual posterior over raw_single_phys UNet"
for seed in 0 1 2; do
  run_one raw_single_phys raw_single_phys_mw3 "${seed}"
done

echo "[3/4] teacher-supervised physics residual posterior over teacher_aux UNet"
for seed in 0 1 2; do
  run_one teacher_aux teacher_aux_mw3 "${seed}"
done

echo "[4/4] summarize diffusion-vs-UNet"
"${PYTHON}" summarize_my_fpp_diffusion_vs_unet.py \
  --direct_results_root "${DIRECT_ROOT}" \
  --residual_results_root "${RESULT_ROOT}"

mkdir -p /root/autodl-tmp/cloud_results
ln -sfn "${RESULT_ROOT}" "/root/autodl-tmp/cloud_results/$(basename "${RESULT_ROOT}")"
tar -czf "${RESULT_ROOT}/diffusion_vs_unet_bundle.tar.gz" \
  -C "${RESULT_ROOT}" \
  diffusion_vs_unet_report.md \
  diffusion_vs_unet_summary.json \
  direct_aggregated_results.csv \
  residual_aggregated_results.csv \
  runs/*/evaluation/summary.json \
  runs/*/evaluation/per_sample_metrics.csv \
  2>/dev/null || true

echo "Done: ${RESULT_ROOT}/diffusion_vs_unet_report.md"
