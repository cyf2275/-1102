#!/usr/bin/env bash
set -euo pipefail

cd /root/diffusion_fpp_v5

PY=/root/miniconda3/envs/dlfpp/bin/python
DATA_DIR=/root/diffusion_fpp_v5/data
CACHE_DIR=/root/autodl-tmp/diffusion_fpp_shearlet_lite_cache
SAVE_DIR=/root/diffusion_fpp_v5/results/shearlet_lite_nguyen_e500
LOG_DIR=/root/diffusion_fpp_v5/results/_logs
LOG_FILE=${LOG_DIR}/shearlet_lite_nguyen_e500.log

mkdir -p "${LOG_DIR}"

{
  echo "===== shearlet-lite Nguyen/Wang full run started $(date '+%F %T') ====="
  echo "data=${DATA_DIR}"
  echo "cache=${CACHE_DIR}"
  echo "save=${SAVE_DIR}"

  if [[ ! -f "${CACHE_DIR}/physics_shearlet_lite_train_float16.npy" || ! -f "${CACHE_DIR}/physics_shearlet_lite_test_float16.npy" ]]; then
    "${PY}" precompute_features_shearlet_lite.py \
      --data_dir "${DATA_DIR}" \
      --cache_dir "${CACHE_DIR}" \
      --dtype float16
  else
    echo "shearlet-lite cache exists, skip precompute"
  fi

  RESUME_ARGS=()
  if [[ -f "${SAVE_DIR}/checkpoints/latest.pt" ]]; then
    RESUME_ARGS=(--resume "${SAVE_DIR}/checkpoints/latest.pt")
    echo "resume from ${SAVE_DIR}/checkpoints/latest.pt"
  fi

  "${PY}" train_shearlet_lite.py \
    --data_dir "${DATA_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --cache_prefix physics_shearlet_lite \
    --save_dir "${SAVE_DIR}" \
    --epochs 500 \
    --batch_size 4 \
    --num_workers 8 \
    --lr 1e-4 \
    --base_channels 48 \
    --timesteps 200 \
    --ddim_steps 50 \
    --ensemble 3 \
    --eval_every 25 \
    --lambda_grad 0.2 \
    --lambda_edge 0.12 \
    --lambda_normal 0.04 \
    --image_h 480 \
    --image_w 640 \
    --seed 42 \
    --require_cache \
    "${RESUME_ARGS[@]}"

  echo "===== shearlet-lite Nguyen/Wang full run finished $(date '+%F %T') ====="
} 2>&1 | tee -a "${LOG_FILE}"
