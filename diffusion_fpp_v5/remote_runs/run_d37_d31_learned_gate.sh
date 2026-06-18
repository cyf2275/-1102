#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5

PY=/root/miniconda3/bin/python
CKPT=results/pip_d31_ch24_lowt015_e3_seed0/checkpoints/epoch_001.pt
CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
SAVE=results/d37_d31_ch24_ep001_a050_learned_gate
SRC=results/d34_ch24_ep001_alpha0.50_edgegate

mkdir -p "${SAVE}"

echo "[D37] started at $(date)"
${PY} -m py_compile eval_adaptive_blend_features.py select_learned_acceptance_gate.py

if [[ ! -f "${SAVE}/train_adaptive_features.csv" ]]; then
  echo "[D37] generating train adaptive features"
  ${PY} eval_adaptive_blend_features.py \
    --checkpoint "${CKPT}" \
    --cache_dir "${CACHE}" \
    --base_prefix base_c4_adapter \
    --save_dir "${SAVE}" \
    --split train \
    --image_h 960 --image_w 960 \
    --ddim_steps 20 \
    --ensemble 1 \
    --eval_batch_size 2 \
    --num_workers 4 \
    --start_ratio 0.05 \
    --alpha 0.5 \
    --require_cache
fi

cp "${SRC}/val_adaptive_features.csv" "${SAVE}/val_adaptive_features.csv"
cp "${SRC}/test_adaptive_features.csv" "${SAVE}/test_adaptive_features.csv"
cp "${SRC}/val_adaptive_summary.json" "${SAVE}/val_adaptive_summary.json"
cp "${SRC}/test_adaptive_summary.json" "${SAVE}/test_adaptive_summary.json"

echo "[D37] selecting learned acceptance gate"
${PY} select_learned_acceptance_gate.py \
  --train_csv "${SAVE}/train_adaptive_features.csv" \
  --val_csv "${SAVE}/val_adaptive_features.csv" \
  --test_csv "${SAVE}/test_adaptive_features.csv" \
  --save_json "${SAVE}/learned_gate_summary.json" \
  --candidate_prefix blend \
  --min_selected 3 \
  --ridge_alphas "0.001 0.01 0.1 1.0 10.0 100.0"

echo "[D37] finished at $(date)"
