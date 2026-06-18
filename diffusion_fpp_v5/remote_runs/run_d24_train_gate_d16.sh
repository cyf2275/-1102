#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
BASE_PREFIX=base_c4_adapter
CKPT=/root/autodl-tmp/diffusion_fpp_v5/results/pip_d16_lowt015_base_residual_e1_seed0/checkpoints/best.pt
OUT=/root/autodl-tmp/diffusion_fpp_v5/results/pip_d24_d16_trainselect_physgate

echo "===== D24 train-selected physical gate for D16 $(date '+%F %T') ====="
/root/miniconda3/bin/python eval_alpha_sweep_adaptive_gate.py \
  --checkpoints "$CKPT" \
  --cache_dir "$CACHE" \
  --base_prefix "$BASE_PREFIX" \
  --save_dir "$OUT" \
  --splits train val test \
  --gate_select_split train \
  --image_h 960 \
  --image_w 960 \
  --ddim_steps 20 \
  --eval_batch_size 1 \
  --num_workers 0 \
  --start_ratio 0.05 \
  --alphas 0.25 0.35 0.50 \
  --min_selected 12 \
  --min_selected_frac 0.25 \
  --gate_features edge_mean:le phase_conf_mean:ge \
  --no_allow_all \
  --save_long_csv \
  --require_cache

echo "===== D24 done $(date '+%F %T') ====="
