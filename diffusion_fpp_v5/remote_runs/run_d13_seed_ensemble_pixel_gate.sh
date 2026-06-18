#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
BASE_PREFIX=base_c4_adapter
OUT=/root/autodl-tmp/diffusion_fpp_v5/results/pip_d13_seed_ensemble_pixel_gate

checkpoints=()
for seed in 0 1 2 3 4 42 123 456; do
  checkpoints+=("/root/autodl-tmp/diffusion_fpp_v5/results/pip_d8_seed${seed}_base_residual_e1_gate050_lr3e5/checkpoints/best.pt")
done

echo "===== D13 START $(date '+%F %T') ====="
/root/miniconda3/bin/python eval_seed_ensemble_pixel_gate.py \
  --checkpoints "${checkpoints[@]}" \
  --cache_dir "$CACHE" \
  --base_prefix "$BASE_PREFIX" \
  --save_dir "$OUT" \
  --splits val test \
  --image_h 960 \
  --image_w 960 \
  --ddim_steps 20 \
  --eval_batch_size 1 \
  --num_workers 0 \
  --start_ratio 0.05 \
  --alpha 0.25 \
  --sample_edge_th 0.4674050956964493 \
  --edge_th 0.8 \
  --delta_min 0.12 \
  --conf_min 0.0 \
  --require_cache
echo "===== D13 END $(date '+%F %T') ====="
