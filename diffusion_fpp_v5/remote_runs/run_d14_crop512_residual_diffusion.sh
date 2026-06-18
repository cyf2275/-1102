#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
BASE_PREFIX=base_c4_adapter
RUN_DIR=/root/autodl-tmp/diffusion_fpp_v5/results/pip_d14_crop512_base_residual_e5_repeat6_seed0
FEAT_DIR=/root/autodl-tmp/diffusion_fpp_v5/results/pip_d14_crop512_base_residual_e5_repeat6_seed0_adaptive_features_a050

echo "===== D14 TRAIN START $(date '+%F %T') ====="
/root/miniconda3/bin/python train_pip_lite.py \
  --dataset fpp_ml_bench \
  --cache_dir "$CACHE" \
  --save_dir "$RUN_DIR" \
  --epochs 5 \
  --eval_every 1 \
  --save_every 0 \
  --skip_final_test \
  --batch_size 4 \
  --eval_batch_size 1 \
  --num_workers 4 \
  --require_cache \
  --image_h 960 \
  --image_w 960 \
  --train_crop_size 512 \
  --train_epoch_repeats 6 \
  --target_mode base_residual \
  --base_prefix "$BASE_PREFIX" \
  --condition_injection adapter \
  --physics_channels 0-8 \
  --adapter_hidden 32 \
  --base_channels 48 \
  --timesteps 200 \
  --ddim_steps 20 \
  --ensemble 1 \
  --sample_start_ratio 0.05 \
  --lr 3e-5 \
  --base_residual_gate 0.5 \
  --seed 0

echo "===== D14 FEATURE EVAL START $(date '+%F %T') ====="
for split in val test; do
  /root/miniconda3/bin/python eval_adaptive_blend_features.py \
    --checkpoint "$RUN_DIR/checkpoints/best.pt" \
    --cache_dir "$CACHE" \
    --base_prefix "$BASE_PREFIX" \
    --save_dir "$FEAT_DIR" \
    --split "$split" \
    --image_h 960 \
    --image_w 960 \
    --ddim_steps 20 \
    --ensemble 1 \
    --eval_batch_size 1 \
    --num_workers 0 \
    --start_ratio 0.05 \
    --alpha 0.5 \
    --require_cache
done

/root/miniconda3/bin/python select_adaptive_gate.py \
  --val_csv "$FEAT_DIR/val_adaptive_features.csv" \
  --test_csv "$FEAT_DIR/test_adaptive_features.csv" \
  --save_json "$FEAT_DIR/selected_gate_summary.json" \
  --min_selected 3

echo "===== D14 ALL DONE $(date '+%F %T') ====="
