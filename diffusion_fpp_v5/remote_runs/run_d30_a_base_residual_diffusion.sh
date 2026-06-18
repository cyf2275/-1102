#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
A_CKPT=/root/autodl-tmp/diffusion_fpp_v5/results/fpp960_a_fringe_unet_control/checkpoints/best.pt
BASE_PREFIX=base_a_fringe
RUN_DIR=/root/autodl-tmp/diffusion_fpp_v5/results/pip_d30_a_base_lowt015_residual_e1_seed0
FEAT_DIR=/root/autodl-tmp/diffusion_fpp_v5/results/pip_d30_a_base_lowt015_residual_e1_seed0_adaptive_features_a050

echo "===== D30 precompute A-control base predictions $(date '+%F %T') ====="
if [ ! -f "$CACHE/${BASE_PREFIX}_stats.json" ]; then
  /root/miniconda3/bin/python precompute_fpp_base_predictions.py \
    --cache_dir "$CACHE" \
    --checkpoint "$A_CKPT" \
    --prefix "$BASE_PREFIX" \
    --model_type official \
    --batch_size 2 \
    --num_workers 4 \
    --image_size 960 \
    --require_cache
else
  echo "A base cache already exists: $CACHE/${BASE_PREFIX}_stats.json"
fi

echo "===== D30 train low-t residual diffusion from A base $(date '+%F %T') ====="
/root/miniconda3/bin/python train_pip_lite.py \
  --dataset fpp_ml_bench \
  --cache_dir "$CACHE" \
  --save_dir "$RUN_DIR" \
  --epochs 1 \
  --eval_every 1 \
  --save_every 0 \
  --skip_final_test \
  --batch_size 1 \
  --eval_batch_size 1 \
  --num_workers 8 \
  --require_cache \
  --image_h 960 \
  --image_w 960 \
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
  --train_t_min_ratio 0.0 \
  --train_t_max_ratio 0.15 \
  --lr 3e-5 \
  --base_residual_gate 0.5 \
  --seed 0

echo "===== D30 evaluate adaptive gate $(date '+%F %T') ====="
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

echo "===== D30 DONE $(date '+%F %T') ====="
