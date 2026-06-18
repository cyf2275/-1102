#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
C7_DIR=/root/autodl-tmp/diffusion_fpp_v5/results/fpp960_c7_adapter_fullfinetune_lr1e5
C7_CKPT=$C7_DIR/checkpoints/best_rmse.pt
BASE_PREFIX=base_c7_ft
RUN_DIR=/root/autodl-tmp/diffusion_fpp_v5/results/pip_d22_c7_lowt015_base_residual_e1_seed0
EVAL_DIR=/root/autodl-tmp/diffusion_fpp_v5/results/pip_d22_c7_lowt015_physgate_alpha050_seed0

echo "===== D22 C7 BASE TEST EVAL $(date '+%F %T') ====="
if [ ! -f "$C7_DIR/evaluation/summary.json" ]; then
  /root/miniconda3/bin/python train_fpp_official_adapter_unet.py \
    --cache_dir "$CACHE" \
    --save_dir "$C7_DIR" \
    --resume "$C7_DIR/checkpoints/latest.pt" \
    --epochs 0 \
    --batch_size 2 \
    --eval_batch_size 2 \
    --num_workers 4 \
    --image_size 960 \
    --lr 1e-5 \
    --physics_channels 1,2,3,4,5,6 \
    --adapter_hidden 32 \
    --require_cache \
    --final_checkpoint best_rmse
else
  cat "$C7_DIR/evaluation/summary.json"
fi

echo "===== D22 PRECOMPUTE C7 BASE $(date '+%F %T') ====="
if [ ! -f "$CACHE/${BASE_PREFIX}_stats.json" ]; then
  /root/miniconda3/bin/python precompute_fpp_base_predictions.py \
    --cache_dir "$CACHE" \
    --checkpoint "$C7_CKPT" \
    --prefix "$BASE_PREFIX" \
    --physics_channels 1,2,3,4,5,6 \
    --adapter_hidden 32 \
    --batch_size 2 \
    --num_workers 4 \
    --image_size 960 \
    --require_cache
else
  cat "$CACHE/${BASE_PREFIX}_stats.json"
fi

echo "===== D22 TRAIN LOWT DIFFUSION ON C7 BASE $(date '+%F %T') ====="
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

echo "===== D22 EVAL C7 LOWT DIFFUSION FIXED PHYS GATE $(date '+%F %T') ====="
/root/miniconda3/bin/python eval_alpha_sweep_adaptive_gate.py \
  --checkpoints "$RUN_DIR/checkpoints/best.pt" \
  --cache_dir "$CACHE" \
  --base_prefix "$BASE_PREFIX" \
  --save_dir "$EVAL_DIR" \
  --splits val test \
  --image_h 960 \
  --image_w 960 \
  --ddim_steps 20 \
  --eval_batch_size 1 \
  --num_workers 0 \
  --start_ratio 0.05 \
  --alphas 0.50 \
  --min_selected 3 \
  --min_selected_frac 0.25 \
  --gate_features edge_mean:le phase_conf_mean:ge \
  --no_allow_all \
  --save_long_csv \
  --require_cache

/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path

base = Path("/root/autodl-tmp/diffusion_fpp_v5/results")
c7 = json.loads((base / "fpp960_c7_adapter_fullfinetune_lr1e5/evaluation/summary.json").read_text(encoding="utf-8"))
stats = json.loads(Path("/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix/base_c7_ft_stats.json").read_text(encoding="utf-8"))
d22 = json.loads((base / "pip_d22_c7_lowt015_physgate_alpha050_seed0/alpha_gate_summary.json").read_text(encoding="utf-8"))
summary = {
    "c7_test_rmse": c7["rmse"]["mean"],
    "c7_test_mae": c7["mae"]["mean"],
    "c7_test_edge_rmse": c7["edge_rmse"]["mean"],
    "c7_precompute_test_rmse": stats["metrics"]["test"]["rmse"]["mean"],
    "d22_selected": d22["selected"],
    "d22_val_base_rmse": d22["val"]["base"]["rmse"]["mean"],
    "d22_val_selected_rmse": d22["val"]["selected"]["rmse"]["mean"],
    "d22_test_base_rmse": d22["test"]["base"]["rmse"]["mean"],
    "d22_test_selected_rmse": d22["test"]["selected"]["rmse"]["mean"],
    "d22_test_selected": d22["test"]["selected"]["selected"],
    "d22_test_diff_rmse": d22["test"]["diff"]["rmse"]["mean"],
}
out = base / "pip_d22_c7_base_lowt_summary.json"
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "===== D22 ALL DONE $(date '+%F %T') ====="
