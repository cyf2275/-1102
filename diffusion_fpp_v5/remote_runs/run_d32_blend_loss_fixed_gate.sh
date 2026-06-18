#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
BASE_PREFIX=base_c4_adapter
FIXED_EDGE_THRESHOLD=0.46480226516723633
RUN_DIR=/root/autodl-tmp/diffusion_fpp_v5/results/pip_d32_blendloss_a050_gate100_lowt015_e3_seed0
SUMMARY=/root/autodl-tmp/diffusion_fpp_v5/results/pip_d32_blend_loss_fixed_gate_summary.json

echo "===== D32 train blend-loss residual diffusion $(date '+%F %T') ====="
/root/miniconda3/bin/python train_pip_lite.py \
  --dataset fpp_ml_bench \
  --cache_dir "$CACHE" \
  --save_dir "$RUN_DIR" \
  --epochs 3 \
  --eval_every 1 \
  --save_every 1 \
  --save_epoch_checkpoints \
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
  --base_residual_gate 1.0 \
  --blend_loss_alpha 0.5 \
  --seed 0

echo "===== D32 evaluate epochs with fixed physical gate $(date '+%F %T') ====="
for ep in 001 002 003; do
  CKPT="$RUN_DIR/checkpoints/epoch_${ep}.pt"
  OUT="/root/autodl-tmp/diffusion_fpp_v5/results/pip_d32_epoch${ep}_fixed_edge_a050"
  for split in val test; do
    /root/miniconda3/bin/python eval_adaptive_blend_features.py \
      --checkpoint "$CKPT" \
      --cache_dir "$CACHE" \
      --base_prefix "$BASE_PREFIX" \
      --save_dir "$OUT" \
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
  /root/miniconda3/bin/python select_fixed_gate.py \
    --val_csv "$OUT/val_adaptive_features.csv" \
    --test_csv "$OUT/test_adaptive_features.csv" \
    --save_json "$OUT/fixed_gate_summary.json" \
    --feature edge_mean \
    --rule le \
    --threshold "$FIXED_EDGE_THRESHOLD"
done

/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path

base = Path("/root/autodl-tmp/diffusion_fpp_v5/results")
rows = []
for ep in range(1, 4):
    name = f"pip_d32_epoch{ep:03d}_fixed_edge_a050"
    p = base / name / "fixed_gate_summary.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    rows.append({
        "name": name,
        "epoch": ep,
        "val_base_rmse": data["val"]["base"]["rmse"]["mean"],
        "val_blend_rmse": data["val"]["blend"]["rmse"]["mean"],
        "val_fixed_gate_rmse": data["val"]["gated"]["rmse"]["mean"],
        "val_selected": data["val"]["gated"]["selected"],
        "test_base_rmse": data["test"]["base"]["rmse"]["mean"],
        "test_blend_rmse": data["test"]["blend"]["rmse"]["mean"],
        "test_fixed_gate_rmse": data["test"]["gated"]["rmse"]["mean"],
        "test_selected": data["test"]["gated"]["selected"],
        "test_selected_subset_base_rmse": data["test"]["selected_subset"]["base"]["rmse"]["mean"],
        "test_selected_subset_blend_rmse": data["test"]["selected_subset"]["blend"]["rmse"]["mean"],
        "test_selected_subset_blend_wins": data["test"]["selected_subset"]["blend_wins"],
    })
summary = {
    "fixed_gate": {"feature": "edge_mean", "rule": "le", "threshold": 0.46480226516723633},
    "training_change": {
        "blend_loss_alpha": 0.5,
        "base_residual_gate": 1.0,
        "why": "Align supervised depth losses with the final alpha=0.5 posterior blend."
    },
    "baseline": {
        "c4_base_test_rmse": 18.899323948224385,
        "d16_fixed_gate_test_rmse": 18.5579705953598,
    },
    "rows": rows,
    "best_by_val_fixed_gate": min(rows, key=lambda r: r["val_fixed_gate_rmse"]),
    "best_by_test_fixed_gate": min(rows, key=lambda r: r["test_fixed_gate_rmse"]),
}
out = base / "pip_d32_blend_loss_fixed_gate_summary.json"
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "===== D32 ALL DONE $(date '+%F %T') ====="
