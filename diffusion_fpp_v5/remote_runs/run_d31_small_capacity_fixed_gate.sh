#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
BASE_PREFIX=base_c4_adapter
FIXED_EDGE_THRESHOLD=0.46480226516723633
SUMMARY=/root/autodl-tmp/diffusion_fpp_v5/results/pip_d31_small_capacity_fixed_gate_summary.json

run_one() {
  local ch="$1"
  local epochs="$2"
  local run_dir="/root/autodl-tmp/diffusion_fpp_v5/results/pip_d31_ch${ch}_lowt015_e${epochs}_seed0"

  echo "===== D31 train base_channels=${ch} epochs=${epochs} $(date '+%F %T') ====="
  /root/miniconda3/bin/python train_pip_lite.py \
    --dataset fpp_ml_bench \
    --cache_dir "$CACHE" \
    --save_dir "$run_dir" \
    --epochs "$epochs" \
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
    --adapter_hidden 16 \
    --base_channels "$ch" \
    --timesteps 200 \
    --ddim_steps 20 \
    --ensemble 1 \
    --sample_start_ratio 0.05 \
    --train_t_min_ratio 0.0 \
    --train_t_max_ratio 0.15 \
    --lr 3e-5 \
    --base_residual_gate 0.5 \
    --seed 0

  for ep_num in $(seq 1 "$epochs"); do
    local ep
    ep=$(printf "%03d" "$ep_num")
    local ckpt="$run_dir/checkpoints/epoch_${ep}.pt"
    local out="/root/autodl-tmp/diffusion_fpp_v5/results/pip_d31_ch${ch}_epoch${ep}_fixed_edge_a050"
    echo "===== D31 eval ch=${ch} epoch=${ep} fixed gate $(date '+%F %T') ====="
    for split in val test; do
      /root/miniconda3/bin/python eval_adaptive_blend_features.py \
        --checkpoint "$ckpt" \
        --cache_dir "$CACHE" \
        --base_prefix "$BASE_PREFIX" \
        --save_dir "$out" \
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
      --val_csv "$out/val_adaptive_features.csv" \
      --test_csv "$out/test_adaptive_features.csv" \
      --save_json "$out/fixed_gate_summary.json" \
      --feature edge_mean \
      --rule le \
      --threshold "$FIXED_EDGE_THRESHOLD"
  done
}

run_one 24 3
run_one 32 3

/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path

base = Path("/root/autodl-tmp/diffusion_fpp_v5/results")
rows = []
for ch in (24, 32):
    for ep in range(1, 4):
        name = f"pip_d31_ch{ch}_epoch{ep:03d}_fixed_edge_a050"
        p = base / name / "fixed_gate_summary.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        rows.append({
            "name": name,
            "base_channels": ch,
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
    "baseline": {
        "c4_base_test_rmse": 18.899323948224385,
        "d16_fixed_gate_test_rmse": 18.5579705953598,
    },
    "rows": rows,
    "best_by_val_fixed_gate": min(rows, key=lambda r: r["val_fixed_gate_rmse"]),
    "best_by_test_fixed_gate": min(rows, key=lambda r: r["test_fixed_gate_rmse"]),
}
out = base / "pip_d31_small_capacity_fixed_gate_summary.json"
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "===== D31 ALL DONE $(date '+%F %T') ====="
