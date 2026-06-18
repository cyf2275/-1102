#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
BASE_PREFIX=base_c4_adapter
EDGE_THRESHOLD=0.46480226516723633
MASTER=results/remote_logs/d35_d31_seed_stability_master.log

echo "START D35 D31 seed stability $(date '+%F %T')" | tee "$MASTER"

run_seed() {
  local seed="$1"
  local run_dir="results/d35_d31_ch24_lowt015_e3_seed${seed}"
  echo "TRAIN seed=${seed} $(date '+%F %T')" | tee -a "$MASTER"
  /root/miniconda3/bin/python train_pip_lite.py \
    --dataset fpp_ml_bench \
    --cache_dir "$CACHE" \
    --save_dir "$run_dir" \
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
    --adapter_hidden 16 \
    --base_channels 24 \
    --timesteps 200 \
    --ddim_steps 20 \
    --ensemble 1 \
    --sample_start_ratio 0.05 \
    --train_t_min_ratio 0.0 \
    --train_t_max_ratio 0.15 \
    --lr 3e-5 \
    --base_residual_gate 0.5 \
    --seed "$seed"

  for ep in 001 002 003; do
    local ckpt="$run_dir/checkpoints/epoch_${ep}.pt"
    local out="results/d35_d31_ch24_seed${seed}_epoch${ep}_a050_edgegate"
    echo "EVAL seed=${seed} epoch=${ep} $(date '+%F %T')" | tee -a "$MASTER"
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
      --threshold "$EDGE_THRESHOLD"
  done
}

run_seed 1
run_seed 2

/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path

base = Path("/root/autodl-tmp/diffusion_fpp_v5/results")
rows = []
for path in sorted(base.glob("d35_d31_ch24_seed*_epoch*_a050_edgegate/fixed_gate_summary.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "name": path.parent.name,
        "val_base_rmse": data["val"]["base"]["rmse"]["mean"],
        "val_blend_rmse": data["val"]["blend"]["rmse"]["mean"],
        "val_gated_rmse": data["val"]["gated"]["rmse"]["mean"],
        "val_selected": data["val"]["gated"]["selected"],
        "test_base_rmse": data["test"]["base"]["rmse"]["mean"],
        "test_blend_rmse": data["test"]["blend"]["rmse"]["mean"],
        "test_gated_rmse": data["test"]["gated"]["rmse"]["mean"],
        "test_selected": data["test"]["gated"]["selected"],
        "test_selected_subset_base_rmse": data["test"]["selected_subset"]["base"]["rmse"]["mean"],
        "test_selected_subset_blend_rmse": data["test"]["selected_subset"]["blend"]["rmse"]["mean"],
        "test_selected_subset_blend_wins": data["test"]["selected_subset"]["blend_wins"],
    })
summary = {
    "purpose": "Seed stability for D31 residual diffusion posterior correction with fixed alpha=0.5 and fixed edge gate.",
    "fixed_gate": {"feature": "edge_mean", "rule": "le", "threshold": 0.46480226516723633},
    "alpha": 0.5,
    "seed0_reference": {
        "name": "pip_d31_ch24_epoch001_fixed_edge_a050",
        "val_gated_rmse": 18.262747359275817,
        "test_gated_rmse": 18.494679013888042,
    },
    "rows": rows,
    "best_by_val_gated": min(rows, key=lambda r: r["val_gated_rmse"]),
    "best_by_test_gated": min(rows, key=lambda r: r["test_gated_rmse"]),
}
out = base / "d35_d31_seed_stability_summary.json"
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "DONE D35 D31 seed stability $(date '+%F %T')" | tee -a "$MASTER"
