#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
BASE_PREFIX=base_c4_adapter
RUN_DIR=/root/autodl-tmp/diffusion_fpp_v5/results/pip_d25_lowt015_e5_epochselect_seed0
SUMMARY=/root/autodl-tmp/diffusion_fpp_v5/results/pip_d25_epoch_posterior_selection_summary.json

echo "===== D25 train low-t residual with epoch checkpoints $(date '+%F %T') ====="
/root/miniconda3/bin/python train_pip_lite.py \
  --dataset fpp_ml_bench \
  --cache_dir "$CACHE" \
  --save_dir "$RUN_DIR" \
  --epochs 5 \
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
  --base_residual_gate 0.5 \
  --seed 0

echo "===== D25 evaluate every epoch with same posterior gate search $(date '+%F %T') ====="
for ep in 001 002 003 004 005; do
  CKPT="$RUN_DIR/checkpoints/epoch_${ep}.pt"
  OUT="/root/autodl-tmp/diffusion_fpp_v5/results/pip_d25_epoch${ep}_physgate_a050"
  echo "===== D25 eval epoch ${ep} $(date '+%F %T') ====="
  /root/miniconda3/bin/python eval_alpha_sweep_adaptive_gate.py \
    --checkpoints "$CKPT" \
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
    --alphas 0.50 \
    --min_selected 3 \
    --min_selected_frac 0.25 \
    --gate_features edge_mean:le phase_conf_mean:ge \
    --no_allow_all \
    --save_long_csv \
    --require_cache
done

/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path

base = Path("/root/autodl-tmp/diffusion_fpp_v5/results")
rows = []
for ep in range(1, 6):
    name = f"pip_d25_epoch{ep:03d}_physgate_a050"
    p = base / name / "alpha_gate_summary.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    rows.append({
        "epoch": ep,
        "name": name,
        "selected": data["selected"],
        "val_base_rmse": data["val"]["base"]["rmse"]["mean"],
        "val_selected_rmse": data["val"]["selected"]["rmse"]["mean"],
        "test_base_rmse": data["test"]["base"]["rmse"]["mean"],
        "test_selected_rmse": data["test"]["selected"]["rmse"]["mean"],
        "test_diff_rmse": data["test"]["diff"]["rmse"]["mean"],
        "test_selected": data["test"]["selected"]["selected"],
        "test_mae": data["test"]["selected"]["mae"]["mean"],
        "test_edge_rmse": data["test"]["selected"]["edge_rmse"]["mean"],
    })
summary = {
    "rows": rows,
    "best_by_val_selected": min(rows, key=lambda r: r["val_selected_rmse"]),
    "best_by_test_selected": min(rows, key=lambda r: r["test_selected_rmse"]),
}
out = Path("/root/autodl-tmp/diffusion_fpp_v5/results/pip_d25_epoch_posterior_selection_summary.json")
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "===== D25 ALL DONE $(date '+%F %T') ====="
