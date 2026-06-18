#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
BASE_PREFIX=base_c4_adapter
ALPHAS=(0.10 0.20 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.70 0.80 1.00)

run_physgate() {
  local name="$1"
  shift
  local checkpoints=("$@")
  local out="/root/autodl-tmp/diffusion_fpp_v5/results/${name}"
  echo "===== D21 PHYSGATE ALPHA ${name} checkpoints=${#checkpoints[@]} $(date '+%F %T') ====="
  /root/miniconda3/bin/python eval_alpha_sweep_adaptive_gate.py \
    --checkpoints "${checkpoints[@]}" \
    --cache_dir "$CACHE" \
    --base_prefix "$BASE_PREFIX" \
    --save_dir "$out" \
    --splits val test \
    --image_h 960 \
    --image_w 960 \
    --ddim_steps 20 \
    --eval_batch_size 1 \
    --num_workers 0 \
    --start_ratio 0.05 \
    --alphas "${ALPHAS[@]}" \
    --min_selected 3 \
    --min_selected_frac 0.25 \
    --gate_features edge_mean:le phase_conf_mean:ge \
    --no_allow_all \
    --save_long_csv \
    --require_cache
}

run_physgate \
  pip_d21_physgate_alpha_d16_seed0 \
  /root/autodl-tmp/diffusion_fpp_v5/results/pip_d16_lowt015_base_residual_e1_seed0/checkpoints/best.pt

run_physgate \
  pip_d21_physgate_alpha_d17_seed42 \
  /root/autodl-tmp/diffusion_fpp_v5/results/pip_d17_lowt015_seed42_base_residual_e1_gate050_lr3e5/checkpoints/best.pt

run_physgate \
  pip_d21_physgate_alpha_d17_valtop2_seed2_0 \
  /root/autodl-tmp/diffusion_fpp_v5/results/pip_d17_lowt015_seed2_base_residual_e1_gate050_lr3e5/checkpoints/best.pt \
  /root/autodl-tmp/diffusion_fpp_v5/results/pip_d17_lowt015_seed0_base_residual_e1_gate050_lr3e5/checkpoints/best.pt

/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path

base = Path("/root/autodl-tmp/diffusion_fpp_v5/results")
names = [
    "pip_d21_physgate_alpha_d16_seed0",
    "pip_d21_physgate_alpha_d17_seed42",
    "pip_d21_physgate_alpha_d17_valtop2_seed2_0",
]
summary = []
for name in names:
    p = base / name / "alpha_gate_summary.json"
    if not p.exists():
        continue
    data = json.loads(p.read_text(encoding="utf-8"))
    summary.append({
        "name": name,
        "selected": data["selected"],
        "val_base_rmse": data["val"]["base"]["rmse"]["mean"],
        "val_selected_rmse": data["val"]["selected"]["rmse"]["mean"],
        "test_base_rmse": data["test"]["base"]["rmse"]["mean"],
        "test_selected_rmse": data["test"]["selected"]["rmse"]["mean"],
        "test_selected": data["test"]["selected"]["selected"],
        "test_diff_rmse": data["test"]["diff"]["rmse"]["mean"],
    })
out = base / "pip_d21_physgate_alpha_sweep_summary.json"
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "===== D21 ALL DONE $(date '+%F %T') ====="
