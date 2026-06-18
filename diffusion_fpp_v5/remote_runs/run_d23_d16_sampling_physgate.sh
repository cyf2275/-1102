#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
BASE_PREFIX=base_c4_adapter
CKPT=/root/autodl-tmp/diffusion_fpp_v5/results/pip_d16_lowt015_base_residual_e1_seed0/checkpoints/best.pt

run_cfg() {
  local name="$1"
  local start_ratio="$2"
  local steps="$3"
  local out="/root/autodl-tmp/diffusion_fpp_v5/results/${name}"
  echo "===== D23 ${name} start=${start_ratio} steps=${steps} $(date '+%F %T') ====="
  /root/miniconda3/bin/python eval_alpha_sweep_adaptive_gate.py \
    --checkpoints "$CKPT" \
    --cache_dir "$CACHE" \
    --base_prefix "$BASE_PREFIX" \
    --save_dir "$out" \
    --splits val test \
    --image_h 960 \
    --image_w 960 \
    --ddim_steps "$steps" \
    --eval_batch_size 1 \
    --num_workers 0 \
    --start_ratio "$start_ratio" \
    --alphas 0.50 \
    --min_selected 3 \
    --min_selected_frac 0.25 \
    --gate_features edge_mean:le phase_conf_mean:ge \
    --no_allow_all \
    --save_long_csv \
    --require_cache
}

run_cfg pip_d23_d16_sr005_s20_physgate_a050 0.05 20
run_cfg pip_d23_d16_sr010_s20_physgate_a050 0.10 20
run_cfg pip_d23_d16_sr015_s20_physgate_a050 0.15 20
run_cfg pip_d23_d16_sr015_s50_physgate_a050 0.15 50
run_cfg pip_d23_d16_sr020_s50_physgate_a050 0.20 50

/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path

base = Path("/root/autodl-tmp/diffusion_fpp_v5/results")
names = [
    "pip_d23_d16_sr005_s20_physgate_a050",
    "pip_d23_d16_sr010_s20_physgate_a050",
    "pip_d23_d16_sr015_s20_physgate_a050",
    "pip_d23_d16_sr015_s50_physgate_a050",
    "pip_d23_d16_sr020_s50_physgate_a050",
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
out = base / "pip_d23_d16_sampling_physgate_summary.json"
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "===== D23 ALL DONE $(date '+%F %T') ====="
