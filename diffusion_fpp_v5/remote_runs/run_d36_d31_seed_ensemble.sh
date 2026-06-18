#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
BASE_PREFIX=base_c4_adapter
EDGE_THRESHOLD=0.46480226516723633
MASTER=results/remote_logs/d36_d31_seed_ensemble_master.log

CKPT_S0=results/pip_d31_ch24_lowt015_e3_seed0/checkpoints/epoch_001.pt
CKPT_S1=results/d35_d31_ch24_lowt015_e3_seed1/checkpoints/epoch_002.pt
CKPT_S2=results/d35_d31_ch24_lowt015_e3_seed2/checkpoints/epoch_002.pt

echo "START D36 D31 seed ensemble $(date '+%F %T')" | tee "$MASTER"

eval_combo() {
  local tag="$1"
  local alpha="$2"
  shift 2
  local out="results/d36_${tag}_alpha${alpha}_edgegate"
  echo "EVAL combo=${tag} alpha=${alpha} $(date '+%F %T')" | tee -a "$MASTER"
  /root/miniconda3/bin/python eval_seed_ensemble_adaptive_features.py \
    --checkpoints "$@" \
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
    --alpha "$alpha" \
    --require_cache
  /root/miniconda3/bin/python select_fixed_gate_prefix.py \
    --val_csv "$out/val_adaptive_features.csv" \
    --test_csv "$out/test_adaptive_features.csv" \
    --save_json "$out/fixed_gate_summary.json" \
    --candidate_prefix ensemble \
    --feature edge_mean \
    --rule le \
    --threshold "$EDGE_THRESHOLD"
}

for alpha in 0.35 0.45 0.50 0.60; do
  eval_combo s0_s1_s2 "$alpha" "$CKPT_S0" "$CKPT_S1" "$CKPT_S2"
done

for alpha in 0.45 0.50; do
  eval_combo s0_s2 "$alpha" "$CKPT_S0" "$CKPT_S2"
done

/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path

base = Path("/root/autodl-tmp/diffusion_fpp_v5/results")
rows = []
for path in sorted(base.glob("d36_*_edgegate/fixed_gate_summary.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    ss = data["test"]["selected_subset"]
    rows.append({
        "name": path.parent.name,
        "val_base_rmse": data["val"]["base"]["rmse"]["mean"],
        "val_ensemble_rmse": data["val"]["ensemble"]["rmse"]["mean"],
        "val_gated_rmse": data["val"]["gated"]["rmse"]["mean"],
        "val_selected": data["val"]["gated"]["selected"],
        "test_base_rmse": data["test"]["base"]["rmse"]["mean"],
        "test_ensemble_rmse": data["test"]["ensemble"]["rmse"]["mean"],
        "test_gated_rmse": data["test"]["gated"]["rmse"]["mean"],
        "test_selected": data["test"]["gated"]["selected"],
        "test_selected_subset_base_rmse": ss["base"]["rmse"]["mean"],
        "test_selected_subset_ensemble_rmse": ss["ensemble"]["rmse"]["mean"],
        "test_selected_subset_ensemble_wins": ss["ensemble_wins"],
    })
summary = {
    "purpose": "Seed/checkpoint ensemble for D31 residual diffusion posterior correction with fixed edge gate.",
    "fixed_gate": {"feature": "edge_mean", "rule": "le", "threshold": 0.46480226516723633},
    "baselines": {
        "c4_base_test_rmse": 18.899323948224385,
        "best_single_d31_test_gated_rmse": 18.494679013888042,
    },
    "rows": rows,
    "best_by_val_gated": min(rows, key=lambda r: r["val_gated_rmse"]),
    "best_by_test_gated": min(rows, key=lambda r: r["test_gated_rmse"]),
}
out = base / "d36_d31_seed_ensemble_summary.json"
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "DONE D36 D31 seed ensemble $(date '+%F %T')" | tee -a "$MASTER"
