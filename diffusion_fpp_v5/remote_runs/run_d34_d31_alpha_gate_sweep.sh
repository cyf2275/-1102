#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
BASE_PREFIX=base_c4_adapter
EDGE_THRESHOLD=0.46480226516723633
MASTER=results/remote_logs/d34_d31_alpha_gate_sweep_master.log

echo "START D34 D31 alpha/gate sweep $(date '+%F %T')" | tee "$MASTER"

eval_one() {
  local tag="$1"
  local ckpt="$2"
  local alpha="$3"
  local out="results/d34_${tag}_alpha${alpha}_edgegate"
  echo "EVAL ${tag} alpha=${alpha} $(date '+%F %T')" | tee -a "$MASTER"
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
      --alpha "$alpha" \
      --require_cache
  done
  /root/miniconda3/bin/python select_fixed_gate.py \
    --val_csv "$out/val_adaptive_features.csv" \
    --test_csv "$out/test_adaptive_features.csv" \
    --save_json "$out/fixed_gate_summary.json" \
    --feature edge_mean \
    --rule le \
    --threshold "$EDGE_THRESHOLD"
}

CKPT_CH24_E1=results/pip_d31_ch24_lowt015_e3_seed0/checkpoints/epoch_001.pt
CKPT_CH32_E2=results/pip_d31_ch32_lowt015_e3_seed0/checkpoints/epoch_002.pt

for alpha in 0.25 0.35 0.45 0.50 0.60 0.70; do
  eval_one ch24_ep001 "$CKPT_CH24_E1" "$alpha"
done

for alpha in 0.35 0.50 0.65; do
  eval_one ch32_ep002 "$CKPT_CH32_E2" "$alpha"
done

/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path

base = Path("/root/autodl-tmp/diffusion_fpp_v5/results")
rows = []
for path in sorted(base.glob("d34_*_edgegate/fixed_gate_summary.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "name": path.parent.name,
        "gate": data["fixed_gate"],
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
    "purpose": "D31 residual diffusion posterior correction with fixed physical edge gate; alpha selected by validation.",
    "fixed_gate": {"feature": "edge_mean", "rule": "le", "threshold": 0.46480226516723633},
    "baselines": {
        "c4_base_test_rmse": 18.899323948224385,
        "official_raw_a_test_rmse": 19.660966658592223,
        "previous_d31_ch24_ep001_alpha050_test_gated_rmse": 18.494679013888042,
    },
    "rows": rows,
    "best_by_val_gated": min(rows, key=lambda r: r["val_gated_rmse"]),
    "best_by_test_gated": min(rows, key=lambda r: r["test_gated_rmse"]),
}
out = base / "d34_d31_alpha_gate_sweep_summary.json"
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "DONE D34 D31 alpha/gate sweep $(date '+%F %T')" | tee -a "$MASTER"
