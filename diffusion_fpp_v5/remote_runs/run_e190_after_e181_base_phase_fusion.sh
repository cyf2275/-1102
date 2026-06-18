#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PSP_CACHE=/root/autodl-tmp/fpp_ml_pspquad_cache_960
E181_DIR=results/fpp960_e181_official_fringe_plus_physics_ftp_e120_bs2
E181_CKPT="$E181_DIR/checkpoints/best_rmse.pt"
BASE_PREFIX=base_e181_physicsftp
PHASE_CKPT=results/fpp960_e117_e63ens_psp_instrxy_adapter_decoder_seed117_e35_preload/checkpoints/best_rmse.pt
FUSION=results/e190_e181base_plus_e117_phase_fusion
GATE=results/e191_e190_e84_rule
MASTER=results/remote_logs/e190_after_e181_master.log

echo "START E190 after E181 $(date '+%F %T')" | tee "$MASTER"

if [[ ! -f "$E181_DIR/evaluation/summary.json" ]]; then
  echo "E181 final summary is missing: $E181_DIR/evaluation/summary.json" | tee -a "$MASTER"
  exit 2
fi
if [[ ! -f "$E181_CKPT" ]]; then
  echo "E181 best checkpoint is missing: $E181_CKPT" | tee -a "$MASTER"
  exit 2
fi

"$PY" -m py_compile precompute_fpp_official_style_predictions.py eval_base_phase_fusion.py select_edge_aware_phase_gate_csv.py

if [[ -f "$BASE_CACHE/${BASE_PREFIX}_stats.json" ]]; then
  echo "SKIP precompute; E181 base cache already exists $(date '+%F %T')" | tee -a "$MASTER"
else
  echo "PRECOMPUTE E181 base cache $(date '+%F %T')" | tee -a "$MASTER"
  "$PY" precompute_fpp_official_style_predictions.py \
    --cache_dir "$BASE_CACHE" \
    --checkpoint "$E181_CKPT" \
    --prefix "$BASE_PREFIX" \
    --input_mode fringe_plus_physics \
    --include_ftp \
    --physics_channels 0-10 \
    --batch_size 2 \
    --num_workers 12 \
    --image_size 960 \
    --require_cache \
    2>&1 | tee results/remote_logs/e190_precompute_e181_base.log
fi

echo "FUSE E181 base with E117 phase-depth branch $(date '+%F %T')" | tee -a "$MASTER"
"$PY" eval_base_phase_fusion.py \
  --phase_depth_checkpoint "$PHASE_CKPT" \
  --cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PSP_CACHE" \
  --base_prefix "$BASE_PREFIX" \
  --save_dir "$FUSION" \
  --image_size 960 \
  --eval_batch_size 2 \
  --num_workers 0 \
  --phase_weights "0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4 0.425 0.45 0.475 0.5 0.525 0.55 0.575 0.6" \
  --splits "val test" \
  --require_cache \
  2>&1 | tee results/remote_logs/e190_base_phase_fusion.log

echo "APPLY strict E84-style gate on E190 rows $(date '+%F %T')" | tee -a "$MASTER"
"$PY" select_edge_aware_phase_gate_csv.py \
  --val_hier_csv "$FUSION/val_hier_phase_rows.csv" \
  --test_hier_csv "$FUSION/test_hier_phase_rows.csv" \
  --val_fused_csv "$FUSION/val_fused_weight_rows.csv" \
  --test_fused_csv "$FUSION/test_fused_weight_rows.csv" \
  --edge_tau 0.42 \
  --edge_op ">=" \
  --delta_max 0.11 \
  --phase_conf_max 0.74 \
  --low_weight 0.0 \
  --high_weight 0.6 \
  --save_dir "$GATE" \
  2>&1 | tee results/remote_logs/e191_e190_e84_rule.log

"$PY" - <<'PY' | tee results/remote_logs/e190_summary.log
import json
from pathlib import Path

paths = {
    "E181_full_physics_ftp_unet": Path("results/fpp960_e181_official_fringe_plus_physics_ftp_e120_bs2/evaluation/summary.json"),
    "E190_E181base_E117phase_fusion": Path("results/e190_e181base_plus_e117_phase_fusion/base_phase_fusion_summary.json"),
    "E191_E84_gate": Path("results/e191_e190_e84_rule/edge_aware_phase_gate_summary.json"),
}

def metric_block(d):
    return {k: (d[k]["mean"] if isinstance(d.get(k), dict) else d.get(k))
            for k in ["rmse", "mae", "edge_rmse", "normal_deg", "ssim"] if k in d}

out = {}
for name, path in paths.items():
    if not path.exists():
        out[name] = {"missing": str(path)}
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    if name == "E190_E181base_E117phase_fusion":
        out[name] = {
            "selected_by_val": data.get("selected_by_val"),
            "test_base": metric_block(data["test"]["branches"]["base"]),
            "test_phase_branch": metric_block(data["test"]["branches"]["phase_branch"]),
        }
    elif name == "E191_E84_gate":
        out[name] = {
            "val": data["val"]["metrics"],
            "test": data["test"]["metrics"],
            "counts": data["test"].get("weight_counts"),
        }
    else:
        out[name] = metric_block(data)

print(json.dumps(out, indent=2, ensure_ascii=False))
Path("results/e190_after_e181_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
PY

echo "DONE E190 after E181 $(date '+%F %T')" | tee -a "$MASTER"
