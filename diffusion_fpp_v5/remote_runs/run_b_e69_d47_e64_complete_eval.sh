#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs remote_runs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PSP_CACHE=/root/autodl-tmp/fpp_ml_pspquad_cache_960
DEPTH_CKPT=results/pip_d31_ch24_lowt015_e3_seed0/checkpoints/epoch_001.pt
E64_CKPT=results/fpp960_e64_e63ens_psp_instrxy_adapter_decoder_bs4_e25/checkpoints/best_rmse.pt
D47_DIR=results/d47_d31_epoch001_hierarchical_physical_gate
SWEEP_DIR=results/e69_d47_existing_gate_sweep
FUSION_DIR=results/e69_d47_plus_e64_e63_phase_fusion
MASTER=results/remote_logs/e69_d47_e64_complete_eval_master.log

echo "START E69 complete D47/E64 eval $(date '+%F %T')" | tee "$MASTER"

"$PY" -m py_compile sweep_existing_hier_gate_csv.py eval_hierarchical_phase_fusion.py

echo "E69-A post-hoc D47 sample-level gate sweep $(date '+%F %T')" | tee -a "$MASTER"
"$PY" sweep_existing_hier_gate_csv.py \
  --val_csv "$D47_DIR/val_hierarchical_gate_metrics.csv" \
  --test_csv "$D47_DIR/test_hierarchical_gate_metrics.csv" \
  --save_dir "$SWEEP_DIR" \
  2>&1 | tee results/remote_logs/e69_d47_existing_gate_sweep.log

echo "E69-B D47 hierarchical + E64 E63-phase depth branch fusion $(date '+%F %T')" | tee -a "$MASTER"
"$PY" eval_hierarchical_phase_fusion.py \
  --depth_checkpoint "$DEPTH_CKPT" \
  --phase_depth_checkpoint "$E64_CKPT" \
  --cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PSP_CACHE" \
  --base_prefix base_c4_adapter \
  --save_dir "$FUSION_DIR" \
  --image_size 960 \
  --eval_batch_size 2 \
  --num_workers 4 \
  --ddim_steps 20 \
  --ensemble 1 \
  --start_ratio 0.05 \
  --phase_weights "0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.25 0.3 0.35 0.4 0.45 0.5" \
  --require_cache \
  2>&1 | tee results/remote_logs/e69_d47_plus_e64_e63_phase_fusion.log

echo "E69-C collect final summary $(date '+%F %T')" | tee -a "$MASTER"
"$PY" - <<'PY' | tee results/remote_logs/e69_final_summary.txt
import json
from pathlib import Path

base = Path("results")

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def metric_block(d):
    return {k: d[k]["mean"] for k in ("rmse", "mae", "edge_rmse", "normal_deg", "ssim") if k in d}

d47 = load(base / "d47_d31_epoch001_hierarchical_physical_gate/hierarchical_gate_summary.json")
e64 = load(base / "fpp960_e64_e63ens_psp_instrxy_adapter_decoder_bs4_e25/evaluation/summary.json")
sweep = load(base / "e69_d47_existing_gate_sweep/d47_existing_gate_sweep_summary.json")
fusion = load(base / "e69_d47_plus_e64_e63_phase_fusion/hierarchical_phase_fusion_summary.json")

selected_w = str(fusion["selected_by_val"]["phase_weight"])
out = {
    "methods": {
        "C4_base_from_D47": metric_block(d47["test"]["base"]),
        "D47_depth_diffusion_hierarchical": metric_block(d47["test"]["hierarchical"]),
        "E64_E63_phase_adapter": metric_block(e64),
        "E69_D47_gate_resweep_selected_by_val": metric_block(sweep["test"]["selected_gate"]),
        "E69_D47_plus_E64_fusion_selected_by_val": metric_block(fusion["test"]["weights"][selected_w]),
        "E69_oracle_best_base_diff_pixel_upper_bound": metric_block(sweep["test"]["oracle_best_of_base_diff_pixel"]),
    },
    "selected": {
        "gate": sweep["selected_gate"],
        "fusion_phase_weight": fusion["selected_by_val"]["phase_weight"],
        "fusion_selected_by_val": fusion["selected_by_val"],
    },
    "notes": [
        "E69 gate sweep selects rules on validation split only, then applies them unchanged to test.",
        "E69 fusion uses D47 depth diffusion posterior branch and E64 depth branch trained from E63 phase-diffusion ensemble predictions.",
    ],
}
out_path = base / "e69_complete_result_summary.json"
out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=2, ensure_ascii=False))
PY

echo "DONE E69 complete D47/E64 eval $(date '+%F %T')" | tee -a "$MASTER"
