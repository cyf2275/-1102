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
PHASE_PREFIX=phase_pred_e63_ens_e28_e62_w055
BASE_PREFIX=base_c4_adapter
DEPTH_CKPT=results/pip_d31_ch24_lowt015_e3_seed0/checkpoints/epoch_001.pt
INIT_RAW=results/fpp960_a_fringe_unet_control/checkpoints/best.pt

SAVE=results/fpp960_e210_e63ens_psp_instrxy_adapter_decoder_seed180_e35_preload_fast
FUSION=results/e210_d47_seed180_phase_fusion_rows
GATE_E89=results/e211_e210_e89_rule
GATE_E84=results/e212_e210_e84_rule
MASTER=results/remote_logs/e210_seed180_phase_depth_fusion_master.log

echo "START E210 seed180 phase-depth branch + D47 fusion $(date '+%F %T')" | tee "$MASTER"

"$PY" -m py_compile train_fpp_psp_adapter_unet.py eval_hierarchical_phase_fusion.py select_edge_aware_phase_gate_csv.py data/dataset_fpp_phase.py

if [[ ! -f "$SAVE/evaluation/summary.json" ]]; then
  echo "TRAIN E210 phase-depth adapter seed180 $(date '+%F %T')" | tee -a "$MASTER"
  "$PY" train_fpp_psp_adapter_unet.py \
    --base_cache_dir "$BASE_CACHE" \
    --phase_cache_dir "$PSP_CACHE" \
    --phase_pred_prefix "$PHASE_PREFIX" \
    --save_dir "$SAVE" \
    --base_checkpoint "$INIT_RAW" \
    --cond_mode phase_pred_instr_xy \
    --instr_channels 1-10 \
    --train_scope adapter_decoder \
    --epochs 35 \
    --batch_size 4 \
    --eval_batch_size 4 \
    --num_workers 12 \
    --image_size 960 \
    --lr 5e-4 \
    --adapter_lr 5e-4 \
    --backbone_lr 2e-5 \
    --weight_decay 1e-5 \
    --adapter_hidden 64 \
    --alpha 0.7 \
    --eval_every 1 \
    --eval_metrics_every 1 \
    --save_every 5 \
    --eval_initial \
    --train_minimal \
    --preload_ram \
    --seed 180 \
    2>&1 | tee results/remote_logs/e210_phase_depth_seed180_train.log
else
  echo "SKIP E210 train; existing $SAVE/evaluation/summary.json" | tee -a "$MASTER"
fi

echo "FUSE E210 phase-depth branch with D47 depth diffusion posterior $(date '+%F %T')" | tee -a "$MASTER"
"$PY" eval_hierarchical_phase_fusion.py \
  --depth_checkpoint "$DEPTH_CKPT" \
  --phase_depth_checkpoint "$SAVE/checkpoints/best_rmse.pt" \
  --cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PSP_CACHE" \
  --base_prefix "$BASE_PREFIX" \
  --save_dir "$FUSION" \
  --image_size 960 \
  --eval_batch_size 2 \
  --num_workers 8 \
  --ddim_steps 20 \
  --ensemble 1 \
  --start_ratio 0.05 \
  --phase_weights "0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4 0.425 0.45 0.475 0.5 0.525 0.55 0.575 0.6 0.625 0.65 0.675 0.7 0.725 0.75 0.775 0.8" \
  --splits "val test" \
  --require_cache \
  2>&1 | tee results/remote_logs/e210_d47_phase_fusion.log

echo "APPLY E89 validation-selected ambiguity gate to E210 rows $(date '+%F %T')" | tee -a "$MASTER"
"$PY" select_edge_aware_phase_gate_csv.py \
  --val_hier_csv "$FUSION/val_hier_phase_rows.csv" \
  --test_hier_csv "$FUSION/test_hier_phase_rows.csv" \
  --val_fused_csv "$FUSION/val_fused_weight_rows.csv" \
  --test_fused_csv "$FUSION/test_fused_weight_rows.csv" \
  --edge_tau 0.42 \
  --edge_op ">=" \
  --delta_max 0.11 \
  --phase_conf_max 0.78 \
  --low_weight 0.0 \
  --high_weight 0.6 \
  --save_dir "$GATE_E89" \
  2>&1 | tee results/remote_logs/e211_e210_e89_rule.log

echo "APPLY E84 strict confidence gate to E210 rows $(date '+%F %T')" | tee -a "$MASTER"
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
  --save_dir "$GATE_E84" \
  2>&1 | tee results/remote_logs/e212_e210_e84_rule.log

"$PY" - <<'PY' | tee results/remote_logs/e210_seed180_summary.txt
import json
from pathlib import Path

paths = {
    "E210_direct_phase_depth": Path("results/fpp960_e210_e63ens_psp_instrxy_adapter_decoder_seed180_e35_preload_fast/evaluation/summary.json"),
    "E210_D47_fusion": Path("results/e210_d47_seed180_phase_fusion_rows/hierarchical_phase_fusion_summary.json"),
    "E211_E89_rule": Path("results/e211_e210_e89_rule/edge_aware_phase_gate_summary.json"),
    "E212_E84_rule": Path("results/e212_e210_e84_rule/edge_aware_phase_gate_summary.json"),
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
    if name == "E210_D47_fusion":
        out[name] = {
            "selected_by_val": data.get("selected_by_val"),
            "test_hierarchical": metric_block(data["test"]["branches"]["hierarchical"]),
        }
    elif "rule" in name:
        out[name] = {
            "val": data["val"]["metrics"],
            "test": data["test"]["metrics"],
            "counts": data["test"].get("weight_counts"),
        }
    else:
        out[name] = metric_block(data)
print(json.dumps(out, indent=2, ensure_ascii=False))
Path("results/e210_seed180_phase_depth_fusion_summary.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
)
PY

echo "DONE E210 seed180 phase-depth branch + D47 fusion $(date '+%F %T')" | tee -a "$MASTER"
