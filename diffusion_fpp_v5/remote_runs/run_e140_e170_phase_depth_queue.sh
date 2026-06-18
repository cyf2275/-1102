#!/usr/bin/env bash
set -uo pipefail

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
MASTER=results/remote_logs/e140_e170_phase_depth_queue_master.log

echo "START E140-E170 phase-depth queue $(date '+%F %T')" | tee "$MASTER"

"$PY" -m py_compile train_fpp_psp_adapter_unet.py eval_hierarchical_phase_fusion.py select_edge_aware_phase_gate_csv.py data/dataset_fpp_phase.py \
  2>&1 | tee -a "$MASTER"

run_experiment() {
  local tag="$1"
  local cond_mode="$2"
  local instr_channels="$3"
  local seed="$4"
  local save_dir="$5"
  local fusion_dir="$6"
  local gate89_dir="$7"
  local gate84_dir="$8"
  local log_prefix="$9"
  local summary_path="results/${tag}_phase_depth_fusion_summary.json"

  if [[ -f "$summary_path" ]]; then
    echo "SKIP $tag existing summary $summary_path $(date '+%F %T')" | tee -a "$MASTER"
    return 0
  fi

  echo "START $tag cond=$cond_mode instr=$instr_channels seed=$seed $(date '+%F %T')" | tee -a "$MASTER"

  "$PY" train_fpp_psp_adapter_unet.py \
    --base_cache_dir "$BASE_CACHE" \
    --phase_cache_dir "$PSP_CACHE" \
    --phase_pred_prefix "$PHASE_PREFIX" \
    --save_dir "$save_dir" \
    --base_checkpoint "$INIT_RAW" \
    --cond_mode "$cond_mode" \
    --instr_channels "$instr_channels" \
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
    --seed "$seed" \
    2>&1 | tee "results/remote_logs/${log_prefix}_train.log"
  local train_status=${PIPESTATUS[0]}
  if [[ "$train_status" -ne 0 ]]; then
    echo "FAILED $tag train status=$train_status $(date '+%F %T')" | tee -a "$MASTER"
    return "$train_status"
  fi

  "$PY" eval_hierarchical_phase_fusion.py \
    --depth_checkpoint "$DEPTH_CKPT" \
    --phase_depth_checkpoint "$save_dir/checkpoints/best_rmse.pt" \
    --cache_dir "$BASE_CACHE" \
    --phase_cache_dir "$PSP_CACHE" \
    --base_prefix "$BASE_PREFIX" \
    --save_dir "$fusion_dir" \
    --image_size 960 \
    --eval_batch_size 2 \
    --num_workers 8 \
    --ddim_steps 20 \
    --ensemble 1 \
    --start_ratio 0.05 \
    --phase_weights "0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4 0.425 0.45 0.475 0.5 0.525 0.55 0.575 0.6 0.625 0.65 0.675 0.7 0.725 0.75 0.775 0.8" \
    --splits "val test" \
    --require_cache \
    2>&1 | tee "results/remote_logs/${log_prefix}_fusion.log"
  local fusion_status=${PIPESTATUS[0]}
  if [[ "$fusion_status" -ne 0 ]]; then
    echo "FAILED $tag fusion status=$fusion_status $(date '+%F %T')" | tee -a "$MASTER"
    return "$fusion_status"
  fi

  "$PY" select_edge_aware_phase_gate_csv.py \
    --val_hier_csv "$fusion_dir/val_hier_phase_rows.csv" \
    --test_hier_csv "$fusion_dir/test_hier_phase_rows.csv" \
    --val_fused_csv "$fusion_dir/val_fused_weight_rows.csv" \
    --test_fused_csv "$fusion_dir/test_fused_weight_rows.csv" \
    --edge_tau 0.42 \
    --edge_op ">=" \
    --delta_max 0.11 \
    --phase_conf_max 0.78 \
    --low_weight 0.0 \
    --high_weight 0.6 \
    --save_dir "$gate89_dir" \
    2>&1 | tee "results/remote_logs/${log_prefix}_gate89.log"

  "$PY" select_edge_aware_phase_gate_csv.py \
    --val_hier_csv "$fusion_dir/val_hier_phase_rows.csv" \
    --test_hier_csv "$fusion_dir/test_hier_phase_rows.csv" \
    --val_fused_csv "$fusion_dir/val_fused_weight_rows.csv" \
    --test_fused_csv "$fusion_dir/test_fused_weight_rows.csv" \
    --edge_tau 0.42 \
    --edge_op ">=" \
    --delta_max 0.11 \
    --phase_conf_max 0.74 \
    --low_weight 0.0 \
    --high_weight 0.6 \
    --save_dir "$gate84_dir" \
    2>&1 | tee "results/remote_logs/${log_prefix}_gate84.log"

  TAG="$tag" SAVE_DIR="$save_dir" FUSION_DIR="$fusion_dir" GATE89_DIR="$gate89_dir" GATE84_DIR="$gate84_dir" SUMMARY_PATH="$summary_path" "$PY" - <<'PY' \
    2>&1 | tee "results/remote_logs/${log_prefix}_summary.log"
import json
import os
from pathlib import Path

tag = os.environ["TAG"]
paths = {
    f"{tag}_direct_phase_depth": Path(os.environ["SAVE_DIR"]) / "evaluation" / "summary.json",
    f"{tag}_D47_fusion": Path(os.environ["FUSION_DIR"]) / "hierarchical_phase_fusion_summary.json",
    f"{tag}_E89_rule": Path(os.environ["GATE89_DIR"]) / "edge_aware_phase_gate_summary.json",
    f"{tag}_E84_rule": Path(os.environ["GATE84_DIR"]) / "edge_aware_phase_gate_summary.json",
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
    if name.endswith("_D47_fusion"):
        out[name] = {
            "selected_by_val": data.get("selected_by_val"),
            "test_hierarchical": metric_block(data["test"]["branches"]["hierarchical"]),
        }
    elif name.endswith("_rule"):
        out[name] = {
            "val": data["val"]["metrics"],
            "test": data["test"]["metrics"],
            "counts": data["test"].get("weight_counts"),
        }
    else:
        out[name] = metric_block(data)
print(json.dumps(out, indent=2, ensure_ascii=False))
Path(os.environ["SUMMARY_PATH"]).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
PY

  echo "DONE $tag $(date '+%F %T')" | tee -a "$MASTER"
}

run_experiment "e140_seed123_full_instrxy" "phase_pred_instr_xy" "1-10" "123" \
  "results/fpp960_e140_e63ens_psp_instrxy_adapter_decoder_seed123_e35_preload_fast" \
  "results/e140_d47_seed123_phase_fusion_rows" \
  "results/e141_e140_e89_rule" \
  "results/e142_e140_e84_rule" \
  "e140_seed123_full_instrxy" || true

run_experiment "e150_ablate_no_instr_xyonly" "phase_pred_xy" "none" "150" \
  "results/fpp960_e150_e63ens_psp_xy_adapter_decoder_seed150_e35_preload_fast" \
  "results/e150_d47_xyonly_phase_fusion_rows" \
  "results/e151_e150_e89_rule" \
  "results/e152_e150_e84_rule" \
  "e150_ablate_no_instr_xyonly" || true

run_experiment "e160_ablate_no_xy_instronly" "phase_pred_instr" "1-10" "160" \
  "results/fpp960_e160_e63ens_psp_instr_adapter_decoder_seed160_e35_preload_fast" \
  "results/e160_d47_instronly_phase_fusion_rows" \
  "results/e161_e160_e89_rule" \
  "results/e162_e160_e84_rule" \
  "e160_ablate_no_xy_instronly" || true

run_experiment "e170_ablate_phase_only" "phase_pred" "none" "170" \
  "results/fpp960_e170_e63ens_psp_phaseonly_adapter_decoder_seed170_e35_preload_fast" \
  "results/e170_d47_phaseonly_phase_fusion_rows" \
  "results/e171_e170_e89_rule" \
  "results/e172_e170_e84_rule" \
  "e170_ablate_phase_only" || true

echo "DONE E140-E170 phase-depth queue $(date '+%F %T')" | tee -a "$MASTER"
