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
TEACHER_PHASE_CACHE=/root/autodl-tmp/fpp_ml_phase_cache_960
PHASE_PREFIX=phase_pred_e63_ens_e28_e62_w055
BASE_PREFIX=base_c4_adapter
DEPTH_CKPT=results/pip_d31_ch24_lowt015_e3_seed0/checkpoints/epoch_001.pt
INIT_RAW=results/fpp960_a_fringe_unet_control/checkpoints/best.pt

EXP_ID=${EXP_ID:-e256}
GATE_ID=${GATE_ID:-e257}
SEED=${SEED:-180}
PHASE_W=${PHASE_W:-0.005}
TAG=${TAG:-l0005}
EPOCHS=${EPOCHS:-35}

PHASE_PROXY_COEFFS="45.452732235850426 0.030703432236726596 -0.001926472379255077 66.69305714209537 -5.830587969787353e-06 -2.542049065133751e-06 2.4473742115755103e-05 -0.0025881002126868897 0.00997907576305477 0.003892213471562195"

SAVE=results/fpp960_${EXP_ID}_e63ens_psp_phaseproxy_${TAG}_seed${SEED}_e${EPOCHS}_preload_fast
FUSION=results/${EXP_ID}_d47_seed${SEED}_phaseproxy_${TAG}_fusion_rows
GATE_E84=results/${GATE_ID}_${EXP_ID}_e84_phaseproxy_${TAG}_rule
MASTER=results/remote_logs/${EXP_ID}_phaseproxy_psp_${TAG}_seed${SEED}_master.log

echo "START ${EXP_ID} teacher-phase proxy PSP lambda=${PHASE_W} seed${SEED} $(date '+%F %T')" | tee "$MASTER"

"$PY" -m py_compile \
  train_fpp_psp_adapter_unet.py \
  eval_hierarchical_phase_fusion.py \
  select_edge_aware_phase_gate_csv.py \
  data/dataset_fpp_phase.py

if [[ ! -f "$SAVE/evaluation/summary.json" ]]; then
  echo "TRAIN ${EXP_ID} PSP + teacher-phase proxy loss seed${SEED} $(date '+%F %T')" | tee -a "$MASTER"
  "$PY" train_fpp_psp_adapter_unet.py \
    --base_cache_dir "$BASE_CACHE" \
    --phase_cache_dir "$PSP_CACHE" \
    --teacher_phase_cache_dir "$TEACHER_PHASE_CACHE" \
    --phase_pred_prefix "$PHASE_PREFIX" \
    --save_dir "$SAVE" \
    --base_checkpoint "$INIT_RAW" \
    --cond_mode phase_pred_instr_xy \
    --instr_channels 1-10 \
    --train_scope adapter_decoder \
    --epochs "$EPOCHS" \
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
    --phase_proxy_loss_weight "$PHASE_W" \
    --phase_proxy_coeffs "$PHASE_PROXY_COEFFS" \
    --eval_every 1 \
    --eval_metrics_every 1 \
    --save_every 5 \
    --eval_initial \
    --train_minimal \
    --preload_ram \
    --seed "$SEED" \
    2>&1 | tee "results/remote_logs/${EXP_ID}_phaseproxy_psp_${TAG}_seed${SEED}_train.log"
else
  echo "SKIP ${EXP_ID} train; existing $SAVE/evaluation/summary.json" | tee -a "$MASTER"
fi

echo "FUSE ${EXP_ID} PSP branch with D47 diffusion posterior $(date '+%F %T')" | tee -a "$MASTER"
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
  2>&1 | tee "results/remote_logs/${EXP_ID}_d47_phaseproxy_${TAG}_fusion.log"

echo "APPLY E84/RCPC rule to ${EXP_ID} rows $(date '+%F %T')" | tee -a "$MASTER"
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
  2>&1 | tee "results/remote_logs/${GATE_ID}_${EXP_ID}_e84_phaseproxy_${TAG}_rule.log"

"$PY" - "$EXP_ID" "$GATE_ID" "$TAG" "$SAVE" "$FUSION" "$GATE_E84" <<'PY' | tee "results/remote_logs/${EXP_ID}_phaseproxy_psp_${TAG}_seed${SEED}_summary.txt"
import json
import sys
from pathlib import Path

exp_id, gate_id, tag, save, fusion, gate = sys.argv[1:]
paths = {
    f"{exp_id}_direct_phaseproxy_PSP_{tag}": Path(save) / "evaluation" / "summary.json",
    f"{exp_id}_D47_fusion": Path(fusion) / "hierarchical_phase_fusion_summary.json",
    f"{gate_id}_E84_RCPC": Path(gate) / "edge_aware_phase_gate_summary.json",
}

def metric_block(d):
    out = {}
    for k in ["rmse", "mae", "edge_rmse", "normal_deg", "ssim", "phase_proxy_mae_rad", "phase_proxy_rmse_rad"]:
        if k in d:
            out[k] = d[k]["mean"] if isinstance(d[k], dict) else d[k]
    return out

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
            "test_phase_branch": metric_block(data["test"]["branches"]["phase_branch"]),
        }
    elif "RCPC" in name:
        out[name] = {
            "val": data["val"]["metrics"],
            "test": data["test"]["metrics"],
            "counts": data["test"].get("weight_counts"),
        }
    else:
        out[name] = metric_block(data)
print(json.dumps(out, indent=2, ensure_ascii=False))
Path(f"results/{exp_id}_phaseproxy_psp_{tag}_seed180_summary.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
)
PY

echo "DONE ${EXP_ID} teacher-phase proxy PSP lambda=${PHASE_W} seed${SEED} $(date '+%F %T')" | tee -a "$MASTER"
