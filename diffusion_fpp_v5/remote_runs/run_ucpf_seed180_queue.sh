#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY=${PY:-/root/miniconda3/bin/python}
BASE_CACHE=${BASE_CACHE:-/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix}
PSP_CACHE=${PSP_CACHE:-/root/autodl-tmp/fpp_ml_pspquad_cache_960}
UCPF_CACHE=${UCPF_CACHE:-/root/autodl-tmp/fpp_ml_ucpf_cache_960_seed180}
RUN_PREFIX=${RUN_PREFIX:-fpp960_ucpf}
DIFF_CANDIDATE_MODE=${DIFF_CANDIDATE_MODE:-hierarchical}
BASE_PREFIX=${BASE_PREFIX:-base_c4_adapter}
DEPTH_CKPT=${DEPTH_CKPT:-results/pip_d31_ch24_lowt015_e3_seed0/checkpoints/epoch_001.pt}
PHASE_DEPTH_CKPT=${PHASE_DEPTH_CKPT:-results/fpp960_e210_e63ens_psp_instrxy_adapter_decoder_seed180_e35_preload_fast/checkpoints/best_rmse.pt}
EPOCHS=${EPOCHS:-160}
BATCH=${BATCH:-4}
EVAL_BATCH=${EVAL_BATCH:-2}
WORKERS=${WORKERS:-8}
SEED=${SEED:-180}
export SEED
export RUN_PREFIX

MASTER=results/remote_logs/${RUN_PREFIX}_seed${SEED}_queue_master.log
echo "START UCPF seed${SEED} queue $(date '+%F %T')" | tee "$MASTER"

"$PY" -m py_compile \
  export_ucpf_candidate_cache.py \
  train_uncertainty_posterior_fusion.py \
  eval_hierarchical_phase_fusion.py \
  train_fpp_psp_adapter_unet.py \
  data/dataset_fpp_ml_bench.py \
  data/dataset_fpp_phase.py

if [[ ! -f "$UCPF_CACHE/ucpf_candidate_manifest.json" ]]; then
  echo "EXPORT frozen UCPF candidate cache $(date '+%F %T')" | tee -a "$MASTER"
  "$PY" export_ucpf_candidate_cache.py \
    --cache_dir "$BASE_CACHE" \
    --phase_cache_dir "$PSP_CACHE" \
    --save_dir "$UCPF_CACHE" \
    --base_prefix "$BASE_PREFIX" \
    --depth_checkpoint "$DEPTH_CKPT" \
    --phase_depth_checkpoint "$PHASE_DEPTH_CKPT" \
    --image_size 960 \
    --eval_batch_size 1 \
    --num_workers "$WORKERS" \
    --ddim_steps 20 \
    --start_ratio 0.05 \
    --diff_candidate_mode "$DIFF_CANDIDATE_MODE" \
    --splits "train val test" \
    --require_cache \
    2>&1 | tee results/remote_logs/ucpf_seed${SEED}_export_cache.log
else
  echo "SKIP export; existing $UCPF_CACHE/ucpf_candidate_manifest.json" | tee -a "$MASTER"
fi

run_ucpf() {
  local name="$1"
  shift
  local save="results/${name}"
  local log="results/remote_logs/${name}.log"
  if [[ -f "$save/ucpf_summary.json" ]]; then
    echo "SKIP $name; existing $save/ucpf_summary.json" | tee -a "$MASTER"
    return
  fi
  echo "RUN $name $(date '+%F %T')" | tee -a "$MASTER"
  "$PY" train_uncertainty_posterior_fusion.py \
    --cache_dir "$UCPF_CACHE" \
    --save_dir "$save" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH" \
    --eval_batch_size "$EVAL_BATCH" \
    --num_workers "$WORKERS" \
    --early_stop_patience 20 \
    --seed "$SEED" \
    --amp \
    "$@" \
    2>&1 | tee "$log"
}

run_ucpf "${RUN_PREFIX}_u0_x0_seed${SEED}" \
  --preset u0_x0

run_ucpf "${RUN_PREFIX}_u1_x0_seed${SEED}" \
  --preset u1_x0

run_ucpf "${RUN_PREFIX}_u2_soft_x0_seed${SEED}" \
  --preset u2_soft_x0

run_ucpf "${RUN_PREFIX}_u2_clip_x0_seed${SEED}" \
  --preset u2_clip_x0

run_ucpf "${RUN_PREFIX}_full_x1_seed${SEED}" \
  --preset ucpf_full_x1

run_ucpf "${RUN_PREFIX}_full_x1_no_dd_seed${SEED}" \
  --preset ucpf_full_x1 \
  --candidates b,p

run_ucpf "${RUN_PREFIX}_full_x1_no_dp_seed${SEED}" \
  --preset ucpf_full_x1 \
  --candidates b,d

run_ucpf "${RUN_PREFIX}_full_x1_no_edge_conf_seed${SEED}" \
  --preset ucpf_full_x1 \
  --drop_edge_conf

run_ucpf "${RUN_PREFIX}_full_x1_no_physics_instr_seed${SEED}" \
  --preset ucpf_full_x1 \
  --drop_physics_instr

run_ucpf "${RUN_PREFIX}_full_x1_no_lcal_seed${SEED}" \
  --preset ucpf_full_x1 \
  --disable_cal

"$PY" - <<'PY' | tee results/remote_logs/${RUN_PREFIX}_seed${SEED}_queue_summary.txt
import json
import os
from pathlib import Path

seed = os.environ.get("SEED", "180")
prefix = os.environ.get("RUN_PREFIX", "fpp960_ucpf")
names = [
    f"{prefix}_u0_x0_seed{seed}",
    f"{prefix}_u1_x0_seed{seed}",
    f"{prefix}_u2_soft_x0_seed{seed}",
    f"{prefix}_u2_clip_x0_seed{seed}",
    f"{prefix}_full_x1_seed{seed}",
    f"{prefix}_full_x1_no_dd_seed{seed}",
    f"{prefix}_full_x1_no_dp_seed{seed}",
    f"{prefix}_full_x1_no_edge_conf_seed{seed}",
    f"{prefix}_full_x1_no_physics_instr_seed{seed}",
    f"{prefix}_full_x1_no_lcal_seed{seed}",
]
out = {}
for name in names:
    path = Path("results") / name / "ucpf_summary.json"
    if not path.exists():
        out[name] = {"missing": str(path)}
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    out[name] = {
        "test_rmse": data["test"]["rmse"]["mean"],
        "test_mae": data["test"]["mae"]["mean"],
        "test_edge_rmse": data["test"]["edge_rmse"]["mean"],
        "test_normal_deg": data["test"]["normal_deg"]["mean"],
        "selected_temperature": data.get("selected_temperature"),
        "selected_kappa": data.get("selected_kappa"),
        "params": data.get("params"),
    }
print(json.dumps(out, indent=2, ensure_ascii=False))
Path(f"results/{prefix}_seed{seed}_queue_summary.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
PY

echo "DONE UCPF seed${SEED} queue $(date '+%F %T')" | tee -a "$MASTER"
