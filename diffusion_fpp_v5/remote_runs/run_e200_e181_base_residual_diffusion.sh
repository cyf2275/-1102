#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY=/root/miniconda3/bin/python
CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
E181_DIR=results/fpp960_e181_official_fringe_plus_physics_ftp_e120_bs2
E181_CKPT="$E181_DIR/checkpoints/best_rmse.pt"
BASE_PREFIX=base_e181_physicsftp
MASTER=results/remote_logs/e200_e181_base_residual_diffusion_master.log
SUMMARY=results/e200_e181_base_residual_diffusion_summary.json

echo "START E200 E181-base residual diffusion $(date '+%F %T')" | tee "$MASTER"

if [[ ! -f "$CACHE/${BASE_PREFIX}_stats.json" ]]; then
  echo "E181 base cache missing; precompute now $(date '+%F %T')" | tee -a "$MASTER"
  "$PY" precompute_fpp_official_style_predictions.py \
    --cache_dir "$CACHE" \
    --checkpoint "$E181_CKPT" \
    --prefix "$BASE_PREFIX" \
    --input_mode fringe_plus_physics \
    --include_ftp \
    --physics_channels 0-10 \
    --batch_size 2 \
    --num_workers 12 \
    --image_size 960 \
    --require_cache \
    2>&1 | tee results/remote_logs/e200_precompute_e181_base.log
fi

run_one() {
  local ch="$1"
  local epochs="$2"
  local run_dir="results/pip_e200_e181base_ch${ch}_lowt015_e${epochs}_seed0"
  local feat_dir="results/pip_e200_e181base_ch${ch}_lowt015_e${epochs}_seed0_adaptive_a050"

  echo "TRAIN E200 base_channels=${ch} epochs=${epochs} $(date '+%F %T')" | tee -a "$MASTER"
  "$PY" train_pip_lite.py \
    --dataset fpp_ml_bench \
    --cache_dir "$CACHE" \
    --save_dir "$run_dir" \
    --epochs "$epochs" \
    --eval_every 1 \
    --save_every 1 \
    --save_epoch_checkpoints \
    --skip_final_test \
    --batch_size 1 \
    --eval_batch_size 1 \
    --num_workers 8 \
    --require_cache \
    --image_h 960 \
    --image_w 960 \
    --target_mode base_residual \
    --base_prefix "$BASE_PREFIX" \
    --condition_injection adapter \
    --include_ftp \
    --physics_channels 0-10 \
    --adapter_hidden 16 \
    --base_channels "$ch" \
    --timesteps 200 \
    --ddim_steps 20 \
    --ensemble 1 \
    --sample_start_ratio 0.05 \
    --train_t_min_ratio 0.0 \
    --train_t_max_ratio 0.15 \
    --lr 3e-5 \
    --base_residual_gate 0.5 \
    --seed 0 \
    2>&1 | tee "results/remote_logs/e200_ch${ch}_train.log"

  for ep_num in $(seq 1 "$epochs"); do
    local ep
    ep=$(printf "%03d" "$ep_num")
    local ckpt="$run_dir/checkpoints/epoch_${ep}.pt"
    local out="${feat_dir}_epoch${ep}"
    echo "EVAL E200 ch=${ch} epoch=${ep} adaptive gate $(date '+%F %T')" | tee -a "$MASTER"
    for split in val test; do
      "$PY" eval_adaptive_blend_features.py \
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
        --alpha 0.5 \
        --require_cache \
        2>&1 | tee "results/remote_logs/e200_ch${ch}_epoch${ep}_${split}_eval.log"
    done
    "$PY" select_adaptive_gate.py \
      --val_csv "$out/val_adaptive_features.csv" \
      --test_csv "$out/test_adaptive_features.csv" \
      --save_json "$out/selected_gate_summary.json" \
      --min_selected 3 \
      2>&1 | tee "results/remote_logs/e200_ch${ch}_epoch${ep}_gate.log"
  done
}

run_one 24 5
run_one 32 5

"$PY" - <<'PY' | tee results/remote_logs/e200_summary.log
import json
from pathlib import Path

base = Path("results")
rows = []
for ch in (24, 32):
    for ep in range(1, 6):
        name = f"pip_e200_e181base_ch{ch}_lowt015_e5_seed0_adaptive_a050_epoch{ep:03d}"
        p = base / name / "selected_gate_summary.json"
        if not p.exists():
            rows.append({"name": name, "missing": str(p)})
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        rows.append({
            "name": name,
            "base_channels": ch,
            "epoch": ep,
            "selected_gate": data["selected_gate"],
            "val_base_rmse": data["val"]["base"]["rmse"]["mean"],
            "val_blend_rmse": data["val"]["blend"]["rmse"]["mean"],
            "val_gated_rmse": data["val"]["gated"]["rmse"]["mean"],
            "test_base_rmse": data["test"]["base"]["rmse"]["mean"],
            "test_blend_rmse": data["test"]["blend"]["rmse"]["mean"],
            "test_gated_rmse": data["test"]["gated"]["rmse"]["mean"],
            "test_selected": data["test"]["gated"]["selected"],
        })

valid = [r for r in rows if "missing" not in r]
summary = {
    "purpose": "Train low-t residual diffusion on the E181 physical-UNet base cache and select a validation-gated residual correction.",
    "base_prefix": "base_e181_physicsftp",
    "rows": rows,
    "best_by_val_gated": min(valid, key=lambda r: r["val_gated_rmse"]) if valid else None,
    "best_by_test_gated": min(valid, key=lambda r: r["test_gated_rmse"]) if valid else None,
}
out = base / "e200_e181_base_residual_diffusion_summary.json"
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "DONE E200 E181-base residual diffusion $(date '+%F %T')" | tee -a "$MASTER"
