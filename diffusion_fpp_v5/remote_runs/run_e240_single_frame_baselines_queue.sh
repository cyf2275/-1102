#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
LOG=results/remote_logs/e240_single_frame_baselines_queue.log

wait_gpu_free() {
  while true; do
    if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
      echo "GPU not visible; sleep 300s $(date '+%F %T')" | tee -a "$LOG"
      sleep 300
      continue
    fi
    active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -E '[0-9]' | wc -l || true)
    mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "unknown")
    if [[ "$active" -eq 0 ]]; then
      echo "GPU free $(date '+%F %T') mem=$mem" | tee -a "$LOG"
      break
    fi
    echo "GPU busy; active=$active mem=$mem; sleep 300s $(date '+%F %T')" | tee -a "$LOG"
    sleep 300
  done
}

run_one() {
  local arch="$1"
  local save="$2"
  local base_ch="$3"
  local seed="$4"
  local train_log="$5"

  if [[ -f "$save/evaluation/summary.json" ]]; then
    echo "SKIP $arch; existing $save/evaluation/summary.json" | tee -a "$LOG"
    return
  fi

  wait_gpu_free
  echo "START $arch baseline $(date '+%F %T')" | tee -a "$LOG"
  "$PY" train_fpp_single_frame_baseline.py \
    --cache_dir "$BASE_CACHE" \
    --save_dir "$save" \
    --arch "$arch" \
    --epochs 80 \
    --batch_size 4 \
    --eval_batch_size 4 \
    --num_workers 10 \
    --image_size 960 \
    --base_channels "$base_ch" \
    --lr 5e-4 \
    --weight_decay 1e-5 \
    --alpha 0.7 \
    --eval_every 1 \
    --eval_metrics_every 1 \
    --save_every 0 \
    --require_cache \
    --seed "$seed" \
    2>&1 | tee "$train_log"
  echo "DONE $arch baseline $(date '+%F %T')" | tee -a "$LOG"
}

echo "QUEUE E240 single-frame baselines $(date '+%F %T')" | tee -a "$LOG"
"$PY" -m py_compile train_fpp_single_frame_baseline.py models/single_frame_baselines.py

run_one resunet \
  results/fpp960_e240_resunet_raw_a0_e80_seed240 \
  48 \
  240 \
  results/remote_logs/e240_resunet_raw_a0_train.log

run_one attention_unet \
  results/fpp960_e241_attention_unet_raw_a0_e80_seed241 \
  48 \
  241 \
  results/remote_logs/e241_attention_unet_raw_a0_train.log

run_one nested_unet \
  results/fpp960_e242_nested_unetpp_raw_a0_e80_seed242 \
  32 \
  242 \
  results/remote_logs/e242_nested_unetpp_raw_a0_train.log

"$PY" - <<'PY' | tee results/remote_logs/e240_single_frame_baselines_summary.txt
import json
from pathlib import Path

paths = {
    "E240_ResUNet_raw_A0": Path("results/fpp960_e240_resunet_raw_a0_e80_seed240/evaluation/summary.json"),
    "E241_AttentionUNet_raw_A0": Path("results/fpp960_e241_attention_unet_raw_a0_e80_seed241/evaluation/summary.json"),
    "E242_UNetPP_raw_A0": Path("results/fpp960_e242_nested_unetpp_raw_a0_e80_seed242/evaluation/summary.json"),
}

def metric_block(d):
    return {
        k: d[k]["mean"]
        for k in ["rmse", "mae", "edge_rmse", "normal_deg", "ssim"]
        if k in d
    }

out = {}
for name, path in paths.items():
    if path.exists():
        out[name] = metric_block(json.loads(path.read_text(encoding="utf-8")))
    else:
        out[name] = {"missing": str(path)}
print(json.dumps(out, indent=2, ensure_ascii=False))
Path("results/e240_single_frame_baselines_summary.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
)
PY

echo "DONE E240 single-frame baseline queue $(date '+%F %T')" | tee -a "$LOG"
