#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs
LOG="results/remote_logs/e243_mps_xnet_baseline_queue.log"
exec > >(tee -a "$LOG") 2>&1

echo "QUEUE E243 MPS-XNet-style baseline $(date '+%F %T')"

while pgrep -af 'run_e240_single_frame_baselines_queue.sh|train_fpp_single_frame_baseline.py' | grep -v grep >/dev/null 2>&1; do
  echo "E240/E241/E242 comparison queue still active; sleep 300s $(date '+%F %T')"
  sleep 300
done

wait_gpu_free() {
  while true; do
    if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
      echo "No visible GPU yet; sleep 300s $(date '+%F %T')"
      sleep 300
      continue
    fi
    local active mem
    active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF{c++} END{print c+0}')
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
    if [ "${active}" -eq 0 ] && [ "${mem}" -lt 1000 ]; then
      echo "GPU free $(date '+%F %T')"
      break
    fi
    echo "GPU busy; active=${active} mem=${mem}; sleep 300s $(date '+%F %T')"
    sleep 300
  done
}

run_one() {
  local name="$1"
  shift
  wait_gpu_free
  echo "START ${name} $(date '+%F %T')"
  /root/miniconda3/bin/python "$@"
  echo "DONE ${name} $(date '+%F %T')"
}

run_one "E243 mps_xnet physical multitask raw-A0 baseline" \
  train_fpp_mps_xnet_baseline.py \
  --base_cache_dir /root/autodl-tmp/fpp_ml_bench_cache_960_fgfix \
  --phase_cache_dir /root/autodl-tmp/fpp_ml_phase_cache_960 \
  --save_dir results/e243_mps_xnet_physical_multitask_seed243_e80 \
  --epochs 80 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --num_workers 10 \
  --image_size 960 \
  --base_channels 8 \
  --lr 5e-4 \
  --weight_decay 1e-5 \
  --alpha 0.7 \
  --fenzi_weight 0.05 \
  --fenmu_weight 0.05 \
  --wrapped_weight 0.05 \
  --eval_every 1 \
  --eval_metrics_every 1 \
  --save_every 0 \
  --require_cache \
  --seed 243

/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path

items = []
summary_path = Path("results/e243_mps_xnet_physical_multitask_seed243_e80/evaluation/summary.json")
if summary_path.exists():
    with open(summary_path, "r", encoding="utf-8") as f:
        s = json.load(f)
    items.append({
        "method": "E243_MPS_XNet_style_physical_multitask",
        "summary_path": str(summary_path),
        "rmse": s.get("rmse", {}).get("mean"),
        "mae": s.get("mae", {}).get("mean"),
        "edge_rmse": s.get("edge_rmse", {}).get("mean"),
        "normal_deg": s.get("normal_deg", {}).get("mean"),
    })
out = Path("results/e243_mps_xnet_baseline_summary.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(items, f, indent=2, ensure_ascii=False)
print(json.dumps(items, indent=2, ensure_ascii=False))
PY

echo "ALL DONE E243 $(date '+%F %T')"
