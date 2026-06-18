#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs
LOG="results/remote_logs/e244_pix2pix_baseline_queue.log"
exec > >(tee -a "$LOG") 2>&1

echo "QUEUE E244 pix2pix baseline $(date '+%F %T')"

while pgrep -af 'run_e240_single_frame_baselines_queue.sh|run_e243_mps_xnet_baseline_queue.sh|train_fpp_single_frame_baseline.py|train_fpp_mps_xnet_baseline.py' | grep -v grep >/dev/null 2>&1; do
  echo "E240/E243 comparison queue still active; sleep 300s $(date '+%F %T')"
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

wait_gpu_free
echo "START E244 pix2pix raw-A0 baseline $(date '+%F %T')"
/root/miniconda3/bin/python train_fpp_pix2pix_baseline.py \
  --cache_dir /root/autodl-tmp/fpp_ml_bench_cache_960_fgfix \
  --save_dir results/e244_pix2pix_raw_a0_seed244_e80 \
  --epochs 80 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --num_workers 10 \
  --image_size 960 \
  --gen_channels 64 \
  --disc_channels 64 \
  --lr_g 2e-4 \
  --lr_d 2e-4 \
  --lambda_l1 1.0 \
  --lambda_smooth_l1 0.5 \
  --lambda_gan 0.01 \
  --eval_every 1 \
  --eval_metrics_every 1 \
  --require_cache \
  --seed 244
echo "DONE E244 pix2pix raw-A0 baseline $(date '+%F %T')"

/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path

summary_path = Path("results/e244_pix2pix_raw_a0_seed244_e80/evaluation/summary.json")
items = []
if summary_path.exists():
    with open(summary_path, "r", encoding="utf-8") as f:
        s = json.load(f)
    items.append({
        "method": "E244_pix2pix_raw_A0",
        "summary_path": str(summary_path),
        "rmse": s.get("rmse", {}).get("mean"),
        "mae": s.get("mae", {}).get("mean"),
        "edge_rmse": s.get("edge_rmse", {}).get("mean"),
        "normal_deg": s.get("normal_deg", {}).get("mean"),
    })
out = Path("results/e244_pix2pix_baseline_summary.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(items, f, indent=2, ensure_ascii=False)
print(json.dumps(items, indent=2, ensure_ascii=False))
PY

echo "ALL DONE E244 $(date '+%F %T')"
