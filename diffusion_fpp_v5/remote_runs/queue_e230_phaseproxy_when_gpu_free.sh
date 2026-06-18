#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

LOG=results/remote_logs/e230_phaseproxy_queue_wait.log

echo "QUEUE E230 waiting for free GPU $(date '+%F %T')" | tee -a "$LOG"
while true; do
  if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
    echo "GPU not visible; sleep 300s $(date '+%F %T')" | tee -a "$LOG"
    sleep 300
    continue
  fi
  active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -E '[0-9]' | wc -l || true)
  mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "unknown")
  if [[ "$active" -eq 0 ]]; then
    echo "GPU free; starting E230 $(date '+%F %T') mem=$mem" | tee -a "$LOG"
    break
  fi
  echo "GPU busy; active=$active mem=$mem; sleep 300s $(date '+%F %T')" | tee -a "$LOG"
  sleep 300
done

bash remote_runs/run_e230_phaseproxy_psp_seed180_pilot.sh
