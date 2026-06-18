#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

MASTER=results/remote_logs/after_e190_queue_master.log
echo "START after-E190 queue watcher $(date '+%F %T')" | tee "$MASTER"

while true; do
  if [[ -f results/e190_after_e181_summary.json ]]; then
    echo "E190 summary found $(date '+%F %T')" | tee -a "$MASTER"
    break
  fi
  if ps -eo cmd | grep -E 'run_e190_after_e181_base_phase_fusion.sh|precompute_fpp_official_style_predictions.py|eval_base_phase_fusion.py|select_edge_aware_phase_gate_csv.py' | grep -v grep >/dev/null; then
    echo "E190 still running $(date '+%F %T')" | tee -a "$MASTER"
  else
    echo "E190 not running and summary missing; restart E190 $(date '+%F %T')" | tee -a "$MASTER"
    nohup bash remote_runs/run_e190_after_e181_base_phase_fusion.sh > results/remote_logs/e190_after_e181_retry_nohup.out 2>&1 &
  fi
  sleep 180
done

if [[ -f results/e200_e181_base_residual_diffusion_summary.json ]]; then
  echo "E200 already complete; skip $(date '+%F %T')" | tee -a "$MASTER"
else
  echo "START E200 from watcher $(date '+%F %T')" | tee -a "$MASTER"
  bash remote_runs/run_e200_e181_base_residual_diffusion.sh 2>&1 | tee results/remote_logs/e200_from_watcher_nohup.out
fi

echo "DONE after-E190 queue watcher $(date '+%F %T')" | tee -a "$MASTER"
