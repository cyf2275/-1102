#!/usr/bin/env bash
set -u

CODE_ROOT="${CODE_ROOT:-/root/autodl-tmp/diffusion_fpp_v5}"
RESULT_ROOT="${RESULT_ROOT:-/root/autodl-tmp/diffusion_fpp_v5/results/A_20260614_single_frame3d_physics_diffusion}"
RUNNER="${RUNNER:-${CODE_ROOT}/remote_runs/run_single_frame3d_physics_diffusion.sh}"
WATCH_INTERVAL="${WATCH_INTERVAL:-300}"
WATCH_HOURS="${WATCH_HOURS:-10}"

mkdir -p "${RESULT_ROOT}/logs"
WATCH_LOG="${RESULT_ROOT}/logs/watchdog.log"
RESTART_LOG="${RESULT_ROOT}/logs/watchdog_restarted_runner.log"
REPORT="${RESULT_ROOT}/single_frame3d_physics_diffusion_report.md"
SUMMARY="${RESULT_ROOT}/single_frame3d_physics_diffusion_summary.json"

deadline=$(( $(date +%s) + WATCH_HOURS * 3600 ))
echo "$(date '+%F %T') watchdog start interval=${WATCH_INTERVAL}s hours=${WATCH_HOURS}" >> "${WATCH_LOG}"

while [[ "$(date +%s)" -lt "${deadline}" ]]; do
  if [[ -f "${REPORT}" && -f "${SUMMARY}" ]]; then
    echo "$(date '+%F %T') final report exists; watchdog exit" >> "${WATCH_LOG}"
    exit 0
  fi

  runner_alive=0
  train_alive=0
  if pgrep -af 'remote_runs/run_single_frame3d_physics_diffusion.sh' >/dev/null 2>&1; then
    runner_alive=1
  fi
  if pgrep -af 'train_single_frame3d_physics_diffusion.py' >/dev/null 2>&1; then
    train_alive=1
  fi

  if [[ "${runner_alive}" -eq 1 || "${train_alive}" -eq 1 ]]; then
    echo "$(date '+%F %T') alive runner=${runner_alive} train=${train_alive}" >> "${WATCH_LOG}"
  else
    echo "$(date '+%F %T') no active runner/train and no final report; restart runner" >> "${WATCH_LOG}"
    (
      cd "${CODE_ROOT}" || exit 1
      nohup bash "${RUNNER}" >> "${RESTART_LOG}" 2>&1 &
    )
  fi

  sleep "${WATCH_INTERVAL}"
done

echo "$(date '+%F %T') watchdog time window ended" >> "${WATCH_LOG}"
