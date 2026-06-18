#!/usr/bin/env bash
set -uo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5 || exit 1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PHASE_CACHE=/root/autodl-tmp/fpp_ml_phase_cache_960
LOGDIR=/root/autodl-tmp/diffusion_fpp_v5/results/_logs
mkdir -p "$LOGDIR"

echo "===== A-copy value queue started $(date '+%F %T') ====="
echo "Waiting for current seed132 job to finish; seed133 continuation will be stopped."

while pgrep -f "train_joint_pip_diffusion.py.*fpp960_f1_masked_residual_seed132" >/dev/null; do
  date '+%F %T waiting seed132'
  sleep 30
done

sleep 5
if pgrep -f "train_joint_pip_diffusion.py.*fpp960_f1_masked_residual_seed133" >/dev/null; then
  echo "Stopping queued seed133 because follow-up seeds are lower priority than method comparison."
  pkill -TERM -f "train_joint_pip_diffusion.py.*fpp960_f1_masked_residual_seed133" || true
  sleep 15
  pkill -KILL -f "train_joint_pip_diffusion.py.*fpp960_f1_masked_residual_seed133" || true
fi
pkill -TERM -f "bash remote_runs/run_f1_continue_seed132_133.sh" || true
pkill -TERM -f "run_f1_continue_seed132_133.sh" || true

while pgrep -f "train_joint_pip_diffusion.py.*fpp960_f1_masked_residual_seed133" >/dev/null; do
  date '+%F %T waiting seed133 termination'
  sleep 10
done

run_mps_phase_proxy() {
  local exp_id="$1"
  local seed="$2"
  local w4="$3"
  local save_dir="/root/autodl-tmp/diffusion_fpp_v5/results/${exp_id}_mps_xnet_phase_proxy_w${w4}_seed${seed}_e100"
  local log="$LOGDIR/${exp_id}_mps_xnet_phase_proxy_w${w4}_seed${seed}_e100.log"
  echo "===== ${exp_id} MPS-XNet phase-first proxy w4=${w4} seed=${seed} $(date '+%F %T') =====" | tee -a "$log"
  nvidia-smi | tee -a "$log" || true
  "$PY" train_fpp_mps_xnet_phase_proxy_baseline.py \
    --base_cache_dir "$BASE_CACHE" \
    --phase_cache_dir "$PHASE_CACHE" \
    --save_dir "$save_dir" \
    --epochs 100 \
    --batch_size 2 \
    --eval_batch_size 2 \
    --num_workers 10 \
    --image_size 960 \
    --base_channels 8 \
    --lr 5e-4 \
    --weight_decay 1e-5 \
    --w1 1.0 \
    --w2 1.0 \
    --w3 1.0 \
    --w4 "$w4" \
    --eval_every 1 \
    --eval_metrics_every 1 \
    --require_cache \
    --seed "$seed" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  echo "===== ${exp_id} finished rc=${rc} $(date '+%F %T') =====" | tee -a "$log"
  return "$rc"
}

run_mps_phase_proxy e251 251 50
rc251=$?
if [ "$rc251" -eq 0 ]; then
  run_mps_phase_proxy e252 252 100
  rc252=$?
else
  echo "E251 failed; skipping E252 to avoid repeating the same failure mode."
  rc252=999
fi

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/root/autodl-tmp/diffusion_fpp_v5/results")
items = {
    "E243_MPS_XNet_depth_adaptation": root / "e243_mps_xnet_baseline_summary.json",
    "E244_Pix2Pix": root / "e244_pix2pix_baseline_summary.json",
    "E251_MPS_XNet_phase_proxy_w50": root / "e251_mps_xnet_phase_proxy_w50_seed251_e100" / "evaluation" / "summary.json",
    "E252_MPS_XNet_phase_proxy_w100": root / "e252_mps_xnet_phase_proxy_w100_seed252_e100" / "evaluation" / "summary.json",
}
summary = {}
for name, path in items.items():
    if not path.exists():
        summary[name] = {"exists": False, "path": str(path)}
        continue
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        summary[name] = {"exists": True, "path": str(path), "error": repr(exc)}
        continue
    summary[name] = {
        "exists": True,
        "path": str(path),
        "rmse_mean": data.get("rmse", {}).get("mean"),
        "mae_mean": data.get("mae", {}).get("mean"),
        "edge_rmse_mean": data.get("edge_rmse", {}).get("mean"),
        "normal_mean": data.get("normal_error", {}).get("mean"),
        "phase_rmse_mean": data.get("phase_rmse", {}).get("mean"),
        "method_note": data.get("method_note", ""),
    }
out = root / "e251_e252_mps_phase_proxy_queue_summary.json"
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "===== A-copy value queue finished $(date '+%F %T') rc251=${rc251} rc252=${rc252} ====="
