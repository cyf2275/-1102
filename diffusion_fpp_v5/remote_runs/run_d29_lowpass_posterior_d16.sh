#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
BASE_PREFIX=base_c4_adapter
CKPT=/root/autodl-tmp/diffusion_fpp_v5/results/pip_d16_lowt015_base_residual_e1_seed0/checkpoints/best.pt
OUT=/root/autodl-tmp/diffusion_fpp_v5/results/pip_d29_lowpass_posterior_d16

echo "===== D29 low-pass edge-aware residual posterior on D16 $(date '+%F %T') ====="

/root/miniconda3/bin/python eval_lowpass_residual_posterior.py \
  --checkpoint "$CKPT" \
  --cache_dir "$CACHE" \
  --base_prefix "$BASE_PREFIX" \
  --save_dir "$OUT" \
  --split val \
  --image_h 960 \
  --image_w 960 \
  --ddim_steps 20 \
  --ensemble 1 \
  --eval_batch_size 1 \
  --num_workers 0 \
  --start_ratio 0.05 \
  --alphas 0,0.05,0.10,0.20,0.35,0.50,0.75 \
  --kernels 1,3,7,15,31 \
  --edge_powers 0,1,2,4 \
  --conf_powers 0,1 \
  --require_cache

read ALPHA KERNEL EDGE_POWER CONF_POWER < <(/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("/root/autodl-tmp/diffusion_fpp_v5/results/pip_d29_lowpass_posterior_d16/val_lowpass_summary.json")
d = json.loads(p.read_text(encoding="utf-8"))
s = d["selected"]
print(s["alpha"], s["kernel"], s["edge_power"], s["conf_power"])
PY
)
echo "===== D29 selected val config alpha=${ALPHA} kernel=${KERNEL} edge=${EDGE_POWER} conf=${CONF_POWER} ====="

/root/miniconda3/bin/python eval_lowpass_residual_posterior.py \
  --checkpoint "$CKPT" \
  --cache_dir "$CACHE" \
  --base_prefix "$BASE_PREFIX" \
  --save_dir "$OUT" \
  --split test \
  --image_h 960 \
  --image_w 960 \
  --ddim_steps 20 \
  --ensemble 1 \
  --eval_batch_size 1 \
  --num_workers 0 \
  --start_ratio 0.05 \
  --alphas "$ALPHA" \
  --kernels "$KERNEL" \
  --edge_powers "$EDGE_POWER" \
  --conf_powers "$CONF_POWER" \
  --require_cache

/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path
out = Path("/root/autodl-tmp/diffusion_fpp_v5/results/pip_d29_lowpass_posterior_d16")
summary = {
    "val": json.loads((out / "val_lowpass_summary.json").read_text(encoding="utf-8")),
    "test": json.loads((out / "test_lowpass_summary.json").read_text(encoding="utf-8")),
}
(out / "lowpass_posterior_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "===== D29 DONE $(date '+%F %T') ====="
