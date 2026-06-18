#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
export PYTHONUNBUFFERED=1

CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
BASE_PREFIX=base_c4_adapter
SEEDS="0 1 2 3 4 42 123 456"
SAMPLE_EDGE_TH=0.4674050956964493

for seed in $SEEDS; do
  name="pip_d12_combined_sample_pixel_gate_d8_seed${seed}"
  ckpt="/root/autodl-tmp/diffusion_fpp_v5/results/pip_d8_seed${seed}_base_residual_e1_gate050_lr3e5/checkpoints/best.pt"
  out_dir="/root/autodl-tmp/diffusion_fpp_v5/results/${name}"
  if [ -f "${out_dir}/pixel_gate_summary.json" ]; then
    echo "===== D12 SKIP seed=${seed}; summary exists $(date '+%F %T') ====="
    continue
  fi
  echo "===== D12 START seed=${seed} $(date '+%F %T') ====="
  /root/miniconda3/bin/python eval_pixel_adaptive_gate.py \
    --checkpoint "$ckpt" \
    --cache_dir "$CACHE" \
    --base_prefix "$BASE_PREFIX" \
    --save_dir "$out_dir" \
    --image_h 960 \
    --image_w 960 \
    --ddim_steps 20 \
    --ensemble 1 \
    --eval_batch_size 1 \
    --num_workers 0 \
    --start_ratio 0.05 \
    --alphas "0.25" \
    --sample_edge_thresholds "$SAMPLE_EDGE_TH" \
    --edge_thresholds "0.8" \
    --delta_mins "0.12" \
    --conf_mins "0.0" \
    --require_cache
  echo "===== D12 END seed=${seed} $(date '+%F %T') ====="
done

/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path

base = Path('/root/autodl-tmp/diffusion_fpp_v5/results')
rows = []
for p in sorted(base.glob('pip_d12_combined_sample_pixel_gate_d8_seed*/pixel_gate_summary.json')):
    seed = p.parent.name.split('seed')[-1]
    d = json.load(open(p, 'r', encoding='utf-8'))
    rows.append({
        'seed': seed,
        'val_base': d['val']['base']['rmse']['mean'],
        'val_combined': d['val']['pixel_gated']['rmse']['mean'],
        'test_base': d['test']['base']['rmse']['mean'],
        'test_combined': d['test']['pixel_gated']['rmse']['mean'],
        'test_edge_base': d['test']['base']['edge_rmse']['mean'],
        'test_edge_combined': d['test']['pixel_gated']['edge_rmse']['mean'],
        'test_mae_base': d['test']['base']['mae']['mean'],
        'test_mae_combined': d['test']['pixel_gated']['mae']['mean'],
        'test_selected_frac': d['test']['selected_frac_mean'],
        'best_gate': d['best_gate'],
    })
if rows:
    def mean(key):
        return sum(float(r[key]) for r in rows) / len(rows)
    summary = {
        'n': len(rows),
        'gate': {
            'alpha': 0.25,
            'sample_edge_th': 0.4674050956964493,
            'edge_th': 0.8,
            'delta_min': 0.12,
            'conf_min': 0.0,
        },
        'test_base_rmse_mean': mean('test_base'),
        'test_combined_rmse_mean': mean('test_combined'),
        'test_base_mae_mean': mean('test_mae_base'),
        'test_combined_mae_mean': mean('test_mae_combined'),
        'test_base_edge_rmse_mean': mean('test_edge_base'),
        'test_combined_edge_rmse_mean': mean('test_edge_combined'),
        'test_selected_frac_mean': mean('test_selected_frac'),
        'wins': sum(float(r['test_combined']) < float(r['test_base']) for r in rows),
        'rows': rows,
    }
    out = base / 'pip_d12_combined_sample_pixel_gate_summary.json'
    json.dump(summary, open(out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))
PY

echo "===== D12 ALL DONE $(date '+%F %T') ====="
