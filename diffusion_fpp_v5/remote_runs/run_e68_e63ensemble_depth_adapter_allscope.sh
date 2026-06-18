#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs remote_runs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PSP_CACHE=/root/autodl-tmp/fpp_ml_pspquad_cache_960
PHASE_PREFIX=phase_pred_e63_ens_e28_e62_w055
INIT_RAW=results/fpp960_a_fringe_unet_control/checkpoints/best.pt
SAVE=results/fpp960_e68_e63ens_psp_instrxy_adapter_all_bs4_e80
MASTER=results/remote_logs/e68_e63ensemble_depth_adapter_allscope_master.log

echo "START E68 E63 ensemble phase depth adapter all-scope $(date '+%F %T')" | tee "$MASTER"

"$PY" train_fpp_psp_adapter_unet.py \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PSP_CACHE" \
  --phase_pred_prefix "$PHASE_PREFIX" \
  --save_dir "$SAVE" \
  --base_checkpoint "$INIT_RAW" \
  --cond_mode phase_pred_instr_xy \
  --instr_channels 1-10 \
  --train_scope all \
  --epochs 80 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --num_workers 16 \
  --image_size 960 \
  --lr 5e-4 \
  --adapter_lr 5e-4 \
  --backbone_lr 5e-6 \
  --weight_decay 1e-5 \
  --adapter_hidden 64 \
  --alpha 0.7 \
  --eval_every 2 \
  --eval_metrics_every 2 \
  --save_every 5 \
  --eval_initial \
  --train_minimal \
  --seed 68 \
  2>&1 | tee results/remote_logs/fpp960_e68_e63ens_psp_instrxy_adapter_all_bs4_e80.log

"$PY" - <<'PY' | tee results/remote_logs/e68_depth_comparison.txt
import json, os

paths = {
    'D47_hierarchical': 'results/d47_d31_epoch001_hierarchical_physical_gate/hierarchical_gate_summary.json',
    'E64_E63_adapter_decoder': 'results/fpp960_e64_e63ens_psp_instrxy_adapter_decoder_bs4_e25/evaluation/summary.json',
    'E68_E63_adapter_all': 'results/fpp960_e68_e63ens_psp_instrxy_adapter_all_bs4_e80/evaluation/summary.json',
}

def extract(path):
    d = json.load(open(path))
    if 'test' in d and 'hierarchical' in d['test']:
        d = d['test']['hierarchical']
    return {k: d[k]['mean'] for k in ['rmse', 'mae', 'edge_rmse', 'normal_deg', 'ssim'] if k in d}

out = {}
for name, path in paths.items():
    if os.path.exists(path):
        out[name] = extract(path)
print(json.dumps(out, indent=2, ensure_ascii=False))
with open('results/e68_depth_comparison_summary.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
PY

echo "DONE E68 $(date '+%F %T')" | tee -a "$MASTER"
