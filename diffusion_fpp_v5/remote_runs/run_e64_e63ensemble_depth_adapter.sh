#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PSP_CACHE=/root/autodl-tmp/fpp_ml_pspquad_cache_960
E62_DIR=results/fpp960_e62_pspquad_phase_diffusion_ch24_reg_e30
E62_PREFIX=phase_pred_e62_best_pspquad_ddim20_e3
ENS_PREFIX=phase_pred_e63_ens_e28_e62_w055
E64_DIR=results/fpp960_e64_e63ens_psp_instrxy_adapter_decoder_bs4_e25
INIT_RAW=results/fpp960_a_fringe_unet_control/checkpoints/best.pt
MASTER_LOG=results/remote_logs/e64_e63ensemble_depth_adapter_master.log

echo "START E64 ensemble phase to depth $(date '+%F %T')" | tee "$MASTER_LOG"

echo "PRECOMPUTE E62 train/val predictions if missing $(date '+%F %T')" | tee -a "$MASTER_LOG"
if [[ ! -f "$PSP_CACHE/${E62_PREFIX}_train_float16.npy" || ! -f "$PSP_CACHE/${E62_PREFIX}_val_float16.npy" ]]; then
  "$PY" precompute_fpp_phase_diffusion_predictions.py \
    --checkpoint "$E62_DIR/checkpoints/best_phase.pt" \
    --base_cache_dir "$BASE_CACHE" \
    --phase_cache_dir "$PSP_CACHE" \
    --output_prefix "$E62_PREFIX" \
    --splits train,val \
    --image_size 960 \
    --batch_size 1 \
    --num_workers 8 \
    --ddim_steps 20 \
    --ensemble 3 \
    --sample_start_from ftp \
    --sample_start_ratio 0.7 \
    2>&1 | tee results/remote_logs/e62_best_train_val_precompute.log
fi

echo "BUILD E63 ensemble predictions w=0.55 $(date '+%F %T')" | tee -a "$MASTER_LOG"
"$PY" - <<'PY' | tee results/remote_logs/e63_build_ensemble_w055.log
import json
from pathlib import Path
import numpy as np

cache = Path('/root/autodl-tmp/fpp_ml_pspquad_cache_960')
e28 = 'phase_pred_e28_pspquad_ddim20_e3'
e62 = 'phase_pred_e62_best_pspquad_ddim20_e3'
outp = 'phase_pred_e63_ens_e28_e62_w055'
w = 0.55
eps = 1e-8

def normalize_sc(sc):
    n = np.sqrt(np.sum(sc * sc, axis=1, keepdims=True))
    return sc / np.maximum(n, eps)

summary = {}
for split in ('train', 'val', 'test'):
    out_path = cache / f'{outp}_{split}_float16.npy'
    p28 = np.load(cache / f'{e28}_{split}_float16.npy', mmap_mode='r')
    p62 = np.load(cache / f'{e62}_{split}_float16.npy', mmap_mode='r')
    if out_path.exists() and tuple(np.load(out_path, mmap_mode='r').shape) == tuple(p28.shape):
        summary[split] = {'path': str(out_path), 'shape': list(p28.shape), 'skipped': True}
        continue
    out = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float16, shape=p28.shape)
    bs = 4
    for s in range(0, p28.shape[0], bs):
        a = p28[s:s+bs].astype(np.float32)
        b = p62[s:s+bs].astype(np.float32)
        y = np.empty_like(a, dtype=np.float32)
        y[:, :2] = w * a[:, :2] + (1.0 - w) * b[:, :2]
        y[:, :2] = normalize_sc(y[:, :2])
        y[:, 2:] = w * a[:, 2:] + (1.0 - w) * b[:, 2:]
        out[s:s+bs] = y.astype(np.float16)
    out.flush()
    summary[split] = {'path': str(out_path), 'shape': list(p28.shape), 'skipped': False}
with open(cache / f'{outp}_manifest.json', 'w', encoding='utf-8') as f:
    json.dump({'source': [e28, e62], 'w_e28': w, 'splits': summary}, f, indent=2)
print(json.dumps(summary, indent=2))
PY

echo "START E64 depth adapter on E63 ensemble phase $(date '+%F %T')" | tee -a "$MASTER_LOG"
"$PY" train_fpp_psp_adapter_unet.py \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PSP_CACHE" \
  --phase_pred_prefix "$ENS_PREFIX" \
  --save_dir "$E64_DIR" \
  --base_checkpoint "$INIT_RAW" \
  --cond_mode phase_pred_instr_xy \
  --instr_channels 1-10 \
  --train_scope adapter_decoder \
  --epochs 25 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --num_workers 16 \
  --image_size 960 \
  --lr 5e-4 \
  --adapter_lr 5e-4 \
  --backbone_lr 2e-5 \
  --weight_decay 1e-5 \
  --adapter_hidden 64 \
  --alpha 0.7 \
  --eval_every 1 \
  --eval_metrics_every 1 \
  --save_every 5 \
  --eval_initial \
  --train_minimal \
  --seed 64 \
  2>&1 | tee results/remote_logs/fpp960_e64_e63ens_psp_instrxy_adapter_decoder_bs4_e25.log

echo "WRITE E64 depth comparison $(date '+%F %T')" | tee -a "$MASTER_LOG"
"$PY" - <<'PY' | tee results/remote_logs/e64_depth_comparison.txt
import json, os

paths = {
    'D47_hierarchical': 'results/d47_d31_epoch001_hierarchical_physical_gate/hierarchical_gate_summary.json',
    'E34_E28_phase_adapter': 'results/fpp960_e34c_pspquad_pred_instr_xy_adapter_decoder_bs4_e70/evaluation/summary.json',
    'E64_E63ensemble_phase_adapter': 'results/fpp960_e64_e63ens_psp_instrxy_adapter_decoder_bs4_e25/evaluation/summary.json',
}

def extract(p):
    d=json.load(open(p))
    if 'test' in d:
        d=d['test']['hierarchical']
    return {k:d[k]['mean'] for k in ['rmse','mae','edge_rmse','normal_deg','ssim'] if k in d}

out={}
for k,p in paths.items():
    if os.path.exists(p):
        out[k]=extract(p)
print(json.dumps(out, indent=2, ensure_ascii=False))
with open('results/e64_depth_comparison_summary.json','w',encoding='utf-8') as f:
    json.dump(out,f,indent=2,ensure_ascii=False)
PY

echo "DONE E64 ensemble phase to depth $(date '+%F %T')" | tee -a "$MASTER_LOG"
