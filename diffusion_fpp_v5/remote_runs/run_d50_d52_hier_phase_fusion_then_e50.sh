#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs remote_runs

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PSP_CACHE=/root/autodl-tmp/fpp_ml_pspquad_cache_960
DEPTH_CKPT=results/pip_d31_ch24_lowt015_e3_seed0/checkpoints/epoch_001.pt
E30_CKPT=results/fpp960_e30_pspquad_pred_plus_fringe_from_rawA_e60/checkpoints/best_rmse.pt
E31_CKPT=results/fpp960_e31_pspquad_gt_plus_fringe_from_rawA_e40/checkpoints/best_rmse.pt
E36_CKPT=results/fpp960_e36c_pspquad_pred_instr_xy_adapter_all_bs4_e60/checkpoints/best_rmse.pt
INIT_RAW=results/fpp960_a_fringe_unet_control/checkpoints/best.pt
PRED_PREFIX=phase_pred_e28_pspquad_ddim20_e3
MASTER=results/remote_logs/d50_d52_hier_phase_fusion_then_e50_master.log

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "START D50-D52/E50 suite $(date '+%F %T')" | tee "$MASTER"
"$PY" -m py_compile eval_hierarchical_phase_fusion.py

run_fusion() {
  local name="$1"
  local ckpt="$2"
  echo "START ${name} $(date '+%F %T')" | tee -a "$MASTER"
  "$PY" eval_hierarchical_phase_fusion.py \
    --depth_checkpoint "$DEPTH_CKPT" \
    --phase_depth_checkpoint "$ckpt" \
    --cache_dir "$BASE_CACHE" \
    --phase_cache_dir "$PSP_CACHE" \
    --base_prefix base_c4_adapter \
    --save_dir "results/${name}" \
    --image_size 960 \
    --eval_batch_size 2 \
    --num_workers 0 \
    --ddim_steps 20 \
    --ensemble 1 \
    --start_ratio 0.05 \
    --phase_weights "0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5" \
    --require_cache \
    2>&1 | tee "results/remote_logs/${name}.log"
  echo "DONE ${name} $(date '+%F %T')" | tee -a "$MASTER"
}

run_fusion d50_d47_hier_plus_e30_phasepred_branch "$E30_CKPT"
if [[ -f "$E36_CKPT" ]]; then
  run_fusion d51_d47_hier_plus_e36c_phasepred_adapter "$E36_CKPT"
fi
run_fusion d52_d47_hier_plus_e31_gtpsp_oracle "$E31_CKPT"

echo "START E50 predicted PSP adapter all-scope long run $(date '+%F %T')" | tee -a "$MASTER"
"$PY" train_fpp_psp_adapter_unet.py \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PSP_CACHE" \
  --phase_pred_prefix "$PRED_PREFIX" \
  --save_dir results/fpp960_e50_pspquad_pred_instr_xy_adapter_all_bs4_e100 \
  --base_checkpoint "$INIT_RAW" \
  --cond_mode phase_pred_instr_xy \
  --instr_channels 1-10 \
  --train_scope all \
  --epochs 100 \
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
  --eval_every 5 \
  --eval_metrics_every 5 \
  --save_every 5 \
  --eval_initial \
  --train_minimal \
  --seed 123 \
  2>&1 | tee results/remote_logs/fpp960_e50_pspquad_pred_instr_xy_adapter_all_bs4_e100.log

echo "DONE D50-D52/E50 suite $(date '+%F %T')" | tee -a "$MASTER"
