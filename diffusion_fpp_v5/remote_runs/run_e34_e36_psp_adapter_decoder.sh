#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PSP_CACHE=/root/autodl-tmp/fpp_ml_pspquad_cache_960
PRED_PREFIX=phase_pred_e28_pspquad_ddim20_e3
INIT_RAW=results/fpp960_a_fringe_unet_control/checkpoints/best.pt

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

MASTER_LOG=results/remote_logs/e34_e36_psp_adapter_decoder_master.log
echo "START E34-E36 PSP adapter-decoder suite $(date '+%F %T')" | tee "$MASTER_LOG"

run_depth() {
  local name="$1"
  local cond_mode="$2"
  local phase_prefix="$3"
  local scope="$4"
  local epochs="$5"
  local adapter_lr="$6"
  local backbone_lr="$7"
  local out="results/${name}"

  echo "START ${name} $(date '+%F %T')" | tee -a "$MASTER_LOG"
  local phase_args=()
  if [[ -n "$phase_prefix" ]]; then
    phase_args=(--phase_pred_prefix "$phase_prefix")
  fi
  "$PY" train_fpp_psp_adapter_unet.py \
    --base_cache_dir "$BASE_CACHE" \
    --phase_cache_dir "$PSP_CACHE" \
    "${phase_args[@]}" \
    --save_dir "$out" \
    --base_checkpoint "$INIT_RAW" \
    --cond_mode "$cond_mode" \
    --instr_channels 1-10 \
    --train_scope "$scope" \
    --epochs "$epochs" \
    --batch_size 4 \
    --eval_batch_size 4 \
    --num_workers 16 \
    --image_size 960 \
    --lr "$adapter_lr" \
    --adapter_lr "$adapter_lr" \
    --backbone_lr "$backbone_lr" \
    --weight_decay 1e-5 \
    --adapter_hidden 64 \
    --alpha 0.7 \
    --eval_every 5 \
    --eval_metrics_every 5 \
    --save_every 5 \
    --eval_initial \
    --train_minimal \
    --seed 42 \
    2>&1 | tee "results/remote_logs/${name}.log"
  echo "DONE ${name} $(date '+%F %T')" | tee -a "$MASTER_LOG"
}

# Main paper-facing route: diffusion-restored PSP + Hilbert/FTP/DWT/gradient
# instructions are injected by zero adapters, while only the bottleneck/decoder
# are softly fine-tuned from the raw-fringe official UNet.
run_depth \
  fpp960_e34c_pspquad_pred_instr_xy_adapter_decoder_bs4_e70 \
  phase_pred_instr_xy \
  "$PRED_PREFIX" \
  adapter_decoder \
  70 \
  5e-4 \
  2e-5

# Oracle upper bound with GT PSP quadrature. If this is not clearly better,
# the depth bottleneck is not phase restoration but the current depth decoder.
run_depth \
  fpp960_e35c_pspquad_gt_instr_xy_adapter_decoder_bs4_e50 \
  gt_psp_instr_xy \
  "" \
  adapter_decoder \
  50 \
  5e-4 \
  2e-5

# Stronger fallback: allow the whole pretrained UNet to adapt at a very low
# learning rate while keeping the adapter learning faster.
run_depth \
  fpp960_e36c_pspquad_pred_instr_xy_adapter_all_bs4_e60 \
  phase_pred_instr_xy \
  "$PRED_PREFIX" \
  all \
  60 \
  5e-4 \
  5e-6

echo "DONE E34-E36 PSP adapter-decoder suite $(date '+%F %T')" | tee -a "$MASTER_LOG"
