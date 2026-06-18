#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5

PY=/root/miniconda3/bin/python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
DATA=/root/autodl-tmp/single_frame_3d_dataset_v1_upload_smalltest
TEACHER=/root/autodl-tmp/single_frame_3d_dataset_v1_teacher_extra
OOD=/root/autodl-tmp/single_frame_3d_dataset_v1_upload_smalltest_ood61_64
OUT=/root/autodl-tmp/diffusion_fpp_v5/results/A_20260618_single_frame3d_baseline_comparison_quick1seed
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

ARCHES=(unet resunet attention_unet unetpp mps_xnet pix2pix)

echo "===== GPU before quick baseline $(date '+%F %T') =====" | tee "$LOGDIR/master.log"
nvidia-smi | tee -a "$LOGDIR/master.log"
pgrep -af 'train_single_frame3d|train_my|diffusion|python' | tee -a "$LOGDIR/master.log" || true

for arch in "${ARCHES[@]}"; do
  echo "===== smoke $arch $(date '+%F %T') =====" | tee -a "$LOGDIR/master.log"
  "$PY" -u train_single_frame3d_backbone_baselines.py \
    --data_root "$DATA" \
    --teacher_extra_root "$TEACHER" \
    --ood_root "$OOD" \
    --save_dir "$OUT/$arch" \
    --arch "$arch" \
    --seed 0 \
    --epochs 40 \
    --batch_size 2 \
    --accum_steps 2 \
    --eval_batch_size 2 \
    --num_workers 8 \
    --eval_every 5 \
    --smoke_only \
    2>&1 | tee "$LOGDIR/${arch}_smoke.log"
done

for arch in "${ARCHES[@]}"; do
  echo "===== train $arch $(date '+%F %T') =====" | tee -a "$LOGDIR/master.log"
  "$PY" -u train_single_frame3d_backbone_baselines.py \
    --data_root "$DATA" \
    --teacher_extra_root "$TEACHER" \
    --ood_root "$OOD" \
    --save_dir "$OUT/$arch" \
    --arch "$arch" \
    --seed 0 \
    --epochs 40 \
    --batch_size 2 \
    --accum_steps 2 \
    --eval_batch_size 2 \
    --num_workers 8 \
    --eval_every 5 \
    2>&1 | tee "$LOGDIR/${arch}.log"
done

"$PY" -u summarize_single_frame3d_baselines.py \
  --result_dir "$OUT" \
  --ours_fullchain_json /root/autodl-tmp/diffusion_fpp_v5/results/A_20260618_paper_ready_anchor_ablation/anchor_ablation_fullchain_summary.json \
  --ours_fixed_json /root/autodl-tmp/diffusion_fpp_v5/results/A_20260618_paper_ready_anchor_ablation/anchor_ablation_fixed_posterior_summary.json \
  2>&1 | tee "$LOGDIR/summary.log"

echo "===== done quick baseline $(date '+%F %T') =====" | tee -a "$LOGDIR/master.log"
