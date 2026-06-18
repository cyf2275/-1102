#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5

PY=/root/miniconda3/bin/python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

DATA=/root/autodl-tmp/single_frame_3d_dataset_v1_upload_smalltest
TEACHER=/root/autodl-tmp/single_frame_3d_dataset_v1_teacher_extra
OOD=/root/autodl-tmp/single_frame_3d_dataset_v1_upload_smalltest_ood61_64
OUT=/root/autodl-tmp/diffusion_fpp_v5/results/A_20260619_formal_strong_backbone_direct_seed012
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

ARCHES=(attention_unet unetpp)
SEEDS=(0 1 2)

echo "===== formal strong backbone direct start $(date '+%F %T') =====" | tee "$LOGDIR/master.log"
nvidia-smi | tee -a "$LOGDIR/master.log"
pgrep -af 'train_single_frame3d|train_my|diffusion|python' | tee -a "$LOGDIR/master.log" || true

for arch in "${ARCHES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    RUN="$OUT/${arch}_seed${seed}"
    if [ -f "$RUN/evaluation/summary.json" ]; then
      echo "===== skip existing $arch seed$seed =====" | tee -a "$LOGDIR/master.log"
      continue
    fi
    echo "===== train $arch seed$seed 80ep $(date '+%F %T') =====" | tee -a "$LOGDIR/master.log"
    "$PY" -u train_single_frame3d_backbone_baselines.py \
      --data_root "$DATA" \
      --teacher_extra_root "$TEACHER" \
      --ood_root "$OOD" \
      --save_dir "$RUN" \
      --arch "$arch" \
      --seed "$seed" \
      --epochs 80 \
      --batch_size 2 \
      --accum_steps 2 \
      --eval_batch_size 2 \
      --num_workers 8 \
      --eval_every 5 \
      2>&1 | tee "$LOGDIR/${arch}_seed${seed}.log"
  done
done

"$PY" -u summarize_single_frame3d_baselines.py \
  --result_dir "$OUT" \
  2>&1 | tee "$LOGDIR/summary.log"

echo "===== formal strong backbone direct done $(date '+%F %T') =====" | tee -a "$LOGDIR/master.log"
