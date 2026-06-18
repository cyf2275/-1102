# Self-built Dataset Comparison Experiment Plan

## Goal

Evaluate whether phase posterior evidence plus RCPC/selector improves
single-frame FPP reconstruction under the same legal input constraint:

```text
input_vertical_0120.bmp -> depth_z
```

The paper claim should not be absolute SOTA. The intended claim is that direct
depth residual diffusion is unstable on this dataset, while structured x-phase
posterior evidence plus constrained fusion is a more reliable route.

## Phase 1: Quick 1-seed Backbone Screening

Purpose: identify competitive direct-depth backbones and avoid comparing only
against a weak UNet.

Fixed setup:

```text
seed = 0
epochs = 40
image size = 480 x 640
batch size = 4
eval batch size = 2
eval every = 5 epochs
checkpoint = best validation object-mask RMSE
main metric = object-mask RMSE on depth_z
auxiliary metric = valid-mask RMSE
```

Methods:

```text
UNet direct
ResUNet direct
Attention UNet direct
UNet++ direct
MPS-XNet-style multitask direct
Pix2Pix-style direct
Ours base+x mean anchor
Ours phase posterior diffusion + RCPC/selector
```

Example run:

```bash
python diffusion_fpp_v5/train_single_frame3d_backbone_baselines.py \
  --data_root /root/autodl-tmp/single_frame_3d_dataset_v1_upload_smalltest \
  --teacher_extra_root /root/autodl-tmp/single_frame_3d_dataset_v1_teacher_extra \
  --ood_root /root/autodl-tmp/single_frame_3d_dataset_v1_upload_smalltest_ood61_64 \
  --save_dir /root/autodl-tmp/diffusion_fpp_v5/results/A_20260618_single_frame3d_baseline_comparison_quick1seed/unet \
  --arch unet \
  --seed 0 \
  --epochs 40 \
  --batch_size 4 \
  --eval_batch_size 2 \
  --num_workers 8
```

Summarize:

```bash
python diffusion_fpp_v5/summarize_single_frame3d_baselines.py \
  --result_dir /root/autodl-tmp/diffusion_fpp_v5/results/A_20260618_single_frame3d_baseline_comparison_quick1seed \
  --ours_fullchain_json /root/autodl-tmp/diffusion_fpp_v5/results/A_20260618_paper_ready_anchor_ablation/anchor_ablation_fullchain_summary.json \
  --ours_fixed_json /root/autodl-tmp/diffusion_fpp_v5/results/A_20260618_paper_ready_anchor_ablation/anchor_ablation_fixed_posterior_summary.json
```

Phase 1 outputs:

```text
baseline_comparison_quick1seed_summary.csv
baseline_comparison_quick1seed_summary.json
baseline_comparison_quick1seed_report.md
baseline_comparison_quick1seed_rmse.png
*/evaluation/summary.json
*/evaluation/per_sample_metrics.csv
*/evaluation/ood_per_sample_metrics.csv
*/visualizations/
```

Important: Phase 1 is quick screening only. Do not use it as the final paper
table if validation is still improving.

## Phase 2: Strong Backbone + Ours

Purpose: answer whether the method only wins because the original UNet is weak.

Choose the strongest one or two direct backbones from Phase 1, typically:

```text
UNet++
Attention UNet
```

Then evaluate:

```text
strong backbone direct depth
strong backbone + x-phase evidence
strong backbone + x-phase posterior diffusion
strong backbone + RCPC/selector
```

If a strong direct backbone outperforms the current full method, the paper
method should be updated to use the stronger backbone as the depth/evidence
backbone instead of defending the older UNet version.

## Phase 3: Paper-level Results

Only methods that are meaningful after Phase 1/2 should be promoted to final
paper experiments.

Final settings:

```text
seeds = 0, 1, 2
epochs = 80 by default, extend to 120 if validation is still improving
early stop = no validation object RMSE improvement for 20 epochs
report = mean +/- std
splits = ordinary test and OOD 61-64
```

Final table should include:

```text
strongest direct backbone
traditional FTP/WFT/Hilbert/Wavelet proxy
MPS-XNet-style or phase-first baseline
ours without diffusion
ours without RCPC
ours full phase posterior diffusion + RCPC/selector
```

Qualitative outputs:

```text
2D reconstruction panels
absolute error maps
posterior uncertainty maps
MLP/selector probability maps
strict camera-projective PLY point clouds
visual-aligned PLY point clouds only for shape inspection
```

## Decision Rules

- If direct UNet++ or Attention UNet beats the current full method, update the
  method to use the stronger backbone and re-run ours.
- If phase posterior diffusion improves x-phase evidence but hurts depth
  without RCPC, report diffusion as evidence posterior rather than direct depth
  residual correction.
- If RCPC/selector improves OOD but slightly hurts ordinary test, analyze
  acceptance probability and report the tradeoff explicitly.
- Do not compare `depth_z` RMSE directly with old `wall_normal_height` RMSE or
  FPP-ML-Bench metrics without target and dataset qualification.
