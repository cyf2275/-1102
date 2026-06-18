# Single-Frame 3D FPP Phase Posterior Experiments

This repository contains code for single-frame fringe projection profilometry
experiments on a self-built real-capture dataset. The current research focus is
not to claim absolute SOTA, but to evaluate whether phase posterior evidence and
RCPC-style constrained fusion are more reliable than direct depth residual
diffusion under the same single-frame input constraint.

## What Is Included

- Manifest-based dataset loaders for the self-built `single_frame_3d` dataset.
- Direct depth baselines for UNet, ResUNet, Attention UNet, UNet++, MPS-XNet-style, and Pix2Pix-style models.
- Phase-evidence and diffusion-posterior experiment scripts.
- Visualization and point-cloud export utilities for qualitative inspection.
- Documentation for dataset format and experiment protocol.

The repository intentionally does not include real captured data, checkpoints,
logs, generated images, Word reports, or large experiment results.

## Dataset Boundary

Legal test-time input is restricted to the single captured frame:

```text
input_vertical_0120.bmp
```

Allowed derived inputs are deterministic single-frame features computed from
that image, such as gradients, Hilbert/DWT/FTP-like proxy features, and x/y
coordinate maps. Teacher phase, PMP phase, confidence maps, fringe order, and
ground-truth depth are labels or diagnostics only. They must not be concatenated
into the test-time input tensor.

See [docs/DATASET_FORMAT_SINGLE_FRAME3D.md](docs/DATASET_FORMAT_SINGLE_FRAME3D.md)
for the full dataset layout and field definitions.

## Main Experiment Entrypoints

Backbone quick screening on the self-built dataset:

```bash
python diffusion_fpp_v5/train_single_frame3d_backbone_baselines.py \
  --data_root /path/to/single_frame_3d_dataset_v1_upload_smalltest \
  --teacher_extra_root /path/to/single_frame_3d_dataset_v1_teacher_extra \
  --ood_root /path/to/single_frame_3d_dataset_v1_upload_smalltest_ood61_64 \
  --save_dir results/A_20260618_single_frame3d_baseline_comparison_quick1seed/unet \
  --arch unet \
  --seed 0 \
  --epochs 40 \
  --batch_size 4 \
  --eval_batch_size 2 \
  --num_workers 8
```

Summarize quick baseline runs:

```bash
python diffusion_fpp_v5/summarize_single_frame3d_baselines.py \
  --result_dir results/A_20260618_single_frame3d_baseline_comparison_quick1seed
```

Existing phase posterior / RCPC scripts:

```text
diffusion_fpp_v5/train_single_frame3d_physics_diffusion.py
diffusion_fpp_v5/train_single_frame3d_xphase_diffusion_rcpc.py
diffusion_fpp_v5/train_single_frame3d_refined_xphase_depth.py
diffusion_fpp_v5/train_refined_xphase_reliability_selector.py
diffusion_fpp_v5/make_best_anchor_reconstruction_visuals.py
diffusion_fpp_v5/export_best_anchor_pose_ply.py
```

## Recommended Paper Protocol

Use 40 epochs and one seed only for quick screening. A method should be used in
the final paper table only after validation has converged and at least three
seeds have been evaluated.

Recommended final comparison groups:

- Direct backbones: UNet, ResUNet, Attention UNet, UNet++, and the strongest available transformer-like backbone if implemented.
- FPP-related baselines: MPS-XNet-style multitask and traditional FTP/WFT/Hilbert/Wavelet proxy baselines.
- Ablations: direct depth, x-phase evidence, phase posterior diffusion, RCPC/selector without and with diffusion.
- Ours full: phase posterior diffusion plus RCPC/selector under legal single-frame input.

See [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) for details.

## Notes

- The main target is `depth_z`, not the older `wall_normal_height` target.
- Results on this dataset must not be numerically mixed with older
  `my_fpp_dataset_v1` or FPP-ML-Bench results without a clear target/metric
  distinction.
- OOD 61-64 samples are different-material samples and should be reported
  separately from the ordinary test split.
