# Checkpoint Backup 2026-06-19

Checkpoints are intentionally not tracked by Git. This note records the minimal
local checkpoint backup for the self-built single-frame3D paper experiments.

## Local Backup

The checkpoint backup was downloaded outside the repository because the
repository and `H:` drive should not store large model weights.

Local machine paths:

```text
F:\selfbuilt_dataset_checkpoints_20260619\selfbuilt_paper_checkpoints_minimal_20260619.tar.gz
F:\selfbuilt_dataset_checkpoints_20260619\extracted
F:\selfbuilt_dataset_checkpoints_20260619\CHECKPOINT_BACKUP_README.md
```

Archive checksum:

```text
sha256 = 38f0317cc15656ce9fc0a03661fb034c08b2d3cea03ea0e6df22cbe132dee7a7
size = 1,272,205,303 bytes
```

Extracted size:

```text
20 files
about 1.30 GB
```

The internal `SHA256SUMS.txt` verified successfully for all checkpoint and
summary entries.

## Included Checkpoints

Formal direct backbones:

```text
A_20260619_formal_strong_backbone_direct_seed012/attention_unet_seed0/checkpoints/best.pt
A_20260619_formal_strong_backbone_direct_seed012/attention_unet_seed1/checkpoints/best.pt
A_20260619_formal_strong_backbone_direct_seed012/attention_unet_seed2/checkpoints/best.pt
A_20260619_formal_strong_backbone_direct_seed012/unetpp_seed0/checkpoints/best.pt
A_20260619_formal_strong_backbone_direct_seed012/unetpp_seed1/checkpoints/best.pt
A_20260619_formal_strong_backbone_direct_seed012/unetpp_seed2/checkpoints/best.pt
```

Formal Attention UNet + ours selectors:

```text
A_20260619_formal_attention_unet_ours_selector_seed012/seed0/reliability_selector_seed0/reliability_selector.pt
A_20260619_formal_attention_unet_ours_selector_seed012/seed1/reliability_selector_seed1/reliability_selector.pt
A_20260619_formal_attention_unet_ours_selector_seed012/seed2/reliability_selector_seed2/reliability_selector.pt
```

Phase posterior and refined-depth components:

```text
A_20260617_single_frame3d_xphase_diffusion_rcpc/x_phase_diffusion/checkpoints/best.pt
A_20260618_refined_xphase_depth/refined_xphase_depth/checkpoints/best.pt
A_20260618_refined_xphase_depth/fullchain_seed1/xphase_diffusion_rcpc/x_phase_diffusion/checkpoints/best.pt
A_20260618_refined_xphase_depth/fullchain_seed2/xphase_diffusion_rcpc/x_phase_diffusion/checkpoints/best.pt
A_20260618_refined_xphase_depth/fullchain_seed1/refined_xphase_depth/refined_xphase_depth/checkpoints/best.pt
A_20260618_refined_xphase_depth/fullchain_seed2/refined_xphase_depth/refined_xphase_depth/checkpoints/best.pt
```

Small metadata included:

```text
A_20260619_formal_strong_backbone_direct_seed012/baseline_comparison_quick1seed_summary.csv
A_20260619_formal_attention_unet_ours_selector_seed012/formal_attention_unet_ours_selector_summary.json
A_20260619_formal_attention_unet_ours_selector_seed012/formal_attention_unet_ours_selector_report.md
```

## Not Included

The backup does not include:

- Real captured data.
- 20260614 early raw/raw_xy/raw_single_phys/teacher_aux checkpoint sweeps.
- Smoke/debug/1-epoch checkpoints.
- Quick-screening ResUNet, Pix2Pix, MPS-XNet, or ordinary UNet checkpoints.
- Intermediate epoch checkpoints such as `epoch_010.pt`.
- Generated result folders, images, PLY files, or Word reports.

Those outputs are stored in local `cloud_results` packages, not in Git.

## What Is Needed To Re-run Inference

To regenerate figures or PLY files, keep:

```text
1. The checkpoint backup above.
2. This repository code.
3. The self-built dataset and teacher_extra/calibration folders.
```

Expected local dataset roots from the experiment thread:

```text
I:\cyf\single_frame_3d_dataset_v1_upload_smalltest
I:\cyf\single_frame_3d_dataset_v1_teacher_extra
```

The OOD 61-64 set must also be available if OOD metrics or PLY exports are
recomputed.

