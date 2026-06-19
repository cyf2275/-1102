# 2026-06-19 Self-built Single-frame3D Results

This note records the paper-facing result package for the self-built
single-frame FPP dataset. It is an experiment log and reproducibility map; it
does not include data, checkpoints, generated images, PLY files, or Word
reports.

## Scope

Dataset target:

```text
input_vertical_0120.bmp -> depth_z
```

Splits:

```text
train = 352
val = 80
test = 31
OOD 61-64 = 12
```

Main metric:

```text
object-mask RMSE in mm
```

Auxiliary metric:

```text
valid-mask RMSE in mm
```

Legal test-time input is only the single frame and deterministic features
derived from that single frame. True phase, confidence, order, and depth are
labels or diagnostics only.

## Formal Direct Baselines

Formal direct backbone settings:

```text
seeds = 0, 1, 2
epochs = 80
checkpoint = best validation object-mask RMSE
resolution = 480 x 640
batch size = 2
gradient accumulation = 2
eval every = 5 epochs
```

Results:

| Method | Test object RMSE | OOD 61-64 object RMSE |
| --- | ---: | ---: |
| Attention UNet direct | 1.7107 +/- 0.0697 | 1.8125 +/- 0.1336 |
| UNet++ direct | 1.7775 +/- 0.0558 | 1.9657 +/- 0.2105 |

## Formal Ours With Attention UNet Base

The formal ours run uses the 80-epoch Attention UNet best-val checkpoints as
the direct depth base, then evaluates phase posterior evidence and
RCPC/selector variants.

| Method | Test object RMSE | OOD 61-64 object RMSE |
| --- | ---: | ---: |
| Base+x mean anchor | 1.5107 +/- 0.0336 | 1.5087 +/- 0.0428 |
| Rule final | 1.4917 +/- 0.0343 | 1.4389 +/- 0.0480 |
| MLP final | 1.4536 +/- 0.0327 | 1.4973 +/- 0.0703 |

Interpretation:

- MLP final is best on the ordinary test split.
- Rule final is most stable on the OOD 61-64 split.
- The diffusion candidate alone is not the final claim. The paper claim should
  be that phase posterior evidence combined with anchor and RCPC/selector is
  more reliable than direct depth regression or direct depth residual diffusion.

## Per-sample Paired Significance

Statistics are computed after averaging each sample over seeds 0/1/2.

| Split | Comparison | Mean improvement | Relative | Wins/n | Sign-test p |
| --- | --- | ---: | ---: | ---: | ---: |
| test | Attention UNet -> MLP final | 0.2572 mm | 15.0% | 30/31 | 2.98e-8 |
| test | UNet++ -> MLP final | 0.3240 mm | 18.2% | 27/31 | 3.40e-5 |
| OOD 61-64 | Attention UNet -> Rule final | 0.3736 mm | 20.6% | 10/12 | 0.0386 |
| OOD 61-64 | UNet++ -> Rule final | 0.5269 mm | 26.8% | 12/12 | 4.88e-4 |

Local result files generated outside this repository:

```text
cloud_results/A_20260619_selfbuilt_dataset_paper_experiments/paper_per_sample_significance.csv
cloud_results/A_20260619_selfbuilt_dataset_paper_experiments/paper_per_sample_mean_rmse.csv
```

## Qualitative Outputs

The local pulled result package contains:

```text
paper_summary_assets/formal_test_ood_comparison.png
paper_summary_assets/selector_ablation_test_ood.png
paper_summary_assets/per_sample_paired_improvement.png
paper_summary_assets/visual_comparison_contact_sheet.png
paper_summary_assets/pointcloud_obj061_pose02_direct_vs_ours_camera_projective.png
paper_summary_assets/pointcloud_obj061_pose02_direct_vs_ours_visual_z10.png
```

PLY outputs use the same sample:

```text
new0612_obj061_pose02
```

Strict camera-projective PLY files use the dataset `camera_matrix_3x4` and
0-based pixels to solve X/Y from predicted `depth_z`. Visual z10 PLY files are
for shape inspection only and must not be used as metrics.

## Source Code Map

| Purpose | Source |
| --- | --- |
| Direct baseline training/evaluation | `diffusion_fpp_v5/train_single_frame3d_backbone_baselines.py` |
| Phase posterior and RCPC workflow | `diffusion_fpp_v5/train_single_frame3d_xphase_diffusion_rcpc.py` |
| Reliability selector | `diffusion_fpp_v5/train_refined_xphase_reliability_selector.py` |
| 2D reconstruction panels | `diffusion_fpp_v5/make_best_anchor_reconstruction_visuals.py` |
| Ours/anchor PLY export | `diffusion_fpp_v5/export_best_anchor_pose_ply.py` |
| Direct baseline PLY export | `diffusion_fpp_v5/export_single_frame3d_direct_baseline_ply.py` |
| Per-sample paired statistics | `tools/add_selfbuilt_dataset_supplementary_analysis_20260619.py` |
| Markdown/Word report generation | `tools/build_selfbuilt_dataset_report_20260619.py` |

## Paper Wording Boundary

Recommended claim:

```text
Under the same single-frame input constraint, phase posterior evidence combined
with RCPC/selector improves both ordinary test and material-OOD reconstruction
over strong direct depth backbones.
```

Avoid claiming:

```text
Direct depth residual diffusion consistently improves depth.
Absolute SOTA over all structured-light methods.
```

