# Self-built Single-Frame 3D Dataset Format

This document describes the expected layout for the self-built real-capture
single-frame FPP dataset used by the scripts in this repository.

## Root Layout

```text
single_frame_3d_dataset_v1_upload_smalltest/
  normalization_stats.json
  train_manifest.csv
  val_manifest.csv
  test_manifest.csv
  samples/
    {domain}/
      obj{object_id:03d}/
        pose{pose_id:02d}/
          input_vertical_0120.bmp
          ablation_horizontal_0048.bmp
          labels.npz
          metadata.json
```

Additional teacher fields are stored separately:

```text
single_frame_3d_dataset_v1_teacher_extra/
  teacher_extra_manifest.csv
  calibration/
    {domain}_calibration.json
  samples/
    {domain}/
      obj{object_id:03d}/
        pose{pose_id:02d}/
          teacher_extra.npz
```

OOD 61-64 samples use the same `samples/{domain}/objXXX/poseYY` layout and are
passed through `--ood_root`.

## Manifest Columns

Each manifest row identifies one sample:

```text
sample_id,object_id,pose_id,split,domain,...
```

The loader resolves the primary input and label path as:

```text
samples/{domain}/obj{object_id:03d}/pose{pose_id:02d}/input_vertical_0120.bmp
samples/{domain}/obj{object_id:03d}/pose{pose_id:02d}/labels.npz
```

For OOD samples, `teacher_extra_manifest.csv` rows with split
`extra_unlisted` are used to locate the object/pose identifiers.

## Input Images

- `input_vertical_0120.bmp`: main legal single-frame input.
- `ablation_horizontal_0048.bmp`: available for controlled ablations, not used by the default legal single-frame setting.

The default paper setting uses only `input_vertical_0120.bmp` at test time.

## `labels.npz` Fields

Required fields:

| field | shape | meaning | test-time input? |
|---|---:|---|---|
| `depth_z` | H x W | target depth/Z value reconstructed from projective calibration | no |
| `valid_mask` | H x W | valid calibrated reconstruction pixels | no |
| `object_mask` | H x W | object ROI used as the main evaluation mask | no |
| `phase_y` | H x W | unwrapped teacher phase in y-related direction | no |
| `phase_x` | H x W | unwrapped teacher phase in x-related direction | no |
| `bc_y` | H x W | modulation/confidence-like weight for `phase_y` | no |
| `bc_x` | H x W | modulation/confidence-like weight for `phase_x` | no |

Optional but useful fields:

| field | shape | meaning |
|---|---:|---|
| `xyz_camera` | H x W x 3 | calibrated 3D camera/projective coordinates |
| `depth_valid_mask` | H x W | duplicate or refined depth-valid mask |
| `foreground_mask` | H x W | foreground segmentation mask |
| `base_support_mask` | H x W | stable support mask used in diagnostics |
| `ac_y`, `ac_x` | H x W | amplitude/intensity terms from PMP processing |

`depth_z` is normalized during training as:

```text
center = normalization_stats["depth_z"]["mean"]
scale = max(abs(p1 - center), abs(p99_5 - center), 1.0)
depth_norm = clip((depth_z - center) / scale, -1, 1)
```

Evaluation converts predictions back with:

```text
pred_depth_z = pred_norm * scale + center
```

## `teacher_extra.npz` Fields

These fields are labels, auxiliary targets, or diagnostics. They are not legal
test-time inputs.

| field | shape | meaning |
|---|---:|---|
| `wrapped_phase_y` | H x W | wrapped y phase |
| `wrapped_phase_x` | H x W | wrapped x phase |
| `phase_conf_y` | H x W | phase confidence for y |
| `phase_conf_x` | H x W | phase confidence for x |
| `fringe_order_y` | H x W | integer fringe order for y |
| `fringe_order_x` | H x W | integer fringe order for x |

Legal use:

- training-time auxiliary supervision
- loss weighting
- oracle or diagnostic evaluation

Illegal use:

- concatenating these maps into the test-time input tensor
- using them to select predictions at test time unless the same quantity is predicted from the single input frame

## Calibration JSON

The current `new0612` data uses projective calibration:

```text
camera_matrix_3x4
projector_projective_matrix_3x4
```

For strict point-cloud export, `export_best_anchor_pose_ply.py` can solve X/Y
from predicted `depth_z` and `camera_matrix_3x4` using 0-based pixel
coordinates. The saved `xyz_camera` field is used only for checking or GT
visualization.

## Evaluation Masks

Reports should include both:

- object-mask ROI: main paper metric
- valid-mask ROI: auxiliary metric over all valid reconstruction pixels

Do not mix these with old `wall_normal_height` metrics from earlier self-built
dataset versions.
