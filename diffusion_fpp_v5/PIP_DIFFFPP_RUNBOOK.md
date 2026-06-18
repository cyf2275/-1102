# PIP-DiffFPP Phase-1 Runbook

This runbook implements the first executable stage of the final PIP-DiffFPP plan
on the existing Nguyen/Wang-style dataset.

## Files Added

- `physics_features_pip.py`: frequency-normalized physics instructions.
- `precompute_features_pip.py`: cache PIP features to SSD.
- `data/dataset_pip.py`: PIP dataset wrapper with phase/confidence/edge maps.
- `models/pip.py`: point-wise phase projection head and low-pass CoarseNet.
- `diffusion_pip.py`: PIP-lite loss and posterior guidance.
- `train_phase_projection_pip.py`: train `P_learned(D,x,y)->sin/cos(phi)` and run scrambled-depth test.
- `train_pip_lite.py`: train PIP-lite full x0 diffusion.
- `sweep_posterior_pip.py`: validation-selected posterior guidance sweep.
- `train_coarse_lowpass_pip.py`: train low-pass CoarseNet and uncertainty.
- `train_pip_target_ablation.py`: full x0 vs normalized residual vs hybrid target ablation.

## Cloud Setup

Use one RTX 4090 first. Recommended:

```bash
cd /root/diffusion_fpp_v5
```

Precompute PIP features:

```bash
/root/miniconda3/bin/python precompute_features_pip.py \
  --data_dir /root/diffusion_fpp_v5/data \
  --cache_dir /root/autodl-tmp/diffusion_fpp_pip_cache
```

## Phase Projection Head

Train the point-wise projection head:

```bash
/root/miniconda3/bin/python train_phase_projection_pip.py \
  --data_dir /root/diffusion_fpp_v5/data \
  --cache_dir /root/autodl-tmp/diffusion_fpp_pip_cache \
  --epochs 120 \
  --batch_size 4 \
  --num_workers 8 \
  --eval_every 10 \
  --require_cache \
  --save_dir /root/diffusion_fpp_v5/results/pip_phase_projection
```

Acceptance:

- `scrambled_depth_summary.json` should show `zero_over_normal` and
  `shuffled_over_normal` clearly above 1.0.
- If not, `P_learned` is not sufficiently depth-dependent.

## PIP-lite

Train PIP-lite with frozen phase projection:

```bash
/root/miniconda3/bin/python train_pip_lite.py \
  --data_dir /root/diffusion_fpp_v5/data \
  --cache_dir /root/autodl-tmp/diffusion_fpp_pip_cache \
  --phase_head /root/diffusion_fpp_v5/results/pip_phase_projection/checkpoints/best.pt \
  --epochs 500 \
  --batch_size 4 \
  --num_workers 8 \
  --base_channels 48 \
  --timesteps 200 \
  --ddim_steps 50 \
  --ensemble 3 \
  --eval_every 25 \
  --require_cache \
  --save_dir /root/diffusion_fpp_v5/results/pip_lite
```

Acceptance:

- Must beat v3.5 RMSE `7.1921mm` to justify continuing.
- Strong target: `<7.0mm`; ideal target: `<=6.7mm`.

## Posterior Guidance Sweep

Run after PIP-lite and phase head are trained:

```bash
/root/miniconda3/bin/python sweep_posterior_pip.py \
  --data_dir /root/diffusion_fpp_v5/data \
  --cache_dir /root/autodl-tmp/diffusion_fpp_pip_cache \
  --checkpoint /root/diffusion_fpp_v5/results/pip_lite/checkpoints/best.pt \
  --phase_head /root/diffusion_fpp_v5/results/pip_phase_projection/checkpoints/best.pt \
  --weights 0,0.005,0.01,0.02,0.05 \
  --starts 0.5,0.7,0.85 \
  --clips 0.03,0.05,0.1 \
  --ddim_steps 20,50 \
  --ensemble 1 \
  --require_cache \
  --out_dir /root/diffusion_fpp_v5/results/pip_posterior_sweep
```

Rules:

- Select only by val RMSE.
- Test is run once with `best_posterior_config.json`.
- If `weight=0` wins, posterior guidance is reported as a failed/neutral
  ablation, not forced into the main model.

## Low-pass CoarseNet

Train low-pass CoarseNet separately:

```bash
/root/miniconda3/bin/python train_coarse_lowpass_pip.py \
  --data_dir /root/diffusion_fpp_v5/data \
  --cache_dir /root/autodl-tmp/diffusion_fpp_pip_cache \
  --epochs 150 \
  --batch_size 4 \
  --num_workers 8 \
  --lowpass_factor 8 \
  --require_cache \
  --save_dir /root/diffusion_fpp_v5/results/pip_coarse_lowpass
```

Acceptance:

- Low-pass RMSE should be reasonable.
- `uncertainty_error_corr` should be positive; otherwise uncertainty is not
  tracking low-frequency error.

## Target Ablation

Run all three targets after PIP features and the phase head are available:

```bash
for mode in full residual hybrid; do
  /root/miniconda3/bin/python train_pip_target_ablation.py \
    --data_dir /root/diffusion_fpp_v5/data \
    --cache_dir /root/autodl-tmp/diffusion_fpp_pip_cache \
    --phase_head /root/diffusion_fpp_v5/results/pip_phase_projection/checkpoints/best.pt \
    --target_mode ${mode} \
    --epochs 300 \
    --batch_size 4 \
    --num_workers 8 \
    --base_channels 48 \
    --timesteps 200 \
    --ddim_steps 50 \
    --ensemble 3 \
    --eval_every 25 \
    --require_cache \
    --save_dir /root/diffusion_fpp_v5/results/pip_target_ablation
done
```

Residual mode uses:

```text
residual_scale = p99(|D_gt - D_low|)
target = clip((D_gt - D_low) / residual_scale, -1, 1)
```

Do not compare unnormalized residual diffusion against full x0 diffusion.

## First Decision Gate

Continue to the full PIP-DiffFPP architecture only if at least one is true:

- PIP-lite beats v3.5 clearly.
- Posterior guidance improves val and test over no-guidance.
- PIP-lite improves hard samples without sacrificing normal samples.
