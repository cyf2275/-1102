# diffusion_fpp_v5

Fringe-only physics-conditioned full-height diffusion for single-frame FPP.

This version does not use speckle or double-fringe data. All condition channels
are derived from the same `X_*_fringe.npy` frame:

- raw fringe
- Hilbert/FFT phase proxy as `sin(phase), cos(phase)`
- analytic amplitude confidence
- local fringe gradient
- normalized `x,y` coordinates
- optional coarse height predicted from the same condition tensor

## Cloud Run

```bash
cd /root/diffusion_fpp_v5
/root/miniconda3/bin/python precompute_features.py
/root/miniconda3/bin/python train_v5.py --epochs 500 --coarse_epochs 50 --batch_size 4 --num_workers 8 --require_cache
/root/miniconda3/bin/python evaluate_v5.py
```

For a quick smoke test:

```bash
cd /root/diffusion_fpp_v5
/root/miniconda3/bin/python train_v5.py --epochs 1 --coarse_epochs 1 --batch_size 1 --eval_samples 1 --ddim_steps 4 --ensemble 1 --save_dir /root/diffusion_fpp_v5/results/smoke
```
