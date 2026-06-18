from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EPS = 1e-8


def corr(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]
    b = b[ok]
    if a.size < 10 or np.std(a) < EPS or np.std(b) < EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def fit_plane(z, mask=None):
    h, w = z.shape
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float64),
        np.linspace(-1.0, 1.0, w, dtype=np.float64),
        indexing="ij",
    )
    valid = np.isfinite(z)
    if mask is not None:
        valid &= mask > 0.5
    x = np.stack([xx[valid], yy[valid], np.ones(int(valid.sum()))], axis=1)
    y = z[valid].astype(np.float64)
    if y.size < 3:
        return np.zeros_like(z, dtype=np.float64)
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    full = coef[0] * xx + coef[1] * yy + coef[2]
    return full


def planar_residual(z, mask=None):
    return np.asarray(z, dtype=np.float64) - fit_plane(np.asarray(z, dtype=np.float64), mask=mask)


def make_xy(h, w, step):
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float64),
        np.linspace(-1.0, 1.0, w, dtype=np.float64),
        indexing="ij",
    )
    return xx[::step, ::step], yy[::step, ::step]


def feature_matrix(uph, x, y, degree=2):
    feats = [np.ones_like(uph), uph, x, y]
    if degree >= 2:
        feats.extend([uph * uph, uph * x, uph * y, x * x, x * y, y * y])
    if degree >= 3:
        feats.extend([uph ** 3, (uph * uph) * x, (uph * uph) * y, uph * x * y, x ** 3, y ** 3])
    return np.stack([f.reshape(-1) for f in feats], axis=1)


def solve_ridge(x, y, alpha=1e-4):
    reg = np.eye(x.shape[1], dtype=np.float64) * float(alpha)
    reg[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + reg, x.T @ y)


def rmse(a, b):
    e = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(e * e)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def load_split(base_dir, phase_dir, split):
    depth01 = np.load(base_dir / f"depth01_{split}_float16.npy", mmap_mode="r")[:, 0]
    depthmm = np.load(base_dir / f"depth_mm_{split}_float32.npy", mmap_mode="r")[:, 0]
    mask = np.load(base_dir / f"mask_{split}_uint8.npy", mmap_mode="r")[:, 0]
    target = np.load(phase_dir / f"phase_target_{split}_float16.npy", mmap_mode="r")
    pminmax = np.load(phase_dir / f"phase_minmax_{split}_float32.npy", mmap_mode="r")
    return depth01, depthmm, mask, target, pminmax


def split_rows(base_dir, phase_dir, split, step):
    depth01, depthmm, mask, target, pminmax = load_split(base_dir, phase_dir, split)
    rows = []
    for i in range(depth01.shape[0]):
        m = mask[i, ::step, ::step] > 0.5
        d01 = np.asarray(depth01[i, ::step, ::step], dtype=np.float64)
        dmm = np.asarray(depthmm[i, ::step, ::step], dtype=np.float64)
        uph01 = np.asarray(target[i, 2, ::step, ::step], dtype=np.float64)
        lo, hi = [float(v) for v in pminmax[i]]
        raw_uph = uph01 * (hi - lo) + lo
        raw_res = planar_residual(raw_uph, mask=m)
        dmm_res = planar_residual(dmm, mask=m)
        d01_res = planar_residual(d01, mask=m)
        rows.append(
            {
                "split": split,
                "sample": i,
                "mask_pixels": int(m.sum()),
                "uph_range": float(hi - lo),
                "depth_range_mm": float(np.max(dmm[m]) - np.min(dmm[m])) if np.any(m) else float("nan"),
                "corr_uph01_depth01": corr(uph01[m], d01[m]),
                "corr_rawuph_depthmm": corr(raw_uph[m], dmm[m]),
                "corr_rawuph_res_depthmm_res": corr(raw_res[m], dmm_res[m]),
                "corr_uph01_res_depth01_res": corr(planar_residual(uph01, mask=m)[m], d01_res[m]),
            }
        )
    return rows


def aggregate(rows):
    out = {"n": len(rows)}
    keys = [k for k in rows[0] if k.startswith("corr_") or k.endswith("_range") or k.endswith("_range_mm")]
    for key in keys:
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        out[key] = {
            "mean": float(np.nanmean(vals)),
            "median": float(np.nanmedian(vals)),
            "std": float(np.nanstd(vals)),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
        }
    return out


def collect_pixels(base_dir, phase_dir, split, step, max_pixels, degree):
    depth01, depthmm, mask, target, pminmax = load_split(base_dir, phase_dir, split)
    xs = []
    ys = []
    rng = np.random.default_rng(1234)
    for i in range(depthmm.shape[0]):
        m = mask[i, ::step, ::step] > 0.5
        if not np.any(m):
            continue
        h, w = m.shape
        xg, yg = make_xy(depthmm.shape[1], depthmm.shape[2], step)
        uph01 = np.asarray(target[i, 2, ::step, ::step], dtype=np.float64)
        lo, hi = [float(v) for v in pminmax[i]]
        raw_uph = uph01 * (hi - lo) + lo
        feat = feature_matrix(raw_uph[m], xg[m], yg[m], degree=degree)
        y = np.asarray(depthmm[i, ::step, ::step], dtype=np.float64)[m].reshape(-1)
        xs.append(feat)
        ys.append(y)
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if max_pixels > 0 and x.shape[0] > max_pixels:
        idx = rng.choice(x.shape[0], size=max_pixels, replace=False)
        x = x[idx]
        y = y[idx]
    return x, y


def evaluate_global_model(base_dir, phase_dir, coef, split, step, degree):
    x, y = collect_pixels(base_dir, phase_dir, split, step=step, max_pixels=0, degree=degree)
    pred = x @ coef
    return {"rmse_mm": rmse(pred, y), "mae_mm": mae(pred, y), "n_pixels": int(y.size)}


def per_sample_affine_upper_bound(base_dir, phase_dir, split, step):
    depth01, depthmm, mask, target, pminmax = load_split(base_dir, phase_dir, split)
    rmses = []
    maes = []
    for i in range(depthmm.shape[0]):
        m = mask[i, ::step, ::step] > 0.5
        if m.sum() < 10:
            continue
        uph01 = np.asarray(target[i, 2, ::step, ::step], dtype=np.float64)
        lo, hi = [float(v) for v in pminmax[i]]
        raw_uph = uph01 * (hi - lo) + lo
        y = np.asarray(depthmm[i, ::step, ::step], dtype=np.float64)
        x = np.stack([raw_uph[m], np.ones(int(m.sum()))], axis=1)
        coef = np.linalg.lstsq(x, y[m], rcond=None)[0]
        pred = coef[0] * raw_uph[m] + coef[1]
        rmses.append(rmse(pred, y[m]))
        maes.append(mae(pred, y[m]))
    return {
        "rmse_mm_mean": float(np.mean(rmses)),
        "rmse_mm_median": float(np.median(rmses)),
        "mae_mm_mean": float(np.mean(maes)),
        "n": len(rmses),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_phase_cache_960")
    parser.add_argument("--save_dir", default="/root/autodl-tmp/diffusion_fpp_v5/results/phase_depth_mapping_diag")
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--ridge_alpha", type=float, default=1e-4)
    parser.add_argument("--max_train_pixels", type=int, default=200000)
    args = parser.parse_args()

    base_dir = Path(args.base_cache_dir)
    phase_dir = Path(args.phase_cache_dir)
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary = {"args": vars(args), "splits": {}}
    for split in ("train", "val", "test"):
        rows = split_rows(base_dir, phase_dir, split, step=args.step)
        all_rows.extend(rows)
        summary["splits"][split] = aggregate(rows)

    with open(out_dir / "per_sample_phase_depth_corr.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    train_x, train_y = collect_pixels(
        base_dir,
        phase_dir,
        "train",
        step=args.step,
        max_pixels=args.max_train_pixels,
        degree=args.degree,
    )
    coef = solve_ridge(train_x, train_y, alpha=args.ridge_alpha)
    summary["global_poly_uph_xy_to_depth"] = {
        "degree": args.degree,
        "ridge_alpha": args.ridge_alpha,
        "coef": [float(v) for v in coef],
        "train_fit_pixels": int(train_y.size),
        "train_sampled": {"rmse_mm": rmse(train_x @ coef, train_y), "mae_mm": mae(train_x @ coef, train_y)},
        "eval": {
            split: evaluate_global_model(base_dir, phase_dir, coef, split, step=args.step, degree=args.degree)
            for split in ("train", "val", "test")
        },
    }
    summary["per_sample_affine_rawuph_to_depth_upper_bound"] = {
        split: per_sample_affine_upper_bound(base_dir, phase_dir, split, step=args.step)
        for split in ("train", "val", "test")
    }

    with open(out_dir / "phase_depth_mapping_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
