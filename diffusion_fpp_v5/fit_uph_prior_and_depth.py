from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def uph_raw(data, idx, step):
    target = data["target"][idx, 2, ::step, ::step].astype(np.float64)
    lo, hi = data["phase_minmax"][idx].astype(np.float64)
    return target * (hi - lo) + lo


def valid_mask(data, idx, step):
    return data["mask"][idx, 0, ::step, ::step] > 0.5


def prior_features(cond, basis):
    x = cond[11].astype(np.float64)
    y = cond[12].astype(np.float64)
    feats = [np.ones_like(x), x, y, x * x, x * y, y * y]
    names = ["1", "x", "y", "x2", "xy", "y2"]
    if basis in {"xy2_phase", "xy2_phase_edge"}:
        hres = cond[3].astype(np.float64)
        fres = cond[7].astype(np.float64)
        feats.extend([hres, fres, hres * x, hres * y, fres * x, fres * y])
        names.extend(["hilbert_res", "ftp_res", "hilbert_res_x", "hilbert_res_y", "ftp_res_x", "ftp_res_y"])
    if basis == "xy2_phase_edge":
        dwt = cond[9].astype(np.float64)
        grad = cond[10].astype(np.float64)
        feats.extend([dwt, grad, dwt * x, dwt * y, grad * x, grad * y])
        names.extend(["dwt", "grad", "dwt_x", "dwt_y", "grad_x", "grad_y"])
    return np.stack(feats, axis=-1), names


def depth_features(raw_uph, cond, degree):
    x = cond[11].astype(np.float64)
    y = cond[12].astype(np.float64)
    feats = [np.ones_like(raw_uph), raw_uph, x, y]
    if degree >= 2:
        feats.extend([raw_uph * raw_uph, raw_uph * x, raw_uph * y, x * x, x * y, y * y])
    if degree >= 3:
        feats.extend([raw_uph ** 3, raw_uph * raw_uph * x, raw_uph * raw_uph * y, raw_uph * x * y, x ** 3, y ** 3])
    return np.stack(feats, axis=-1)


def load_split(base_dir, phase_dir, split):
    return {
        "cond": np.load(phase_dir / f"phase_instr_{split}_float16.npy", mmap_mode="r"),
        "target": np.load(phase_dir / f"phase_target_{split}_float16.npy", mmap_mode="r"),
        "phase_minmax": np.load(phase_dir / f"phase_minmax_{split}_float32.npy", mmap_mode="r"),
        "height": np.load(base_dir / f"depth_mm_{split}_float32.npy", mmap_mode="r"),
        "mask": np.load(base_dir / f"mask_{split}_uint8.npy", mmap_mode="r"),
    }


def ridge_solve(x, y, alpha):
    xtx = x.T @ x
    if alpha > 0:
        xtx = xtx + float(alpha) * np.eye(xtx.shape[0], dtype=np.float64)
    return np.linalg.solve(xtx, x.T @ y)


def sample_prior_fit(data, basis, step, max_pixels, seed):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for i in range(data["target"].shape[0]):
        mask = valid_mask(data, i, step)
        if not np.any(mask):
            continue
        feat, names = prior_features(data["cond"][i, :, ::step, ::step], basis)
        y = uph_raw(data, i, step)
        xs.append(feat[mask])
        ys.append(y[mask])
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if max_pixels > 0 and x.shape[0] > max_pixels:
        sel = rng.choice(x.shape[0], size=max_pixels, replace=False)
        x = x[sel]
        y = y[sel]
    return x, y, names


def sample_depth_fit(data, prior_coef, basis, degree, step, max_pixels, seed, use_gt_raw=False):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for i in range(data["target"].shape[0]):
        mask = valid_mask(data, i, step)
        if not np.any(mask):
            continue
        cond = data["cond"][i, :, ::step, ::step]
        raw = uph_raw(data, i, step) if use_gt_raw else prior_features(cond, basis)[0] @ prior_coef
        feat = depth_features(raw, cond, degree)
        height = data["height"][i, 0, ::step, ::step].astype(np.float64)
        xs.append(feat[mask])
        ys.append(height[mask])
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if max_pixels > 0 and x.shape[0] > max_pixels:
        sel = rng.choice(x.shape[0], size=max_pixels, replace=False)
        x = x[sel]
        y = y[sel]
    return x, y


def eval_uph_split(data, prior_coef, basis, step):
    rows = []
    abs_all = []
    sq_all = []
    for i in range(data["target"].shape[0]):
        mask = valid_mask(data, i, step)
        if not np.any(mask):
            continue
        cond = data["cond"][i, :, ::step, ::step]
        pred = prior_features(cond, basis)[0] @ prior_coef
        gt = uph_raw(data, i, step)
        err = pred[mask] - gt[mask]
        abs_err = np.abs(err)
        rows.append({
            "mae": float(abs_err.mean()),
            "rmse": float(np.sqrt(np.mean(err * err))),
            "res_p95": float(np.percentile(abs_err, 95)),
            "res_p99": float(np.percentile(abs_err, 99)),
        })
        abs_all.append(abs_err)
        sq_all.append(err * err)
    abs_cat = np.concatenate(abs_all)
    sq_cat = np.concatenate(sq_all)
    return {
        "mae_pixel": float(abs_cat.mean()),
        "rmse_pixel": float(np.sqrt(sq_cat.mean())),
        "mae_sample_mean": float(np.mean([r["mae"] for r in rows])),
        "rmse_sample_mean": float(np.mean([r["rmse"] for r in rows])),
        "residual_abs_p95_pixel": float(np.percentile(abs_cat, 95)),
        "residual_abs_p99_pixel": float(np.percentile(abs_cat, 99)),
        "n": len(rows),
    }


def eval_depth_split(data, prior_coef, basis, depth_coef, degree, step, use_gt_raw=False):
    sample_rmse = []
    sample_mae = []
    abs_all = []
    sq_all = []
    for i in range(data["target"].shape[0]):
        mask = valid_mask(data, i, step)
        if not np.any(mask):
            continue
        cond = data["cond"][i, :, ::step, ::step]
        raw = uph_raw(data, i, step) if use_gt_raw else prior_features(cond, basis)[0] @ prior_coef
        pred = depth_features(raw, cond, degree) @ depth_coef
        gt = data["height"][i, 0, ::step, ::step].astype(np.float64)
        err = pred[mask] - gt[mask]
        abs_err = np.abs(err)
        abs_all.append(abs_err)
        sq_all.append(err * err)
        sample_mae.append(float(abs_err.mean()))
        sample_rmse.append(float(np.sqrt(np.mean(err * err))))
    abs_cat = np.concatenate(abs_all)
    sq_cat = np.concatenate(sq_all)
    return {
        "rmse_pixel": float(np.sqrt(sq_cat.mean())),
        "mae_pixel": float(abs_cat.mean()),
        "rmse_sample_mean": float(np.mean(sample_rmse)),
        "mae_sample_mean": float(np.mean(sample_mae)),
        "n": len(sample_rmse),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_phase_cache_960")
    parser.add_argument("--save_dir", default="results/e54_uph_prior_fit")
    parser.add_argument("--basis", choices=["xy2", "xy2_phase", "xy2_phase_edge"], default="xy2_phase")
    parser.add_argument("--prior_step", type=int, default=8)
    parser.add_argument("--depth_fit_step", type=int, default=8)
    parser.add_argument("--eval_step", type=int, default=2)
    parser.add_argument("--max_prior_pixels", type=int, default=300000)
    parser.add_argument("--max_depth_pixels", type=int, default=300000)
    parser.add_argument("--prior_ridge", type=float, default=1e-4)
    parser.add_argument("--depth_ridge", type=float, default=1e-4)
    parser.add_argument("--depth_degree", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = Path(args.base_cache_dir)
    phase_dir = Path(args.phase_cache_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    data = {split: load_split(base_dir, phase_dir, split) for split in ("train", "val", "test")}

    x_prior, y_prior, names = sample_prior_fit(
        data["train"], args.basis, args.prior_step, args.max_prior_pixels, args.seed
    )
    prior_coef = ridge_solve(x_prior, y_prior, args.prior_ridge)

    x_depth_prior, y_depth_prior = sample_depth_fit(
        data["train"], prior_coef, args.basis, args.depth_degree, args.depth_fit_step,
        args.max_depth_pixels, args.seed, use_gt_raw=False
    )
    depth_coef_prior = ridge_solve(x_depth_prior, y_depth_prior, args.depth_ridge)
    x_depth_gt, y_depth_gt = sample_depth_fit(
        data["train"], prior_coef, args.basis, args.depth_degree, args.depth_fit_step,
        args.max_depth_pixels, args.seed, use_gt_raw=True
    )
    depth_coef_gt = ridge_solve(x_depth_gt, y_depth_gt, args.depth_ridge)

    summary = {
        "args": vars(args),
        "basis_names": names,
        "prior_coef": prior_coef.tolist(),
        "depth_coef_prior": depth_coef_prior.tolist(),
        "depth_coef_gt_raw": depth_coef_gt.tolist(),
        "uph_prior": {},
        "depth_from_prior_raw": {},
        "depth_from_gt_raw": {},
    }
    for split, split_data in data.items():
        summary["uph_prior"][split] = eval_uph_split(split_data, prior_coef, args.basis, args.eval_step)
        summary["depth_from_prior_raw"][split] = eval_depth_split(
            split_data, prior_coef, args.basis, depth_coef_prior, args.depth_degree, args.eval_step, use_gt_raw=False
        )
        summary["depth_from_gt_raw"][split] = eval_depth_split(
            split_data, prior_coef, args.basis, depth_coef_gt, args.depth_degree, args.eval_step, use_gt_raw=True
        )
    train_p99 = summary["uph_prior"]["train"]["residual_abs_p99_pixel"]
    summary["recommended_residual_scale"] = float(max(train_p99, 1e-6))

    np.save(save_dir / "uph_prior_coef.npy", prior_coef.astype(np.float64))
    np.save(save_dir / "depth_coef_prior.npy", depth_coef_prior.astype(np.float64))
    with open(save_dir / "uph_prior_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
