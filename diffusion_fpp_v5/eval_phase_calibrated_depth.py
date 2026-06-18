from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EPS = 1e-8


def feature_matrix(phase, x, y, degree=2):
    phase = np.asarray(phase, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    feats = [np.ones_like(phase), phase, x, y]
    if degree >= 2:
        feats.extend([phase * phase, phase * x, phase * y, x * x, x * y, y * y])
    if degree >= 3:
        feats.extend([phase ** 3, phase * phase * x, phase * phase * y, phase * x * y, x ** 3, y ** 3])
    return np.stack(feats, axis=1)


def uph_prior_features(cond, basis="xy2_phase"):
    x = cond[11].astype(np.float64)
    y = cond[12].astype(np.float64)
    feats = [np.ones_like(x), x, y, x * x, x * y, y * y]
    basis = str(basis)
    if basis in {"xy2_phase", "xy2_phase_edge"}:
        hres = cond[3].astype(np.float64)
        fres = cond[7].astype(np.float64)
        feats.extend([hres, fres, hres * x, hres * y, fres * x, fres * y])
    if basis == "xy2_phase_edge":
        dwt = cond[9].astype(np.float64)
        grad = cond[10].astype(np.float64)
        feats.extend([dwt, grad, dwt * x, dwt * y, grad * x, grad * y])
    return np.stack(feats, axis=0)


def uph_prior_raw(cond, coef, basis="xy2_phase"):
    feats = uph_prior_features(cond, basis=basis)
    coef = np.asarray(coef, dtype=np.float64).reshape(-1, 1, 1)
    if feats.shape[0] != coef.shape[0]:
        raise ValueError(f"UPH prior feature count {feats.shape[0]} != coef count {coef.shape[0]}")
    return np.sum(feats * coef, axis=0)


def load_split(base_cache_dir, phase_cache_dir, split, phase_pred_prefix=None):
    out = {
        "depth": np.load(base_cache_dir / f"depth_mm_{split}_float32.npy", mmap_mode="r"),
        "mask": np.load(base_cache_dir / f"mask_{split}_uint8.npy", mmap_mode="r"),
        "cond": np.load(phase_cache_dir / f"phase_instr_{split}_float16.npy", mmap_mode="r"),
        "target": np.load(phase_cache_dir / f"phase_target_{split}_float16.npy", mmap_mode="r"),
        "phase_minmax": np.load(phase_cache_dir / f"phase_minmax_{split}_float32.npy", mmap_mode="r"),
    }
    if phase_pred_prefix:
        out["phase_pred"] = np.load(phase_cache_dir / f"{phase_pred_prefix}_{split}_float16.npy", mmap_mode="r")
    return out


def phase_source_arrays(
    data,
    source,
    idx,
    step,
    pred_global_min=0.0,
    pred_global_max=1.0,
    pred_uph_representation="absolute",
    uph_prior_coef=None,
    uph_prior_basis="xy2_phase",
    uph_residual_scale=1.0,
):
    target = data["target"][idx, 2, ::step, ::step].astype(np.float64)
    lo, hi = data["phase_minmax"][idx].astype(np.float64)
    x = data["cond"][idx, 11, ::step, ::step].astype(np.float64)
    y = data["cond"][idx, 12, ::step, ::step].astype(np.float64)
    if source == "gt_raw":
        phase = target * (hi - lo) + lo
    elif source == "gt_01":
        phase = target
    elif source == "pred_raw_gt_minmax":
        pred = np.clip(data["phase_pred"][idx, 2, ::step, ::step].astype(np.float64), 0.0, 1.0)
        phase = pred * (hi - lo) + lo
    elif source == "pred_01":
        phase = np.clip(data["phase_pred"][idx, 2, ::step, ::step].astype(np.float64), 0.0, 1.0)
    elif source == "pred_raw_global":
        pred = np.clip(data["phase_pred"][idx, 2, ::step, ::step].astype(np.float64), 0.0, 1.0)
        phase = pred * (float(pred_global_max) - float(pred_global_min)) + float(pred_global_min)
    elif source == "pred_raw_prior_residual":
        pred = np.clip(data["phase_pred"][idx, 2, ::step, ::step].astype(np.float64), 0.0, 1.0)
        cond = data["cond"][idx, :, ::step, ::step].astype(np.float64)
        prior = uph_prior_raw(cond, uph_prior_coef, basis=uph_prior_basis)
        phase = prior + (pred - 0.5) * (2.0 * max(float(uph_residual_scale), EPS))
    else:
        raise ValueError(f"unknown source: {source}")
    return phase, x, y


def sample_fit_pixels(
    data,
    source,
    degree,
    step,
    max_pixels,
    rng,
    pred_global_min=0.0,
    pred_global_max=1.0,
    pred_uph_representation="absolute",
    uph_prior_coef=None,
    uph_prior_basis="xy2_phase",
    uph_residual_scale=1.0,
):
    xs = []
    ys = []
    n = int(data["depth"].shape[0])
    per_sample = max(1, int(max_pixels) // max(1, n))
    for i in range(n):
        depth = data["depth"][i, 0, ::step, ::step].astype(np.float64)
        mask = data["mask"][i, 0, ::step, ::step] > 0
        if not np.any(mask):
            continue
        phase, xg, yg = phase_source_arrays(
            data,
            source,
            i,
            step,
            pred_global_min=pred_global_min,
            pred_global_max=pred_global_max,
            pred_uph_representation=pred_uph_representation,
            uph_prior_coef=uph_prior_coef,
            uph_prior_basis=uph_prior_basis,
            uph_residual_scale=uph_residual_scale,
        )
        valid = np.where(mask.reshape(-1))[0]
        if valid.size > per_sample:
            valid = rng.choice(valid, size=per_sample, replace=False)
        feat = feature_matrix(phase.reshape(-1)[valid], xg.reshape(-1)[valid], yg.reshape(-1)[valid], degree=degree)
        xs.append(feat)
        ys.append(depth.reshape(-1)[valid])
    if not xs:
        raise RuntimeError("no valid fit pixels")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def fit_ridge(x, y, alpha):
    xtx = x.T @ x
    reg = float(alpha) * np.eye(xtx.shape[0], dtype=np.float64)
    reg[0, 0] = 0.0
    return np.linalg.solve(xtx + reg, x.T @ y)


def eval_split(
    data,
    source,
    coef,
    degree,
    step,
    chunk_pixels=1_000_000,
    pred_global_min=0.0,
    pred_global_max=1.0,
    pred_uph_representation="absolute",
    uph_prior_coef=None,
    uph_prior_basis="xy2_phase",
    uph_residual_scale=1.0,
):
    rows = []
    sum_sq = 0.0
    sum_abs = 0.0
    count = 0
    n = int(data["depth"].shape[0])
    for i in range(n):
        depth = data["depth"][i, 0, ::step, ::step].astype(np.float64)
        mask = data["mask"][i, 0, ::step, ::step] > 0
        phase, xg, yg = phase_source_arrays(
            data,
            source,
            i,
            step,
            pred_global_min=pred_global_min,
            pred_global_max=pred_global_max,
            pred_uph_representation=pred_uph_representation,
            uph_prior_coef=uph_prior_coef,
            uph_prior_basis=uph_prior_basis,
            uph_residual_scale=uph_residual_scale,
        )
        valid = np.where(mask.reshape(-1))[0]
        if valid.size == 0:
            continue
        pred = np.empty(valid.size, dtype=np.float64)
        phase_f = phase.reshape(-1)
        x_f = xg.reshape(-1)
        y_f = yg.reshape(-1)
        for start in range(0, valid.size, int(chunk_pixels)):
            sel = valid[start:start + int(chunk_pixels)]
            pred[start:start + sel.size] = feature_matrix(phase_f[sel], x_f[sel], y_f[sel], degree=degree) @ coef
        gt = depth.reshape(-1)[valid]
        err = pred - gt
        rmse = float(np.sqrt(np.mean(err * err)))
        mae = float(np.mean(np.abs(err)))
        rows.append({"sample": i, "rmse": rmse, "mae": mae, "valid_pixels": int(valid.size)})
        sum_sq += float(np.sum(err * err))
        sum_abs += float(np.sum(np.abs(err)))
        count += int(valid.size)
    return {
        "rmse_pixel": float(np.sqrt(sum_sq / max(1, count))),
        "mae_pixel": float(sum_abs / max(1, count)),
        "rmse_sample_mean": float(np.mean([r["rmse"] for r in rows])) if rows else float("nan"),
        "mae_sample_mean": float(np.mean([r["mae"] for r in rows])) if rows else float("nan"),
        "n": len(rows),
        "rows": rows,
    }


def write_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_phase_cache_960")
    parser.add_argument("--phase_pred_prefix", default="")
    parser.add_argument("--save_dir", default="/root/autodl-tmp/diffusion_fpp_v5/results/phase_calibrated_depth")
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--fit_step", type=int, default=8)
    parser.add_argument("--eval_step", type=int, default=2)
    parser.add_argument("--max_train_pixels", type=int, default=300000)
    parser.add_argument("--ridge_alpha", type=float, default=1e-4)
    parser.add_argument("--pred_global_min", type=float, default=0.0)
    parser.add_argument("--pred_global_max", type=float, default=0.0)
    parser.add_argument("--pred_uph_representation", choices=["absolute", "prior_residual"], default="absolute")
    parser.add_argument("--uph_prior_summary", default="")
    parser.add_argument("--uph_prior_basis", default="")
    parser.add_argument("--uph_residual_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_cache_dir = Path(args.base_cache_dir)
    phase_cache_dir = Path(args.phase_cache_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    has_pred = bool(args.phase_pred_prefix)
    uph_prior_coef = None
    if args.pred_uph_representation == "prior_residual":
        if not args.uph_prior_summary:
            raise ValueError("--uph_prior_summary is required for prior_residual predictions")
        with open(args.uph_prior_summary, "r", encoding="utf-8") as f:
            prior_summary = json.load(f)
        uph_prior_coef = [float(v) for v in prior_summary["prior_coef"]]
        if not args.uph_prior_basis:
            args.uph_prior_basis = str(prior_summary.get("args", {}).get("basis", "xy2_phase"))
        if args.uph_residual_scale <= 0:
            args.uph_residual_scale = float(prior_summary.get("recommended_residual_scale", 1.0))
    sources = ["gt_raw", "gt_01"]
    if has_pred:
        sources.extend(["pred_raw_gt_minmax", "pred_01"])
        if args.pred_global_max > args.pred_global_min:
            sources.append("pred_raw_global")
        if args.pred_uph_representation == "prior_residual":
            sources.append("pred_raw_prior_residual")

    data = {
        split: load_split(base_cache_dir, phase_cache_dir, split, args.phase_pred_prefix if has_pred else None)
        for split in ("train", "val", "test")
    }
    rng = np.random.default_rng(args.seed)
    summary = {"args": vars(args), "sources": {}}
    for source in sources:
        x_train, y_train = sample_fit_pixels(
            data["train"],
            source=source,
            degree=args.degree,
            step=args.fit_step,
            max_pixels=args.max_train_pixels,
            rng=rng,
            pred_global_min=args.pred_global_min,
            pred_global_max=args.pred_global_max,
            pred_uph_representation=args.pred_uph_representation,
            uph_prior_coef=uph_prior_coef,
            uph_prior_basis=args.uph_prior_basis,
            uph_residual_scale=args.uph_residual_scale,
        )
        coef = fit_ridge(x_train, y_train, args.ridge_alpha)
        source_summary = {
            "fit_pixels": int(x_train.shape[0]),
            "coef": [float(v) for v in coef],
            "splits": {},
        }
        for split in ("train", "val", "test"):
            res = eval_split(
                data[split],
                source,
                coef,
                degree=args.degree,
                step=args.eval_step,
                pred_global_min=args.pred_global_min,
                pred_global_max=args.pred_global_max,
                pred_uph_representation=args.pred_uph_representation,
                uph_prior_coef=uph_prior_coef,
                uph_prior_basis=args.uph_prior_basis,
                uph_residual_scale=args.uph_residual_scale,
            )
            write_rows(res.pop("rows"), save_dir / source / f"{split}_per_sample_metrics.csv")
            source_summary["splits"][split] = res
        summary["sources"][source] = source_summary
        print(source, json.dumps(source_summary["splits"], ensure_ascii=False))

    with open(save_dir / "phase_calibrated_depth_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
