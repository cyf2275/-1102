from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


SPLITS = ("train", "val", "test")
EPS = 1e-8


def sample_parts(sample: str) -> tuple[str, str]:
    obj, angle = sample.rsplit("_", 1)
    return obj, angle


def load_phase(data_root: Path, sample: str, step: int) -> np.ndarray:
    obj, angle = sample_parts(sample)
    path = data_root / "fpp_synthetic_dataset" / obj / angle / "unwrapped_phase.mat"
    mat = loadmat(path)
    for key in ("uph", "unwrapped_phase"):
        if key in mat:
            return np.asarray(mat[key], dtype=np.float64)[::step, ::step]
    keys = [k for k in mat.keys() if not k.startswith("__")]
    raise KeyError(f"no unwrapped phase in {path}; keys={keys}")


def normalized_xy(h: int, w: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float64),
        np.linspace(-1.0, 1.0, w, dtype=np.float64),
        indexing="ij",
    )
    return xx[::step, ::step], yy[::step, ::step]


def poly_features(depth: np.ndarray, x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    feats = [np.ones_like(depth), depth, x, y]
    if degree >= 2:
        feats.extend([depth * depth, depth * x, depth * y, x * x, x * y, y * y])
    if degree >= 3:
        feats.extend([depth**3, depth * depth * x, depth * depth * y, depth * x * y, x**3, y**3])
    return np.stack([f.reshape(-1) for f in feats], axis=1)


def solve_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x.T @ x
    reg = np.eye(xtx.shape[0], dtype=np.float64) * float(alpha)
    reg[0, 0] = 0.0
    return np.linalg.solve(xtx + reg, x.T @ y)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    err = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(err * err)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def masked_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.asarray(arr, dtype=np.float64)[valid]))


def load_manifest(base_cache_dir: Path) -> dict:
    with (base_cache_dir / "manifest.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_array(cache_dir: Path, name: str, split: str, suffix: str):
    return np.load(cache_dir / f"{name}_{split}_{suffix}.npy", mmap_mode="r")


def candidate_norm_to_mm(candidate_norm: np.ndarray, depth_minmax: np.ndarray) -> np.ndarray:
    d01 = np.clip((np.asarray(candidate_norm, dtype=np.float64) + 1.0) * 0.5, 0.0, 1.0)
    lo = float(depth_minmax[0])
    hi = float(depth_minmax[1])
    return d01 * (hi - lo) + lo


def collect_train_fit_pixels(
    data_root: Path,
    base_cache_dir: Path,
    candidate_cache_dir: Path,
    manifest: dict,
    step: int,
    max_pixels: int,
    degree: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    target_mm = load_array(candidate_cache_dir, "target_mm", "train", "float32")
    mask = load_array(candidate_cache_dir, "mask", "train", "uint8")
    samples = manifest["splits"]["train"]["samples"]
    xpix, ypix = normalized_xy(target_mm.shape[-2], target_mm.shape[-1], step)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    per_sample = max(1, int(max_pixels) // max(1, len(samples)))
    for i, sample_meta in enumerate(samples):
        sample = str(sample_meta["sample"])
        depth = np.asarray(target_mm[i, 0, ::step, ::step], dtype=np.float64)
        valid = mask[i, 0, ::step, ::step] > 0
        if not np.any(valid):
            continue
        phase = load_phase(data_root, sample, step=step)
        idx = np.where(valid.reshape(-1))[0]
        if idx.size > per_sample:
            idx = rng.choice(idx, size=per_sample, replace=False)
        xs.append(poly_features(depth.reshape(-1)[idx], xpix.reshape(-1)[idx], ypix.reshape(-1)[idx], degree=degree))
        ys.append(phase.reshape(-1)[idx])
    if not xs:
        raise RuntimeError("no train pixels for proxy phase fit")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def summarize_metric(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    return {
        "mean": float(np.nanmean(vals)),
        "median": float(np.nanmedian(vals)),
        "std": float(np.nanstd(vals)),
        "min": float(np.nanmin(vals)),
        "max": float(np.nanmax(vals)),
    }


def eval_split(
    data_root: Path,
    base_cache_dir: Path,
    candidate_cache_dir: Path,
    manifest: dict,
    split: str,
    coef: np.ndarray,
    step: int,
    degree: int,
    rcpc_edge_tau: float,
    rcpc_delta_max: float,
    rcpc_phase_conf_max: float,
    rcpc_high_weight: float,
    blend_weight: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    d_b = load_array(candidate_cache_dir, "d_b", split, "float16")
    d_p = load_array(candidate_cache_dir, "d_p", split, "float16")
    d_d = load_array(candidate_cache_dir, "d_d", split, "float16")
    target_mm = load_array(candidate_cache_dir, "target_mm", split, "float32")
    mask = load_array(candidate_cache_dir, "mask", split, "uint8")
    edge = load_array(candidate_cache_dir, "edge", split, "float16")
    phase_conf = load_array(candidate_cache_dir, "phase_conf", split, "float16")
    depth_minmax = load_array(candidate_cache_dir, "depth_minmax", split, "float32")
    samples = manifest["splits"][split]["samples"]
    xpix, ypix = normalized_xy(d_b.shape[-2], d_b.shape[-1], step)

    rows: list[dict[str, object]] = []
    branches = ("target", "d_b", "d_p", "d_d", "blend_w03", "rcpc_e84")
    for i, sample_meta in enumerate(samples):
        sample = str(sample_meta["sample"])
        valid = mask[i, 0, ::step, ::step] > 0
        if not np.any(valid):
            continue
        phase = load_phase(data_root, sample, step=step)
        target = np.asarray(target_mm[i, 0, ::step, ::step], dtype=np.float64)
        db_norm = np.asarray(d_b[i, 0, ::step, ::step], dtype=np.float64)
        dp_norm = np.asarray(d_p[i, 0, ::step, ::step], dtype=np.float64)
        dd_norm = np.asarray(d_d[i, 0, ::step, ::step], dtype=np.float64)
        db = candidate_norm_to_mm(db_norm, depth_minmax[i])
        dp = candidate_norm_to_mm(dp_norm, depth_minmax[i])
        dd = candidate_norm_to_mm(dd_norm, depth_minmax[i])
        blend_norm = np.clip(dd_norm + float(blend_weight) * (dp_norm - dd_norm), -1.0, 1.0)
        blend = candidate_norm_to_mm(blend_norm, depth_minmax[i])

        full_mask = mask[i, 0] > 0
        edge_mean = masked_mean(edge[i, 0], full_mask)
        conf_mean = masked_mean(phase_conf[i, 0], full_mask)
        delta_mean = masked_mean(np.abs(np.asarray(d_p[i, 0], dtype=np.float64) - np.asarray(d_d[i, 0], dtype=np.float64)), full_mask)
        accept_phase = (
            edge_mean >= float(rcpc_edge_tau)
            and delta_mean <= float(rcpc_delta_max)
            and conf_mean <= float(rcpc_phase_conf_max)
        )
        rcpc_norm = dd_norm
        if accept_phase:
            rcpc_norm = np.clip(dd_norm + float(rcpc_high_weight) * (dp_norm - dd_norm), -1.0, 1.0)
        rcpc = candidate_norm_to_mm(rcpc_norm, depth_minmax[i])

        depth_by_branch = {
            "target": target,
            "d_b": db,
            "d_p": dp,
            "d_d": dd,
            "blend_w03": blend,
            "rcpc_e84": rcpc,
        }
        idx = np.where(valid.reshape(-1))[0]
        for branch in branches:
            pred_depth = depth_by_branch[branch]
            pred_phase = poly_features(
                pred_depth.reshape(-1)[idx],
                xpix.reshape(-1)[idx],
                ypix.reshape(-1)[idx],
                degree=degree,
            ) @ coef
            gt_phase = phase.reshape(-1)[idx]
            gt_depth = target.reshape(-1)[idx]
            pd = pred_depth.reshape(-1)[idx]
            rows.append(
                {
                    "split": split,
                    "sample": sample,
                    "branch": branch,
                    "phase_rmse_rad": rmse(pred_phase, gt_phase),
                    "phase_mae_rad": mae(pred_phase, gt_phase),
                    "depth_rmse_mm": rmse(pd, gt_depth),
                    "depth_mae_mm": mae(pd, gt_depth),
                    "edge_mean": edge_mean,
                    "delta_mean_norm": delta_mean,
                    "phase_conf_mean": conf_mean,
                    "rcpc_accept_phase": int(bool(accept_phase)),
                    "valid_pixels": int(idx.size),
                }
            )
    summary = {"split": split, "branches": {}}
    for branch in branches:
        br = [row for row in rows if row["branch"] == branch]
        summary["branches"][branch] = {
            "n": len(br),
            "phase_rmse_rad": summarize_metric(br, "phase_rmse_rad"),
            "phase_mae_rad": summarize_metric(br, "phase_mae_rad"),
            "depth_rmse_mm": summarize_metric(br, "depth_rmse_mm"),
            "depth_mae_mm": summarize_metric(br, "depth_mae_mm"),
        }
    summary["rcpc_accept_count"] = int(sum(1 for row in rows if row["branch"] == "rcpc_e84" and int(row["rcpc_accept_phase"]) == 1))
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/root/autodl-tmp/datasets/fpp-ml-bench")
    parser.add_argument("--base_cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--candidate_cache_dir", default="/root/autodl-tmp/fpp_ml_ucpf_hier_orderfix_cache_960_seed180")
    parser.add_argument("--save_dir", default="/root/autodl-tmp/diffusion_fpp_v5/results/candidate_proxy_phase_consistency")
    parser.add_argument("--fit_step", type=int, default=8)
    parser.add_argument("--eval_step", type=int, default=4)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--max_train_pixels", type=int, default=250000)
    parser.add_argument("--ridge_alpha", type=float, default=1e-4)
    parser.add_argument("--blend_weight", type=float, default=0.3)
    parser.add_argument("--rcpc_edge_tau", type=float, default=0.42)
    parser.add_argument("--rcpc_delta_max", type=float, default=0.11)
    parser.add_argument("--rcpc_phase_conf_max", type=float, default=0.74)
    parser.add_argument("--rcpc_high_weight", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=20260606)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    base_cache_dir = Path(args.base_cache_dir)
    candidate_cache_dir = Path(args.candidate_cache_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(base_cache_dir)
    rng = np.random.default_rng(int(args.seed))

    x_train, y_train = collect_train_fit_pixels(
        data_root=data_root,
        base_cache_dir=base_cache_dir,
        candidate_cache_dir=candidate_cache_dir,
        manifest=manifest,
        step=args.fit_step,
        max_pixels=args.max_train_pixels,
        degree=args.degree,
        rng=rng,
    )
    coef = solve_ridge(x_train, y_train, alpha=args.ridge_alpha)
    summary: dict[str, object] = {
        "args": vars(args),
        "proxy_model": {
            "type": "global polynomial depth+xy -> unwrapped phase",
            "degree": int(args.degree),
            "coef": [float(v) for v in coef],
            "train_fit_pixels": int(y_train.size),
            "train_fit_phase_rmse_rad": rmse(x_train @ coef, y_train),
            "train_fit_phase_mae_rad": mae(x_train @ coef, y_train),
        },
        "splits": {},
    }

    all_rows: list[dict[str, object]] = []
    for split in ("val", "test"):
        rows, split_summary = eval_split(
            data_root=data_root,
            base_cache_dir=base_cache_dir,
            candidate_cache_dir=candidate_cache_dir,
            manifest=manifest,
            split=split,
            coef=coef,
            step=args.eval_step,
            degree=args.degree,
            rcpc_edge_tau=args.rcpc_edge_tau,
            rcpc_delta_max=args.rcpc_delta_max,
            rcpc_phase_conf_max=args.rcpc_phase_conf_max,
            rcpc_high_weight=args.rcpc_high_weight,
            blend_weight=args.blend_weight,
        )
        write_csv(save_dir / f"{split}_candidate_proxy_phase_rows.csv", rows)
        all_rows.extend(rows)
        summary["splits"][split] = split_summary

    write_csv(save_dir / "candidate_proxy_phase_rows.csv", all_rows)
    with (save_dir / "candidate_proxy_phase_consistency_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
