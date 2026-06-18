from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat


SPLITS = ("train", "val", "test")
EPS = 1e-8


def load_mat_array(path: Path, keys: tuple[str, ...]) -> np.ndarray:
    mat = loadmat(path)
    for key in keys:
        if key in mat:
            return np.asarray(mat[key], dtype=np.float64)
    found = [k for k in mat.keys() if not k.startswith("__")]
    raise KeyError(f"none of {keys} found in {path}; keys={found}")


def sample_parts(stem: str) -> tuple[str, str]:
    obj, angle = stem.rsplit("_", 1)
    return obj, angle


def raw_sample_dir(data_root: Path, sample_stem: str) -> Path:
    obj, angle = sample_parts(sample_stem)
    return data_root / "fpp_synthetic_dataset" / obj / angle


def normalized_xy(h: int, w: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float64),
        np.linspace(-1.0, 1.0, w, dtype=np.float64),
        indexing="ij",
    )
    return xx[::step, ::step], yy[::step, ::step]


def poly_features(a: np.ndarray, x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    feats = [np.ones_like(a), a, x, y]
    if degree >= 2:
        feats.extend([a * a, a * x, a * y, x * x, x * y, y * y])
    if degree >= 3:
        feats.extend([a**3, a * a * x, a * a * y, a * x * y, x**3, y**3])
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


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]
    b = b[ok]
    if a.size < 10 or np.std(a) < EPS or np.std(b) < EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_split_samples(data_root: Path, split: str) -> list[Path]:
    fringe_dir = (
        data_root
        / "training_datasets"
        / "training_data_depth_raw"
        / split
        / "fringe"
    )
    paths = sorted(fringe_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no official split fringe files under {fringe_dir}")
    return paths


def load_sample(
    data_root: Path,
    split: str,
    fringe_path: Path,
    step: int,
    need_z: bool = False,
) -> dict[str, np.ndarray | str | int]:
    stem = fringe_path.stem
    raw_dir = raw_sample_dir(data_root, stem)
    raw_depth_path = (
        data_root
        / "training_datasets"
        / "training_data_depth_raw"
        / split
        / "depth"
        / f"{stem}.mat"
    )
    depth = load_mat_array(raw_depth_path, ("depthMap", "depthMapNormalized"))[::step, ::step]
    valid = depth > 0.0
    uph = load_mat_array(raw_dir / "unwrapped_phase.mat", ("uph", "unwrapped_phase"))[::step, ::step]
    wph = load_mat_array(raw_dir / "wrapped_phase.mat", ("wph", "wrapped_phase"))[::step, ::step]
    fringe = np.asarray(Image.open(raw_dir / "A_0.png").convert("L"), dtype=np.float64)[::step, ::step]
    xpix, ypix = normalized_xy(depth.shape[0] * step, depth.shape[1] * step, step)
    sample = {
        "sample": stem,
        "depth": depth,
        "valid": valid,
        "uph": uph,
        "wph": wph,
        "fringe": fringe / 255.0,
        "xpix": xpix,
        "ypix": ypix,
        "valid_pixels": int(valid.sum()),
    }
    if need_z:
        sample["z"] = np.loadtxt(raw_dir / "z.csv", delimiter=",", dtype=np.float64)[::step, ::step]
    return sample


def collect_model_pixels(
    data_root: Path,
    split: str,
    step: int,
    max_pixels: int,
    degree: int,
    source: str,
    target: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    paths = load_split_samples(data_root, split)
    per_sample = max(1, int(max_pixels) // max(1, len(paths))) if max_pixels > 0 else 0
    for path in paths:
        sample = load_sample(data_root, split, path, step, need_z=(source == "z" or target == "z"))
        valid = np.asarray(sample["valid"], dtype=bool).reshape(-1)
        if not np.any(valid):
            continue
        if per_sample > 0:
            idx = np.where(valid)[0]
            if idx.size > per_sample:
                idx = rng.choice(idx, size=per_sample, replace=False)
        else:
            idx = np.where(valid)[0]
        src = np.asarray(sample[source], dtype=np.float64).reshape(-1)[idx]
        xpix = np.asarray(sample["xpix"], dtype=np.float64).reshape(-1)[idx]
        ypix = np.asarray(sample["ypix"], dtype=np.float64).reshape(-1)[idx]
        tgt = np.asarray(sample[target], dtype=np.float64).reshape(-1)[idx]
        xs.append(poly_features(src, xpix, ypix, degree=degree))
        ys.append(tgt)
    if not xs:
        raise RuntimeError(f"no valid pixels for {split} {source}->{target}")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def eval_poly_model(
    data_root: Path,
    split: str,
    coef: np.ndarray,
    step: int,
    degree: int,
    source: str,
    target: str,
) -> dict[str, object]:
    rows = []
    sum_sq = 0.0
    sum_abs = 0.0
    count = 0
    for path in load_split_samples(data_root, split):
        sample = load_sample(data_root, split, path, step, need_z=(source == "z" or target == "z"))
        valid = np.asarray(sample["valid"], dtype=bool).reshape(-1)
        idx = np.where(valid)[0]
        if idx.size == 0:
            continue
        src = np.asarray(sample[source], dtype=np.float64).reshape(-1)[idx]
        xpix = np.asarray(sample["xpix"], dtype=np.float64).reshape(-1)[idx]
        ypix = np.asarray(sample["ypix"], dtype=np.float64).reshape(-1)[idx]
        tgt = np.asarray(sample[target], dtype=np.float64).reshape(-1)[idx]
        pred = poly_features(src, xpix, ypix, degree=degree) @ coef
        err = pred - tgt
        sample_rmse = float(np.sqrt(np.mean(err * err)))
        sample_mae = float(np.mean(np.abs(err)))
        rows.append(
            {
                "split": split,
                "sample": str(sample["sample"]),
                "source": source,
                "target": target,
                "rmse": sample_rmse,
                "mae": sample_mae,
                "valid_pixels": int(idx.size),
            }
        )
        sum_sq += float(np.sum(err * err))
        sum_abs += float(np.sum(np.abs(err)))
        count += int(idx.size)
    return {
        "pixel_rmse": float(np.sqrt(sum_sq / max(1, count))),
        "pixel_mae": float(sum_abs / max(1, count)),
        "sample_rmse_mean": float(np.mean([r["rmse"] for r in rows])) if rows else float("nan"),
        "sample_rmse_median": float(np.median([r["rmse"] for r in rows])) if rows else float("nan"),
        "sample_mae_mean": float(np.mean([r["mae"] for r in rows])) if rows else float("nan"),
        "n_samples": len(rows),
        "n_pixels": int(count),
        "rows": rows,
    }


def fit_linear_z_depth(
    data_root: Path,
    split: str,
    step: int,
    max_pixels: int,
    rng: np.random.Generator,
    source: str,
    target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    ys = []
    paths = load_split_samples(data_root, split)
    per_sample = max(1, int(max_pixels) // max(1, len(paths))) if max_pixels > 0 else 0
    for path in paths:
        sample = load_sample(data_root, split, path, step, need_z=(source == "z" or target == "z"))
        valid = np.asarray(sample["valid"], dtype=bool).reshape(-1)
        idx = np.where(valid)[0]
        if per_sample > 0 and idx.size > per_sample:
            idx = rng.choice(idx, size=per_sample, replace=False)
        src = np.asarray(sample[source], dtype=np.float64).reshape(-1)[idx]
        tgt = np.asarray(sample[target], dtype=np.float64).reshape(-1)[idx]
        xs.append(np.stack([src, np.ones_like(src)], axis=1))
        ys.append(tgt)
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    return coef, x, y


def eval_linear_relation(data_root: Path, split: str, step: int, coef: np.ndarray, source: str, target: str) -> dict[str, object]:
    rows = []
    sum_sq = 0.0
    sum_abs = 0.0
    count = 0
    for path in load_split_samples(data_root, split):
        sample = load_sample(data_root, split, path, step, need_z=(source == "z" or target == "z"))
        valid = np.asarray(sample["valid"], dtype=bool).reshape(-1)
        idx = np.where(valid)[0]
        if idx.size == 0:
            continue
        src = np.asarray(sample[source], dtype=np.float64).reshape(-1)[idx]
        tgt = np.asarray(sample[target], dtype=np.float64).reshape(-1)[idx]
        pred = coef[0] * src + coef[1]
        err = pred - tgt
        rows.append(
            {
                "split": split,
                "sample": str(sample["sample"]),
                "source": source,
                "target": target,
                "rmse": float(np.sqrt(np.mean(err * err))),
                "mae": float(np.mean(np.abs(err))),
                "corr": corr(src, tgt),
                "valid_pixels": int(idx.size),
            }
        )
        sum_sq += float(np.sum(err * err))
        sum_abs += float(np.sum(np.abs(err)))
        count += int(idx.size)
    return {
        "pixel_rmse": float(np.sqrt(sum_sq / max(1, count))),
        "pixel_mae": float(sum_abs / max(1, count)),
        "sample_rmse_mean": float(np.mean([r["rmse"] for r in rows])) if rows else float("nan"),
        "sample_corr_mean": float(np.nanmean([r["corr"] for r in rows])) if rows else float("nan"),
        "n_samples": len(rows),
        "n_pixels": int(count),
        "rows": rows,
    }


def per_sample_stats(data_root: Path, split: str, step: int) -> list[dict[str, object]]:
    rows = []
    for path in load_split_samples(data_root, split):
        sample = load_sample(data_root, split, path, step, need_z=True)
        valid = np.asarray(sample["valid"], dtype=bool)
        if not np.any(valid):
            continue
        depth = np.asarray(sample["depth"], dtype=np.float64)
        uph = np.asarray(sample["uph"], dtype=np.float64)
        z = np.asarray(sample["z"], dtype=np.float64)
        fringe = np.asarray(sample["fringe"], dtype=np.float64)
        rows.append(
            {
                "split": split,
                "sample": str(sample["sample"]),
                "valid_pixels": int(valid.sum()),
                "depth_min": float(depth[valid].min()),
                "depth_max": float(depth[valid].max()),
                "depth_range": float(depth[valid].max() - depth[valid].min()),
                "uph_min": float(uph[valid].min()),
                "uph_max": float(uph[valid].max()),
                "uph_range": float(uph[valid].max() - uph[valid].min()),
                "z_min": float(z[valid].min()),
                "z_max": float(z[valid].max()),
                "z_range": float(z[valid].max() - z[valid].min()),
                "corr_uph_depth": corr(uph[valid], depth[valid]),
                "corr_z_depth": corr(z[valid], depth[valid]),
                "corr_fringe_depth": corr(fringe[valid], depth[valid]),
                "corr_fringe_uph": corr(fringe[valid], uph[valid]),
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {"n": len(rows)}
    if not rows:
        return out
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and key not in {"valid_pixels"}
    ]
    for key in numeric_keys:
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        out[key] = {
            "mean": float(np.nanmean(vals)),
            "median": float(np.nanmedian(vals)),
            "std": float(np.nanstd(vals)),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
        }
    return out


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
    parser.add_argument("--data_root", default=r"D:\shujuji\fpp-ml-bench")
    parser.add_argument("--save_dir", default=r"H:\yjs\实验室\sdxx\cloud_results\A_20260606_measurement_proxy_consistency")
    parser.add_argument("--fit_step", type=int, default=8)
    parser.add_argument("--eval_step", type=int, default=4)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--max_train_pixels", type=int, default=250000)
    parser.add_argument("--ridge_alpha", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260606)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    per_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {"args": vars(args), "splits": {}}
    for split in SPLITS:
        rows = per_sample_stats(data_root, split, step=args.eval_step)
        per_rows.extend(rows)
        summary["splits"][split] = summarize_rows(rows)
    write_csv(save_dir / "per_sample_measurement_proxy_stats.csv", per_rows)

    models = {}
    for name, source, target in [
        ("phase_xy_to_depth", "uph", "depth"),
        ("depth_xy_to_phase", "depth", "uph"),
        ("fringe_xy_to_depth_control", "fringe", "depth"),
    ]:
        x_train, y_train = collect_model_pixels(
            data_root,
            "train",
            step=args.fit_step,
            max_pixels=args.max_train_pixels,
            degree=args.degree,
            source=source,
            target=target,
            rng=rng,
        )
        coef = solve_ridge(x_train, y_train, alpha=args.ridge_alpha)
        model_summary = {
            "source": source,
            "target": target,
            "degree": int(args.degree),
            "coef": [float(v) for v in coef],
            "train_fit_pixels": int(y_train.size),
            "train_fit_rmse": rmse(x_train @ coef, y_train),
            "train_fit_mae": mae(x_train @ coef, y_train),
            "eval": {},
        }
        rows_out = []
        for split in SPLITS:
            ev = eval_poly_model(data_root, split, coef, step=args.eval_step, degree=args.degree, source=source, target=target)
            rows_out.extend(ev.pop("rows"))
            model_summary["eval"][split] = ev
        write_csv(save_dir / f"{name}_per_sample.csv", rows_out)
        models[name] = model_summary

    for name, source, target in [
        ("z_to_depth_linear", "z", "depth"),
        ("depth_to_z_linear", "depth", "z"),
    ]:
        coef, x_train, y_train = fit_linear_z_depth(
            data_root,
            "train",
            step=args.fit_step,
            max_pixels=args.max_train_pixels,
            rng=rng,
            source=source,
            target=target,
        )
        model_summary = {
            "source": source,
            "target": target,
            "coef": [float(v) for v in coef],
            "train_fit_pixels": int(y_train.size),
            "train_fit_rmse": rmse(x_train @ coef, y_train),
            "train_fit_mae": mae(x_train @ coef, y_train),
            "eval": {},
        }
        rows_out = []
        for split in SPLITS:
            ev = eval_linear_relation(data_root, split, step=args.eval_step, coef=coef, source=source, target=target)
            rows_out.extend(ev.pop("rows"))
            model_summary["eval"][split] = ev
        write_csv(save_dir / f"{name}_per_sample.csv", rows_out)
        models[name] = model_summary

    summary["models"] = models
    with (save_dir / "measurement_proxy_consistency_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
