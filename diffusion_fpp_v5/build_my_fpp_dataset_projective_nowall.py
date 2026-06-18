from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from build_my_fpp_dataset import (
    PMP_FRAME_COUNT,
    SINGLE_INPUT_INDEX,
    compute_multifrequency_phase,
    compute_pmp_orderfix,
    image_stats,
    numeric_key,
    sample_id,
)
from reconstruct_my_dataset_pmp_ftp import load_pmp_stack, parse_pmp_calibration, read_gray


@dataclass(frozen=True)
class SessionSpec:
    session_id: str
    min_object: int
    max_object: int
    roots: tuple[Path, ...]
    calibration: Path
    calibration_model: str
    phase_order: str
    pixel_origin: str


def discover_samples(spec: SessionSpec) -> list[tuple[str, str, Path, Path]]:
    samples: list[tuple[str, str, Path, Path]] = []
    for obj_num in range(spec.min_object, spec.max_object + 1):
        obj = str(obj_num)
        obj_root = None
        for root in spec.roots:
            if (root / obj).is_dir():
                obj_root = root
                break
        if obj_root is None:
            continue
        for pose_dir in sorted((obj_root / obj).glob("pose*"), key=lambda p: numeric_key(p.name.replace("pose", ""))):
            pmp_dir = pose_dir / "pmp"
            if not pmp_dir.is_dir():
                continue
            if len(list(pmp_dir.glob("*.bmp"))) != PMP_FRAME_COUNT:
                continue
            samples.append((obj, pose_dir.name, pose_dir, obj_root))
    return samples


def split_for_object(object_id: str) -> str:
    obj = int(object_id)
    if obj <= 8:
        return "train"
    if obj <= 12:
        return "val" if obj <= 10 else "test"
    if obj >= 61:
        return "test"
    if obj >= 55:
        return "val"
    return "train"


def pixel_grid_0based(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices((h, w), dtype=np.float64)
    return xx, yy


def solve_projective_4eq_0based(
    cm: np.ndarray,
    pm: np.ndarray,
    phase_x: np.ndarray,
    phase_y: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = phase_y.shape
    xc, yc = pixel_grid_0based(h, w)
    valid = mask & np.isfinite(phase_x) & np.isfinite(phase_y)
    x = xc[valid]
    y = yc[valid]
    xp = phase_x[valid].astype(np.float64)
    yp = phase_y[valid].astype(np.float64)

    rows = np.stack(
        [
            np.stack([cm[0, 0] - cm[2, 0] * x, cm[0, 1] - cm[2, 1] * x, cm[0, 2] - cm[2, 2] * x], axis=1),
            np.stack([cm[1, 0] - cm[2, 0] * y, cm[1, 1] - cm[2, 1] * y, cm[1, 2] - cm[2, 2] * y], axis=1),
            np.stack([pm[0, 0] - pm[2, 0] * xp, pm[0, 1] - pm[2, 1] * xp, pm[0, 2] - pm[2, 2] * xp], axis=1),
            np.stack([pm[1, 0] - pm[2, 0] * yp, pm[1, 1] - pm[2, 1] * yp, pm[1, 2] - pm[2, 2] * yp], axis=1),
        ],
        axis=1,
    )
    rhs = np.stack([x - cm[0, 3], y - cm[1, 3], xp - pm[0, 3], yp - pm[1, 3]], axis=1)
    ata = np.einsum("nki,nkj->nij", rows, rows)
    atb = np.einsum("nki,nk->ni", rows, rhs)
    coords = np.linalg.solve(ata, atb)
    residual = np.sqrt(np.mean((np.einsum("nki,ni->nk", rows, coords) - rhs) ** 2, axis=1))

    X = np.full((h, w), np.nan, dtype=np.float32)
    Y = np.full((h, w), np.nan, dtype=np.float32)
    Z = np.full((h, w), np.nan, dtype=np.float32)
    R = np.full((h, w), np.nan, dtype=np.float32)
    idx = np.flatnonzero(valid)
    X.flat[idx] = coords[:, 0].astype(np.float32)
    Y.flat[idx] = coords[:, 1].astype(np.float32)
    Z.flat[idx] = coords[:, 2].astype(np.float32)
    R.flat[idx] = residual.astype(np.float32)
    return X, Y, Z, R


def compute_full_projective(pose_dir: Path, cm: np.ndarray, pm: np.ndarray) -> dict[str, np.ndarray]:
    stack = load_pmp_stack(pose_dir)
    phase_y, ac_y, bc_y = compute_multifrequency_phase(stack, 0)
    phase_x, ac_x, bc_x = compute_multifrequency_phase(stack, 72)
    valid = (bc_y > 5.0) & (bc_x > 5.0)
    X, Y, Z, residual = solve_projective_4eq_0based(cm, pm, phase_x, phase_y, valid)
    valid = valid & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z) & np.isfinite(residual)
    return {
        "phase_y_capture": phase_y.astype(np.float32),
        "phase_x_capture": phase_x.astype(np.float32),
        "ac_y": ac_y.astype(np.float32),
        "ac_x": ac_x.astype(np.float32),
        "bc_y": bc_y.astype(np.float32),
        "bc_x": bc_x.astype(np.float32),
        "X": X.astype(np.float32),
        "Y": Y.astype(np.float32),
        "Z": Z.astype(np.float32),
        "residual": residual.astype(np.float32),
        "valid_mask": valid,
    }


def clean_foreground_from_z(
    Z: np.ndarray,
    valid: np.ndarray,
    z_delta: float,
    min_area: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    vals = Z[valid & np.isfinite(Z)]
    if vals.size == 0:
        empty = np.zeros_like(valid, dtype=bool)
        return empty, empty, {"foreground_z_threshold": float("nan"), "foreground_pixels": 0, "foreground_clean_pixels": 0}
    z_med = float(np.median(vals))
    threshold = z_med - z_delta
    raw = valid & np.isfinite(Z) & (Z < threshold)
    if int(raw.sum()) < min_area:
        threshold = float(np.percentile(vals, 15))
        raw = valid & np.isfinite(Z) & (Z < threshold)

    clean = ndimage.binary_opening(raw, iterations=1)
    clean = ndimage.binary_closing(clean, iterations=2)
    labels, nlab = ndimage.label(clean)
    if nlab > 0:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        keep = sizes >= min_area
        clean = keep[labels]
        labels, nlab = ndimage.label(clean)
        if nlab > 0:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            # Keep the largest few components. This preserves supports/bases without swallowing the whole wall.
            keep_labels = np.argsort(sizes)[-4:]
            clean = np.isin(labels, keep_labels) & clean
    meta = {
        "foreground_z_threshold": float(threshold),
        "foreground_pixels": int(raw.sum()),
        "foreground_clean_pixels": int(clean.sum()),
        "z_median": z_med,
    }
    return raw.astype(bool), clean.astype(bool), meta


def robust_limits(arr: np.ndarray, mask: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple[float, float]:
    vals = arr[mask & np.isfinite(arr)]
    if vals.size == 0:
        return 0.0, 1.0
    a, b = np.percentile(vals, [lo, hi])
    if not np.isfinite(a) or not np.isfinite(b) or abs(float(b - a)) < 1e-8:
        a, b = float(np.nanmin(vals)), float(np.nanmax(vals))
    if abs(float(b - a)) < 1e-8:
        b = a + 1.0
    return float(a), float(b)


def write_ply(path: Path, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, mask: np.ndarray, max_points: int = 180_000) -> int:
    idx = np.flatnonzero(mask & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z))
    if idx.size > max_points:
        rng = np.random.default_rng(20260612)
        idx = rng.choice(idx, size=max_points, replace=False)
    x = X.ravel()[idx].astype(np.float32)
    y = Y.ravel()[idx].astype(np.float32)
    z = Z.ravel()[idx].astype(np.float32)
    zlo, zhi = np.percentile(z, [2, 98]) if z.size else (0.0, 1.0)
    t = np.clip((z - zlo) / (zhi - zlo + 1e-9), 0.0, 1.0)
    rgb = (plt.get_cmap("turbo")(t)[:, :3] * 255).astype(np.uint8)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {x.size}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for xi, yi, zi, (r, g, b) in zip(x, y, z, rgb):
            f.write(f"{xi:.6f} {yi:.6f} {zi:.6f} {int(r)} {int(g)} {int(b)}\n")
    return int(x.size)


def set_equal_3d(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> None:
    mins = np.array([np.nanmin(xs), np.nanmin(ys), np.nanmin(zs)])
    maxs = np.array([np.nanmax(xs), np.nanmax(ys), np.nanmax(zs)])
    center = (mins + maxs) / 2.0
    radius = float(np.nanmax(maxs - mins) / 2.0)
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_preview(out_dir: Path, sid: str, rec: dict[str, np.ndarray], single: np.ndarray, raw_fg: np.ndarray, clean_fg: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    X, Y, Z = rec["X"], rec["Y"], rec["Z"]
    valid = rec["valid_mask"].astype(bool)
    residual = rec["residual"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].imshow(single, cmap="gray")
    axes[0, 0].set_title("single input pmp/0048")
    axes[0, 1].imshow(rec["bc_y"], cmap="magma")
    axes[0, 1].set_title("Bc-Y modulation")
    vmin, vmax = robust_limits(Z, valid)
    im = axes[0, 2].imshow(np.where(valid, Z, np.nan), cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 2].set_title("absolute Z, full valid")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)
    overlay = np.dstack([single, single, single]).astype(np.float32) / 255.0
    overlay[raw_fg] = overlay[raw_fg] * 0.45 + np.array([1.0, 0.2, 0.1]) * 0.55
    overlay[clean_fg] = overlay[clean_fg] * 0.35 + np.array([0.0, 0.85, 1.0]) * 0.65
    axes[1, 0].imshow(np.clip(overlay, 0, 1))
    axes[1, 0].set_title("foreground mask: red raw, cyan clean")
    rmin, rmax = robust_limits(residual, valid)
    im = axes[1, 1].imshow(np.where(valid, residual, np.nan), cmap="inferno", vmin=rmin, vmax=rmax)
    axes[1, 1].set_title("4-eq residual")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
    im = axes[1, 2].imshow(np.where(clean_fg, Z, np.nan), cmap="turbo", vmin=vmin, vmax=vmax)
    axes[1, 2].set_title("foreground Z")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(sid)
    fig.savefig(out_dir / f"{sid}_maps.png", dpi=160)
    plt.close(fig)

    idx = np.flatnonzero(clean_fg & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z))
    if idx.size == 0:
        idx = np.flatnonzero(valid & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z))
    if idx.size > 60_000:
        rng = np.random.default_rng(7)
        idx = rng.choice(idx, size=60_000, replace=False)
    xs, ys, zs = X.ravel()[idx], Y.ravel()[idx], Z.ravel()[idx]
    fig = plt.figure(figsize=(6.5, 5.6), dpi=170)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, c=zs, s=0.25, cmap="turbo", alpha=0.9, linewidths=0)
    set_equal_3d(ax, xs, ys, zs)
    ax.view_init(elev=22, azim=-63)
    ax.set_title(f"{sid} foreground point cloud")
    fig.tight_layout()
    fig.savefig(out_dir / f"{sid}_pointcloud.png")
    plt.close(fig)


def row_stats(arr: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    vals = arr[mask & np.isfinite(arr)]
    if vals.size == 0:
        return {"p02": None, "p50": None, "p98": None}
    q = np.percentile(vals, [2, 50, 98])
    return {"p02": float(q[0]), "p50": float(q[1]), "p98": float(q[2])}


def parse_sample_key(text: str) -> tuple[int, int]:
    if ":" in text:
        a, b = text.split(":", 1)
        return int(a), int(b.lower().replace("pose", ""))
    if "_pose" in text:
        a, b = text.split("_pose", 1)
        return int(a.replace("obj", "")), int(b)
    raise ValueError(text)


def process_one_item(task: tuple[SessionSpec, str, str, Path, Path, str, str, float, int, tuple[tuple[int, int], ...]]) -> tuple[bool, dict[str, object]]:
    spec, obj, pose, pose_dir, root, out_root_s, npz_compression, z_foreground_delta, min_foreground_area, preview_key_tuple = task
    out = Path(out_root_s)
    processed = out / "processed" / "projective_nowall_v1"
    preview_dir = out / "preview"
    ply_dir = out / "ply"
    preview_keys = set(preview_key_tuple)
    sid = sample_id(obj, pose)
    out_npz = processed / f"{sid}.npz"
    try:
        cm, pm, _ = parse_pmp_calibration(spec.calibration)
        if spec.session_id == "session0612_projective":
            rec = compute_full_projective(pose_dir, cm, pm)
        else:
            rec = compute_pmp_orderfix(root, obj, pose, cm, pm, phase_order=spec.phase_order)
            rec["residual"] = np.full_like(rec["Z"], np.nan, dtype=np.float32)

        single = read_gray(pose_dir / "pmp" / f"{SINGLE_INPUT_INDEX:04d}.bmp").astype(np.uint8)
        raw_fg, clean_fg, fg_meta = clean_foreground_from_z(
            rec["Z"],
            rec["valid_mask"].astype(bool),
            z_delta=z_foreground_delta,
            min_area=min_foreground_area,
        )
        valid = rec["valid_mask"].astype(bool)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        save_npz = np.savez_compressed if npz_compression == "compressed" else np.savez
        save_npz(
            out_npz,
            single_input=single,
            phase_y_capture=rec["phase_y_capture"],
            phase_x_capture=rec["phase_x_capture"],
            ac_y=rec["ac_y"],
            ac_x=rec["ac_x"],
            bc_y=rec["bc_y"],
            bc_x=rec["bc_x"],
            X=rec["X"],
            Y=rec["Y"],
            Z=rec["Z"],
            depth_z=rec["Z"],
            reconstruction_residual=rec["residual"],
            valid_mask=valid.astype(np.uint8),
            foreground_mask_z=raw_fg.astype(np.uint8),
            object_mask_clean_v2=clean_fg.astype(np.uint8),
        )

        split = split_for_object(obj)
        z_valid = row_stats(rec["Z"], valid)
        z_fg = row_stats(rec["Z"], clean_fg)
        residual_stats = row_stats(rec["residual"], valid)
        row: dict[str, object] = {
            "sample_id": sid,
            "object_id": obj,
            "pose": pose,
            "split": split,
            "session_id": spec.session_id,
            "calibration_model": spec.calibration_model,
            "phase_order": spec.phase_order,
            "pixel_origin": spec.pixel_origin,
            "source_root": str(root),
            "pose_dir": str(pose_dir),
            "npz": str(out_npz),
            "single_input_raw": str(pose_dir / "pmp" / f"{SINGLE_INPUT_INDEX:04d}.bmp"),
            "valid_pixels": int(valid.sum()),
            "foreground_pixels": int(raw_fg.sum()),
            "foreground_clean_pixels": int(clean_fg.sum()),
            "z_valid_p02": z_valid["p02"],
            "z_valid_p50": z_valid["p50"],
            "z_valid_p98": z_valid["p98"],
            "z_fg_p02": z_fg["p02"],
            "z_fg_p50": z_fg["p50"],
            "z_fg_p98": z_fg["p98"],
            "residual_p50": residual_stats["p50"],
            "residual_p98": residual_stats["p98"],
            "single_input_mean": image_stats(single)["mean"],
            "single_input_std": image_stats(single)["std"],
            **fg_meta,
        }
        if (int(obj), int(pose.replace("pose", ""))) in preview_keys:
            sample_preview = preview_dir / sid
            save_preview(sample_preview, sid, rec, single, raw_fg, clean_fg)
            ply_dir.mkdir(parents=True, exist_ok=True)
            write_ply(ply_dir / f"{sid}_valid.ply", rec["X"], rec["Y"], rec["Z"], valid)
            write_ply(ply_dir / f"{sid}_foreground_clean.ply", rec["X"], rec["Y"], rec["Z"], clean_fg)
        return True, row
    except Exception as exc:
        return False, {"sample_id": sid, "pose_dir": str(pose_dir), "error": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, default=Path(r"H:\yjs\实验室\sdxx\data_my"))
    parser.add_argument("--new-root-a", type=Path, default=Path(r"H:\yjs\实验室\sdxx\data_my"))
    parser.add_argument("--new-root-b", type=Path, default=Path(r"I:\cyf"))
    parser.add_argument("--old-calibration", type=Path, default=Path(r"H:\yjs\实验室\sdxx\data_my\calibrate_0610.pmp"))
    parser.add_argument("--new-calibration", type=Path, default=Path(r"H:\yjs\实验室\sdxx\data_my\calibrate_0612.cp"))
    parser.add_argument("--out-root", type=Path, default=Path(r"I:\cyf\my_fpp_dataset_projective_nowall_v1"))
    parser.add_argument("--npz-compression", choices=["compressed", "stored"], default="compressed")
    parser.add_argument("--z-foreground-delta", type=float, default=4.0)
    parser.add_argument("--min-foreground-area", type=int, default=500)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--only-samples",
        nargs="+",
        default=None,
        help="Optional sample list like 3:pose5 28:pose1. When set, only these samples are processed.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--preview-samples", nargs="+", default=["3:pose5", "7:pose5", "11:pose1", "12:pose1", "28:pose1", "40:pose1", "50:pose2", "61:pose2", "64:pose2"])
    args = parser.parse_args()

    out = args.out_root
    processed = out / "processed" / "projective_nowall_v1"
    preview_dir = out / "preview"
    ply_dir = out / "ply"
    splits_dir = out / "splits"
    for d in [processed, preview_dir, ply_dir, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)

    specs = [
        SessionSpec(
            "session0610_legacy",
            1,
            12,
            (args.old_root,),
            args.old_calibration,
            "legacy_0610_projector_phase_matrix_compat",
            "yx",
            "legacy builder convention",
        ),
        SessionSpec(
            "session0612_projective",
            13,
            64,
            (args.new_root_a, args.new_root_b),
            args.new_calibration,
            "full_3x4_projector_phase_coordinate_matrix",
            "xy",
            "0-based camera pixels",
        ),
    ]
    all_items: list[tuple[SessionSpec, str, str, Path, Path]] = []
    for spec in specs:
        for obj, pose, pose_dir, root in discover_samples(spec):
            all_items.append((spec, obj, pose, pose_dir, root))
    all_items.sort(key=lambda x: (int(x[1]), numeric_key(x[2].replace("pose", ""))))
    if args.only_samples:
        requested = {parse_sample_key(s) for s in args.only_samples}
        all_items = [
            item
            for item in all_items
            if (int(item[1]), int(item[2].replace("pose", ""))) in requested
        ]
    if args.max_samples is not None:
        all_items = all_items[: args.max_samples]

    preview_keys = {parse_sample_key(s) for s in args.preview_samples}
    split_rows: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    tasks = [
        (spec, obj, pose, pose_dir, root, str(out), args.npz_compression, args.z_foreground_delta, args.min_foreground_area, tuple(sorted(preview_keys)))
        for spec, obj, pose, pose_dir, root in all_items
    ]
    if args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            iterator = executor.map(process_one_item, tasks)
            for idx, (ok, payload) in enumerate(iterator, 1):
                if ok:
                    rows.append(payload)
                    split_rows[str(payload["split"])].append(str(payload["npz"]))
                    print(f"[{idx}/{len(tasks)}] wrote {payload['sample_id']} {payload['session_id']} valid={payload['valid_pixels']} fg={payload['foreground_clean_pixels']}")
                else:
                    skipped.append(payload)
                    print(f"[{idx}/{len(tasks)}] skip {payload['sample_id']}: {payload['error']}")
    else:
        for idx, task in enumerate(tasks, 1):
            ok, payload = process_one_item(task)
            if ok:
                rows.append(payload)
                split_rows[str(payload["split"])].append(str(payload["npz"]))
                print(f"[{idx}/{len(tasks)}] wrote {payload['sample_id']} {payload['session_id']} valid={payload['valid_pixels']} fg={payload['foreground_clean_pixels']}")
            else:
                skipped.append(payload)
                print(f"[{idx}/{len(tasks)}] skip {payload['sample_id']}: {payload['error']}")

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with (out / "samples.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for split, paths in split_rows.items():
        (splits_dir / f"{split}.txt").write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")
    (out / "skipped.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "dataset": "my_fpp_dataset_projective_nowall_v1",
        "output_root": str(out),
        "sample_count": len(rows),
        "skipped_count": len(skipped),
        "split_counts": {k: len(v) for k, v in split_rows.items()},
        "sessions": [
            {
                "session_id": s.session_id,
                "object_range": [s.min_object, s.max_object],
                "roots": [str(r) for r in s.roots],
                "calibration": str(s.calibration),
                "calibration_model": s.calibration_model,
                "phase_order": s.phase_order,
                "pixel_origin": s.pixel_origin,
            }
            for s in specs
        ],
        "npz_fields": [
            "single_input",
            "phase_y_capture",
            "phase_x_capture",
            "ac_y",
            "ac_x",
            "bc_y",
            "bc_x",
            "X",
            "Y",
            "Z",
            "depth_z",
            "reconstruction_residual",
            "valid_mask",
            "foreground_mask_z",
            "object_mask_clean_v2",
        ],
        "notes": [
            "No wall reference is used in this dataset.",
            "Objects 13-64 use the 0612 full ProjectorProjectiveMatrix-style calibration with 0-based camera pixels.",
            "Objects 1-12 use the available 0610 legacy projector phase matrix because no 0610 full 3x4 projector-coordinate matrix was found.",
            "object_mask_clean_v2 is a coarse Z-foreground mask for visualization/training selection, not a calibrated wall-difference mask.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# My FPP Projective No-Wall Dataset v1\n\n"
        "This dataset reconstructs absolute XYZ/depth from PMP without using a reference wall.\n\n"
        "Old samples `1-12` use `calibrate_0610.pmp` through the legacy compatibility reconstruction.\n"
        "New samples `13-64` use `calibrate_0612.cp` as a full 3x4 projector phase-coordinate matrix with 0-based camera pixels.\n\n"
        "Main training fields are `single_input`, `depth_z`/`Z`, `valid_mask`, and optionally `object_mask_clean_v2`.\n"
        "The foreground mask is intentionally coarse and should be inspected before object-only training.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
