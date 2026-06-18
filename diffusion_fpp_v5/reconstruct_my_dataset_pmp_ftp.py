from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import unwrap_phase


FREQUENCIES = [1, 4, 8, 16, 32, 64]
N_STEPS = 12
TWO_PI = 2.0 * np.pi


def parse_pmp_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    values: dict[str, float] = {}
    meta: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("["):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            values[k] = float(v)
        except ValueError:
            meta[k] = v
    cm = np.array([[values[f"cm{i}{j}"] for j in range(1, 5)] for i in range(1, 4)], dtype=np.float64)
    projector_prefix = "pm" if "pm11" in values else "pcm"
    pm = np.array([[values[f"{projector_prefix}{i}{j}"] for j in range(1, 5)] for i in range(1, 4)], dtype=np.float64)
    meta["projector_matrix_prefix"] = projector_prefix
    return cm, pm, meta


def read_gray(path: Path) -> np.ndarray:
    # cv2.imread can fail on Windows paths containing non-ASCII characters.
    # np.fromfile + imdecode keeps the filesystem path handling in Python.
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(path)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32)


def load_pmp_stack(pose_dir: Path) -> np.ndarray:
    pmp_dir = pose_dir / "pmp"
    files = [pmp_dir / f"{i:04d}.bmp" for i in range(144)]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing PMP frames: {missing[:5]}")
    stack = np.stack([read_gray(p) for p in files], axis=0)
    return stack


def compute_multifrequency_phase(stack: np.ndarray, start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase_main = None
    ac = None
    bc = None
    for j, freq in enumerate(FREQUENCIES):
        frames = stack[start + j * N_STEPS : start + (j + 1) * N_STEPS].astype(np.float64)
        n = np.arange(N_STEPS, dtype=np.float64)[:, None, None]
        sin_sum = np.sum(frames * np.sin(TWO_PI * n / N_STEPS), axis=0)
        cos_sum = np.sum(frames * np.cos(TWO_PI * n / N_STEPS), axis=0)
        phase_h = np.arctan2(-sin_sum, -cos_sum) + np.pi
        if j == 0:
            phase_main = phase_h
            ac = np.mean(frames, axis=0)
            bc = 2.0 / N_STEPS * np.sqrt(sin_sum * sin_sum + cos_sum * cos_sum)
        else:
            k_phase = np.round((freq * phase_main - phase_h) / TWO_PI)
            phase_main = (phase_h + TWO_PI * k_phase) / freq
    assert phase_main is not None and ac is not None and bc is not None
    return phase_main.astype(np.float32), ac.astype(np.float32), bc.astype(np.float32)


def pixel_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices((h, w), dtype=np.float64)
    # The original MATLAB reconstruction used 1-based image coordinates.
    return xx + 1.0, yy + 1.0


def solve_projective(cm: np.ndarray, pm: np.ndarray, phase_x: np.ndarray, phase_y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = phase_y.shape
    xc, yc = pixel_grid(h, w)
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
    coords = np.full((valid.sum(), 3), np.nan, dtype=np.float64)
    ok = np.ones(valid.sum(), dtype=bool)
    try:
        coords = np.linalg.solve(ata, atb)
    except np.linalg.LinAlgError:
        ok[:] = False
        for i in range(valid.sum()):
            try:
                coords[i] = np.linalg.solve(ata[i], atb[i])
                ok[i] = True
            except np.linalg.LinAlgError:
                coords[i] = np.linalg.lstsq(rows[i], rhs[i], rcond=None)[0]
                ok[i] = np.all(np.isfinite(coords[i]))

    X = np.full((h, w), np.nan, dtype=np.float32)
    Y = np.full((h, w), np.nan, dtype=np.float32)
    Z = np.full((h, w), np.nan, dtype=np.float32)
    flat_idx = np.flatnonzero(valid)
    X.flat[flat_idx[ok]] = coords[ok, 0].astype(np.float32)
    Y.flat[flat_idx[ok]] = coords[ok, 1].astype(np.float32)
    Z.flat[flat_idx[ok]] = coords[ok, 2].astype(np.float32)
    return X, Y, Z


def solve_single_phase_y(cm: np.ndarray, pm: np.ndarray, phase_y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = phase_y.shape
    xc, yc = pixel_grid(h, w)
    valid = mask & np.isfinite(phase_y)
    x = xc[valid]
    y = yc[valid]
    yp = phase_y[valid].astype(np.float64)
    rows = np.stack(
        [
            np.stack([cm[0, 0] - cm[2, 0] * x, cm[0, 1] - cm[2, 1] * x, cm[0, 2] - cm[2, 2] * x], axis=1),
            np.stack([cm[1, 0] - cm[2, 0] * y, cm[1, 1] - cm[2, 1] * y, cm[1, 2] - cm[2, 2] * y], axis=1),
            np.stack([pm[1, 0] - pm[2, 0] * yp, pm[1, 1] - pm[2, 1] * yp, pm[1, 2] - pm[2, 2] * yp], axis=1),
        ],
        axis=1,
    )
    rhs = np.stack([x - cm[0, 3], y - cm[1, 3], yp - pm[1, 3]], axis=1)
    coords = np.linalg.solve(rows, rhs)
    X = np.full((h, w), np.nan, dtype=np.float32)
    Y = np.full((h, w), np.nan, dtype=np.float32)
    Z = np.full((h, w), np.nan, dtype=np.float32)
    flat_idx = np.flatnonzero(valid)
    X.flat[flat_idx] = coords[:, 0].astype(np.float32)
    Y.flat[flat_idx] = coords[:, 1].astype(np.float32)
    Z.flat[flat_idx] = coords[:, 2].astype(np.float32)
    return X, Y, Z


def robust_range(arr: np.ndarray, mask: np.ndarray | None = None, lo: float = 2.0, hi: float = 98.0) -> tuple[float, float]:
    vals = arr[np.isfinite(arr)]
    if mask is not None:
        vals = arr[mask & np.isfinite(arr)]
    if vals.size == 0:
        return 0.0, 1.0
    a, b = np.percentile(vals, [lo, hi])
    if not np.isfinite(a) or not np.isfinite(b) or abs(b - a) < 1e-9:
        a, b = float(np.nanmin(vals)), float(np.nanmax(vals))
    if abs(b - a) < 1e-9:
        b = a + 1.0
    return float(a), float(b)


def save_depth_png(path: Path, z: np.ndarray, mask: np.ndarray, title: str) -> None:
    vmin, vmax = robust_range(z, mask)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    im = ax.imshow(np.where(mask, z, np.nan), cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Z")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_compare_png(path: Path, pmp_z: np.ndarray, ftp_z: np.ndarray, rel_z: np.ndarray, mask: np.ndarray, title: str) -> None:
    vmin, vmax = robust_range(pmp_z, mask)
    rvmin, rvmax = robust_range(rel_z, mask)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    panels = [
        (pmp_z, "PMP Z", vmin, vmax, "viridis"),
        (ftp_z, "FTP Z", vmin, vmax, "viridis"),
        (rel_z, "FTP - wall Z", rvmin, rvmax, "coolwarm"),
    ]
    for ax, (arr, name, a, b, cmap) in zip(axes, panels):
        im = ax.imshow(np.where(mask, arr, np.nan), cmap=cmap, vmin=a, vmax=b)
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_ply(path: Path, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, mask: np.ndarray, stride: int = 2) -> int:
    sel = mask & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    sel[::stride, ::stride] = sel[::stride, ::stride]
    sparse = np.zeros_like(sel)
    sparse[::stride, ::stride] = sel[::stride, ::stride]
    pts = np.stack([X[sparse], Y[sparse], Z[sparse]], axis=1)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    return int(pts.shape[0])


def fft_band_phase(
    img: np.ndarray,
    expected_freq: int = 32,
    sigma: float = 5.0,
    forced_peak: tuple[int, int] | None = None,
    axis: str = "y",
) -> tuple[np.ndarray, np.ndarray, dict]:
    h, w = img.shape
    centered = img.astype(np.float64) - np.mean(img)
    win_y = np.hanning(h)[:, None]
    win_x = np.hanning(w)[None, :]
    F = np.fft.fftshift(np.fft.fft2(centered * win_y * win_x))
    mag = np.abs(F)
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    if forced_peak is None:
        if axis == "y":
            # Horizontal fringes: phase changes along y. Search near +/- expected vertical carrier.
            search = (np.abs(xx - cx) <= 24) & (np.abs(np.abs(yy - cy) - expected_freq) <= 12)
        elif axis == "x":
            # Vertical fringes: phase changes along x. Search near +/- expected horizontal carrier.
            search = (np.abs(yy - cy) <= 24) & (np.abs(np.abs(xx - cx) - expected_freq) <= 12)
        else:
            raise ValueError(f"Unsupported FTP axis: {axis}")
        search[cy - 3 : cy + 4, cx - 3 : cx + 4] = False
        peak_flat = np.argmax(np.where(search, mag, 0.0))
        py, px = np.unravel_index(peak_flat, mag.shape)
    else:
        py, px = forced_peak
    gaussian = np.exp(-(((yy - py) ** 2 + (xx - px) ** 2) / (2.0 * sigma * sigma)))
    comp = np.fft.ifft2(np.fft.ifftshift(F * gaussian))
    phase = np.angle(comp).astype(np.float32)
    amp = np.abs(comp).astype(np.float32)
    info = {"axis": axis, "peak_y": int(py), "peak_x": int(px), "peak_dy": int(py - cy), "peak_dx": int(px - cx), "sigma": sigma}
    return phase, amp, info


def compute_ftp_phase_axis(
    ref_single: np.ndarray,
    obj_single: np.ndarray,
    ref_pmp_phase: np.ndarray,
    ref_mask: np.ndarray,
    obj_mask: np.ndarray,
    obj_pmp_phase: np.ndarray | None = None,
    frequency: int = 32,
    axis: str = "y",
) -> tuple[np.ndarray, np.ndarray, dict]:
    ref_phase, ref_amp, ref_info = fft_band_phase(ref_single, expected_freq=frequency, axis=axis)
    # Use the same Fourier sideband as the wall reference. Letting each object
    # choose its own maximum can flip to the conjugate sideband, which destroys
    # the reference-relative FTP phase.
    obj_phase, obj_amp, obj_info = fft_band_phase(
        obj_single,
        expected_freq=frequency,
        forced_peak=(ref_info["peak_y"], ref_info["peak_x"]),
        axis=axis,
    )
    delta_wrapped = np.angle(np.exp(1j * (obj_phase - ref_phase))).astype(np.float32)
    delta_unwrapped = unwrap_phase(delta_wrapped).astype(np.float32)
    phase_plus = ref_pmp_phase + delta_unwrapped / float(frequency)
    phase_minus = ref_pmp_phase - delta_unwrapped / float(frequency)
    select_sign = "plus"
    diagnostic = {}
    if obj_pmp_phase is not None:
        diag_mask = obj_mask & ref_mask & np.isfinite(obj_pmp_phase) & np.isfinite(phase_plus) & np.isfinite(phase_minus)
        if diag_mask.sum() > 1000:
            rmse_plus = float(np.sqrt(np.mean((phase_plus[diag_mask] - obj_pmp_phase[diag_mask]) ** 2)))
            rmse_minus = float(np.sqrt(np.mean((phase_minus[diag_mask] - obj_pmp_phase[diag_mask]) ** 2)))
            diagnostic = {
                "phase_rmse_plus_to_pmp_y": rmse_plus,
                "phase_rmse_minus_to_pmp_y": rmse_minus,
                "diagnostic_pixels": int(diag_mask.sum()),
            }
            select_sign = "plus" if rmse_plus <= rmse_minus else "minus"
    ftp_phase_y = phase_plus if select_sign == "plus" else phase_minus
    amp_mask = (ref_amp > np.percentile(ref_amp[ref_mask], 25)) & (obj_amp > np.percentile(obj_amp[obj_mask], 25))
    ftp_mask = obj_mask & ref_mask & amp_mask & np.isfinite(ftp_phase_y)
    info = {
        "ref_peak": ref_info,
        "obj_peak": obj_info,
        "selected_delta_sign": select_sign,
        "diagnostic": diagnostic,
        "ftp_valid_pixels": int(ftp_mask.sum()),
        "note": "FTP uses object and wall f=32 step0 single frames; wall PMP phase anchors absolute projector phase. Both +/- delta/f signs are evaluated; selection uses PMP phase only as a diagnostic for this capture test.",
    }
    return ftp_phase_y.astype(np.float32), ftp_mask, info


def compute_ftp_phase_y(
    ref_single: np.ndarray,
    obj_single: np.ndarray,
    ref_pmp_phase_y: np.ndarray,
    ref_mask: np.ndarray,
    obj_mask: np.ndarray,
    obj_pmp_phase_y: np.ndarray | None = None,
    frequency: int = 32,
) -> tuple[np.ndarray, np.ndarray, dict]:
    return compute_ftp_phase_axis(
        ref_single,
        obj_single,
        ref_pmp_phase_y,
        ref_mask,
        obj_mask,
        obj_pmp_phase_y,
        frequency=frequency,
        axis="y",
    )


def summarize_z(z: np.ndarray, mask: np.ndarray) -> dict:
    vals = z[mask & np.isfinite(z)]
    if vals.size == 0:
        return {"valid_pixels": 0}
    return {
        "valid_pixels": int(vals.size),
        "z_mean": float(np.mean(vals)),
        "z_median": float(np.median(vals)),
        "z_std": float(np.std(vals)),
        "z_p02": float(np.percentile(vals, 2)),
        "z_p98": float(np.percentile(vals, 98)),
    }


def process_pose(obj_id: str, data_root: Path, out_root: Path, cm: np.ndarray, pm: np.ndarray) -> dict:
    pose_dir = data_root / obj_id / "pose1"
    out_dir = out_root / obj_id / "pose1"
    out_dir.mkdir(parents=True, exist_ok=True)
    stack = load_pmp_stack(pose_dir)
    phase_y, ac_y, bc_y = compute_multifrequency_phase(stack, 0)
    phase_x, ac_x, bc_x = compute_multifrequency_phase(stack, 72)
    mask = (bc_y > 5.0) & (bc_x > 5.0)
    X, Y, Z = solve_projective(cm, pm, phase_x, phase_y, mask)
    z_good = np.isfinite(Z) & (np.abs(Z) < np.nanpercentile(np.abs(Z[np.isfinite(Z)]), 99.5))
    mask = mask & z_good
    np.savez_compressed(
        out_dir / "pmp_reconstruction.npz",
        phase_x=phase_x,
        phase_y=phase_y,
        ac_y=ac_y,
        bc_y=bc_y,
        bc_x=bc_x,
        mask=mask,
        X=X,
        Y=Y,
        Z=Z,
    )
    save_depth_png(out_dir / "pmp_depth_Z.png", Z, mask, f"Object {obj_id} PMP absolute Z")
    ply_n = write_ply(out_dir / "pmp_pointcloud_stride2.ply", X, Y, Z, mask, stride=2)
    summary = {
        "object_id": obj_id,
        "pose": "pose1",
        "method": "PMP 6-frequency x 12-step, X/Y double-direction reconstruction",
        "pmp": summarize_z(Z, mask),
        "pmp_pointcloud_vertices_stride2": ply_n,
        "phase_y_range_valid": [float(np.nanpercentile(phase_y[mask], 2)), float(np.nanpercentile(phase_y[mask], 98))] if mask.any() else None,
        "phase_x_range_valid": [float(np.nanpercentile(phase_x[mask], 2)), float(np.nanpercentile(phase_x[mask], 98))] if mask.any() else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def add_relative_to_wall(out_root: Path, obj_id: str, wall_z: np.ndarray, wall_mask: np.ndarray) -> None:
    out_dir = out_root / obj_id / "pose1"
    data = np.load(out_dir / "pmp_reconstruction.npz")
    z = data["Z"]
    mask = data["mask"] & wall_mask & np.isfinite(wall_z)
    rel = z - wall_z
    np.save(out_dir / "pmp_relative_Z_minus_wall.npy", rel.astype(np.float32))
    save_depth_png(out_dir / "pmp_relative_Z_minus_wall.png", rel, mask, f"Object {obj_id} PMP Z - wall Z")


def process_ftp(obj_id: str, data_root: Path, out_root: Path, cm: np.ndarray, pm: np.ndarray, wall_npz: np.lib.npyio.NpzFile) -> dict:
    out_dir = out_root / obj_id / "pose1"
    out_dir.mkdir(parents=True, exist_ok=True)
    wall_single = read_gray(data_root / "0" / "pose1" / "pmp" / "0048.bmp")
    obj_single = read_gray(data_root / obj_id / "pose1" / "pmp" / "0048.bmp")
    obj_npz = np.load(out_dir / "pmp_reconstruction.npz")
    ftp_phase_y, ftp_mask, ftp_info = compute_ftp_phase_y(
        wall_single,
        obj_single,
        wall_npz["phase_y"],
        wall_npz["mask"],
        obj_npz["mask"],
        obj_npz["phase_y"],
        frequency=32,
    )
    X, Y, Z = solve_single_phase_y(cm, pm, ftp_phase_y, ftp_mask)
    wall_z = wall_npz["Z"]
    rel = Z - wall_z
    good = ftp_mask & np.isfinite(Z) & np.isfinite(wall_z)
    if np.any(good):
        lo, hi = np.nanpercentile(Z[good], [0.5, 99.5])
        good = good & (Z >= lo) & (Z <= hi)
    np.savez_compressed(out_dir / "ftp_reconstruction_from_wall0048.npz", phase_y=ftp_phase_y, mask=good, X=X, Y=Y, Z=Z, relative_Z_minus_wall=rel)
    save_depth_png(out_dir / "ftp_depth_Z.png", Z, good, f"Object {obj_id} FTP absolute Z")
    save_depth_png(out_dir / "ftp_relative_Z_minus_wall.png", rel, good, f"Object {obj_id} FTP Z - wall Z")
    save_compare_png(out_dir / "ftp_vs_pmp_depth_compare.png", obj_npz["Z"], Z, rel, good & obj_npz["mask"], f"Object {obj_id}: PMP vs FTP")
    ply_n = write_ply(out_dir / "ftp_pointcloud_stride2.ply", X, Y, Z, good, stride=2)
    summary = {
        "object_id": obj_id,
        "pose": "pose1",
        "method": "FTP single-frame reconstruction with object pmp/0048.bmp and wall 0/pmp/0048.bmp reference",
        "boundary": "Uses wall PMP phase only to anchor absolute projector phase for calibrated 3D; object uses single-frame FTP phase.",
        "ftp": summarize_z(Z, good),
        "ftp_relative_to_wall": summarize_z(rel, good),
        "ftp_pointcloud_vertices_stride2": ply_n,
        "ftp_info": ftp_info,
    }
    (out_dir / "ftp_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def process_ftp_xy(obj_id: str, data_root: Path, out_root: Path, cm: np.ndarray, pm: np.ndarray, wall_npz: np.lib.npyio.NpzFile) -> dict:
    out_dir = out_root / obj_id / "pose1"
    obj_npz = np.load(out_dir / "pmp_reconstruction.npz")
    wall_y_single = read_gray(data_root / "0" / "pose1" / "pmp" / "0048.bmp")
    obj_y_single = read_gray(data_root / obj_id / "pose1" / "pmp" / "0048.bmp")
    wall_x_single = read_gray(data_root / "0" / "pose1" / "pmp" / "0120.bmp")
    obj_x_single = read_gray(data_root / obj_id / "pose1" / "pmp" / "0120.bmp")

    ftp_phase_y, mask_y, info_y = compute_ftp_phase_axis(
        wall_y_single,
        obj_y_single,
        wall_npz["phase_y"],
        wall_npz["mask"],
        obj_npz["mask"],
        obj_npz["phase_y"],
        frequency=32,
        axis="y",
    )
    ftp_phase_x, mask_x, info_x = compute_ftp_phase_axis(
        wall_x_single,
        obj_x_single,
        wall_npz["phase_x"],
        wall_npz["mask"],
        obj_npz["mask"],
        obj_npz["phase_x"],
        frequency=32,
        axis="x",
    )
    mask = mask_y & mask_x & wall_npz["mask"] & obj_npz["mask"]
    X, Y, Z = solve_projective(cm, pm, ftp_phase_x, ftp_phase_y, mask)
    wall_z = wall_npz["Z"]
    rel = Z - wall_z
    good = mask & np.isfinite(Z) & np.isfinite(obj_npz["Z"])
    if np.any(good):
        err = Z[good] - obj_npz["Z"][good]
        z_rmse_to_pmp = float(np.sqrt(np.mean(err * err)))
        rel_err = rel[good] - (obj_npz["Z"][good] - wall_z[good])
        rel_rmse_to_pmp = float(np.sqrt(np.mean(rel_err * rel_err)))
    else:
        z_rmse_to_pmp = None
        rel_rmse_to_pmp = None

    np.savez_compressed(
        out_dir / "ftp_xy_reconstruction_from_wall_f32_step0.npz",
        phase_x=ftp_phase_x,
        phase_y=ftp_phase_y,
        mask=good,
        X=X,
        Y=Y,
        Z=Z,
        relative_Z_minus_wall=rel,
    )
    save_depth_png(out_dir / "ftp_xy_depth_Z.png", Z, good, f"Object {obj_id} FTP-XY absolute Z")
    save_depth_png(out_dir / "ftp_xy_relative_Z_minus_wall.png", rel, good, f"Object {obj_id} FTP-XY Z - wall Z")
    save_compare_png(out_dir / "ftp_xy_vs_pmp_depth_compare.png", obj_npz["Z"], Z, rel, good, f"Object {obj_id}: PMP vs FTP-XY")
    ply_n = write_ply(out_dir / "ftp_xy_pointcloud_stride2.ply", X, Y, Z, good, stride=2)
    summary = {
        "object_id": obj_id,
        "pose": "pose1",
        "method": "Two-direction FTP using f=32 step0 horizontal frame 0048 and vertical frame 0120 with wall reference",
        "boundary": "This is a two-frame FTP diagnostic, not a strict one-frame method. Sign selection uses PMP phase only to verify this capture test.",
        "ftp_xy": summarize_z(Z, good),
        "ftp_xy_relative_to_wall": summarize_z(rel, good),
        "rmse_to_pmp_Z": z_rmse_to_pmp,
        "rmse_to_pmp_relative_Z": rel_rmse_to_pmp,
        "ftp_xy_pointcloud_vertices_stride2": ply_n,
        "phase_y_info": info_y,
        "phase_x_info": info_x,
    }
    (out_dir / "ftp_xy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def make_overview(out_root: Path, object_ids: list[str]) -> None:
    panels = []
    titles = []
    for obj_id in object_ids:
        out_dir = out_root / obj_id / "pose1"
        pmp = np.load(out_dir / "pmp_reconstruction.npz")
        panels.append((pmp["Z"], pmp["mask"]))
        titles.append(f"{obj_id} PMP Z")
        ftp_path = out_dir / "ftp_reconstruction_from_wall0048.npz"
        if ftp_path.exists():
            ftp = np.load(ftp_path)
            panels.append((ftp["Z"], ftp["mask"]))
            titles.append(f"{obj_id} FTP-Y Z")
        ftp_xy_path = out_dir / "ftp_xy_reconstruction_from_wall_f32_step0.npz"
        if ftp_xy_path.exists():
            ftp_xy = np.load(ftp_xy_path)
            panels.append((ftp_xy["Z"], ftp_xy["mask"]))
            titles.append(f"{obj_id} FTP-XY Z")
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    if len(panels) == 1:
        axes = [axes]
    for ax, (arr, mask), title in zip(axes, panels, titles):
        vmin, vmax = robust_range(arr, mask)
        im = ax.imshow(np.where(mask, arr, np.nan), cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_root / "overview_depths.png", dpi=160)
    plt.close(fig)


def make_ftp_xy_compact(out_root: Path, object_ids: list[str]) -> None:
    fig, axes = plt.subplots(len(object_ids), 3, figsize=(10.5, 3.8 * len(object_ids)))
    if len(object_ids) == 1:
        axes = axes[None, :]
    for row, obj_id in enumerate(object_ids):
        out_dir = out_root / obj_id / "pose1"
        pmp = np.load(out_dir / "pmp_reconstruction.npz")
        ftp = np.load(out_dir / "ftp_xy_reconstruction_from_wall_f32_step0.npz")
        mask = pmp["mask"] & ftp["mask"] & np.isfinite(pmp["Z"]) & np.isfinite(ftp["Z"])
        err = np.abs(ftp["Z"] - pmp["Z"])
        vmin, vmax = robust_range(pmp["Z"], mask)
        panels = [
            (pmp["Z"], f"{obj_id} PMP Z", vmin, vmax, "viridis"),
            (ftp["Z"], f"{obj_id} FTP-XY Z", vmin, vmax, "viridis"),
            (err, f"{obj_id} |FTP-XY - PMP|", 0.0, float(np.nanpercentile(err[mask], 98)) if mask.any() else 1.0, "magma"),
        ]
        for col, (arr, title, a, b, cmap) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(np.where(mask, arr, np.nan), cmap=cmap, vmin=a, vmax=b)
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_root / "ftp_xy_vs_pmp_compact.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data_my"))
    parser.add_argument("--calibration", type=Path, default=Path("data_my") / "calibrate_0609.pmp")
    parser.add_argument("--out-root", type=Path, default=Path("cloud_results") / "A_20260609_data_my_reconstruction_test")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    cm, pm, cal_meta = parse_pmp_calibration(args.calibration)
    (args.out_root / "calibration_parsed.json").write_text(
        json.dumps({"cm": cm.tolist(), "pm": pm.tolist(), "meta": cal_meta}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    all_summaries = {"data_root": str(args.data_root), "calibration": str(args.calibration), "objects": {}, "ftp_y": {}, "ftp_xy": {}}
    for obj_id in ["0", "1", "2"]:
        all_summaries["objects"][obj_id] = process_pose(obj_id, args.data_root, args.out_root, cm, pm)

    wall_npz = np.load(args.out_root / "0" / "pose1" / "pmp_reconstruction.npz")
    for obj_id in ["1", "2"]:
        add_relative_to_wall(args.out_root, obj_id, wall_npz["Z"], wall_npz["mask"])
        all_summaries["ftp_y"][obj_id] = process_ftp(obj_id, args.data_root, args.out_root, cm, pm, wall_npz)
        all_summaries["ftp_xy"][obj_id] = process_ftp_xy(obj_id, args.data_root, args.out_root, cm, pm, wall_npz)

    make_overview(args.out_root, ["0", "1", "2"])
    make_ftp_xy_compact(args.out_root, ["1", "2"])
    (args.out_root / "reconstruction_summary.json").write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(all_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
