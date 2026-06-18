from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy import ndimage

    SCIPY_OK = True
except Exception:
    ndimage = None
    SCIPY_OK = False


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in str(text).replace(",", " ").split() if x.strip()]


def load_array(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r")


def norm_to_mm(depth_norm: np.ndarray, depth_minmax: np.ndarray) -> np.ndarray:
    depth01 = np.clip((np.asarray(depth_norm, dtype=np.float32) + 1.0) * 0.5, 0.0, 1.0)
    lo = float(depth_minmax[0])
    hi = float(depth_minmax[1])
    return depth01 * max(hi - lo, 1e-6) + lo


def masked_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool)
    return float(np.asarray(arr, dtype=np.float64)[valid].mean()) if np.any(valid) else float("nan")


def metric_pair(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    valid = mask.astype(bool)
    diff = np.asarray(pred, dtype=np.float64)[valid] - np.asarray(target, dtype=np.float64)[valid]
    if diff.size == 0:
        return float("nan"), float("nan")
    return float(np.sqrt(np.mean(diff * diff))), float(np.mean(np.abs(diff)))


def masked_img(arr: np.ndarray, mask: np.ndarray):
    return np.ma.masked_where(~mask.astype(bool), arr)


def set_mpl_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 7,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig, path_base: Path, dpi: int = 450):
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight")


def save_compact(fig, path_base: Path):
    save_figure(fig, path_base, dpi=300)


def rcpc_fuse(
    d_d_norm: np.ndarray,
    d_p_norm: np.ndarray,
    edge: np.ndarray,
    phase_conf: np.ndarray,
    mask: np.ndarray,
    edge_tau: float,
    delta_max: float,
    phase_conf_max: float,
    high_weight: float,
    low_weight: float,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    delta = np.abs(np.asarray(d_p_norm, dtype=np.float32) - np.asarray(d_d_norm, dtype=np.float32))
    edge_mean = masked_mean(edge, mask)
    delta_mean = masked_mean(delta, mask)
    conf_mean = masked_mean(phase_conf, mask)
    selected = edge_mean >= edge_tau and delta_mean <= delta_max and conf_mean <= phase_conf_max
    weight = float(high_weight if selected else low_weight)
    final = np.clip((1.0 - weight) * d_d_norm + weight * d_p_norm, -1.0, 1.0)
    return final, {
        "selected": bool(selected),
        "weight": weight,
        "edge_mean": edge_mean,
        "delta_mean_norm": delta_mean,
        "phase_conf_mean": conf_mean,
    }


class CacheView:
    def __init__(self, base_cache_dir: Path, phase_cache_dir: Path, candidate_cache_dir: Path, split: str):
        self.base_cache_dir = Path(base_cache_dir)
        self.phase_cache_dir = Path(phase_cache_dir)
        self.candidate_cache_dir = Path(candidate_cache_dir)
        self.split = split
        self.fringe_u8 = load_array(self.base_cache_dir / f"fringe_{split}_uint8.npy")
        self.phase_instr = load_array(self.phase_cache_dir / f"phase_instr_{split}_float16.npy")
        self.depth_mm = load_array(self.base_cache_dir / f"depth_mm_{split}_float32.npy")
        self.mask = load_array(self.base_cache_dir / f"mask_{split}_uint8.npy")
        self.depth_minmax = load_array(self.base_cache_dir / f"depth_minmax_{split}_float32.npy")
        self.d_b = load_array(self.candidate_cache_dir / f"d_b_{split}_float16.npy")
        self.d_p = load_array(self.candidate_cache_dir / f"d_p_{split}_float16.npy")
        self.d_d = load_array(self.candidate_cache_dir / f"d_d_{split}_float16.npy")
        self.edge = load_array(self.candidate_cache_dir / f"edge_{split}_float16.npy")
        self.phase_conf = load_array(self.candidate_cache_dir / f"phase_conf_{split}_float16.npy")
        self.sample_index = load_array(self.candidate_cache_dir / f"sample_index_{split}_int32.npy")
        self.id_to_pos = {int(self.sample_index[i]): i for i in range(len(self.sample_index))}

    def sample(self, sample_id: int, args: argparse.Namespace) -> dict[str, np.ndarray | dict]:
        pos = self.id_to_pos[int(sample_id)]
        mask = np.asarray(self.mask[pos, 0], dtype=bool)
        final_norm, rcpc_meta = rcpc_fuse(
            np.asarray(self.d_d[pos, 0], dtype=np.float32),
            np.asarray(self.d_p[pos, 0], dtype=np.float32),
            np.asarray(self.edge[pos, 0], dtype=np.float32),
            np.asarray(self.phase_conf[pos, 0], dtype=np.float32),
            mask,
            args.edge_tau,
            args.delta_max,
            args.phase_conf_max,
            args.high_weight,
            args.low_weight,
        )
        minmax = np.asarray(self.depth_minmax[pos], dtype=np.float32)
        return {
            "pos": pos,
            "sample": int(sample_id),
            "fringe": np.asarray(self.fringe_u8[pos, 0], dtype=np.float32) / 255.0,
            "instr": np.asarray(self.phase_instr[pos], dtype=np.float32),
            "target": np.asarray(self.depth_mm[pos, 0], dtype=np.float32),
            "mask": mask,
            "depth_minmax": minmax,
            "d_b": norm_to_mm(self.d_b[pos, 0], minmax),
            "d_p": norm_to_mm(self.d_p[pos, 0], minmax),
            "d_d": norm_to_mm(self.d_d[pos, 0], minmax),
            "rcpc": norm_to_mm(final_norm, minmax),
            "rcpc_meta": rcpc_meta,
        }


def estimate_fx(img: np.ndarray) -> float:
    row = img.mean(axis=0).astype(np.float32)
    row = row - row.mean()
    spec = np.abs(np.fft.rfft(row))
    lo = max(2, int(0.005 * row.size))
    hi = max(lo + 1, int(0.45 * row.size))
    k = int(np.argmax(spec[lo:hi]) + lo)
    return k / float(row.size)


def gaussian_complex(z: np.ndarray, sigma: float) -> np.ndarray:
    if SCIPY_OK:
        return ndimage.gaussian_filter(z.real, sigma=sigma, mode="reflect") + 1j * ndimage.gaussian_filter(
            z.imag, sigma=sigma, mode="reflect"
        )
    h, w = z.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    filt = np.exp(-2 * (math.pi ** 2) * (sigma ** 2) * (fx * fx + fy * fy))
    return np.fft.ifft2(np.fft.fft2(z) * filt)


def wft_features(img: np.ndarray, sigmas=(5.0, 11.0)) -> list[np.ndarray]:
    h, w = img.shape
    imgn = (img.astype(np.float32) - float(img.mean())) / (float(img.std()) + 1e-6)
    fx = estimate_fx(imgn)
    x = np.arange(w, dtype=np.float32)[None, :]
    carrier = np.exp(-1j * 2.0 * np.pi * fx * x).astype(np.complex64)
    z = imgn.astype(np.complex64) * carrier
    feats = []
    for sigma in sigmas:
        local = gaussian_complex(z, sigma)
        phase_ang = np.angle(local).astype(np.float32)
        amp = np.abs(local).astype(np.float32)
        amp = amp / (np.percentile(amp, 99.0) + 1e-6)
        feats.extend([np.sin(phase_ang), np.cos(phase_ang), amp, phase_ang / np.pi])
    return feats


def laplace_gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    if SCIPY_OK:
        return ndimage.gaussian_laplace(img, sigma=sigma, mode="reflect").astype(np.float32)
    h, w = img.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    rr = fx * fx + fy * fy
    filt = -(4 * math.pi * math.pi) * rr * np.exp(-2 * (math.pi ** 2) * (sigma ** 2) * rr)
    return np.fft.ifft2(np.fft.fft2(img) * filt).real.astype(np.float32)


def e245_features(instr_chw: np.ndarray, variant: str) -> np.ndarray:
    ins = instr_chw.astype(np.float32)
    raw = ins[0].reshape(-1)
    hs, hc, hr, hq = [ins[i].reshape(-1) for i in [1, 2, 3, 4]]
    fs, fc, fr, fq = [ins[i].reshape(-1) for i in [5, 6, 7, 8]]
    x = ins[11].reshape(-1)
    y = ins[12].reshape(-1)
    one = np.ones_like(x, dtype=np.float32)
    if variant == "ftp_phase_xy":
        cols = [one, fs, fc, fr, fq, x, y, x * x, y * y, x * y, fr * x, fr * y]
    elif variant == "hilbert_phase_xy":
        cols = [one, hs, hc, hr, hq, x, y, x * x, y * y, x * y, hr * x, hr * y]
    elif variant == "hilbert_ftp_phase_xy":
        cols = [one, hs, hc, hr, hq, fs, fc, fr, fq, x, y, x * x, y * y, x * y, hr * x, hr * y, fr * x, fr * y]
    elif variant == "raw_xy":
        cols = [one, raw, x, y, raw * x, raw * y, x * x, y * y, x * y]
    else:
        raise ValueError(variant)
    return np.stack(cols, axis=1)


def e246_features(fringe: np.ndarray, instr_chw: np.ndarray, variant: str) -> np.ndarray:
    img = fringe.astype(np.float32)
    ins = instr_chw.astype(np.float32)
    x = ins[11].reshape(-1)
    y = ins[12].reshape(-1)
    one = np.ones_like(x, dtype=np.float32)
    raw = ins[0].reshape(-1)
    hs, hc, hr, hq = [ins[i].reshape(-1) for i in [1, 2, 3, 4]]
    fs, fc, fr, fq = [ins[i].reshape(-1) for i in [5, 6, 7, 8]]
    dwt = ins[9].reshape(-1)
    grad = ins[10].reshape(-1)
    cols = [one, x, y, x * x, y * y, x * y]
    if variant in ("wft_gaussian_xy", "all_traditional_features_xy"):
        cols.extend([f.reshape(-1).astype(np.float32) for f in wft_features(img, sigmas=(5.0, 11.0))])
    if variant in ("wavelet_gabor_bank_xy", "all_traditional_features_xy"):
        imgn = (img - float(img.mean())) / (float(img.std()) + 1e-6)
        for sigma in (2.0, 4.0, 8.0, 16.0):
            lg = laplace_gauss(imgn, sigma)
            scale = np.percentile(np.abs(lg), 99.0) + 1e-6
            cols.append(np.clip(lg / scale, -3, 3).reshape(-1).astype(np.float32))
        cols.extend([dwt, grad])
    if variant in ("dwt_grad_phase_xy", "all_traditional_features_xy"):
        cols.extend([hs, hc, hr, hq, fs, fc, fr, fq, dwt, grad, raw, hr * x, hr * y, fr * x, fr * y])
    return np.stack(cols, axis=1)


class TraditionalPredictor:
    def __init__(self, e245_dir: Path, e246_dir: Path):
        self.coefs: dict[str, np.ndarray] = {}
        for name in ("ftp_phase_xy", "hilbert_ftp_phase_xy"):
            self.coefs[name] = np.load(Path(e245_dir) / f"{name}_coef.npy")
        for name in ("wft_gaussian_xy", "wavelet_gabor_bank_xy", "dwt_grad_phase_xy", "all_traditional_features_xy"):
            self.coefs[name] = np.load(Path(e246_dir) / f"{name}_coef.npy")

    def predict(self, sample: dict, variant: str) -> np.ndarray:
        if variant in {"ftp_phase_xy", "hilbert_ftp_phase_xy"}:
            x = e245_features(sample["instr"], variant)
        else:
            x = e246_features(sample["fringe"], sample["instr"], variant)
        coef = self.coefs[variant]
        pred01 = np.clip((x.astype(np.float64) @ coef).astype(np.float32), 0.0, 1.0)
        h, w = sample["target"].shape
        pred01 = pred01.reshape(h, w)
        lo, hi = map(float, sample["depth_minmax"])
        return pred01 * max(hi - lo, 1e-6) + lo


def depth_limits(target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    valid = target[mask.astype(bool)]
    if valid.size == 0:
        return float(np.nanmin(target)), float(np.nanmax(target))
    return float(np.nanpercentile(valid, 1)), float(np.nanpercentile(valid, 99))


def surface_arrays(depth: np.ndarray, mask: np.ndarray, step: int):
    z = np.asarray(depth, dtype=np.float32)[::step, ::step].copy()
    m = mask.astype(bool)[::step, ::step]
    z[~m] = np.nan
    h, w = z.shape
    yy, xx = np.mgrid[0:h, 0:w]
    return xx * step, yy * step, z


def style_3d(ax, zmin: float, zmax: float, title: str):
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("x", labelpad=-5)
    ax.set_ylabel("y", labelpad=-5)
    ax.set_zlabel("depth", labelpad=-4)
    ax.set_zlim(zmin, zmax)
    ax.view_init(elev=30, azim=-58)
    ax.tick_params(labelsize=6, pad=-2)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.42))
    except Exception:
        pass


def add_depth_panel(ax, img, mask, title, zmin, zmax, cmap="viridis"):
    im = ax.imshow(masked_img(img, mask), cmap=cmap, vmin=zmin, vmax=zmax)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    return im


def add_error_panel(ax, pred, target, mask, title, vmax):
    err = np.abs(pred - target)
    im = ax.imshow(masked_img(err, mask), cmap="magma", vmin=0.0, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    return im


def make_rcpc_gain_figure(samples: list[dict], out_dir: Path, surface_step: int):
    # Keep this figure focused: input, GT, before/after error, and one 3D pair.
    n = len(samples)
    fig = plt.figure(figsize=(7.1, 2.55 * n), constrained_layout=True)
    gs = fig.add_gridspec(n, 7, width_ratios=[1.0, 1.0, 1.0, 1.0, 1.05, 1.05, 0.08])
    cbar_depth = None
    cbar_err = None
    rows = []
    for r, s in enumerate(samples):
        target = s["target"]
        mask = s["mask"]
        zmin, zmax = depth_limits(target, mask)
        err_vmax = max(float(np.nanpercentile(np.abs(s["d_d"] - target)[mask], 95)), 1e-3)
        rmse_d, mae_d = metric_pair(s["d_d"], target, mask)
        rmse_f, mae_f = metric_pair(s["rcpc"], target, mask)
        rows.append({
            "sample": s["sample"],
            "d47_rmse": rmse_d,
            "rcpc_rmse": rmse_f,
            "delta_rmse": rmse_f - rmse_d,
            "selected": s["rcpc_meta"]["selected"],
        })
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(s["fringe"], cmap="gray")
        ax.set_title("single fringe" if r == 0 else "", fontsize=8)
        ax.set_ylabel(f"test {s['sample']}\nΔRMSE {rmse_f - rmse_d:+.2f}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar_depth = add_depth_panel(fig.add_subplot(gs[r, 1]), target, mask, "GT depth" if r == 0 else "", zmin, zmax)
        add_depth_panel(fig.add_subplot(gs[r, 2]), s["d_d"], mask, f"D47\n{rmse_d:.2f} mm" if r == 0 else f"{rmse_d:.2f} mm", zmin, zmax)
        add_depth_panel(fig.add_subplot(gs[r, 3]), s["rcpc"], mask, f"RCPC\n{rmse_f:.2f} mm" if r == 0 else f"{rmse_f:.2f} mm", zmin, zmax)
        cbar_err = add_error_panel(fig.add_subplot(gs[r, 4]), s["d_d"], target, mask, "|D47-GT|" if r == 0 else "", err_vmax)
        add_error_panel(fig.add_subplot(gs[r, 5]), s["rcpc"], target, mask, "|RCPC-GT|" if r == 0 else "", err_vmax)
        ax3 = fig.add_subplot(gs[r, 6], projection="3d")
        xx, yy, zz = surface_arrays(s["rcpc"], mask, surface_step)
        ax3.plot_surface(xx, yy, zz, cmap="viridis", linewidth=0, antialiased=False, vmin=zmin, vmax=zmax)
        style_3d(ax3, zmin, zmax, "RCPC 3D" if r == 0 else "")
    fig.suptitle("Risk-controlled posterior correction reduces selected reconstruction errors", fontsize=9)
    save_figure(fig, out_dir / "figure_rcpc_gain")
    plt.close(fig)
    return rows


def make_traditional_depth_plate(samples: list[dict], trad: TraditionalPredictor, out_dir: Path):
    variants = [
        ("ftp_phase_xy", "FTP proxy"),
        ("wft_gaussian_xy", "WFT proxy"),
        ("wavelet_gabor_bank_xy", "wavelet proxy"),
        ("dwt_grad_phase_xy", "DWT+phase proxy"),
        ("rcpc", "RCPC"),
    ]
    n = len(samples)
    fig, axes = plt.subplots(n, 7, figsize=(7.15, 1.55 * n), constrained_layout=True)
    if n == 1:
        axes = axes[None, :]
    rows = []
    for r, s in enumerate(samples):
        target = s["target"]
        mask = s["mask"]
        zmin, zmax = depth_limits(target, mask)
        axes[r, 0].imshow(s["fringe"], cmap="gray")
        axes[r, 0].set_title("single fringe" if r == 0 else "", fontsize=8)
        axes[r, 0].set_ylabel(f"test {s['sample']}", fontsize=7)
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        add_depth_panel(axes[r, 1], target, mask, "GT" if r == 0 else "", zmin, zmax)
        for c, (variant, label) in enumerate(variants, start=2):
            pred = s["rcpc"] if variant == "rcpc" else trad.predict(s, variant)
            rmse, mae = metric_pair(pred, target, mask)
            title = f"{label}\n{rmse:.1f} mm" if r == 0 else f"{rmse:.1f}"
            add_depth_panel(axes[r, c], pred, mask, title, zmin, zmax)
            rows.append({"sample": s["sample"], "variant": variant, "rmse": rmse, "mae": mae})
    fig.suptitle("Traditional single-frame analytic proxies recover only coarse depth compared with RCPC", fontsize=9)
    save_figure(fig, out_dir / "figure_traditional_depth_plate")
    plt.close(fig)
    return rows


def make_traditional_error_plate(samples: list[dict], trad: TraditionalPredictor, out_dir: Path):
    variants = [
        ("ftp_phase_xy", "FTP"),
        ("wft_gaussian_xy", "WFT"),
        ("wavelet_gabor_bank_xy", "wavelet"),
        ("dwt_grad_phase_xy", "DWT+phase"),
        ("rcpc", "RCPC"),
    ]
    n = len(samples)
    fig, axes = plt.subplots(n, len(variants) + 1, figsize=(7.15, 1.45 * n), constrained_layout=True)
    if n == 1:
        axes = axes[None, :]
    rows = []
    for r, s in enumerate(samples):
        target = s["target"]
        mask = s["mask"]
        axes[r, 0].imshow(s["fringe"], cmap="gray")
        axes[r, 0].set_title("single fringe" if r == 0 else "", fontsize=8)
        axes[r, 0].set_ylabel(f"test {s['sample']}", fontsize=7)
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        preds = []
        for variant, label in variants:
            pred = s["rcpc"] if variant == "rcpc" else trad.predict(s, variant)
            preds.append((variant, label, pred))
        vmax = max(1e-3, float(np.nanpercentile(np.concatenate([np.abs(p - target)[mask] for _, _, p in preds]), 95)))
        for c, (variant, label, pred) in enumerate(preds, start=1):
            rmse, mae = metric_pair(pred, target, mask)
            title = f"{label}\n{rmse:.1f} mm" if r == 0 else f"{rmse:.1f}"
            add_error_panel(axes[r, c], pred, target, mask, title, vmax)
            rows.append({"sample": s["sample"], "variant": variant, "rmse": rmse, "mae": mae})
    fig.suptitle("Error maps reveal why traditional single-frame proxies are weak baselines", fontsize=9)
    save_figure(fig, out_dir / "figure_traditional_error_plate")
    plt.close(fig)
    return rows


def make_traditional_3d_sample(sample: dict, trad: TraditionalPredictor, out_dir: Path, surface_step: int):
    variants = [
        ("target", "GT"),
        ("ftp_phase_xy", "FTP proxy"),
        ("wft_gaussian_xy", "WFT proxy"),
        ("wavelet_gabor_bank_xy", "wavelet proxy"),
        ("dwt_grad_phase_xy", "DWT+phase proxy"),
        ("rcpc", "RCPC"),
    ]
    target = sample["target"]
    mask = sample["mask"]
    zmin, zmax = depth_limits(target, mask)
    fig = plt.figure(figsize=(7.15, 3.0), constrained_layout=True)
    gs = fig.add_gridspec(1, len(variants))
    rows = []
    for c, (variant, label) in enumerate(variants):
        if variant == "target":
            pred = target
            metric = "GT"
        elif variant == "rcpc":
            pred = sample["rcpc"]
            rmse, _ = metric_pair(pred, target, mask)
            metric = f"{rmse:.1f} mm"
        else:
            pred = trad.predict(sample, variant)
            rmse, _ = metric_pair(pred, target, mask)
            metric = f"{rmse:.1f} mm"
        rows.append({"sample": sample["sample"], "variant": variant, "rmse_label": metric})
        ax = fig.add_subplot(gs[0, c], projection="3d")
        xx, yy, zz = surface_arrays(pred, mask, surface_step)
        ax.plot_surface(xx, yy, zz, cmap="viridis", linewidth=0, antialiased=False, vmin=zmin, vmax=zmax)
        style_3d(ax, zmin, zmax, f"{label}\n{metric}")
    fig.suptitle(f"3D reconstruction surfaces for traditional single-frame proxies (test {sample['sample']})", fontsize=9)
    save_figure(fig, out_dir / f"figure_traditional_3d_test_{sample['sample']:03d}")
    plt.close(fig)
    return rows


def make_rcpc_gain_compact(samples: list[dict], out_dir: Path):
    cols = [
        ("fringe", "Input"),
        ("target", "GT"),
        ("d_d", "D47"),
        ("rcpc", "RCPC"),
        ("err_d", "|D47-GT|"),
        ("err_f", "|RCPC-GT|"),
    ]
    n = len(samples)
    fig, axes = plt.subplots(
        n,
        len(cols),
        figsize=(7.2, 1.25 * n),
        gridspec_kw={"width_ratios": [1.05, 1, 1, 1, 1, 1]},
    )
    if n == 1:
        axes = axes[None, :]
    rows = []
    for r, s in enumerate(samples):
        target = s["target"]
        mask = s["mask"]
        zmin, zmax = depth_limits(target, mask)
        rmse_d, _ = metric_pair(s["d_d"], target, mask)
        rmse_f, _ = metric_pair(s["rcpc"], target, mask)
        err_vmax = max(float(np.nanpercentile(np.abs(s["d_d"] - target)[mask], 95)), 1e-3)
        rows.append({"sample": s["sample"], "d47_rmse": rmse_d, "rcpc_rmse": rmse_f, "delta_rmse": rmse_f - rmse_d})
        data = {
            "fringe": s["fringe"],
            "target": target,
            "d_d": s["d_d"],
            "rcpc": s["rcpc"],
            "err_d": np.abs(s["d_d"] - target),
            "err_f": np.abs(s["rcpc"] - target),
        }
        for c, (key, label) in enumerate(cols):
            ax = axes[r, c]
            if key == "fringe":
                ax.imshow(data[key], cmap="gray")
                title = label if r == 0 else ""
            elif key.startswith("err"):
                ax.imshow(masked_img(data[key], mask), cmap="magma", vmin=0.0, vmax=err_vmax)
                title = label if r == 0 else ""
            else:
                ax.imshow(masked_img(data[key], mask), cmap="viridis", vmin=zmin, vmax=zmax)
                if key == "d_d":
                    title = f"{label}\n{rmse_d:.2f} mm" if r == 0 else f"{rmse_d:.2f}"
                elif key == "rcpc":
                    title = f"{label}\n{rmse_f:.2f} mm" if r == 0 else f"{rmse_f:.2f}"
                else:
                    title = label if r == 0 else ""
            ax.set_title(title, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[r, 0].set_ylabel(f"test {s['sample']}\n{rmse_f - rmse_d:+.2f}", fontsize=7, rotation=0, labelpad=22, va="center")
    fig.subplots_adjust(left=0.085, right=0.995, top=0.92, bottom=0.02, wspace=0.06, hspace=0.12)
    save_compact(fig, out_dir / "figure_rcpc_gain_compact")
    plt.close(fig)
    return rows


def make_rcpc_3d_compact(sample: dict, out_dir: Path, surface_step: int):
    target = sample["target"]
    mask = sample["mask"]
    zmin, zmax = depth_limits(target, mask)
    variants = [("target", "GT"), ("d_d", "D47"), ("rcpc", "RCPC")]
    fig = plt.figure(figsize=(5.4, 1.85))
    for i, (key, label) in enumerate(variants, start=1):
        depth = sample[key]
        ax = fig.add_subplot(1, 3, i, projection="3d")
        xx, yy, zz = surface_arrays(depth, mask, surface_step)
        ax.plot_surface(xx, yy, zz, cmap="viridis", linewidth=0, antialiased=False, vmin=zmin, vmax=zmax)
        if key == "target":
            metric = "GT"
        else:
            rmse, _ = metric_pair(depth, target, mask)
            metric = f"{rmse:.2f} mm"
        style_3d(ax, zmin, zmax, f"{label}\n{metric}")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
    fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.0, wspace=0.02)
    save_compact(fig, out_dir / f"figure_rcpc_3d_compact_test_{sample['sample']:03d}")
    plt.close(fig)


def make_traditional_depth_compact(samples: list[dict], trad: TraditionalPredictor, out_dir: Path):
    variants = [
        ("ftp_phase_xy", "FTP"),
        ("wft_gaussian_xy", "WFT"),
        ("wavelet_gabor_bank_xy", "Wavelet"),
        ("dwt_grad_phase_xy", "DWT+phase"),
        ("rcpc", "RCPC"),
    ]
    n = len(samples)
    fig, axes = plt.subplots(
        n,
        7,
        figsize=(7.2, 1.2 * n),
        gridspec_kw={"width_ratios": [1.05, 1, 1, 1, 1, 1, 1]},
    )
    if n == 1:
        axes = axes[None, :]
    rows = []
    for r, s in enumerate(samples):
        target = s["target"]
        mask = s["mask"]
        zmin, zmax = depth_limits(target, mask)
        axes[r, 0].imshow(s["fringe"], cmap="gray")
        axes[r, 0].set_title("Input" if r == 0 else "", fontsize=7)
        add_depth_panel(axes[r, 1], target, mask, "GT" if r == 0 else "", zmin, zmax)
        for c, (variant, label) in enumerate(variants, start=2):
            pred = s["rcpc"] if variant == "rcpc" else trad.predict(s, variant)
            rmse, mae = metric_pair(pred, target, mask)
            title = f"{label}\n{rmse:.1f}" if r == 0 else f"{rmse:.1f}"
            add_depth_panel(axes[r, c], pred, mask, title, zmin, zmax)
            rows.append({"sample": s["sample"], "variant": variant, "rmse": rmse, "mae": mae})
        axes[r, 0].set_ylabel(f"test {s['sample']}", fontsize=7, rotation=0, labelpad=20, va="center")
        for ax in axes[r, :]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.91, bottom=0.02, wspace=0.06, hspace=0.12)
    save_compact(fig, out_dir / "figure_traditional_depth_compact")
    plt.close(fig)
    return rows


def make_traditional_error_compact(samples: list[dict], trad: TraditionalPredictor, out_dir: Path):
    variants = [
        ("ftp_phase_xy", "FTP"),
        ("wft_gaussian_xy", "WFT"),
        ("wavelet_gabor_bank_xy", "Wavelet"),
        ("dwt_grad_phase_xy", "DWT+phase"),
        ("rcpc", "RCPC"),
    ]
    n = len(samples)
    fig, axes = plt.subplots(n, 6, figsize=(7.2, 1.2 * n), gridspec_kw={"width_ratios": [1.05, 1, 1, 1, 1, 1]})
    if n == 1:
        axes = axes[None, :]
    rows = []
    for r, s in enumerate(samples):
        target = s["target"]
        mask = s["mask"]
        axes[r, 0].imshow(s["fringe"], cmap="gray")
        axes[r, 0].set_title("Input" if r == 0 else "", fontsize=7)
        preds = []
        for variant, label in variants:
            pred = s["rcpc"] if variant == "rcpc" else trad.predict(s, variant)
            preds.append((variant, label, pred))
        vmax = max(1e-3, float(np.nanpercentile(np.concatenate([np.abs(p - target)[mask] for _, _, p in preds]), 92)))
        for c, (variant, label, pred) in enumerate(preds, start=1):
            rmse, mae = metric_pair(pred, target, mask)
            title = f"{label}\n{rmse:.1f}" if r == 0 else f"{rmse:.1f}"
            add_error_panel(axes[r, c], pred, target, mask, title, vmax)
            rows.append({"sample": s["sample"], "variant": variant, "rmse": rmse, "mae": mae})
        axes[r, 0].set_ylabel(f"test {s['sample']}", fontsize=7, rotation=0, labelpad=20, va="center")
        for ax in axes[r, :]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.91, bottom=0.02, wspace=0.06, hspace=0.12)
    save_compact(fig, out_dir / "figure_traditional_error_compact")
    plt.close(fig)
    return rows


def load_csv_metrics(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    out = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["sample"])] = float(row["rmse"])
    return out


def main():
    parser = argparse.ArgumentParser(description="Create focused paper figures for RCPC and traditional single-frame baselines.")
    parser.add_argument("--base_cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_phase_cache_960")
    parser.add_argument("--candidate_cache_dir", default="/root/autodl-tmp/fpp_ml_ucpf_hier_orderfix_cache_960_seed180")
    parser.add_argument("--e245_dir", default="results/e245_traditional_single_frame_proxy_baselines")
    parser.add_argument("--e246_dir", default="results/e246_traditional_wft_wavelet_proxy_baselines")
    parser.add_argument("--save_dir", default="results/e248_paper_reconstruction_figures")
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", default="7 19 21")
    parser.add_argument("--traditional_3d_sample", type=int, default=7)
    parser.add_argument("--surface_step", type=int, default=10)
    parser.add_argument("--edge_tau", type=float, default=0.42)
    parser.add_argument("--delta_max", type=float, default=0.11)
    parser.add_argument("--phase_conf_max", type=float, default=0.74)
    parser.add_argument("--high_weight", type=float, default=0.6)
    parser.add_argument("--low_weight", type=float, default=0.0)
    args = parser.parse_args()

    set_mpl_style()
    save_dir = Path(args.save_dir)
    view = CacheView(Path(args.base_cache_dir), Path(args.phase_cache_dir), Path(args.candidate_cache_dir), args.split)
    samples = [view.sample(i, args) for i in parse_int_list(args.samples)]
    trad = TraditionalPredictor(Path(args.e245_dir), Path(args.e246_dir))

    rcpc_rows = make_rcpc_gain_figure(samples, save_dir, max(1, int(args.surface_step)))
    trad_depth_rows = make_traditional_depth_plate(samples, trad, save_dir)
    trad_err_rows = make_traditional_error_plate(samples, trad, save_dir)
    sample3d = view.sample(args.traditional_3d_sample, args)
    trad_3d_rows = make_traditional_3d_sample(sample3d, trad, save_dir, max(1, int(args.surface_step)))
    compact_rcpc_rows = make_rcpc_gain_compact(samples, save_dir)
    make_rcpc_3d_compact(sample3d, save_dir, max(1, int(args.surface_step)))
    compact_trad_depth_rows = make_traditional_depth_compact(samples, trad, save_dir)
    compact_trad_err_rows = make_traditional_error_compact(samples, trad, save_dir)

    # Check one reproducibility link against saved E245/E246 rows for the drawn samples.
    reference = {
        "ftp_phase_xy": load_csv_metrics(Path(args.e245_dir) / f"{args.split}_ftp_phase_xy_rows.csv"),
        "wft_gaussian_xy": load_csv_metrics(Path(args.e246_dir) / f"{args.split}_wft_gaussian_xy_rows.csv"),
        "wavelet_gabor_bank_xy": load_csv_metrics(Path(args.e246_dir) / f"{args.split}_wavelet_gabor_bank_xy_rows.csv"),
        "dwt_grad_phase_xy": load_csv_metrics(Path(args.e246_dir) / f"{args.split}_dwt_grad_phase_xy_rows.csv"),
    }
    checks = []
    for row in trad_depth_rows:
        variant = row["variant"]
        sample = int(row["sample"])
        if variant in reference and sample in reference[variant]:
            checks.append({
                "sample": sample,
                "variant": variant,
                "rendered_rmse": row["rmse"],
                "reference_rmse": reference[variant][sample],
                "abs_diff": abs(row["rmse"] - reference[variant][sample]),
            })

    summary = {
        "figure_intent": {
            "figure_rcpc_gain": "Focused before/after evidence for RCPC over the diffusion posterior candidate.",
            "figure_traditional_depth_plate": "Depth-map comparison of traditional single-frame analytic proxy baselines against RCPC.",
            "figure_traditional_error_plate": "Error-map comparison showing where traditional proxies fail.",
            "figure_traditional_3d": "3D surface view of traditional proxy reconstructions for one representative sample.",
        },
        "traditional_boundary": (
            "Traditional results are single-frame analytic feature proxy baselines calibrated on the train split. "
            "They are not full calibrated FTP/WFT/WT FPP reconstruction because the released files do not provide "
            "the full reference fringe, projector-camera calibration, stripe period, and phase-to-height model."
        ),
        "samples": parse_int_list(args.samples),
        "rcpc_rows": rcpc_rows,
        "traditional_depth_rows": trad_depth_rows,
        "traditional_error_rows": trad_err_rows,
        "traditional_3d_rows": trad_3d_rows,
        "compact_rcpc_rows": compact_rcpc_rows,
        "compact_traditional_depth_rows": compact_trad_depth_rows,
        "compact_traditional_error_rows": compact_trad_err_rows,
        "reference_rmse_checks": checks,
        "scipy_available": SCIPY_OK,
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "paper_reconstruction_figure_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({"save_dir": str(save_dir), "figures": 4, "checks": checks[:6]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
