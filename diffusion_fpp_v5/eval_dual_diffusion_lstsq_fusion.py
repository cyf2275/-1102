from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from data.dataset_fpp_phase import create_fpp_phase_loaders
from diffusion_pip import PIPDiffusion
from eval_adaptive_blend_features import _saved_arg, build_model
from eval_dual_diffusion_fusion import pixel_gate
from eval_hierarchical_phase_fusion import aux_predict01, build_aux_depth_model
from train_fpp_official_style_unet import METRIC_KEYS, parse_channel_spec, summarize
from train_pip_lite import prediction_to_mm
from utils.metrics import compute_metrics


def parse_float_list(text):
    return [float(x) for x in str(text).replace(",", " ").split() if x]


def build_depth_diffusion(args, device):
    ckpt = torch.load(args.depth_checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})
    include_ftp = bool(_saved_arg(saved_args, "include_ftp", False))
    physics_indices = _saved_arg(saved_args, "physics_channel_indices", None)
    if physics_indices is None:
        physics_indices = parse_channel_spec(str(_saved_arg(saved_args, "physics_channels", "")), include_ftp)
    model_cond_channels = int(ckpt.get("model_cond_channels", len(physics_indices)))
    model = build_model(saved_args, model_cond_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    diffusion = PIPDiffusion(
        model,
        timesteps=int(_saved_arg(saved_args, "timesteps", 200)),
        image_h=args.image_size,
        image_w=args.image_size,
        device=device,
        cond_indices=physics_indices,
        target_mode=str(_saved_arg(saved_args, "target_mode", "base_residual")),
        residual_scale=float(_saved_arg(saved_args, "resolved_residual_scale", 1.0)),
        base_residual_gate=float(_saved_arg(saved_args, "base_residual_gate", 1.0)),
    )
    return diffusion, include_ftp


def build_phase_model(args, device):
    # Reuse the loader from hierarchical fusion so both plain phase2depth UNet
    # and PSP adapter checkpoints are supported.
    args.phase_depth_checkpoint = args.phase_checkpoint
    return build_aux_depth_model(args, device)


def fusion_features(base, depth_branch, phase_branch, edge, conf, gate):
    ones = torch.ones_like(base)
    depth_delta = depth_branch - base
    phase_delta = phase_branch - base
    return torch.cat(
        [
            ones,
            base,
            depth_branch,
            phase_branch,
            depth_delta,
            phase_delta,
            edge,
            conf,
            gate,
            depth_delta * edge,
            phase_delta * edge,
            depth_delta * conf,
            phase_delta * conf,
            depth_delta * gate,
            phase_delta * gate,
        ],
        dim=1,
    )


@torch.no_grad()
def branch_batches(args, split, depth_diffusion, phase_model, phase_args, phase_kind, phase_mode,
                   loaders_depth, loaders_phase, device):
    loader_key = "train_eval" if split == "train" else split
    for depth_batch, phase_batch in tqdm(
        zip(loaders_depth[loader_key], loaders_phase[loader_key]),
        total=len(loaders_depth[loader_key]),
        desc=f"branches {split}",
    ):
        if not torch.equal(depth_batch["sample_index"], phase_batch["sample_index"]):
            raise RuntimeError("depth and phase loaders are not aligned")
        base = torch.clamp(depth_batch["base_height"].to(device, non_blocking=True), -1.0, 1.0)
        diff = depth_diffusion.sample_ddim(
            depth_batch,
            steps=args.ddim_steps,
            ensemble_size=1,
            start_from_base=True,
            start_ratio=args.start_ratio,
        )
        mask = torch.clamp(depth_batch["mask"].to(device, non_blocking=True), 0.0, 1.0)
        edge = torch.clamp(depth_batch["edge_score"].to(device, non_blocking=True), 0.0, 1.0)
        conf = torch.clamp(depth_batch["phase_conf"].to(device, non_blocking=True), 0.0, 1.0)
        gate = pixel_gate(base, diff, edge, conf, mask, args)
        depth_branch = torch.clamp(base + args.depth_alpha * gate * (diff - base), -1.0, 1.0)
        phase_pred01 = aux_predict01(
            phase_model, phase_batch, device, phase_args, phase_kind, phase_mode
        )
        phase_branch = phase_pred01 * 2.0 - 1.0
        feats = fusion_features(base, depth_branch, phase_branch, edge, conf, gate)
        yield depth_batch, feats, base, depth_branch, phase_branch, mask


@torch.no_grad()
def fit_lstsq(args, depth_diffusion, phase_model, phase_args, phase_kind, phase_mode,
              loaders_depth, loaders_phase, device):
    n_feat = 15
    xtx = torch.zeros(n_feat, n_feat, dtype=torch.float64, device=device)
    xty = torch.zeros(n_feat, 1, dtype=torch.float64, device=device)
    count = 0
    for depth_batch, feats, _base, _depth, _phase, mask in branch_batches(
        args, args.fit_split, depth_diffusion, phase_model, phase_args, phase_kind, phase_mode,
        loaders_depth, loaders_phase, device
    ):
        target = depth_batch["height"].to(device, non_blocking=True)
        x = feats.permute(0, 2, 3, 1).reshape(-1, n_feat)
        y = target.reshape(-1, 1)
        m = mask.reshape(-1) > 0.5
        x = x[m].to(dtype=torch.float64)
        y = y[m].to(dtype=torch.float64)
        xtx += x.T @ x
        xty += x.T @ y
        count += int(m.sum().detach().cpu())
    coefs = {}
    eye = torch.eye(n_feat, dtype=torch.float64, device=device)
    for ridge in parse_float_list(args.ridge_alphas):
        reg = eye * float(ridge)
        reg[0, 0] = 0.0
        coef = torch.linalg.solve(xtx + reg, xty).flatten().to(dtype=torch.float32)
        coefs[str(ridge)] = coef
    return coefs, count


@torch.no_grad()
def evaluate_coefs(args, split, coefs, depth_diffusion, phase_model, phase_args, phase_kind, phase_mode,
                   loaders_depth, loaders_phase, device):
    rows = {key: [] for key in coefs}
    branch_rows = {"base": [], "depth_branch": [], "phase_branch": []}
    for depth_batch, feats, base, depth_branch, phase_branch, mask in branch_batches(
        args, split, depth_diffusion, phase_model, phase_args, phase_kind, phase_mode,
        loaders_depth, loaders_phase, device
    ):
        target = depth_batch["height_raw"].to(device, non_blocking=True)
        for name, pred in (("base", base), ("depth_branch", depth_branch), ("phase_branch", phase_branch)):
            pred_mm = prediction_to_mm(pred, depth_batch, loaders_depth["height_scale"])
            for j in range(pred_mm.shape[0]):
                branch_rows[name].append(compute_metrics(pred_mm[j:j + 1], target[j:j + 1], mask=mask[j:j + 1]))
        for key, coef in coefs.items():
            pred = torch.clamp((feats * coef.view(1, -1, 1, 1).to(device)).sum(dim=1, keepdim=True), -1.0, 1.0)
            pred_mm = prediction_to_mm(pred, depth_batch, loaders_depth["height_scale"])
            for j in range(pred_mm.shape[0]):
                rows[key].append(compute_metrics(pred_mm[j:j + 1], target[j:j + 1], mask=mask[j:j + 1]))
    out = {
        "branches": {name: summarize(vals) for name, vals in branch_rows.items()},
        "ridge": {},
    }
    for key, vals in rows.items():
        item = summarize(vals)
        item["n"] = len(vals)
        out["ridge"][key] = item
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_checkpoint", required=True)
    parser.add_argument("--phase_checkpoint", required=True)
    parser.add_argument("--cache_dir", default="/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_pspquad_cache_960")
    parser.add_argument("--base_prefix", default="base_c4_adapter")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--fit_split", choices=["train", "val"], default="train")
    parser.add_argument("--image_size", type=int, default=960)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--start_ratio", type=float, default=0.05)
    parser.add_argument("--depth_alpha", type=float, default=0.7)
    parser.add_argument("--sample_edge_th", type=float, default=0.47)
    parser.add_argument("--edge_th", type=float, default=1.0)
    parser.add_argument("--delta_min", type=float, default=0.12)
    parser.add_argument("--conf_min", type=float, default=0.0)
    parser.add_argument("--ridge_alphas", default="0 1e-6 1e-4 1e-2 1 10 100")
    parser.add_argument("--require_cache", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    depth_diffusion, include_ftp = build_depth_diffusion(args, device)
    phase_model, phase_args, phase_kind, phase_mode = build_phase_model(args, device)
    phase_pred_prefix = getattr(phase_args, "phase_pred_prefix", None)
    loaders_depth = create_fpp_ml_bench_loaders(
        cache_dir=args.cache_dir,
        batch_size=1,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        include_ftp=include_ftp,
        image_h=args.image_size,
        image_w=args.image_size,
        require_cache=args.require_cache,
        base_prefix=args.base_prefix,
    )
    loaders_phase = create_fpp_phase_loaders(
        base_cache_dir=args.cache_dir,
        phase_cache_dir=args.phase_cache_dir,
        batch_size=1,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        image_h=args.image_size,
        image_w=args.image_size,
        phase_pred_prefix=phase_pred_prefix,
        require_cache=args.require_cache,
    )

    coefs, fit_pixels = fit_lstsq(
        args, depth_diffusion, phase_model, phase_args, phase_kind, phase_mode,
        loaders_depth, loaders_phase, device
    )
    val = evaluate_coefs(
        args, "val", coefs, depth_diffusion, phase_model, phase_args, phase_kind, phase_mode,
        loaders_depth, loaders_phase, device
    )
    test = evaluate_coefs(
        args, "test", coefs, depth_diffusion, phase_model, phase_args, phase_kind, phase_mode,
        loaders_depth, loaders_phase, device
    )
    best_alpha = min(val["ridge"], key=lambda key: val["ridge"][key]["rmse"]["mean"])
    out = {
        "fit_split": args.fit_split,
        "fit_pixels": fit_pixels,
        "phase_kind": phase_kind,
        "phase_mode": phase_mode,
        "phase_pred_prefix": phase_pred_prefix,
        "ridge_alphas": parse_float_list(args.ridge_alphas),
        "coef": {key: [float(x) for x in coef.detach().cpu()] for key, coef in coefs.items()},
        "val": val,
        "test": test,
        "selected_by_val": {
            "ridge_alpha": float(best_alpha),
            "val_rmse": val["ridge"][best_alpha]["rmse"]["mean"],
            "test_rmse": test["ridge"][best_alpha]["rmse"]["mean"],
        },
    }
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "lstsq_fusion_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out["selected_by_val"], ensure_ascii=False))


if __name__ == "__main__":
    main()
