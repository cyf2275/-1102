from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from train_fpp_official_style_unet import parse_channel_spec
from train_joint_pip_diffusion import (
    JointDiffusionRunner,
    JointPIPDiffFPP,
    METRIC_KEYS,
    summarize_prefixed,
)
from train_pip_lite import prediction_to_mm
from utils.metrics import compute_metrics


FEATURE_RULES = [
    ("edge_mean", "le"),
    ("delta_mean", "ge"),
    ("delta_edge_mean", "ge"),
    ("delta_lowconf_mean", "ge"),
    ("phase_conf_mean", "ge"),
]


def saved_arg(saved_args, key, default=None):
    return saved_args.get(key, default) if isinstance(saved_args, dict) else default


def alpha_tag(alpha: float) -> str:
    return f"a{int(round(float(alpha) * 1000)):03d}"


def masked_scalar(x: torch.Tensor, mask: torch.Tensor) -> float:
    mask = torch.clamp(mask.to(device=x.device, dtype=x.dtype), 0.0, 1.0)
    denom = mask.sum().clamp(min=1.0)
    return float(((x * mask).sum() / denom).detach().cpu())


def direct_summary(rows, prefix):
    out = {"n": len(rows)}
    for key in METRIC_KEYS:
        vals = np.asarray([float(row[f"{prefix}_{key}"]) for row in rows], dtype=np.float64)
        out[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
        }
    return out


def threshold_candidates(rows, feature):
    vals = sorted({float(row[feature]) for row in rows})
    if not vals:
        return [0.0]
    mids = [(a + b) * 0.5 for a, b in zip(vals[:-1], vals[1:])]
    return [vals[0] - 1e-6, *vals, *mids, vals[-1] + 1e-6]


def row_selected(row, feature, rule, threshold):
    if rule == "all":
        return True
    value = float(row[feature])
    if rule == "le":
        return value <= threshold
    if rule == "ge":
        return value >= threshold
    raise ValueError(f"unknown rule: {rule}")


def gated_summary(rows, prefix, feature, rule, threshold):
    out = {
        "n": len(rows),
        "selected": int(sum(row_selected(row, feature, rule, threshold) for row in rows)),
    }
    for key in METRIC_KEYS:
        vals = []
        for row in rows:
            use_prefix = prefix if row_selected(row, feature, rule, threshold) else "base"
            vals.append(float(row[f"{use_prefix}_{key}"]))
        vals = np.asarray(vals, dtype=np.float64)
        out[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
        }
    return out


def parse_gate_features(values):
    if not values or (len(values) == 1 and values[0].lower() == "all"):
        return FEATURE_RULES
    out = []
    for value in values:
        if ":" in value:
            feature, rule = value.split(":", 1)
        else:
            feature = value
            rule = "le" if value == "edge_mean" else "ge"
        if rule not in {"le", "ge"}:
            raise ValueError(f"unknown gate rule: {value}")
        out.append((feature, rule))
    return out


def build_runner(ckpt_path, loaders, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    include_ftp = bool(saved_arg(ckpt_args, "include_ftp", False))
    physics_indices = saved_arg(ckpt_args, "physics_channel_indices", None)
    if physics_indices is None:
        physics_indices = parse_channel_spec(str(saved_arg(ckpt_args, "physics_channels", "")), include_ftp)
    residual_scale = float(saved_arg(ckpt_args, "resolved_residual_scale", saved_arg(ckpt_args, "residual_scale", 1.0)))
    model = JointPIPDiffFPP(
        cond_channels=int(loaders["cond_channels"]),
        cond_indices=physics_indices,
        joint_mode=str(saved_arg(ckpt_args, "joint_mode", "full")),
        base_channels=int(saved_arg(ckpt_args, "base_channels", 24)),
        adapter_hidden=int(saved_arg(ckpt_args, "adapter_hidden", 24)),
        coarse_channels=int(saved_arg(ckpt_args, "coarse_channels", 24)),
        learned_residual_gate=bool(saved_arg(ckpt_args, "learned_residual_gate", False)),
        gate_init=float(saved_arg(ckpt_args, "gate_init", 0.05)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    runner = JointDiffusionRunner(
        model=model,
        timesteps=int(saved_arg(ckpt_args, "timesteps", 200)),
        device=device,
        residual_scale=residual_scale,
        base_residual_gate=float(saved_arg(ckpt_args, "base_residual_gate", 1.0)),
        train_t_min_ratio=float(saved_arg(ckpt_args, "train_t_min_ratio", 0.0)),
        train_t_max_ratio=float(saved_arg(ckpt_args, "train_t_max_ratio", 1.0)),
    )
    return runner, ckpt_args


@torch.no_grad()
def evaluate_split(args, loaders, split, device):
    loader = loaders["train_eval" if split == "train" else split]
    pred_sum = None
    base_store = []
    target_store = []
    mask_store = []
    edge_store = []
    conf_store = []
    minmax_store = []

    for ckpt_idx, ckpt_path in enumerate(args.checkpoints):
        runner, _ = build_runner(ckpt_path, loaders, device)
        preds = []
        for batch in tqdm(loader, desc=f"{Path(ckpt_path).parts[-3]} {split}"):
            pred = runner.sample_ddim(
                batch,
                steps=args.ddim_steps,
                start_ratio=args.start_ratio,
            ).detach().cpu()
            preds.append(pred)
            if ckpt_idx == 0:
                base_store.append(torch.clamp(batch["base_height"], -1.0, 1.0).cpu())
                target_store.append(batch["height_raw"].cpu())
                mask_store.append(torch.clamp(batch["mask"], 0.0, 1.0).cpu())
                edge_store.append(torch.clamp(batch["edge_score"], 0.0, 1.0).cpu())
                conf_store.append(torch.clamp(batch["phase_conf"], 0.0, 1.0).cpu())
                minmax_store.append(batch["depth_minmax"].cpu())
        pred_tensor = torch.cat(preds, dim=0)
        pred_sum = pred_tensor if pred_sum is None else pred_sum + pred_tensor
        del runner
        if device.type == "cuda":
            torch.cuda.empty_cache()

    diff_all = pred_sum / float(len(args.checkpoints))
    base_all = torch.cat(base_store, dim=0)
    target_all = torch.cat(target_store, dim=0)
    mask_all = torch.cat(mask_store, dim=0)
    edge_all = torch.cat(edge_store, dim=0)
    conf_all = torch.cat(conf_store, dim=0)
    minmax_all = torch.cat(minmax_store, dim=0)

    rows_by_alpha = {float(alpha): [] for alpha in args.alphas}
    for i in range(diff_all.shape[0]):
        base = base_all[i:i + 1].to(device, non_blocking=True)
        diff = diff_all[i:i + 1].to(device, non_blocking=True)
        target = target_all[i:i + 1].to(device, non_blocking=True)
        mask = mask_all[i:i + 1].to(device, non_blocking=True)
        edge = edge_all[i:i + 1].to(device, non_blocking=True)
        conf = conf_all[i:i + 1].to(device, non_blocking=True)
        minmax = minmax_all[i:i + 1].to(device, non_blocking=True)
        batch_stub = {"depth_minmax": minmax}
        delta = torch.abs(diff - base)
        base_mm = prediction_to_mm(base, batch_stub, loaders["height_scale"])
        diff_mm = prediction_to_mm(diff, batch_stub, loaders["height_scale"])
        base_metrics = compute_metrics(base_mm, target, mask=mask)
        diff_metrics = compute_metrics(diff_mm, target, mask=mask)
        common = {
            "sample": int(i),
            "delta_mean": masked_scalar(delta, mask),
            "delta_edge_mean": masked_scalar(delta * edge, mask),
            "delta_lowconf_mean": masked_scalar(delta * (1.0 - conf), mask),
            "phase_conf_mean": masked_scalar(conf, mask),
            "edge_mean": masked_scalar(edge, mask),
        }
        for alpha in args.alphas:
            prefix = f"blend_{alpha_tag(alpha)}"
            blend = torch.clamp(base + float(alpha) * (diff - base), -1.0, 1.0)
            blend_mm = prediction_to_mm(blend, batch_stub, loaders["height_scale"])
            blend_metrics = compute_metrics(blend_mm, target, mask=mask)
            row = {**common, "alpha": float(alpha)}
            row.update({f"base_{key}": base_metrics[key] for key in METRIC_KEYS})
            row.update({f"diff_{key}": diff_metrics[key] for key in METRIC_KEYS})
            row.update({f"{prefix}_{key}": blend_metrics[key] for key in METRIC_KEYS})
            rows_by_alpha[float(alpha)].append(row)
    return rows_by_alpha


def find_best_candidate(rows_by_alpha, feature_rules, min_selected, allow_all):
    candidates = []
    for alpha, rows in rows_by_alpha.items():
        prefix = f"blend_{alpha_tag(alpha)}"
        if allow_all:
            summary = direct_summary(rows, prefix)
            summary["selected"] = len(rows)
            candidates.append({
                "alpha": float(alpha),
                "prefix": prefix,
                "feature": "none",
                "rule": "all",
                "threshold": 0.0,
                "summary": summary,
            })
        for feature, rule in feature_rules:
            for threshold in threshold_candidates(rows, feature):
                summary = gated_summary(rows, prefix, feature, rule, threshold)
                if summary["selected"] < int(min_selected):
                    continue
                candidates.append({
                    "alpha": float(alpha),
                    "prefix": prefix,
                    "feature": feature,
                    "rule": rule,
                    "threshold": float(threshold),
                    "summary": summary,
                })
    return min(candidates, key=lambda c: c["summary"]["rmse"]["mean"]), candidates


def apply_candidate(candidate, rows_by_alpha):
    rows = rows_by_alpha[float(candidate["alpha"])]
    if candidate["rule"] == "all":
        summary = direct_summary(rows, candidate["prefix"])
        summary["selected"] = len(rows)
        return summary
    return gated_summary(
        rows,
        candidate["prefix"],
        candidate["feature"],
        candidate["rule"],
        candidate["threshold"],
    )


def summarize_alphas(rows_by_alpha):
    return {
        alpha_tag(alpha): direct_summary(rows, f"blend_{alpha_tag(alpha)}")
        for alpha, rows in rows_by_alpha.items()
    }


def save_rows_long(rows_by_alpha, path):
    keys = [
        "sample",
        "alpha",
        "delta_mean",
        "delta_edge_mean",
        "delta_lowconf_mean",
        "phase_conf_mean",
        "edge_mean",
    ]
    for prefix in ("base", "candidate", "diff"):
        keys.extend(f"{prefix}_{key}" for key in METRIC_KEYS)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for alpha, rows in rows_by_alpha.items():
            prefix = f"blend_{alpha_tag(alpha)}"
            for row in rows:
                out = {key: row[key] for key in keys[:7]}
                for metric in METRIC_KEYS:
                    out[f"base_{metric}"] = row[f"base_{metric}"]
                    out[f"candidate_{metric}"] = row[f"{prefix}_{metric}"]
                    out[f"diff_{metric}"] = row[f"diff_{metric}"]
                writer.writerow(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--phase_cache_dir", default="/root/autodl-tmp/fpp_ml_phase_cache_960")
    parser.add_argument("--base_prefix", default="base_c4_adapter")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["val", "test"])
    parser.add_argument("--gate_select_split", choices=["train", "val"], default="val")
    parser.add_argument("--image_h", type=int, default=960)
    parser.add_argument("--image_w", type=int, default=1280)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--start_ratio", type=float, default=0.05)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0])
    parser.add_argument("--gate_features", nargs="+", default=["all"])
    parser.add_argument("--min_selected", type=int, default=3)
    parser.add_argument("--no_allow_all", action="store_true")
    parser.add_argument("--save_long_csv", action="store_true")
    parser.add_argument("--require_cache", action="store_true")
    args = parser.parse_args()

    args.alphas = sorted({float(a) for a in args.alphas})
    feature_rules = parse_gate_features(args.gate_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first_ckpt = torch.load(args.checkpoints[0], map_location="cpu")
    saved_args = first_ckpt.get("args", {})
    include_ftp = bool(saved_arg(saved_args, "include_ftp", False))
    loaders = create_fpp_ml_bench_loaders(
        cache_dir=args.cache_dir,
        batch_size=1,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        include_ftp=include_ftp,
        image_h=args.image_h,
        image_w=args.image_w,
        require_cache=args.require_cache,
        base_prefix=args.base_prefix,
        phase_cache_dir=args.phase_cache_dir,
    )

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split = {}
    summaries = {}
    for split in args.splits:
        rows_by_alpha = evaluate_split(args, loaders, split, device)
        rows_by_split[split] = rows_by_alpha
        first_rows = next(iter(rows_by_alpha.values()))
        summaries[split] = {
            "base": direct_summary(first_rows, "base"),
            "diff": direct_summary(first_rows, "diff"),
            "alphas": summarize_alphas(rows_by_alpha),
        }
        if args.save_long_csv:
            save_rows_long(rows_by_alpha, out_dir / f"{split}_alpha_rows.csv")

    best, candidates = find_best_candidate(
        rows_by_split[args.gate_select_split],
        feature_rules,
        args.min_selected,
        allow_all=not args.no_allow_all,
    )
    val_selected = apply_candidate(best, rows_by_split["val"]) if "val" in rows_by_split else None
    test_selected = apply_candidate(best, rows_by_split["test"]) if "test" in rows_by_split else None

    result = {
        "checkpoints": args.checkpoints,
        "alphas": args.alphas,
        "gate_select_split": args.gate_select_split,
        "feature_rules": feature_rules,
        "selected": {
            "alpha": best["alpha"],
            "feature": best["feature"],
            "rule": best["rule"],
            "threshold": best["threshold"],
            "select_rmse": best["summary"]["rmse"]["mean"],
            "val_rmse": val_selected["rmse"]["mean"] if val_selected else None,
            "test_rmse": test_selected["rmse"]["mean"] if test_selected else None,
            "test_selected": test_selected["selected"] if test_selected else None,
        },
        "val": ({**summaries["val"], "selected": val_selected} if "val" in summaries else None),
        "test": ({**summaries["test"], "selected": test_selected} if "test" in summaries else None),
        "top_select_candidates": [
            {
                "alpha": c["alpha"],
                "feature": c["feature"],
                "rule": c["rule"],
                "threshold": c["threshold"],
                "val_summary": c["summary"],
                "test_summary": apply_candidate(c, rows_by_split["test"]) if "test" in rows_by_split else None,
            }
            for c in sorted(candidates, key=lambda x: x["summary"]["rmse"]["mean"])[:20]
        ],
    }
    (out_dir / "joint_alpha_gate_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "selected": result["selected"],
        "val_base": result["val"]["base"]["rmse"]["mean"] if result["val"] else None,
        "test_base": result["test"]["base"]["rmse"]["mean"] if result["test"] else None,
        "test_selected": result["test"]["selected"]["rmse"]["mean"] if result["test"] else None,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
