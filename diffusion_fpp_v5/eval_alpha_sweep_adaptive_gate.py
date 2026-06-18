from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.dataset_fpp_ml_bench import create_fpp_ml_bench_loaders
from diffusion_pip import PIPDiffusion
from eval_adaptive_blend_features import _saved_arg, build_model, masked_scalar
from train_fpp_official_style_unet import METRIC_KEYS, parse_channel_spec, summarize
from train_pip_lite import prediction_to_mm
from utils.metrics import compute_metrics


FEATURE_RULES = [
    ("edge_mean", "le"),
    ("delta_mean", "ge"),
    ("delta_edge_mean", "ge"),
    ("delta_lowconf_mean", "ge"),
    ("phase_conf_mean", "ge"),
]


def alpha_tag(alpha: float) -> str:
    return f"a{int(round(alpha * 1000)):03d}"


def threshold_candidates(rows, feature):
    vals = sorted({float(row[feature]) for row in rows})
    if not vals:
        return [0.0]
    mids = [(a + b) * 0.5 for a, b in zip(vals[:-1], vals[1:])]
    return [vals[0] - 1e-6, *vals, *mids, vals[-1] + 1e-6]


def selected(row, feature, rule, threshold):
    value = float(row[feature])
    if rule == "le":
        return value <= threshold
    if rule == "ge":
        return value >= threshold
    raise ValueError(f"unknown rule: {rule}")


def direct_summary(rows, prefix):
    out = {"n": len(rows)}
    for metric in METRIC_KEYS:
        arr = np.asarray([float(row[f"{prefix}_{metric}"]) for row in rows], dtype=np.float64)
        out[metric] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        }
    return out


def gated_summary(rows, candidate_prefix, feature, rule, threshold):
    out = {
        "n": len(rows),
        "selected": int(sum(selected(row, feature, rule, threshold) for row in rows)),
    }
    for metric in METRIC_KEYS:
        vals = []
        for row in rows:
            prefix = candidate_prefix if selected(row, feature, rule, threshold) else "base"
            vals.append(float(row[f"{prefix}_{metric}"]))
        arr = np.asarray(vals, dtype=np.float64)
        out[metric] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        }
    return out


def save_selected_rows(rows, candidate_prefix, feature, rule, threshold, path):
    keys = [
        "sample",
        "selected",
        "alpha",
        "gate_feature",
        "gate_rule",
        "gate_threshold",
        "delta_mean",
        "delta_edge_mean",
        "delta_lowconf_mean",
        "phase_conf_mean",
        "edge_mean",
    ]
    for prefix in ("base", candidate_prefix, "diff", "final"):
        keys.extend(f"{prefix}_{key}" for key in METRIC_KEYS)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            use = selected(row, feature, rule, threshold)
            out = {
                "sample": row["sample"],
                "selected": int(use),
                "alpha": row["alpha"],
                "gate_feature": feature,
                "gate_rule": rule,
                "gate_threshold": threshold,
                "delta_mean": row["delta_mean"],
                "delta_edge_mean": row["delta_edge_mean"],
                "delta_lowconf_mean": row["delta_lowconf_mean"],
                "phase_conf_mean": row["phase_conf_mean"],
                "edge_mean": row["edge_mean"],
            }
            for metric in METRIC_KEYS:
                out[f"base_{metric}"] = row[f"base_{metric}"]
                out[f"{candidate_prefix}_{metric}"] = row[f"{candidate_prefix}_{metric}"]
                out[f"diff_{metric}"] = row[f"diff_{metric}"]
                final_prefix = candidate_prefix if use else "base"
                out[f"final_{metric}"] = row[f"{final_prefix}_{metric}"]
            writer.writerow(out)


def save_long_rows(rows_by_alpha, path):
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
            candidate_prefix = f"blend_{alpha_tag(alpha)}"
            for row in rows:
                out = {
                    "sample": row["sample"],
                    "alpha": alpha,
                    "delta_mean": row["delta_mean"],
                    "delta_edge_mean": row["delta_edge_mean"],
                    "delta_lowconf_mean": row["delta_lowconf_mean"],
                    "phase_conf_mean": row["phase_conf_mean"],
                    "edge_mean": row["edge_mean"],
                }
                for metric in METRIC_KEYS:
                    out[f"base_{metric}"] = row[f"base_{metric}"]
                    out[f"candidate_{metric}"] = row[f"{candidate_prefix}_{metric}"]
                    out[f"diff_{metric}"] = row[f"diff_{metric}"]
                writer.writerow(out)


def parse_gate_features(values):
    if not values:
        return FEATURE_RULES
    if len(values) == 1 and values[0].lower() == "all":
        return FEATURE_RULES
    out = []
    for value in values:
        if ":" in value:
            feature, rule = value.split(":", 1)
        else:
            feature = value
            rule = "le" if value == "edge_mean" else "ge"
        if rule not in {"le", "ge"}:
            raise ValueError(f"unknown gate rule in {value}: {rule}")
        out.append((feature, rule))
    return out


@torch.no_grad()
def evaluate_split(args, loaders, split, saved_args, physics_indices, model_cond_channels, device):
    loader = loaders["train_eval" if split == "train" else split]
    base_store = []
    target_store = []
    mask_store = []
    edge_store = []
    conf_store = []
    minmax_store = []
    pred_sum = None

    for ckpt_idx, ckpt_path in enumerate(args.checkpoints):
        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt_args = ckpt.get("args", saved_args)
        model = build_model(ckpt_args, model_cond_channels).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        diffusion = PIPDiffusion(
            model,
            timesteps=int(_saved_arg(ckpt_args, "timesteps", 200)),
            image_h=args.image_h,
            image_w=args.image_w,
            device=device,
            cond_indices=physics_indices,
            target_mode=str(_saved_arg(ckpt_args, "target_mode", "base_residual")),
            residual_scale=float(_saved_arg(ckpt_args, "resolved_residual_scale", 1.0)),
            base_residual_gate=float(_saved_arg(ckpt_args, "base_residual_gate", 1.0)),
        )
        preds = []
        for batch in tqdm(loader, desc=f"{Path(ckpt_path).parts[-3]} {split}"):
            pred = diffusion.sample_ddim(
                batch,
                steps=args.ddim_steps,
                ensemble_size=1,
                start_from_base=True,
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
        del model, diffusion
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
        diff_raw = diff_all[i:i + 1].to(device, non_blocking=True)
        mask = mask_all[i:i + 1].to(device, non_blocking=True)
        edge = edge_all[i:i + 1].to(device, non_blocking=True)
        conf = conf_all[i:i + 1].to(device, non_blocking=True)
        delta = torch.abs(diff_raw - base)
        batch_stub = {"depth_minmax": minmax_all[i:i + 1].to(device, non_blocking=True)}
        target = target_all[i:i + 1].to(device, non_blocking=True)
        base_mm = prediction_to_mm(base, batch_stub, loaders["height_scale"])
        diff_mm = prediction_to_mm(diff_raw, batch_stub, loaders["height_scale"])
        base_metrics = compute_metrics(base_mm, target, mask=mask)
        diff_metrics = compute_metrics(diff_mm, target, mask=mask)
        common = {
            "sample": i,
            "delta_mean": masked_scalar(delta, mask),
            "delta_edge_mean": masked_scalar(delta * edge, mask),
            "delta_lowconf_mean": masked_scalar(delta * (1.0 - conf), mask),
            "phase_conf_mean": masked_scalar(conf, mask),
            "edge_mean": masked_scalar(edge, mask),
        }
        for alpha in args.alphas:
            candidate_prefix = f"blend_{alpha_tag(alpha)}"
            blend = torch.clamp(base + float(alpha) * (diff_raw - base), -1.0, 1.0)
            blend_mm = prediction_to_mm(blend, batch_stub, loaders["height_scale"])
            blend_metrics = compute_metrics(blend_mm, target, mask=mask)
            row = {**common, "alpha": float(alpha)}
            row.update({f"base_{key}": base_metrics[key] for key in METRIC_KEYS})
            row.update({f"diff_{key}": diff_metrics[key] for key in METRIC_KEYS})
            row.update({f"{candidate_prefix}_{key}": blend_metrics[key] for key in METRIC_KEYS})
            rows_by_alpha[float(alpha)].append(row)
    return rows_by_alpha


def find_best_candidate(val_rows_by_alpha, min_selected, min_selected_frac, feature_rules, allow_all=True):
    candidates = []
    for alpha, rows in val_rows_by_alpha.items():
        prefix = f"blend_{alpha_tag(alpha)}"
        effective_min = max(int(min_selected), int(np.ceil(len(rows) * float(min_selected_frac))))
        if allow_all:
            direct = direct_summary(rows, prefix)
            direct["selected"] = len(rows)
            candidates.append({
                "alpha": float(alpha),
                "feature": "none",
                "rule": "all",
                "threshold": 0.0,
                "candidate_prefix": prefix,
                "summary": direct,
            })
        for feature, rule in feature_rules:
            for threshold in threshold_candidates(rows, feature):
                summary = gated_summary(rows, prefix, feature, rule, threshold)
                if summary["selected"] < effective_min:
                    continue
                candidates.append({
                    "alpha": float(alpha),
                    "feature": feature,
                    "rule": rule,
                    "threshold": float(threshold),
                    "candidate_prefix": prefix,
                    "summary": summary,
                })
    return min(candidates, key=lambda c: c["summary"]["rmse"]["mean"]), candidates


def candidate_test_summary(candidate, test_rows_by_alpha):
    rows = test_rows_by_alpha[candidate["alpha"]]
    prefix = candidate["candidate_prefix"]
    if candidate["rule"] == "all":
        summary = direct_summary(rows, prefix)
        summary["selected"] = len(rows)
        return summary
    return gated_summary(rows, prefix, candidate["feature"], candidate["rule"], candidate["threshold"])


def summarize_alpha_rows(rows_by_alpha):
    out = {}
    for alpha, rows in rows_by_alpha.items():
        prefix = f"blend_{alpha_tag(alpha)}"
        out[alpha_tag(alpha)] = direct_summary(rows, prefix)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--base_prefix", default="base_c4_adapter")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["val", "test"])
    parser.add_argument("--gate_select_split", choices=["train", "val"], default="val",
                        help="Split used to select alpha/gate. The selected rule is then applied to val/test.")
    parser.add_argument("--image_h", type=int, default=960)
    parser.add_argument("--image_w", type=int, default=960)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--start_ratio", type=float, default=0.05)
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.75, 1.0])
    parser.add_argument("--min_selected", type=int, default=3)
    parser.add_argument("--min_selected_frac", type=float, default=0.0)
    parser.add_argument("--gate_features", nargs="+", default=["all"],
                        help="Gate feature rules, e.g. edge_mean:le phase_conf_mean:ge. Use 'all' for the default search space.")
    parser.add_argument("--no_allow_all", action="store_true",
                        help="Disable global alpha candidates that apply diffusion correction to every sample.")
    parser.add_argument("--save_long_csv", action="store_true")
    parser.add_argument("--require_cache", action="store_true")
    args = parser.parse_args()

    args.alphas = sorted({float(a) for a in args.alphas})
    feature_rules = parse_gate_features(args.gate_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first_ckpt = torch.load(args.checkpoints[0], map_location=device)
    saved_args = first_ckpt.get("args", {})
    include_ftp = bool(_saved_arg(saved_args, "include_ftp", False))
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
    )
    physics_indices = _saved_arg(saved_args, "physics_channel_indices", None)
    if physics_indices is None:
        physics_indices = parse_channel_spec(str(_saved_arg(saved_args, "physics_channels", "")), include_ftp)
    model_cond_channels = int(first_ckpt.get(
        "model_cond_channels",
        len(physics_indices) if physics_indices is not None else loaders["cond_channels"],
    ))

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split = {}
    split_summaries = {}
    for split in args.splits:
        rows_by_alpha = evaluate_split(args, loaders, split, saved_args, physics_indices, model_cond_channels, device)
        rows_by_split[split] = rows_by_alpha
        if args.save_long_csv:
            save_long_rows(rows_by_alpha, out_dir / f"{split}_alpha_rows.csv")
        first_rows = next(iter(rows_by_alpha.values()))
        split_summaries[split] = {
            "base": direct_summary(first_rows, "base"),
            "diff": direct_summary(first_rows, "diff"),
            "alphas": summarize_alpha_rows(rows_by_alpha),
        }

    if args.gate_select_split not in rows_by_split or "test" not in rows_by_split:
        raise RuntimeError("gate_select_split and test splits are required for adaptive alpha gate selection")
    best, candidates = find_best_candidate(
        rows_by_split[args.gate_select_split],
        args.min_selected,
        args.min_selected_frac,
        feature_rules,
        allow_all=not args.no_allow_all,
    )
    prefix = best["candidate_prefix"]
    if best["rule"] == "all":
        val_final = direct_summary(rows_by_split["val"][best["alpha"]], prefix) if "val" in rows_by_split else None
        if val_final is not None:
            val_final["selected"] = len(rows_by_split["val"][best["alpha"]])
        test_final = direct_summary(rows_by_split["test"][best["alpha"]], prefix)
        test_final["selected"] = len(rows_by_split["test"][best["alpha"]])
        save_selected_rows(
            rows_by_split["test"][best["alpha"]],
            prefix,
            "edge_mean",
            "ge",
            -1.0,
            out_dir / "test_selected_rows.csv",
        )
    else:
        val_final = (
            gated_summary(rows_by_split["val"][best["alpha"]], prefix, best["feature"], best["rule"], best["threshold"])
            if "val" in rows_by_split else None
        )
        test_final = gated_summary(rows_by_split["test"][best["alpha"]], prefix, best["feature"], best["rule"], best["threshold"])
        save_selected_rows(
            rows_by_split["test"][best["alpha"]],
            prefix,
            best["feature"],
            best["rule"],
            best["threshold"],
            out_dir / "test_selected_rows.csv",
        )

    result = {
        "checkpoints": args.checkpoints,
        "gate_search": {
            "feature_rules": feature_rules,
            "allow_all": not args.no_allow_all,
            "min_selected": args.min_selected,
            "min_selected_frac": args.min_selected_frac,
            "select_split": args.gate_select_split,
        },
        "selected": {
            "alpha": best["alpha"],
            "candidate_prefix": prefix,
            "feature": best["feature"],
            "rule": best["rule"],
            "threshold": best["threshold"],
            "select_split": args.gate_select_split,
            "select_rmse": best["summary"]["rmse"]["mean"],
            "val_rmse": val_final["rmse"]["mean"] if val_final is not None else None,
        },
        "train": split_summaries.get("train"),
        "val": ({**split_summaries["val"], "selected": val_final} if "val" in split_summaries else None),
        "test": {
            **split_summaries["test"],
            "selected": test_final,
        },
        "top_select_candidates": [
            {**c, "test_summary": candidate_test_summary(c, rows_by_split["test"])}
            for c in sorted(candidates, key=lambda c: c["summary"]["rmse"]["mean"])[:20]
        ],
    }
    (out_dir / "alpha_gate_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "selected": result["selected"],
        "val_base": result["val"]["base"]["rmse"]["mean"] if result["val"] is not None else None,
        "val_selected": result["val"]["selected"]["rmse"]["mean"] if result["val"] is not None else None,
        "test_base": result["test"]["base"]["rmse"]["mean"],
        "test_selected": result["test"]["selected"]["rmse"]["mean"],
        "test_selected_n": result["test"]["selected"]["selected"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
