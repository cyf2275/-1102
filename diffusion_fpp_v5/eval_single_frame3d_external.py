"""Evaluate trained SingleFrame3D checkpoints on an external test split."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from train_single_frame3d_physics_diffusion import (
    ConditionalUNet,
    ResidualPosterior,
    SingleFrame3DDataset,
    build_model,
    canonical_config,
    collate_single_frame,
    evaluate_direct,
    evaluate_residual_split,
    load_base_model,
    save_rows,
    summarize_rows,
)


def namespace_from_dict(data: Dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(**data)


def load_direct_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, SimpleNamespace]:
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = dict(ckpt.get("args", {}))
    args = namespace_from_dict(saved_args)
    cond_channels = int(saved_args["cond_channels"])
    model = build_model(cond_channels, 5 if saved_args.get("config") == "teacher_aux" else 1, args).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, args


def make_loader(args: argparse.Namespace, config: str) -> DataLoader:
    ds = SingleFrame3DDataset(
        data_root=args.data_root,
        split=args.split,
        config=config,
        cache_features=args.cache_features,
        feature_cache_dir=args.feature_cache_dir or None,
    )

    def collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
        return collate_single_frame(batch, image_h=args.image_h, image_w=args.image_w)

    return DataLoader(
        ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )


def write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def eval_direct_run(args: argparse.Namespace, ckpt_path: Path, config: str, seed: int, device: torch.device) -> Dict[str, object]:
    loader = make_loader(args, config)
    model, model_args = load_direct_model(ckpt_path, device)
    model_args.config = config
    rows = evaluate_direct(model, loader, device, model_args, mode=args.mode_name)
    out_dir = args.out_dir / "runs" / f"direct_{config}_seed{seed}"
    save_rows(rows, out_dir / "per_sample_metrics.csv")
    summary = {
        "stage": "direct_external",
        "config": config,
        "seed": seed,
        "checkpoint": str(ckpt_path),
        "data_root": str(args.data_root),
        "split": args.split,
        "mode": args.mode_name,
        "n": len(rows),
        "object": summarize_rows(rows)["object"],
        "valid": summarize_rows(rows)["valid"],
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def eval_residual_run(args: argparse.Namespace, ckpt_path: Path, config: str, seed: int, device: torch.device) -> Dict[str, object]:
    loader = make_loader(args, config)
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = dict(ckpt.get("args", {}))
    model_args = namespace_from_dict(saved_args)
    model_args.config = config
    base_ckpt = Path(str(saved_args["base_ckpt"]))
    base_model, _ = load_base_model(base_ckpt, int(saved_args["cond_channels"]), device)
    model = ConditionalUNet(
        in_channels=1,
        cond_channels=int(saved_args.get("posterior_cond_channels", int(saved_args["cond_channels"]) + 1)),
        out_channels=1,
        base_ch=int(saved_args["base_channels"]),
        ch_mult=tuple(saved_args["ch_mult"]),
        num_res_blocks=int(saved_args["num_res_blocks"]),
        dropout=float(saved_args["dropout"]),
        time_emb_dim=int(saved_args["time_emb_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    posterior = ResidualPosterior(
        model,
        timesteps=int(saved_args["timesteps"]),
        residual_scale=float(saved_args["residual_scale"]),
        device=device,
    )
    gate = None
    old_summary = ckpt_path.parents[1] / "evaluation" / "summary.json"
    if old_summary.exists():
        old = json.loads(old_summary.read_text(encoding="utf-8"))
        gate = old.get("gate_applied_to_test") or old.get("gate_selected_on_val")
        if isinstance(gate, dict):
            gate = {k: v for k, v in gate.items() if k in {"threshold", "alpha"}}
    model_args.sample_steps = args.sample_steps
    model_args.ensemble_size = args.ensemble_size
    result = evaluate_residual_split(posterior, base_model, loader, device, model_args, gate=gate, split=args.mode_name)
    out_dir = args.out_dir / "runs" / f"residual_{config}_seed{seed}"
    save_rows(result["base_unet"], out_dir / "base_unet_per_sample_metrics.csv")
    save_rows(result["posterior_mean"], out_dir / "posterior_mean_per_sample_metrics.csv")
    save_rows(result["posterior_gate"], out_dir / "posterior_gate_per_sample_metrics.csv")
    summary = {
        "stage": "residual_external",
        "config": config,
        "seed": seed,
        "checkpoint": str(ckpt_path),
        "base_checkpoint": str(base_ckpt),
        "data_root": str(args.data_root),
        "split": args.split,
        "mode": args.mode_name,
        "n": len(result["posterior_gate"]),
        "gate": result["gate"],
        "base_unet": summarize_rows(result["base_unet"]),
        "posterior_mean": summarize_rows(result["posterior_mean"]),
        "posterior_gate": summarize_rows(result["posterior_gate"]),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def aggregate(args: argparse.Namespace, summaries: List[Dict[str, object]]) -> None:
    rows: List[Dict[str, object]] = []
    for s in summaries:
        if s["stage"] == "direct_external":
            rows.append({
                "stage": "direct",
                "config": s["config"],
                "seed": s["seed"],
                "object_rmse": s["object"]["rmse"]["mean"],
                "valid_rmse": s["valid"]["rmse"]["mean"],
            })
        else:
            base = float(s["base_unet"]["object"]["rmse"]["mean"])
            gate = float(s["posterior_gate"]["object"]["rmse"]["mean"])
            rows.append({
                "stage": "residual",
                "config": s["config"],
                "seed": s["seed"],
                "base_object_rmse": base,
                "posterior_mean_object_rmse": s["posterior_mean"]["object"]["rmse"]["mean"],
                "posterior_gate_object_rmse": gate,
                "gate_gain_percent": (base - gate) / base * 100.0 if base else float("nan"),
                "valid_base_rmse": s["base_unet"]["valid"]["rmse"]["mean"],
                "valid_gate_rmse": s["posterior_gate"]["valid"]["rmse"]["mean"],
            })
    write_csv(rows, args.out_dir / "external_eval_runs.csv")
    write_json(args.out_dir / "external_eval_summary.json", {
        "data_root": str(args.data_root),
        "split": args.split,
        "mode": args.mode_name,
        "n_runs": len(rows),
        "runs": rows,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--result_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--mode_name", default="ood61_64")
    parser.add_argument("--configs", nargs="+", default=["raw", "raw_single_phys", "teacher_aux"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_h", type=int, default=480)
    parser.add_argument("--image_w", type=int, default=640)
    parser.add_argument("--sample_steps", type=int, default=12)
    parser.add_argument("--ensemble_size", type=int, default=3)
    parser.add_argument("--cache_features", action="store_true")
    parser.add_argument("--feature_cache_dir", default="")
    parser.add_argument("--include_residual", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries: List[Dict[str, object]] = []
    for cfg_raw in args.configs:
        cfg = canonical_config(cfg_raw)
        for seed in args.seeds:
            direct_ckpt = args.result_root / "runs" / f"direct_{cfg}_seed{seed}" / "checkpoints" / "best.pt"
            if direct_ckpt.exists():
                summaries.append(eval_direct_run(args, direct_ckpt, cfg, seed, device))
            if args.include_residual and cfg in {"raw", "raw_single_phys", "teacher_aux"}:
                residual_ckpt = args.result_root / "runs" / f"residual_{cfg}_seed{seed}" / "checkpoints" / "best.pt"
                if residual_ckpt.exists():
                    summaries.append(eval_residual_run(args, residual_ckpt, cfg, seed, device))
    aggregate(args, summaries)
    print(args.out_dir / "external_eval_summary.json")


if __name__ == "__main__":
    main()
