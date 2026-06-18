from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from data.dataset_fpp_phase import create_fpp_phase_loaders
from models import OfficialUNetFPPAdapter
from models.official_unet import OfficialUNetFPP
from train_fpp_official_style_unet import METRIC_KEYS, summarize
from train_fpp_psp_adapter_unet import cond_channel_count, evaluate_metrics, parse_indices
from train_fpp_phase2depth_unet import (
    evaluate_metrics as evaluate_phase2depth_metrics,
    input_channels,
)


def save_rows(rows, path):
    keys = ["sample"] + METRIC_KEYS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in keys})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved = dict(ckpt.get("args", {}))
    saved["num_workers"] = args.num_workers
    saved.setdefault("base_cache_dir", "/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix")
    saved.setdefault("phase_cache_dir", "/root/autodl-tmp/fpp_ml_pspquad_cache_960")
    saved.setdefault("phase_pred_prefix", None)
    saved.setdefault("batch_size", 2)
    saved.setdefault("eval_batch_size", 2)
    saved.setdefault("image_size", 960)
    saved.setdefault("preload_ram", False)
    saved.setdefault("train_minimal", False)
    saved.setdefault("instr_channels", "1-6")
    is_phase2depth = "input_mode" in saved and str(saved["input_mode"]) in {
        "gt_phase",
        "phase_pred",
        "gt_phase_plus_fringe",
        "phase_pred_plus_fringe",
    }
    if "cond_mode" not in saved and "input_mode" in saved and not is_phase2depth:
        saved["cond_mode"] = saved["input_mode"]
    saved["instr_channel_indices"] = saved.get("instr_channel_indices") or parse_indices(saved["instr_channels"])
    run_args = SimpleNamespace(**saved)

    loaders = create_fpp_phase_loaders(
        base_cache_dir=run_args.base_cache_dir,
        phase_cache_dir=run_args.phase_cache_dir,
        batch_size=run_args.batch_size,
        eval_batch_size=run_args.eval_batch_size,
        num_workers=run_args.num_workers,
        image_h=run_args.image_size,
        image_w=run_args.image_size,
        phase_pred_prefix=run_args.phase_pred_prefix,
        require_cache=True,
        preload_ram=run_args.preload_ram,
        train_minimal=run_args.train_minimal,
    )
    if is_phase2depth:
        model = OfficialUNetFPP(
            in_channels=input_channels(run_args.input_mode),
            out_channels=1,
            dropout_rate=float(run_args.dropout),
        ).to(device)
    else:
        model = OfficialUNetFPPAdapter(
            cond_channels=cond_channel_count(run_args.cond_mode, run_args.instr_channel_indices),
            out_channels=1,
            dropout_rate=float(run_args.dropout),
            adapter_hidden=int(run_args.adapter_hidden),
        ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"checkpoint": args.checkpoint, "splits": {}}
    for split in args.splits:
        if is_phase2depth:
            rows = evaluate_phase2depth_metrics(model, loaders[split], device, run_args.input_mode)
        else:
            rows = evaluate_metrics(model, loaders[split], device, run_args)
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        save_rows(rows, split_dir / "per_sample_metrics.csv")
        split_summary = summarize(rows)
        split_summary["n"] = len(rows)
        summary["splits"][split] = split_summary
        with open(split_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(split_summary, f, indent=2, ensure_ascii=False)
        print(json.dumps({"split": split, "rmse": split_summary["rmse"]["mean"]}, ensure_ascii=False))
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
