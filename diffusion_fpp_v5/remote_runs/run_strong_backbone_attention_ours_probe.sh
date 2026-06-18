#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5

PY=/root/miniconda3/bin/python
DATA=/root/autodl-tmp/single_frame_3d_dataset_v1_upload_smalltest
EXTRA=/root/autodl-tmp/single_frame_3d_dataset_v1_teacher_extra
OOD=/root/autodl-tmp/single_frame_3d_dataset_v1_upload_smalltest_ood61_64
CACHE=/root/autodl-tmp/single_frame_3d_dataset_v1_upload_smalltest/physics_feature_cache_pip
BASE=/root/autodl-tmp/diffusion_fpp_v5/results/A_20260618_single_frame3d_baseline_comparison_quick1seed/attention_unet/checkpoints/best.pt
XDIAG=/root/autodl-tmp/diffusion_fpp_v5/results/A_20260617_single_frame3d_phase_residual_diagnosis_xonly
OUT=/root/autodl-tmp/diffusion_fpp_v5/results/A_20260619_strong_backbone_ours_probe/attention_unet_base
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

run_selector() {
  local seed="$1"
  local phase="$2"
  local refined="$3"
  local summary="$4"
  local run="$OUT/seed${seed}"
  mkdir -p "$run"
  echo "===== attention-unet base selector seed${seed} $(date '+%F %T') =====" | tee -a "$LOGDIR/master.log"
  "$PY" -u train_refined_xphase_reliability_selector.py \
    --data_root "$DATA" \
    --teacher_extra_root "$EXTRA" \
    --ood_root "$OOD" \
    --save_dir "$run" \
    --base_ckpt "$BASE" \
    --x_diag_dir "$XDIAG" \
    --phase_posterior_ckpt "$phase" \
    --refined_depth_ckpt "$refined" \
    --summary_path "$summary" \
    --seed "$seed" \
    --batch_size 1 \
    --eval_batch_size 1 \
    --num_workers 2 \
    --cache_features \
    --feature_cache_dir "$CACHE" \
    --train_pixels_per_sample 2048 \
    --max_train_pixels 700000 \
    --selector_epochs 25 \
    --anchor_mode base_x_mean \
    2>&1 | tee "$LOGDIR/selector_seed${seed}.log"
}

ROOT=/root/autodl-tmp/diffusion_fpp_v5/results/A_20260618_refined_xphase_depth
run_selector 0 \
  /root/autodl-tmp/diffusion_fpp_v5/results/A_20260617_single_frame3d_xphase_diffusion_rcpc/x_phase_diffusion/checkpoints/best.pt \
  "$ROOT/refined_xphase_depth/checkpoints/best.pt" \
  "$ROOT/refined_xphase_depth_summary.json"

run_selector 1 \
  "$ROOT/fullchain_seed1/xphase_diffusion_rcpc/x_phase_diffusion/checkpoints/best.pt" \
  "$ROOT/fullchain_seed1/refined_xphase_depth/refined_xphase_depth/checkpoints/best.pt" \
  "$ROOT/fullchain_seed1/refined_xphase_depth/refined_xphase_depth_summary.json"

run_selector 2 \
  "$ROOT/fullchain_seed2/xphase_diffusion_rcpc/x_phase_diffusion/checkpoints/best.pt" \
  "$ROOT/fullchain_seed2/refined_xphase_depth/refined_xphase_depth/checkpoints/best.pt" \
  "$ROOT/fullchain_seed2/refined_xphase_depth/refined_xphase_depth_summary.json"

"$PY" - <<'PY' | tee "$OUT/attention_unet_base_ours_summary.json"
import json
from pathlib import Path

out = Path("/root/autodl-tmp/diffusion_fpp_v5/results/A_20260619_strong_backbone_ours_probe/attention_unet_base")
rows = []
for p in sorted(out.glob("seed*/reliability_selector_seed*/reliability_selector_summary.json")):
    data = json.loads(p.read_text(encoding="utf-8"))
    seed = data.get("seed")
    for split, metrics in data["splits"].items():
        rows.append({
            "seed": seed,
            "split": split,
            "anchor": metrics["anchor"],
            "base": metrics["base"],
            "x_phase": metrics["x_phase"],
            "refined": metrics["refined"],
            "sample_rcpc": metrics["sample_rcpc"],
            "rule": metrics["rule_gate"]["gate_rmse"],
            "mlp": metrics["mlp_gate"]["gate_rmse"],
            "true_x_oracle": metrics["true_x_oracle"],
        })

def stats(split, key):
    vals = [float(r[key]) for r in rows if r["split"] == split]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
    return {"mean": mean, "std": var ** 0.5, "n": len(vals)}

summary = {
    "stage": "strong_backbone_attention_unet_base_ours_probe",
    "note": "Attention UNet direct base is fixed from quick screening seed0; phase posterior/refined-depth/selector seeds are 0,1,2.",
    "rows": rows,
    "aggregate": {
        split: {key: stats(split, key) for key in ["base", "x_phase", "anchor", "refined", "sample_rcpc", "rule", "mlp", "true_x_oracle"]}
        for split in ["val", "test", "ood"]
    },
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "===== done attention-unet base probe $(date '+%F %T') =====" | tee -a "$LOGDIR/master.log"
