#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5

PY=/root/miniconda3/bin/python
DATA=/root/autodl-tmp/single_frame_3d_dataset_v1_upload_smalltest
EXTRA=/root/autodl-tmp/single_frame_3d_dataset_v1_teacher_extra
OOD=/root/autodl-tmp/single_frame_3d_dataset_v1_upload_smalltest_ood61_64
CACHE=/root/autodl-tmp/single_frame_3d_dataset_v1_upload_smalltest/physics_feature_cache_pip
XDIAG=/root/autodl-tmp/diffusion_fpp_v5/results/A_20260617_single_frame3d_phase_residual_diagnosis_xonly
DIRECT=/root/autodl-tmp/diffusion_fpp_v5/results/A_20260619_formal_strong_backbone_direct_seed012
OLD=/root/autodl-tmp/diffusion_fpp_v5/results/A_20260618_refined_xphase_depth
OUT=/root/autodl-tmp/diffusion_fpp_v5/results/A_20260619_formal_attention_unet_ours_selector_seed012
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

echo "===== waiting for formal strong direct queue $(date '+%F %T') =====" | tee "$LOGDIR/master.log"
while pgrep -af 'run_formal_strong_backbone_direct_seed012.sh' >/dev/null; do
  sleep 60
done
echo "===== formal strong direct queue finished $(date '+%F %T') =====" | tee -a "$LOGDIR/master.log"

run_selector() {
  local seed="$1"
  local phase="$2"
  local refined="$3"
  local summary="$4"
  local base="$DIRECT/attention_unet_seed${seed}/checkpoints/best.pt"
  local run="$OUT/seed${seed}"
  mkdir -p "$run"
  while [ ! -f "$base" ]; do
    echo "waiting for base checkpoint: $base" | tee -a "$LOGDIR/master.log"
    sleep 60
  done
  if [ -f "$run/reliability_selector_seed${seed}/reliability_selector_summary.json" ]; then
    echo "===== skip existing formal attention selector seed${seed} =====" | tee -a "$LOGDIR/master.log"
    return
  fi
  echo "===== formal attention selector seed${seed} $(date '+%F %T') =====" | tee -a "$LOGDIR/master.log"
  "$PY" -u train_refined_xphase_reliability_selector.py \
    --data_root "$DATA" \
    --teacher_extra_root "$EXTRA" \
    --ood_root "$OOD" \
    --save_dir "$run" \
    --base_ckpt "$base" \
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

run_selector 0 \
  /root/autodl-tmp/diffusion_fpp_v5/results/A_20260617_single_frame3d_xphase_diffusion_rcpc/x_phase_diffusion/checkpoints/best.pt \
  "$OLD/refined_xphase_depth/checkpoints/best.pt" \
  "$OLD/refined_xphase_depth_summary.json"

run_selector 1 \
  "$OLD/fullchain_seed1/xphase_diffusion_rcpc/x_phase_diffusion/checkpoints/best.pt" \
  "$OLD/fullchain_seed1/refined_xphase_depth/refined_xphase_depth/checkpoints/best.pt" \
  "$OLD/fullchain_seed1/refined_xphase_depth/refined_xphase_depth_summary.json"

run_selector 2 \
  "$OLD/fullchain_seed2/xphase_diffusion_rcpc/x_phase_diffusion/checkpoints/best.pt" \
  "$OLD/fullchain_seed2/refined_xphase_depth/refined_xphase_depth/checkpoints/best.pt" \
  "$OLD/fullchain_seed2/refined_xphase_depth/refined_xphase_depth_summary.json"

"$PY" - <<'PY' > "$OUT/formal_attention_unet_ours_selector_summary.json"
import json
import statistics
from pathlib import Path

out = Path("/root/autodl-tmp/diffusion_fpp_v5/results/A_20260619_formal_attention_unet_ours_selector_seed012")
rows = []
for p in sorted(out.glob("seed*/reliability_selector_seed*/reliability_selector_summary.json")):
    data = json.loads(p.read_text(encoding="utf-8"))
    seed = int(data["seed"])
    for split, m in data["splits"].items():
        rows.append({
            "seed": seed,
            "split": split,
            "base": float(m["base"]),
            "x_phase": float(m["x_phase"]),
            "anchor": float(m["anchor"]),
            "refined": float(m["refined"]),
            "sample_rcpc": float(m["sample_rcpc"]),
            "rule": float(m["rule_gate"]["gate_rmse"]),
            "mlp": float(m["mlp_gate"]["gate_rmse"]),
            "true_x_oracle": float(m["true_x_oracle"]),
        })

def stat(split, key):
    vals = [r[key] for r in rows if r["split"] == split]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }

summary = {
    "stage": "formal_attention_unet_ours_selector_seed012",
    "note": "Formal Attention UNet direct base checkpoints are trained up to 80 epochs with best-val checkpoint selection.",
    "rows": rows,
    "aggregate": {
        split: {key: stat(split, key) for key in ["base", "x_phase", "anchor", "refined", "sample_rcpc", "rule", "mlp", "true_x_oracle"]}
        for split in ["val", "test", "ood"]
    },
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

"$PY" - <<'PY' > "$OUT/formal_attention_unet_ours_selector_report.md"
import json
from pathlib import Path

out = Path("/root/autodl-tmp/diffusion_fpp_v5/results/A_20260619_formal_attention_unet_ours_selector_seed012")
data = json.loads((out / "formal_attention_unet_ours_selector_summary.json").read_text(encoding="utf-8"))
lines = [
    "# Formal Attention UNet + Phase Evidence Selector",
    "",
    "Attention UNet direct base uses formal 80-epoch runs with best validation checkpoint selection.",
    "Phase/refined-depth evidence uses the existing x-phase posterior/refined-depth seeds.",
    "",
    "| split | base | x phase | anchor | refined | rule | MLP | true x oracle |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for split in ["test", "ood", "val"]:
    agg = data["aggregate"][split]
    def cell(key):
        s = agg[key]
        return f"{s['mean']:.4f} +/- {s['std']:.4f}"
    lines.append(
        f"| {split} | {cell('base')} | {cell('x_phase')} | {cell('anchor')} | "
        f"{cell('refined')} | {cell('rule')} | {cell('mlp')} | {cell('true_x_oracle')} |"
    )
lines += ["", "This table is suitable as a paper-facing strong-backbone probe, but final use should note that the phase/refined candidates reuse existing x-phase evidence checkpoints."]
print("\\n".join(lines))
PY

echo "===== done formal attention selector $(date '+%F %T') =====" | tee -a "$LOGDIR/master.log"
