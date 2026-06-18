#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/diffusion_fpp_v5
mkdir -p results/remote_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY=/root/miniconda3/bin/python
BASE_CACHE=/root/autodl-tmp/fpp_ml_bench_cache_960_fgfix
PSP_CACHE=/root/autodl-tmp/fpp_ml_pspquad_cache_960

E60_DIR=results/fpp960_e60_pspquad_phase_diffusion_ch32_e30
LATEST_PREFIX=phase_pred_e60_latest_pspquad_ddim20_e3

E62_DIR=results/fpp960_e62_pspquad_phase_diffusion_ch24_reg_e30
MASTER_LOG=results/remote_logs/e62_phase_latest_eval_and_ch24_reg_master.log

echo "START E62 phase latest-eval + ch24 regularized $(date '+%F %T')" | tee "$MASTER_LOG"

echo "EVAL E60 latest checkpoint on test $(date '+%F %T')" | tee -a "$MASTER_LOG"
if [[ ! -f "$PSP_CACHE/${LATEST_PREFIX}_test_summary.json" ]]; then
  "$PY" precompute_fpp_phase_diffusion_predictions.py \
    --checkpoint "$E60_DIR/checkpoints/latest.pt" \
    --base_cache_dir "$BASE_CACHE" \
    --phase_cache_dir "$PSP_CACHE" \
    --output_prefix "$LATEST_PREFIX" \
    --splits test \
    --image_size 960 \
    --batch_size 1 \
    --num_workers 8 \
    --ddim_steps 20 \
    --ensemble 3 \
    --sample_start_from ftp \
    --sample_start_ratio 0.7 \
    2>&1 | tee results/remote_logs/e60_latest_test_precompute.log
fi

echo "START E62 ch24 regularized PSP diffusion $(date '+%F %T')" | tee -a "$MASTER_LOG"
"$PY" train_fpp_phase_diffusion.py \
  --base_cache_dir "$BASE_CACHE" \
  --phase_cache_dir "$PSP_CACHE" \
  --save_dir "$E62_DIR" \
  --phase_channels 0-12 \
  --epochs 30 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --num_workers 8 \
  --image_size 960 \
  --base_channels 24 \
  --ch_mult 1,2,4,8,8 \
  --adapter_hidden 24 \
  --dropout 0.10 \
  --target_channels 3 \
  --timesteps 200 \
  --ddim_steps 20 \
  --ensemble 3 \
  --sample_start_from ftp \
  --sample_start_ratio 0.7 \
  --train_start_from target \
  --train_t_min_ratio 0.0 \
  --train_t_max_ratio 1.0 \
  --phase_weight 1.0 \
  --grad_weight 0.05 \
  --unit_weight 0.02 \
  --uph_norm sample \
  --uph_start_from half \
  --uph_weight 0.5 \
  --uph_grad_weight 0.02 \
  --selection_metric phase_aligned_mae_rad \
  --lr 6e-5 \
  --weight_decay 2e-5 \
  --grad_clip 1.0 \
  --eval_every 1 \
  --save_every 1 \
  --seed 62 \
  2>&1 | tee results/remote_logs/fpp960_e62_pspquad_phase_diffusion_ch24_reg_e30.log

echo "WRITE E62 phase comparison $(date '+%F %T')" | tee -a "$MASTER_LOG"
"$PY" - <<'PY' | tee results/remote_logs/e62_phase_comparison.txt
import json, os

paths = {
    "FTP/Hilbert baselines are inside E28/E60 summaries": None,
    "E28_ch24_test": "results/fpp960_e28_pspquad_phase_diffusion_ch24_e20/evaluation/test/phase_summary.json",
    "E28_precompute_test": "/root/autodl-tmp/fpp_ml_pspquad_cache_960/phase_pred_e28_pspquad_ddim20_e3_test_summary.json",
    "E60_best_ch32_test": "results/fpp960_e60_pspquad_phase_diffusion_ch32_e30/evaluation/test/phase_summary.json",
    "E60_latest_ch32_test": "/root/autodl-tmp/fpp_ml_pspquad_cache_960/phase_pred_e60_latest_pspquad_ddim20_e3_test_summary.json",
    "E62_ch24_reg_test": "results/fpp960_e62_pspquad_phase_diffusion_ch24_reg_e30/evaluation/test/phase_summary.json",
}

out = {}
for name, path in paths.items():
    if not path or not os.path.exists(path):
        continue
    d = json.load(open(path))
    out[name] = {
        "phase_aligned_mae_rad": d["phase_aligned_mae_rad"]["mean"],
        "phase_mae_rad": d["phase_mae_rad"]["mean"],
        "uph_mae_01": d.get("uph_mae_01", {}).get("mean"),
        "n": d.get("n", 30),
    }
    for key in ("ftp_phase_aligned_mae_rad", "hilbert_phase_aligned_mae_rad"):
        if key in d:
            out[name][key] = d[key]["mean"]
print(json.dumps(out, indent=2, ensure_ascii=False))
with open("results/e62_phase_comparison_summary.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
PY

echo "DONE E62 phase latest-eval + ch24 regularized $(date '+%F %T')" | tee -a "$MASTER_LOG"
