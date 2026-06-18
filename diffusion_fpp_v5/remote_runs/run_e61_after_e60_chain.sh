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
E60_CKPT="$E60_DIR/checkpoints/best_phase.pt"
PRED_PREFIX=phase_pred_e60_pspquad_ddim20_e3
INIT_RAW=results/fpp960_a_fringe_unet_control/checkpoints/best.pt
MASTER_LOG=results/remote_logs/e61_after_e60_chain_master.log

echo "START E61 after-E60 chain $(date '+%F %T')" | tee "$MASTER_LOG"

echo "WAIT E60 training/final eval $(date '+%F %T')" | tee -a "$MASTER_LOG"
while pgrep -af 'train_fpp_phase_diffusion.py' | grep -q 'fpp960_e60_pspquad_phase_diffusion_ch32_e30'; do
  sleep 60
done

if [[ ! -f "$E60_CKPT" ]]; then
  echo "ERROR: missing E60 checkpoint: $E60_CKPT" | tee -a "$MASTER_LOG"
  exit 1
fi

echo "E60 checkpoint ready: $E60_CKPT $(date '+%F %T')" | tee -a "$MASTER_LOG"
if [[ -f "$E60_DIR/evaluation/test/phase_summary.json" ]]; then
  "$PY" - <<'PY' | tee -a results/remote_logs/e61_after_e60_chain_master.log
import json
p='results/fpp960_e60_pspquad_phase_diffusion_ch32_e30/evaluation/test/phase_summary.json'
d=json.load(open(p))
print('E60 test phase:', {
    'phase_aligned_mae_rad': d['phase_aligned_mae_rad']['mean'],
    'phase_mae_rad': d['phase_mae_rad']['mean'],
    'uph_mae_01': d.get('uph_mae_01', {}).get('mean'),
})
PY
fi

echo "EVAL old E36 checkpoint if needed $(date '+%F %T')" | tee -a "$MASTER_LOG"
if [[ ! -f results/e36c_best_checkpoint_eval/summary.json ]]; then
  "$PY" eval_fpp_psp_adapter_checkpoint.py \
    --checkpoint results/fpp960_e36c_pspquad_pred_instr_xy_adapter_all_bs4_e60/checkpoints/best_rmse.pt \
    --save_dir results/e36c_best_checkpoint_eval \
    --splits val test \
    --num_workers 8 \
    2>&1 | tee results/remote_logs/e36c_best_checkpoint_eval.log
fi

echo "PRECOMPUTE E60 phase predictions $(date '+%F %T')" | tee -a "$MASTER_LOG"
if [[ ! -f "$PSP_CACHE/${PRED_PREFIX}_test_float16.npy" ]]; then
  "$PY" precompute_fpp_phase_diffusion_predictions.py \
    --checkpoint "$E60_CKPT" \
    --base_cache_dir "$BASE_CACHE" \
    --phase_cache_dir "$PSP_CACHE" \
    --output_prefix "$PRED_PREFIX" \
    --splits train,val,test \
    --image_size 960 \
    --batch_size 1 \
    --num_workers 8 \
    --ddim_steps 20 \
    --ensemble 3 \
    --sample_start_from ftp \
    --sample_start_ratio 0.7 \
    2>&1 | tee results/remote_logs/e60_precompute_${PRED_PREFIX}.log
fi

run_adapter() {
  local name="$1"
  local scope="$2"
  local epochs="$3"
  local adapter_lr="$4"
  local backbone_lr="$5"
  local out="results/${name}"

  echo "START ${name} $(date '+%F %T')" | tee -a "$MASTER_LOG"
  "$PY" train_fpp_psp_adapter_unet.py \
    --base_cache_dir "$BASE_CACHE" \
    --phase_cache_dir "$PSP_CACHE" \
    --phase_pred_prefix "$PRED_PREFIX" \
    --save_dir "$out" \
    --base_checkpoint "$INIT_RAW" \
    --cond_mode phase_pred_instr_xy \
    --instr_channels 1-10 \
    --train_scope "$scope" \
    --epochs "$epochs" \
    --batch_size 4 \
    --eval_batch_size 4 \
    --num_workers 16 \
    --image_size 960 \
    --lr "$adapter_lr" \
    --adapter_lr "$adapter_lr" \
    --backbone_lr "$backbone_lr" \
    --weight_decay 1e-5 \
    --adapter_hidden 64 \
    --alpha 0.7 \
    --eval_every 1 \
    --eval_metrics_every 1 \
    --save_every 5 \
    --eval_initial \
    --train_minimal \
    --seed 61 \
    2>&1 | tee "results/remote_logs/${name}.log"
  echo "DONE ${name} $(date '+%F %T')" | tee -a "$MASTER_LOG"
}

run_adapter fpp960_e61a_e60psp_pred_instrxy_adapter_decoder_bs4_e30 adapter_decoder 30 5e-4 2e-5
run_adapter fpp960_e61b_e60psp_pred_instrxy_all_bs4_e30 all 30 5e-4 5e-6

echo "WRITE E61 comparison summary $(date '+%F %T')" | tee -a "$MASTER_LOG"
"$PY" - <<'PY' | tee results/remote_logs/e61_after_e60_comparison.txt
import json, os

paths = {
    "D47_hier": "results/d47_d31_epoch001_hierarchical_physical_gate/hierarchical_gate_summary.json",
    "E30_e28_phasepred_plus_fringe": "results/fpp960_e30_pspquad_pred_plus_fringe_from_rawA_e60/evaluation/summary.json",
    "E34_e28_psp_adapter_decoder": "results/fpp960_e34c_pspquad_pred_instr_xy_adapter_decoder_bs4_e70/evaluation/summary.json",
    "E35_gt_psp_oracle": "results/fpp960_e35c_pspquad_gt_instr_xy_adapter_decoder_bs4_e50/evaluation/summary.json",
    "E36_e28_psp_all_eval": "results/e36c_best_checkpoint_eval/summary.json",
    "E61a_e60_adapter_decoder": "results/fpp960_e61a_e60psp_pred_instrxy_adapter_decoder_bs4_e30/evaluation/summary.json",
    "E61b_e60_all": "results/fpp960_e61b_e60psp_pred_instrxy_all_bs4_e30/evaluation/summary.json",
}

def pick(path):
    d = json.load(open(path))
    if "test" in d and "splits" not in d:
        d = d["test"]["hierarchical"]
    if "splits" in d:
        d = d["splits"]["test"]
    return {
        "rmse": d["rmse"]["mean"],
        "mae": d["mae"]["mean"],
        "edge_rmse": d["edge_rmse"]["mean"],
        "normal_deg": d["normal_deg"]["mean"],
        "n": d.get("n", 30),
    }

out = {}
for name, path in paths.items():
    if os.path.exists(path):
        out[name] = pick(path)
print(json.dumps(out, indent=2, ensure_ascii=False))
with open("results/e61_after_e60_comparison_summary.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
PY

echo "DONE E61 after-E60 chain $(date '+%F %T')" | tee -a "$MASTER_LOG"
