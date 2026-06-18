#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp
mkdir -p /root/autodl-tmp/diffusion_fpp_v5/results/remote_logs

B_HOST=connect.cqa1.seetacloud.com
B_PORT=47765
B_ROOT=/root/autodl-tmp
SSH_B="ssh -p ${B_PORT} -o StrictHostKeyChecking=no"
RSYNC_BW=80m
LOG=/root/autodl-tmp/diffusion_fpp_v5/results/remote_logs/transfer_minimal_to_b.log

{
  echo "START transfer minimal A->B $(date '+%F %T')"
  echo "B target: ${B_HOST}:${B_PORT}:${B_ROOT}"

  ${SSH_B} root@${B_HOST} "mkdir -p ${B_ROOT}/diffusion_fpp_v5 ${B_ROOT}/fpp_ml_bench_cache_960_fgfix ${B_ROOT}/fpp_ml_pspquad_cache_960"

  echo "SYNC code without heavy results $(date '+%F %T')"
  rsync -a --partial --info=progress2 --bwlimit="${RSYNC_BW}" \
    -e "ssh -p ${B_PORT} -o StrictHostKeyChecking=no" \
    --exclude "results/**" \
    --exclude "logs/**" \
    --exclude "__pycache__/**" \
    --exclude "*.pyc" \
    /root/autodl-tmp/diffusion_fpp_v5/ \
    root@${B_HOST}:${B_ROOT}/diffusion_fpp_v5/

  echo "SYNC base FPP cache $(date '+%F %T')"
  rsync -a --partial --info=progress2 --bwlimit="${RSYNC_BW}" \
    -e "ssh -p ${B_PORT} -o StrictHostKeyChecking=no" \
    /root/autodl-tmp/fpp_ml_bench_cache_960_fgfix/ \
    root@${B_HOST}:${B_ROOT}/fpp_ml_bench_cache_960_fgfix/

  echo "SYNC PSP phase cache $(date '+%F %T')"
  rsync -a --partial --info=progress2 --bwlimit="${RSYNC_BW}" \
    -e "ssh -p ${B_PORT} -o StrictHostKeyChecking=no" \
    /root/autodl-tmp/fpp_ml_pspquad_cache_960/ \
    root@${B_HOST}:${B_ROOT}/fpp_ml_pspquad_cache_960/

  echo "SYNC selected result summaries/checkpoints $(date '+%F %T')"
  ${SSH_B} root@${B_HOST} "mkdir -p ${B_ROOT}/diffusion_fpp_v5/results"
  for path in \
    results/e62_phase_comparison_summary.json \
    results/e63_phase_ensemble_e28_e62_test \
    results/e64_depth_comparison_summary.json \
    results/fpp960_e28_pspquad_phase_diffusion_ch24_e20/evaluation \
    results/fpp960_e62_pspquad_phase_diffusion_ch24_reg_e30/evaluation \
    results/fpp960_e62_pspquad_phase_diffusion_ch24_reg_e30/checkpoints/best_phase.pt \
    results/fpp960_e64_e63ens_psp_instrxy_adapter_decoder_bs4_e25/evaluation \
    results/d47_d31_epoch001_hierarchical_physical_gate \
    results/fpp960_a_fringe_unet_control/checkpoints/best.pt \
    results/fpp960_c4_adapter_hilbert_dwt_grad_no_xy_freeze/checkpoints/best_rmse.pt; do
    if [[ -e "/root/autodl-tmp/diffusion_fpp_v5/${path}" ]]; then
      rsync -a --relative --partial --info=progress2 --bwlimit="${RSYNC_BW}" \
        -e "ssh -p ${B_PORT} -o StrictHostKeyChecking=no" \
        "/root/autodl-tmp/diffusion_fpp_v5/./${path}" \
        root@${B_HOST}:${B_ROOT}/diffusion_fpp_v5/
    else
      echo "SKIP missing ${path}"
    fi
  done

  ${SSH_B} root@${B_HOST} "du -sh ${B_ROOT}/diffusion_fpp_v5 ${B_ROOT}/fpp_ml_bench_cache_960_fgfix ${B_ROOT}/fpp_ml_pspquad_cache_960 2>/dev/null; df -h ${B_ROOT}; date '+DONE %F %T' > ${B_ROOT}/pip_diffusion_transfer_done.txt"
  echo "DONE transfer minimal A->B $(date '+%F %T')"
} 2>&1 | tee "${LOG}"
