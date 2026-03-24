#!/usr/bin/env bash
set -euo pipefail

set +u
source /GenSIvePFS/users/yfwang/miniconda3/etc/profile.d/conda.sh
conda activate verl
set -u

cd /GenSIvePFS/users/yfwang/lowrank_spectral_es_rl

CONFIG_PATH="${1:-configs/spectral_es_vllm_mutant_parallel.yaml}"
RUN_ID="${RUN_ID:-baseline_eval_$(date -u +%m%d_%H%M%S)}"

torchrun --standalone --nproc_per_node="${GPUS_PER_NODE:-4}" train.py \
  --config "${CONFIG_PATH}" \
  --mode baseline \
  --baseline-split "${BASELINE_SPLIT:-test}" \
  --baseline-max-examples "${BASELINE_MAX_EXAMPLES:-0}" \
  --override "output.run_id=${RUN_ID}"
