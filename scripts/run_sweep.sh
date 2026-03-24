#!/usr/bin/env bash
set -euo pipefail

set +u
source /GenSIvePFS/users/yfwang/miniconda3/etc/profile.d/conda.sh
conda activate verl
set -u

cd /GenSIvePFS/users/yfwang/lowrank_spectral_es_rl

CONFIGS=("$@")
if [ "${#CONFIGS[@]}" -eq 0 ]; then
  CONFIGS=(
    configs/spectral_es_vllm_mutant_parallel.yaml
  )
fi

for config_path in "${CONFIGS[@]}"; do
  echo "[run_sweep] launching ${config_path}"
  torchrun --standalone --nproc_per_node="${GPUS_PER_NODE:-4}" train.py \
    --config "${config_path}" \
    --override "execution.world_size=${GPUS_PER_NODE:-4}" \
    --override "execution.gpus_per_node=${GPUS_PER_NODE:-4}"
done
