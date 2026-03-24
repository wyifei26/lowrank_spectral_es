#!/usr/bin/env bash
set -euo pipefail

set +u
source /GenSIvePFS/users/yfwang/miniconda3/etc/profile.d/conda.sh
conda activate verl
set -u

cd /GenSIvePFS/users/yfwang/lowrank_spectral_es_rl

CONFIG_PATH="${1:-configs/spectral_es_vllm_mutant_parallel.yaml}"
RUN_ID="${RUN_ID:-local_debug_$(date -u +%m%d_%H%M%S)}"

torchrun --standalone --nproc_per_node="${GPUS_PER_NODE:-4}" train.py \
  --config "${CONFIG_PATH}" \
  --override "output.run_id=${RUN_ID}" \
  --override "execution.world_size=${GPUS_PER_NODE:-4}" \
  --override "execution.gpus_per_node=${GPUS_PER_NODE:-4}" \
  --override "train.train_steps=${TRAIN_STEPS:-2}" \
  --override "train.effective_question_batch=${QUESTION_BATCH:-32}" \
  --override "train.micro_batch=${MICRO_BATCH:-2}" \
  --override "eval.eval_every_steps=${EVAL_EVERY:-1}" \
  --override "data.train_max_examples=${TRAIN_MAX_EXAMPLES:-64}" \
  --override "data.val_max_examples=${VAL_MAX_EXAMPLES:-16}"
