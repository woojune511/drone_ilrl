#!/usr/bin/env bash
set -euo pipefail

BC_CHECKPOINT="${1:-artifacts/checkpoints/bc/bc_20260608_132020/checkpoint.pt}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-300000}"
EVAL_FREQ="${EVAL_FREQ:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
LOG_STD_INIT="${LOG_STD_INIT:--0.5}"
BC_KL_COEF="${BC_KL_COEF:-0.0003}"
N_ENVS="${N_ENVS:-1}"
N_STEPS="${N_STEPS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TORCH_THREADS="${TORCH_THREADS:-1}"
SEEDS="${SEEDS:-7 11 19 23 31}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${TORCH_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${TORCH_THREADS}}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-${TORCH_THREADS}}"

for SEED in ${SEEDS}; do
  uv run --all-extras python scripts/train_ppo.py \
    --task-variant detour \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --eval-freq "${EVAL_FREQ}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --seed "${SEED}" \
    --n-steps "${N_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --n-envs "${N_ENVS}" \
    --log-std-init "${LOG_STD_INIT}" \
    --obs-norm-bc-checkpoint "${BC_CHECKPOINT}"

  uv run --all-extras python scripts/fine_tune_ppo_from_bc.py \
    --task-variant detour \
    --bc-checkpoint "${BC_CHECKPOINT}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --eval-freq "${EVAL_FREQ}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --seed "${SEED}" \
    --n-steps "${N_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --n-envs "${N_ENVS}" \
    --log-std-init "${LOG_STD_INIT}" \
    --bc-kl-coef "${BC_KL_COEF}"
done
