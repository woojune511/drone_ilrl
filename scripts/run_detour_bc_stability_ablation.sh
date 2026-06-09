#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 BC_CHECKPOINT" >&2
  exit 2
fi

BC_CHECKPOINT="$1"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000}"
EVAL_FREQ="${EVAL_FREQ:-25000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
SEEDS="${SEEDS:-7 19}"
N_ENVS="${N_ENVS:-4}"
N_STEPS="${N_STEPS:-256}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TORCH_THREADS="${TORCH_THREADS:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/checkpoints/ppo_bc_stability_ablation}"

export OMP_NUM_THREADS="${TORCH_THREADS}"
export MKL_NUM_THREADS="${TORCH_THREADS}"
export TORCH_NUM_THREADS="${TORCH_THREADS}"

run_variant() {
  local name="$1"
  local log_std_init="$2"
  local bc_kl_coef="$3"
  local freeze_actor_steps="$4"

  for seed in ${SEEDS}; do
    echo "=== variant=${name} seed=${seed} log_std=${log_std_init} bc_kl=${bc_kl_coef} freeze=${freeze_actor_steps} ==="
    uv run --all-extras python scripts/fine_tune_ppo_from_bc.py \
      --task-variant detour \
      --bc-checkpoint "${BC_CHECKPOINT}" \
      --output-dir "${OUTPUT_ROOT}/${name}" \
      --total-timesteps "${TOTAL_TIMESTEPS}" \
      --eval-freq "${EVAL_FREQ}" \
      --eval-episodes "${EVAL_EPISODES}" \
      --seed "${seed}" \
      --n-steps "${N_STEPS}" \
      --batch-size "${BATCH_SIZE}" \
      --n-envs "${N_ENVS}" \
      --log-std-init "${log_std_init}" \
      --bc-kl-coef "${bc_kl_coef}" \
      --freeze-actor-steps "${freeze_actor_steps}"
  done
}

run_variant low_std_kl1e3 -1.0 0.001 0
run_variant low_std_kl1e3_freeze10k -1.0 0.001 10000
run_variant low_std_kl3e3_freeze25k -1.0 0.003 25000
