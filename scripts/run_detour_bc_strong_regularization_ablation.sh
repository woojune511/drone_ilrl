#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 BC_CHECKPOINT" >&2
  exit 2
fi

BC_CHECKPOINT="$1"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-50000}"
EVAL_FREQ="${EVAL_FREQ:-12500}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
SEEDS="${SEEDS:-7 19}"
N_ENVS="${N_ENVS:-4}"
N_STEPS="${N_STEPS:-256}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TORCH_THREADS="${TORCH_THREADS:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/checkpoints/ppo_bc_strong_regularization_ablation}"
LOG_ROOT="${LOG_ROOT:-artifacts/logs/ppo_bc_strong_regularization_ablation_runs/$(date +%Y%m%d_%H%M%S)}"
FAIL_FAST="${FAIL_FAST:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

export OMP_NUM_THREADS="${TORCH_THREADS}"
export MKL_NUM_THREADS="${TORCH_THREADS}"
export TORCH_NUM_THREADS="${TORCH_THREADS}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PWD}/.uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

mkdir -p "${LOG_ROOT}"

RUN_FAILURES=0

run_one() {
  local name="$1"
  local log_std_init="$2"
  local bc_kl_coef="$3"
  local freeze_actor_steps="$4"
  local seed="$5"
  local run_log="${LOG_ROOT}/${name}_seed${seed}.log"
  local exit_file="${LOG_ROOT}/${name}_seed${seed}.exit"

  if [[ "${SKIP_COMPLETED}" == "1" ]]; then
    shopt -s nullglob
    local completed_summaries=("${OUTPUT_ROOT}/${name}/detour/ppo_bc_init_seed${seed}_"*/summary.json)
    shopt -u nullglob
    if [[ "${#completed_summaries[@]}" -gt 0 ]]; then
      echo "=== variant=${name} seed=${seed} skipped: summary exists at ${completed_summaries[0]} ==="
      echo "0" > "${exit_file}"
      return
    fi
  fi

  echo "=== variant=${name} seed=${seed} log_std=${log_std_init} bc_kl=${bc_kl_coef} freeze=${freeze_actor_steps} ==="
  echo "log=${run_log}"

  set +e
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
    --freeze-actor-steps "${freeze_actor_steps}" 2>&1 | tee "${run_log}"
  local status="${PIPESTATUS[0]}"
  set -e

  echo "${status}" > "${exit_file}"
  echo "=== variant=${name} seed=${seed} exit=${status} ==="

  if [[ "${status}" -ne 0 ]]; then
    RUN_FAILURES=$((RUN_FAILURES + 1))
    if [[ "${FAIL_FAST}" == "1" ]]; then
      exit "${status}"
    fi
  fi
}

run_variant() {
  local name="$1"
  local log_std_init="$2"
  local bc_kl_coef="$3"
  local freeze_actor_steps="$4"

  for seed in ${SEEDS}; do
    run_one "${name}" "${log_std_init}" "${bc_kl_coef}" "${freeze_actor_steps}" "${seed}"
  done
}

run_variant low_std_kl3e3_freeze50k -1.0 0.003 50000
run_variant low_std_kl1e2_freeze25k -1.0 0.01 25000
run_variant very_low_std_kl1e2_freeze25k -1.5 0.01 25000

if [[ "${RUN_FAILURES}" -ne 0 ]]; then
  echo "Failed runs: ${RUN_FAILURES}" >&2
  exit 1
fi
