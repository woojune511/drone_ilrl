# Detour Planar BC Baseline

This note records the first validation run for the deployment-oriented planar action interface.

## Setup

Task variant:

```text
detour_planar
```

Policy-facing action:

```text
[vx_body, vy_body, yaw_rate]
```

The environment holds altitude internally and rate-limits planar velocity and yaw-rate commands.

## Expert Data Collection

Command:

```bash
uv run --all-extras python scripts/collect_expert_rollouts.py \
  --task-variant detour_planar \
  --episodes 50 \
  --quality-tag clean_planar
```

Dataset:

```text
artifacts/datasets/detour_planar_clean_planar_expert_20260609_170503.npz
```

Expert summary:

| Metric | Value |
|---|---:|
| Episodes | `50` |
| Transitions | `8705` |
| Success rate | `1.0` |
| Mean episode length | `174.1` |
| Mean final distance | `0.0805m` |

## BC Training

Command:

```bash
CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1 \
  uv run --all-extras python scripts/train_bc.py \
  --dataset artifacts/datasets/detour_planar_clean_planar_expert_20260609_170503.npz \
  --epochs 20
```

Checkpoint:

```text
artifacts/checkpoints/bc/bc_20260609_171249/checkpoint.pt
```

Training summary:

| Metric | Value |
|---|---:|
| Train transitions | `6970` |
| Validation transitions | `1735` |
| Best validation loss | `0.0276` |
| Final validation MAE | `0.1038` |

The first attempt without thread limits was much slower because PyTorch oversubscribed CPU threads. The thread-limited command above is the recommended default for these small MLP experiments.

## BC Evaluation

Command:

```bash
CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1 \
  uv run --all-extras python scripts/evaluate_policy.py \
  --policy-type bc \
  --checkpoint artifacts/checkpoints/bc/bc_20260609_171249/checkpoint.pt \
  --task-variant detour_planar \
  --episodes 50 \
  --seed 20000 \
  --output artifacts/evals/bc_detour_planar_clean50_50eps_seed20000_20260609.json
```

Evaluation summary:

| Metric | Value |
|---|---:|
| Episodes | `50` |
| Success rate | `1.0` |
| Mean episode return | `52.93` |
| Mean episode length | `171.32` |
| Mean final distance | `0.0552m` |

## Interpretation

The new action interface is viable: a clean scripted expert can solve the task, and BC can imitate it under the same `50`-episode evaluation protocol.

This also means the next useful experiment is not PPO improvement on the same setting. BC already reaches `1.0` success, so PPO has little headroom. The next controlled axis should be observation difficulty, not task difficulty and action difficulty at the same time.

Recommended next step:

1. keep `detour_planar`
2. keep clean scripted demonstrations
3. reduce observation information toward partial observability
4. only then test BC and PPO stability again

