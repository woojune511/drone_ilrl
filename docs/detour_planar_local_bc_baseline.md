# Detour Planar Local-Observation BC Baseline

This note records the first observation-reduction experiment after validating the planar velocity action interface.

## Setup

Task variant:

```text
detour_planar_local
```

Action:

```text
[vx_body, vy_body, yaw_rate]
```

Observation layout:

```text
body_vel_xy(2)
altitude_error(1)
z_vel(1)
sin_yaw, cos_yaw(2)
rel_goal_body_xyz(3)
rel_detour_target_body_xyz(3)
previous_action(3)
```

This removes absolute position and absolute goal from the policy observation, but still provides a route-conditioned local detour target. It is therefore not a depth-only or raw-perception setting yet.

## Expert Data Collection

Command:

```bash
uv run --all-extras python scripts/collect_expert_rollouts.py \
  --task-variant detour_planar_local \
  --episodes 50 \
  --quality-tag clean_planar_local
```

Dataset:

```text
artifacts/datasets/detour_planar_local_clean_planar_local_expert_20260609_173411.npz
```

Expert summary:

| Metric | Value |
|---|---:|
| Episodes | `50` |
| Transitions | `8882` |
| Success rate | `1.0` |
| Mean episode length | `177.64` |
| Mean final distance | `0.0837m` |

## BC Training

Command:

```bash
CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1 \
  uv run --all-extras python scripts/train_bc.py \
  --dataset artifacts/datasets/detour_planar_local_clean_planar_local_expert_20260609_173411.npz \
  --epochs 20
```

Checkpoint:

```text
artifacts/checkpoints/bc/bc_20260609_173437/checkpoint.pt
```

Training summary:

| Metric | Value |
|---|---:|
| Train transitions | `7117` |
| Validation transitions | `1765` |
| Best validation loss | `0.00082` |
| Final validation MAE | `0.0206` |

## BC Evaluation

Command:

```bash
CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1 \
  uv run --all-extras python scripts/evaluate_policy.py \
  --policy-type bc \
  --checkpoint artifacts/checkpoints/bc/bc_20260609_173437/checkpoint.pt \
  --task-variant detour_planar_local \
  --episodes 50 \
  --seed 20000 \
  --output artifacts/evals/bc_detour_planar_local_clean50_50eps_seed20000_20260609.json
```

Evaluation summary:

| Metric | Value |
|---|---:|
| Episodes | `50` |
| Success rate | `1.0` |
| Mean episode return | `74.59` |
| Mean episode length | `185.54` |
| Mean final distance | `0.0659m` |

## Interpretation

Removing absolute position and absolute goal is not enough to create a difficult imitation problem when the observation still contains a local detour target. The policy can treat the problem as local waypoint tracking and BC solves it cleanly.

This is a useful intermediate result because it isolates the action interface and confirms that local route-conditioned control is stable. It also shows that the next meaningful difficulty increase should target perception/planning information, not PPO fine-tuning.

Recommended next step:

1. keep `detour_planar`
2. remove `rel_detour_target_body_xyz`
3. add depth/raycast features for wall and corridor perception
4. keep clean scripted expert labels
5. test whether BC still solves the task

