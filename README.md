# ILRL Drone Lab

This repo is a compact drone navigation lab for testing whether imitation learning can make downstream reinforcement learning more sample-efficient.

The current headline result comes from a harder **detour navigation task**: under a matched `300k`-step budget, `BC + PPO` outperformed `PPO from scratch` on success rate, return, and final goal distance.

## Current Result

Final matched comparison:

- task: `DetourWaypointVelocityAviary`
- timesteps: `300,000`
- seeds: `7, 11, 19`
- evaluation: `30` episodes every `50k`
- exploration init: `log_std_init=-0.5`
- BC regularization: `bc_kl_coef=0.0003`

Final `30`-episode evaluation mean:

| Method | Success | Mean final distance | Mean return |
|---|---:|---:|---:|
| PPO scratch | `0.00` | `0.741` | `37.01` |
| BC + PPO | `0.30` | `0.307` | `124.25` |

Best-checkpoint `50`-episode re-evaluation mean:

| Method | Success | Mean final distance | Mean return |
|---|---:|---:|---:|
| PPO scratch | `0.00` | `0.775` | `22.64` |
| BC + PPO | `0.27` | `0.406` | `91.16` |

Main figures:

- `artifacts/figures/detour_reward_v2_eval30_matched_compare/success_rate_vs_steps.png`
- `artifacts/figures/detour_reward_v2_eval30_matched_compare/final_distance_vs_steps.png`
- `artifacts/figures/detour_reward_v2_eval30_matched_compare/trajectory_comparison.png`
- `artifacts/figures/detour_reward_v2_eval30_matched_compare/portfolio_overview.png`

Detailed write-up:

- `docs/experiment_results.md`
- `docs/portfolio_summary.md`

## Repo Structure

- `src/ilrl_lab/envs/waypoint_vel_aviary.py`: base waypoint-reaching task
- `src/ilrl_lab/envs/detour_vel_aviary.py`: harder detour-constrained task
- `src/ilrl_lab/experts/velocity.py`: waypoint and detour scripted experts
- `src/ilrl_lab/bc.py`: behavior cloning policy
- `src/ilrl_lab/ppo_training.py`: PPO helpers, BC actor init, weak BC regularization
- `scripts/collect_expert_rollouts.py`: expert data collection
- `scripts/train_bc.py`: BC training
- `scripts/evaluate_bc.py`: BC evaluation
- `scripts/train_ppo.py`: scratch PPO training
- `scripts/fine_tune_ppo_from_bc.py`: BC-initialized PPO fine-tuning
- `scripts/plot_experiment_results.py`: comparison plots

## Quickstart

### 1. Bootstrap the environment

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap.ps1
```

### 2. Activate the environment

```powershell
. .\.venv\Scripts\Activate.ps1
```

### 3. Smoke test

```powershell
uv run python scripts/check_env.py
pytest tests/test_waypoint_env.py -q
```

## Reproducing the Pipeline

### Collect expert rollouts

Waypoint:

```powershell
python scripts/collect_expert_rollouts.py --episodes 50 --task-variant waypoint
```

Detour:

```powershell
python scripts/collect_expert_rollouts.py --episodes 50 --task-variant detour
```

### Train behavior cloning

```powershell
python scripts/train_bc.py --epochs 15 --task-variant detour
```

### Evaluate BC

```powershell
python scripts/evaluate_bc.py --episodes 20 --task-variant detour
```

### Train scratch PPO

```powershell
python scripts/train_ppo.py --task-variant detour --total-timesteps 300000 --eval-episodes 30 --log-std-init -0.5
```

### Fine-tune PPO from BC

```powershell
python scripts/fine_tune_ppo_from_bc.py --task-variant detour --total-timesteps 300000 --eval-episodes 30 --log-std-init -0.5 --bc-kl-coef 0.0003
```

## Why the Detour Task Matters

The original waypoint task was useful for validating the pipeline, but it was still fairly PPO-friendly:

- full-state observation
- dense reward
- high-level velocity action
- no real path-planning bottleneck

The detour task adds:

- left-to-right start/goal structure
- a blocking wall
- a single open corridor
- collision-aware truncation
- modest detour-stage reward shaping

That turns the problem into a non-greedy navigation task where imitation pretraining matters much more.

## Notes

- Python is pinned to `3.10` for compatibility with `gym-pybullet-drones`
- `vendor/gym-pybullet-drones-main/` is used as a local editable dependency
- On Windows, the setup uses `pybullet-arm64` as a practical replacement for `pybullet`
