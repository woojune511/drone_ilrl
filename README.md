# ILRL Drone Lab

This repo is a compact drone navigation lab for testing whether imitation learning can make downstream reinforcement learning more sample-efficient and more stable.

The headline result comes from a harder **detour navigation task**. Plain BC can solve part of the task, but PPO fine-tuning initially suffered from BC -> PPO distribution shift. The strongest current method adds local expert-state augmentation with scripted expert relabeling during PPO updates.

The intended claim is narrow and reproducible: PPO did not improve BC by default. It improved after the distribution-shift failure mode was measured and the PPO update was given local expert-state recovery supervision.

## Current Result

Final portfolio comparison:

- task: `DetourWaypointVelocityAviary`
- evaluation: `50` deterministic episodes with evaluation seed `20000`
- PPO seeds: `7, 11, 19, 23, 31`
- PPO budget: `50,000` steps for expert-state augmentation runs
- model selection: validation-best checkpoint from periodic evaluation, with final-checkpoint results reported separately
- best variant: expert trajectory state augmentation with scripted relabeling, `position_noise_std=0.05`, `velocity_noise_std=0.05`, `2` augmented copies

Headline `50`-episode evaluation:

| Method | Success | Mean final distance | Mean return |
|---|---:|---:|---:|
| BC-only | `0.54` | `0.269` | `84.67` |
| BC+PPO KL/freeze best | `0.54` | `0.269` | `84.67` |
| BC+PPO expert-state aug final | `0.728 +/- 0.109` | `0.159 +/- 0.028` | `89.71 +/- 14.73` |
| BC+PPO expert-state aug best | `0.792 +/- 0.104` | `0.153 +/- 0.041` | `81.00 +/- 10.79` |

Interpretation:

- BC-only is useful but limited on the randomized detour task.
- KL/freeze mostly preserves BC behavior instead of improving it.
- Expert-state augmentation plus expert relabeling gives PPO local recovery supervision around the demonstration manifold.
- Validation-best checkpoint selection is part of the official protocol because late PPO updates can still degrade final approach behavior.

Claim boundary:

- This is not presented as a pure RL-from-scratch win.
- The expert-state augmentation still uses the scripted expert during training.
- The main contribution is diagnosing why naive BC -> PPO transfer was unstable and showing a targeted mitigation that improves average success and final distance under the shared evaluation protocol.

Main figures:

- `artifacts/analysis/portfolio_final_20260609/headline_50ep_success_rate.png`
- `artifacts/analysis/portfolio_final_20260609/headline_50ep_mean_final_distance.png`
- `artifacts/analysis/portfolio_final_20260609/online_diagnostics_success_rate.png`
- `artifacts/analysis/aug005_seed31_final_rollouts_20260609/failure_trajectories.png`

Detailed write-up:

- `docs/portfolio_summary.md`
- `docs/portfolio_one_page.md`
- `docs/interview_qa.md`
- `docs/action_space_roadmap.md`
- `docs/detour_planar_bc_baseline.md`
- `docs/detour_planar_local_bc_baseline.md`
- `docs/submission_file_manifest.md`
- `docs/bc_to_ppo_distribution_shift.md`
- `docs/experiment_results.md`
- `docs/normalization_fix_rerun.md`
- `docs/detour_quick_100k_results.md`

## Repo Structure

- `src/ilrl_lab/envs/waypoint_vel_aviary.py`: base waypoint-reaching task
- `src/ilrl_lab/envs/detour_vel_aviary.py`: harder detour-constrained task
- `src/ilrl_lab/envs/detour_planar_vel_aviary.py`: detour task with body-frame planar velocity, yaw-rate, and altitude hold
- `src/ilrl_lab/experts/velocity.py`: waypoint and detour scripted experts
- `src/ilrl_lab/bc.py`: behavior cloning policy
- `src/ilrl_lab/ppo_training.py`: PPO helpers, BC actor init, weak BC regularization
- `scripts/collect_expert_rollouts.py`: expert data collection
- `scripts/train_bc.py`: BC training
- `scripts/evaluate_bc.py`: BC evaluation
- `scripts/train_ppo.py`: scratch PPO training
- `scripts/fine_tune_ppo_from_bc.py`: BC-initialized PPO fine-tuning
- `scripts/plot_training_diagnostics.py`: PPO training diagnostics from eval JSON and TensorBoard logs
- `scripts/build_portfolio_results.py`: final portfolio comparison tables and plots
- `scripts/analyze_policy_rollouts.py`: episode-level rollout failure analysis

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

### Evaluate any saved policy with the shared protocol

BC checkpoint:

```powershell
uv run --all-extras python scripts/evaluate_policy.py --policy-type bc --checkpoint artifacts/checkpoints/bc/<run>/checkpoint.pt --episodes 50 --task-variant detour
```

PPO checkpoint with run metadata:

```powershell
uv run --all-extras python scripts/evaluate_policy.py --run-summary artifacts/checkpoints/ppo_bc_init/detour/<run>/summary.json --checkpoint-selector best --episodes 50
```

PPO checkpoint with explicit BC observation normalization:

```powershell
uv run --all-extras python scripts/evaluate_policy.py --policy-type ppo --checkpoint artifacts/checkpoints/ppo_scratch/detour/<run>/best_model.zip --obs-norm-bc-checkpoint artifacts/checkpoints/bc/<run>/checkpoint.pt --episodes 50 --task-variant detour
```

### Train scratch PPO

```powershell
python scripts/train_ppo.py --task-variant detour --total-timesteps 300000 --eval-episodes 30 --log-std-init -0.5
```

### Fine-tune PPO from BC

```powershell
python scripts/fine_tune_ppo_from_bc.py --task-variant detour --total-timesteps 300000 --eval-episodes 30 --log-std-init -0.5 --bc-kl-coef 0.0003
```

### Fine-tune PPO with expert-state augmentation

```bash
uv run --all-extras python scripts/fine_tune_ppo_from_bc.py \
  --task-variant detour \
  --bc-checkpoint artifacts/checkpoints/bc/bc_20260608_132020/checkpoint.pt \
  --total-timesteps 50000 \
  --eval-freq 12500 \
  --eval-episodes 10 \
  --seed 7 \
  --n-envs 4 \
  --n-steps 256 \
  --batch-size 256 \
  --log-std-init -1.0 \
  --bc-kl-coef 0.003 \
  --freeze-actor-steps 25000 \
  --expert-bc-loss-coef 1.0 \
  --expert-bc-loss-batch-size 256 \
  --expert-bc-augment-copies 2 \
  --expert-bc-position-noise-std 0.05 \
  --expert-bc-velocity-noise-std 0.05
```

Batch script for the current best variant:

```bash
RUN_AUG003=0 RUN_AUG005=1 SEEDS="7 11 19 23 31" \
  scripts/run_detour_expert_bc_aug_ablation.sh artifacts/checkpoints/bc/bc_20260608_132020/checkpoint.pt
```

### Try the deployment-oriented planar action variant

This keeps the detour task but changes the policy-facing action to body-frame planar velocity plus yaw-rate. Altitude is held by the low-level controller.

```bash
uv run --all-extras python scripts/collect_expert_rollouts.py \
  --task-variant detour_planar \
  --episodes 20 \
  --quality-tag clean_planar
```

Initial clean BC baseline for this variant:

- expert success: `1.0`
- BC success: `1.0` over `50` deterministic episodes
- BC mean final distance: `0.055m`

See `docs/detour_planar_bc_baseline.md`.

Next observation-reduction variant:

```bash
uv run --all-extras python scripts/collect_expert_rollouts.py \
  --task-variant detour_planar_local \
  --episodes 20 \
  --quality-tag clean_planar_local
```

Initial clean BC baseline for this local-observation variant:

- expert success: `1.0`
- BC success: `1.0` over `50` deterministic episodes
- BC mean final distance: `0.066m`

See `docs/detour_planar_local_bc_baseline.md`.

### Run the main detour seed sweep

Linux/macOS:

```bash
scripts/run_detour_main_sweep.sh artifacts/checkpoints/bc/bc_20260608_132020/checkpoint.pt
```

Optional overrides:

```bash
SEEDS="7 11 19" TOTAL_TIMESTEPS=300000 EVAL_EPISODES=30 scripts/run_detour_main_sweep.sh artifacts/checkpoints/bc/bc_20260608_132020/checkpoint.pt
```

Parallel rollout collection can speed up PyBullet PPO runs:

```bash
N_ENVS=4 N_STEPS=256 TORCH_THREADS=1 SEEDS="7 11 19" TOTAL_TIMESTEPS=300000 scripts/run_detour_main_sweep.sh artifacts/checkpoints/bc/bc_20260608_132020/checkpoint.pt
```

### Build portfolio tables and figures

```bash
uv run --all-extras python scripts/build_portfolio_results.py --output-dir artifacts/analysis/portfolio_final_20260609
```

### Analyze a weak final checkpoint

```bash
uv run --all-extras python scripts/analyze_policy_rollouts.py \
  --run-summary artifacts/checkpoints/ppo_expert_bc_aug005_more_seeds_20260609_135343/expertbc_aug005_coef1_freeze25k/detour/ppo_bc_init_seed31_20260609_141521/summary.json \
  --checkpoint-selector final \
  --episodes 50 \
  --seed 20000 \
  --output-dir artifacts/analysis/aug005_seed31_final_rollouts_20260609
```

## Model Selection Protocol

PPO runs are evaluated periodically during training. The official portfolio result uses the validation-best checkpoint selected by:

1. highest success rate
2. lowest mean final distance as a tie-breaker
3. highest mean return as a second tie-breaker

The final checkpoint is still reported as a robustness diagnostic. In the current best variant, final-checkpoint success is `0.728`, while validation-best success is `0.792`. This gap is evidence of late-training drift, not a reason to hide the final result.

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
