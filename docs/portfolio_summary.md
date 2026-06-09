# Portfolio Summary

## One-line project summary

Built an imitation-learning-to-reinforcement-learning pipeline for drone navigation, diagnosed BC-to-PPO distribution shift, and improved detour-task success over BC-only using expert-state augmentation with scripted relabeling.

## Claim boundary

The main claim is not that PPO automatically improves a BC policy. The evidence showed the opposite for the naive version: PPO could preserve or damage the BC prior depending on regularization and training stage. The defensible claim is that after measuring the distribution-shift failure mode, adding local expert-state recovery supervision during PPO improved average success and final distance over BC-only under the same `50`-episode evaluation protocol.

## Problem

The original waypoint task was too PPO-friendly:

- full-state observation
- dense reward
- high-level velocity action
- no real path-planning bottleneck

That made it hard to show a clear advantage for imitation pretraining because scratch PPO could mostly catch up with enough steps.

## Challenge

To make the problem more realistic, the task was extended into a detour-constrained setting:

- starts sampled on the left side of the map
- goals sampled on the right side
- a blocking wall in the middle
- only one upper corridor available

This created a non-greedy navigation problem where the agent had to move away from the straight-line goal path before it could succeed.

## What I built

- a new `DetourWaypointVelocityAviary` environment
- a scripted detour expert for demonstration collection
- BC training and evaluation scripts
- scratch PPO and BC-initialized PPO training pipelines
- weak BC-prior regularization for PPO (`bc_kl_coef=0.0003`)
- detour-aware reward shaping
- a less noisy evaluation protocol (`30` eval episodes during training, `50` for best-checkpoint re-evaluation)

## Current portfolio result

Setup:

- task: detour navigation
- base policy: BC trained from scripted expert trajectories
- PPO fine-tuning budget: `50k` steps
- PPO seeds: `7, 11, 19, 23, 31`
- evaluation: `50` deterministic episodes with seed `20000`
- model selection: validation-best checkpoint, with final checkpoint reported separately
- method: expert-state augmentation with scripted relabeling during PPO updates

| Method | Success | Mean final distance | Mean return |
|---|---:|---:|---:|
| BC-only | `0.54` | `0.269` | `84.67` |
| BC+PPO KL/freeze best | `0.54` | `0.269` | `84.67` |
| BC+PPO expert-state aug final | `0.728 +/- 0.109` | `0.159 +/- 0.028` | `89.71 +/- 14.73` |
| BC+PPO expert-state aug best | `0.792 +/- 0.104` | `0.153 +/- 0.041` | `81.00 +/- 10.79` |

This is the current portfolio headline because it shows not just BC initialization, but a targeted fix for the BC -> PPO distribution-shift failure mode.

What PPO added in the final method:

- online interaction still updated the policy through PPO
- auxiliary expert-state loss constrained local recovery near demonstration states
- scripted relabeling made augmented states carry corrective actions, not stale demonstration actions
- validation-best selection exposed the useful checkpoint while final-checkpoint reporting kept the late-drift limitation visible

## Earlier matched experiment

Setup:

- task: detour navigation
- budget: `300k` PPO steps
- seeds: `7, 11, 19`
- scratch PPO and BC+PPO matched on reward, horizon, and exploration init
- `log_std_init=-0.5`

### Main result: final `30`-episode evaluation mean after the transfer fix

| Method | Success | Mean final distance | Mean return |
|---|---:|---:|---:|
| PPO scratch | `0.00` | `0.732` | `21.15` |
| BC + PPO | `0.111` | `0.476` | `81.83` |

## Why this result matters

- BC pretraining helped PPO overcome a real exploration bottleneck
- the detour task made imitation useful in a way the easy waypoint task did not
- task design, reward shaping, and evaluation protocol turned out to be just as important as the learning algorithm

## Current technical diagnosis

The most important remaining AI problem is **BC -> PPO distribution shift**.

After the transfer alignment fix, the PPO actor starts from the same action function as BC. The remaining failure mode is later fine-tuning:

- early checkpoints can preserve BC-like detour behavior
- PPO updates can drift away from the BC controller after actor unfreezing
- final success can drop even when the policy still reaches the corridor or goal stage
- collision and final-settling failures increase in unstable seeds

This is now documented as a separate technical analysis:

- `docs/bc_to_ppo_distribution_shift.md`

That document records:

- the AI failure mode
- the interventions tried so far
- the diagnostic metrics added
- the current ablation evidence
- candidate next interventions

The latest strong-regularization ablation sharpened the conclusion:

- best checkpoints re-evaluated over `50` episodes reached up to `0.54` success
- final checkpoints often collapsed after actor unfreezing
- `freeze50k + bc_kl_coef=0.003` was the most stable final setting
- aggressive low exploration did not solve the issue and increased collisions in final policies

The latest expert-state augmentation ablation produced the first stronger PPO-improvement signal:

- method: perturb expert trajectory states, recompute relative-goal features, reject invalid wall states, and relabel actions with the scripted expert
- setup: `50k` PPO steps, `freeze_actor_steps=25k`, `bc_kl_coef=0.003`, `expert_bc_loss_coef=1.0`, `log_std_init=-1.0`
- best current variant: `position_noise_std=0.05`, `velocity_noise_std=0.05`, `2` augmented copies
- final online eval across seeds `7, 19`: mean success `0.80`, mean final distance `0.129`, collision `0.00`
- final `50`-episode re-evaluation across seeds `7, 11, 19, 23, 31`: mean success `0.728`, mean final distance `0.159`
- validation-best `50`-episode re-evaluation across seeds `7, 11, 19, 23, 31`: mean success `0.792`, mean final distance `0.153`

This is now stronger than the standalone BC reference under the same `50`-episode evaluation seed (`0.54` success, `0.269m` final distance). The current portfolio claim should still mention residual late-training sensitivity because seed `31` final only reached `0.56` while its validation-best checkpoint reached `0.82`, but the technical story is now much better: PPO did not merely preserve BC, but improved on average after adding local recovery supervision around expert states.

Final portfolio artifacts:

- comparison tables and plots: `artifacts/analysis/portfolio_final_20260609/`
- headline table: `artifacts/analysis/portfolio_final_20260609/headline_50ep_comparison.csv`
- headline plots:
  - `artifacts/analysis/portfolio_final_20260609/headline_50ep_success_rate.png`
  - `artifacts/analysis/portfolio_final_20260609/headline_50ep_mean_final_distance.png`
- diagnostics table: `artifacts/analysis/portfolio_final_20260609/online_diagnostics_comparison.csv`

Seed `31` failure analysis:

- output directory: `artifacts/analysis/aug005_seed31_final_rollouts_20260609/`
- success rate: `0.56`
- validation-best success rate: `0.82`
- collision rate: `0.00`
- reached exit stage rate: `1.00`
- reached goal stage rate: `1.00`
- failures: `22 / 50`
- all failures still reached the goal stage
- failure mean min distance: `0.105m`
- failure mean final distance: `0.358m`
- failure mean final speed: `0.076m/s`

Interpretation: seed `31` is not primarily a wall-navigation or exploration failure. It reaches the corridor and goal stage without collisions, but the final checkpoint fails the success condition after approaching the target. The validation-best checkpoint recovers to `0.82`, so the remaining weakness is late-training drift/final approach stability, not detour discovery.

## Resume bullet options

### Direct, metric-heavy

- Built a PyBullet drone IL->RL benchmark for detour-constrained navigation and improved `50`-episode success from a BC-only `0.54` baseline to `0.792 +/- 0.104` with BC-initialized PPO, expert-state augmentation, and validation-best checkpoint selection across `5` PPO seeds.

### More conservative

- Designed a detour-constrained drone navigation benchmark and evaluation pipeline, then showed that expert-state augmentation with scripted relabeling improved BC-initialized PPO over standalone BC under a shared `50`-episode protocol, reducing mean final distance from `0.269m` to `0.153m`.

### Most defensible

- Diagnosed BC-to-PPO distribution shift in a detour-constrained drone navigation task and improved BC-initialized PPO over a standalone BC baseline by adding expert-state augmentation with scripted relabeling, raising validation-best success from `0.54` to `0.792 +/- 0.104` across `5` PPO seeds.

### More engineering-focused

- Implemented a drone imitation-learning and PPO fine-tuning pipeline with custom PyBullet tasks, scripted expert rollout collection, BC actor initialization, auxiliary expert losses, state augmentation, checkpoint selection, and experiment visualization to study stable IL->RL transfer.

## Interview framing

If asked what was learned from the project, the strongest answer is:

> The main lesson was that getting BC initialization correct was only the first step. After I fixed the BC-to-PPO transfer so the initial policies matched exactly, the real bottleneck became distribution shift during PPO fine-tuning. KL regularization and actor freezing mostly preserved BC, but did not improve it. The improvement came from adding local recovery supervision: perturbing expert trajectory states, recomputing task features, rejecting invalid wall states, and relabeling actions with the scripted expert. That moved the result from BC-only `0.54` success to `0.792` validation-best success across five PPO seeds, while also exposing the remaining weakness as late-training final-approach drift.

If challenged on whether PPO itself improved the policy, the precise answer is:

> PPO alone was not enough. The meaningful result is that PPO became useful only after I added a supervised recovery signal on augmented expert states. KL/freeze preserved the BC prior, but expert-state augmentation gave PPO a safer local region to explore around the demonstration manifold. I still report final checkpoints separately because late PPO updates can degrade settling behavior.

## Key artifacts

- Main report: `docs/experiment_results.md`
- Portfolio comparison: `artifacts/analysis/portfolio_final_20260609/`
- Distribution-shift diagnosis: `docs/bc_to_ppo_distribution_shift.md`
- Matched comparison figures: `artifacts/figures/detour_aligned_matched_compare/`
- Main rerun note: `docs/normalization_fix_rerun.md`
