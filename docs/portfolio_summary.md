# Portfolio Summary

## One-line project summary

Built an imitation-learning-to-reinforcement-learning pipeline for drone navigation, then designed a harder detour-constrained benchmark where BC-initialized PPO outperformed scratch PPO under a matched training budget.

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

## Final matched experiment

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

## Resume bullet options

### Direct, metric-heavy

- Built a drone IL->RL benchmark in PyBullet and showed that BC-initialized PPO outperformed scratch PPO on a detour-constrained navigation task, improving final success from `0.00` to `0.111` and reducing mean final goal distance from `0.732m` to `0.476m` under a matched `300k`-step budget after fixing BC->PPO transfer alignment.

### More conservative

- Designed a detour-constrained drone navigation benchmark and evaluation pipeline, then demonstrated that BC-initialized PPO achieved nonzero mean success across `3` seeds while scratch PPO remained at `0.00`, with mean final distance reduced from `0.732m` to `0.476m` in the aligned rerun.

### More engineering-focused

- Implemented a drone imitation-learning and PPO fine-tuning pipeline with custom PyBullet tasks, expert rollout collection, BC actor initialization, reward shaping, and experiment visualization to study sample-efficiency gains from IL priors.

## Interview framing

If asked what was learned from the project, the strongest answer is:

> The main lesson was that algorithm choice alone was not enough. The big unlock came from redesigning the task so imitation had something meaningful to contribute, then making reward shaping and evaluation robust enough that the comparison stopped being dominated by noise.

## Key artifacts

- Main report: `docs/experiment_results.md`
- Matched comparison figures: `artifacts/figures/detour_aligned_matched_compare/`
- Main rerun note: `docs/normalization_fix_rerun.md`
