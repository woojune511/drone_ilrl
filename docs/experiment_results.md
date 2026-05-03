# Drone IL -> RL Experiment Results

## Final takeaway

The most convincing result in this repo now comes from the **detour navigation task** rather than the original waypoint task.

Under a matched setup:

- same detour environment
- same reward function
- same PPO horizon (`300k`)
- same PPO exploration init (`log_std_init=-0.5`)
- same evaluation protocol (`30` episodes every `50k` steps)

`BC + PPO` clearly outperformed `PPO from scratch`.

The simplest honest summary is:

> On a harder detour-constrained drone navigation task, imitation-pretrained PPO learned successful behavior where scratch PPO failed to do so within the same training budget.

## Repo components

- Base waypoint environment: `src/ilrl_lab/envs/waypoint_vel_aviary.py`
- Harder detour variant: `src/ilrl_lab/envs/detour_vel_aviary.py`
- Scripted experts: `src/ilrl_lab/experts/velocity.py`
- BC model: `src/ilrl_lab/bc.py`
- Shared PPO utilities: `src/ilrl_lab/ppo_training.py`
- Scratch PPO entrypoint: `scripts/train_ppo.py`
- BC-initialized PPO entrypoint: `scripts/fine_tune_ppo_from_bc.py`
- Plotting: `scripts/plot_experiment_results.py`

## Phase 1: Easy waypoint baseline

The original waypoint task was useful for validating the pipeline, but it was still fairly PPO-friendly:

- full-state observation
- dense reward
- high-level velocity action
- single goal with no true path-planning bottleneck

That setting produced a nuanced result: BC improved early behavior, but scratch PPO mostly caught up by long horizon.

Reference figures are kept in:

- `artifacts/figures/ppo_vs_bc_init_300k/`

This phase was valuable as a sanity check, but it was not strong enough for a clean portfolio story.

## Phase 2: Detour task

To make the imitation prior matter more, the environment was extended into `DetourWaypointVelocityAviary`.

Task structure:

- start sampled on the left side of the workspace
- goal sampled on the right side
- a blocking wall prevents direct flight
- only an upper corridor is open

This turns the problem into a simple planning task:

1. align with the corridor
2. pass the wall
3. return to the final goal and stabilize

### Why the first detour version was unstable

The main issue was task design rather than model capacity.

- reward mostly tracked final-goal distance
- the correct early behavior was often to move away from the straight-line goal path
- collision feedback was sparse and late
- success was strict because the drone had to both reach the goal and slow down

That made exploration noisy and seed-sensitive.

## Detour task fixes that mattered

### 1. Better expert

The detour expert was upgraded so it no longer stopped at intermediate corridor waypoints. Instead, it treats corridor waypoints as transit targets and switches to the final goal before velocity collapses.

### 2. BC-aware PPO regularization

The most useful PPO-side changes were:

- weak BC regularization: `bc_kl_coef = 0.0003`
- smaller exploration at initialization: `log_std_init = -0.5`

This combination preserved some imitation prior without freezing the actor too aggressively.

### 3. Detour-aware reward shaping, version 2

The strongest reward version overfit badly across seeds, so it was softened.

The final detour shaping adds only modest incentives for:

- moving toward the corridor entry/exit subgoal
- making progress within the current detour stage
- receiving a small bonus when transitioning between stages

This kept the reward aligned with the task while preserving final-goal pressure.

## Final matched comparison

### Protocol

- task: `detour`
- total timesteps: `300,000`
- seeds: `7`, `11`
- evaluation every `50,000` steps
- `30` evaluation episodes per checkpoint
- scratch PPO: tuned only with the same `log_std_init=-0.5`
- BC+PPO: same PPO settings plus BC actor initialization and `bc_kl_coef=0.0003`

### Learning-curve summary

Figures:

- summary JSON: `artifacts/figures/detour_reward_v2_eval30_matched_compare/experiment_summary.json`
- success curve: `artifacts/figures/detour_reward_v2_eval30_matched_compare/success_rate_vs_steps.png`
- return curve: `artifacts/figures/detour_reward_v2_eval30_matched_compare/return_vs_steps.png`
- distance curve: `artifacts/figures/detour_reward_v2_eval30_matched_compare/final_distance_vs_steps.png`
- trajectory comparison: `artifacts/figures/detour_reward_v2_eval30_matched_compare/trajectory_comparison.png`

Final `30`-episode evaluation mean:

| Method | Success | Mean final distance | Mean return |
|---|---:|---:|---:|
| PPO scratch | `0.00` | `0.736` | `36.99` |
| BC + PPO | `0.42` | `0.312` | `108.29` |

This is the main result worth carrying forward.

### Conservative best-checkpoint re-evaluation

Because the task is noisy, the selected best checkpoints were re-evaluated on `50` episodes.

Seed-level results:

| Method | Seed | Success | Mean final distance | Mean return |
|---|---:|---:|---:|---:|
| PPO scratch | `7` | `0.00` | `0.773` | `25.18` |
| PPO scratch | `11` | `0.00` | `0.793` | `16.60` |
| BC + PPO | `7` | `0.36` | `0.384` | `89.72` |
| BC + PPO | `11` | `0.12` | `0.360` | `131.52` |

Conservative mean over the two seeds:

| Method | Success | Mean final distance | Mean return |
|---|---:|---:|---:|
| PPO scratch | `0.00` | `0.783` | `20.89` |
| BC + PPO | `0.24` | `0.372` | `110.62` |

Even under this stricter re-evaluation, BC-initialized PPO remains clearly ahead.

## Interpretation

There are three results that matter most.

1. The easy waypoint task was not enough to separate IL from scratch RL in a decisive way.
2. The detour task exposed a real exploration bottleneck that scratch PPO struggled with.
3. Once the task reward and evaluation protocol were cleaned up, BC initialization gave a meaningful advantage on both success and final distance.

The cleanest technical message is:

> IL mattered most when the task required a non-greedy detour policy and when PPO was prevented from destroying the imitation prior too quickly.

## Portfolio framing

A strong portfolio summary would be:

> Built an imitation-learning-to-reinforcement-learning pipeline for drone navigation and extended it with a detour-constrained navigation task to stress exploration. After redesigning the task reward and evaluation protocol, BC-initialized PPO outperformed scratch PPO under a matched 300k-step budget, improving final success from `0.00` to `0.42` on 30-episode evaluations and reducing mean final distance from `0.736` to `0.312`.

A more conservative version using the `50`-episode re-evaluation is:

> Designed a harder detour navigation benchmark for drone control and showed that BC-initialized PPO achieved nonzero success (`0.24` mean across 2 seeds) while scratch PPO remained at `0.00`, with mean final distance reduced from `0.783` to `0.372`.
