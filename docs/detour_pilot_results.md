# Detour Experiment Notes

This document records the practical lessons from turning the original waypoint benchmark into a harder detour task.

## What changed

The detour task adds:

- left-to-right start/goal structure
- a blocking wall in the center of the workspace
- a single upper corridor
- collision-aware truncation

The intended effect was to make straight-line goal reaching insufficient and force a real path choice.

## Main issues discovered

### 1. The first expert was not reliable enough

The initial detour expert sometimes stopped at the corridor exit instead of committing to the final goal. That made the demonstrations too weak to support a stable BC prior.

Fix:

- treat corridor waypoints as transit waypoints
- switch to the final goal before velocity collapses

Result:

- quick validation improved to `10/10` expert successes on seeds `0..9`

### 2. Pure BC stayed fragile

Even after improving expert data quality, standalone BC still struggled in closed loop on the randomized detour distribution. The demonstrations were useful as a warm start, but not enough to solve the task by imitation alone.

### 3. Reward design mattered as much as algorithm design

The first detour reward was too close to the original final-goal reward, so the agent had weak incentive to discover the corridor subgoal. A stronger detour-shaped reward fixed that partially, but overfit badly across seeds.

The more balanced result came from the softer detour reward (`v2`), which:

- kept a detour-stage reward
- lowered corridor target reward magnitude
- lowered stage-progress reward magnitude
- lowered transition bonus

This was the first version that produced useful matched-comparison results across more than one seed.

### 4. Evaluation noise hid real progress

Using only `10` evaluation episodes made checkpoint selection too noisy. The same run could look strong during training and then collapse under a larger re-evaluation.

Fix:

- increase online evaluation to `30` episodes
- re-check selected best models on `50` episodes

This made the final comparison much more trustworthy.

## Most useful final setting

The strongest detour setting so far is:

- task: `DetourWaypointVelocityAviary`
- reward: softer detour-aware shaping (`v2`)
- PPO horizon: `300k`
- evaluation: `30` episodes every `50k`
- PPO exploration init: `log_std_init = -0.5`
- BC regularization: `bc_kl_coef = 0.0003`

## Why this setting matters

Under the final matched `3`-seed setup, scratch PPO stayed at `0.00` mean success, while BC-initialized PPO reached `0.30` mean success on the online `30`-episode evaluations and `0.27` mean success on the stricter `50`-episode best-checkpoint re-evaluations.

The final comparison figures are in:

- `artifacts/figures/detour_reward_v2_eval30_matched_compare/`

The main report is:

- `docs/experiment_results.md`

## Bottom line

The detour extension did what it was supposed to do:

- it exposed a real exploration bottleneck
- it made task design a first-class issue
- and it created a setting where imitation pretraining had a clearer benefit than in the original waypoint benchmark
