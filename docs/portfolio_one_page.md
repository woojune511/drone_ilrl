# Drone IL->RL Portfolio Summary

## Project

Built a PyBullet drone navigation benchmark to study when imitation learning helps reinforcement learning. The final task is a detour-constrained waypoint problem: the drone starts on the left, the goal is on the right, and a wall blocks the direct path except for a narrow upper corridor.

## Technical Problem

The original waypoint task was too easy for PPO, so imitation pretraining did not show a clear advantage. The harder detour task exposed a more interesting issue: behavior cloning gave a useful navigation prior, but PPO fine-tuning could drift away from that prior and reduce closed-loop success.

The main AI failure mode was **BC -> PPO distribution shift**:

- BC is trained on expert states.
- PPO collects on-policy states from its own slightly changed policy.
- Small policy changes can move the drone into states not well covered by demonstrations.
- KL regularization and actor freezing preserved BC behavior but did not reliably improve it.

## Method

I implemented:

- `DetourWaypointVelocityAviary`, a non-greedy drone navigation task with a blocking wall and corridor.
- A scripted detour expert for automatic demonstration collection.
- Behavior cloning with fixed observation normalization.
- PPO actor initialization from the BC policy.
- Diagnostics for policy drift, action saturation, entropy, KL, collision, stage progress, final distance, and success.
- Expert-state augmentation during PPO: perturb expert trajectory states, recompute task-relative features, reject invalid wall states, and relabel actions with the scripted expert.

The key change was using augmented expert states as local recovery supervision during PPO updates. This directly targeted the distribution-shift region around the expert trajectory instead of only constraining the policy on the original demonstrations.

## Result

Evaluation protocol: `50` deterministic episodes, evaluation seed `20000`.

| Method | Policy seeds | Success | Mean final distance |
|---|---:|---:|---:|
| BC-only | 1 | `0.54` | `0.269m` |
| BC+PPO KL/freeze best | 2 | `0.54` | `0.269m` |
| BC+PPO expert-state aug final | 5 | `0.728 +/- 0.109` | `0.159 +/- 0.028m` |
| BC+PPO expert-state aug validation-best | 5 | `0.792 +/- 0.104` | `0.153 +/- 0.041m` |

Main artifacts:

- `artifacts/analysis/portfolio_final_20260609/headline_50ep_comparison.csv`
- `artifacts/analysis/portfolio_final_20260609/headline_50ep_success_rate.png`
- `artifacts/analysis/portfolio_final_20260609/headline_50ep_mean_final_distance.png`
- `docs/bc_to_ppo_distribution_shift.md`

## Failure Analysis

The weakest final checkpoint was seed `31`, with `0.56` success. Episode-level analysis showed:

- collision rate: `0.00`
- reached exit stage rate: `1.00`
- reached goal stage rate: `1.00`
- failures: `22 / 50`
- all failures reached the goal stage
- validation-best checkpoint recovered to `0.82` success

This means the remaining weakness is not detour discovery or obstacle avoidance. It is late-training drift and final approach/settling stability.

## Interview Framing

The project moved from a basic IL->RL pipeline to a concrete diagnosis-and-fix story. I first made BC-to-PPO transfer exact, then measured how PPO drift damaged the BC prior. Standard regularization mostly preserved BC, but expert-state augmentation with scripted relabeling provided local recovery supervision and improved success beyond BC-only across five PPO seeds.
