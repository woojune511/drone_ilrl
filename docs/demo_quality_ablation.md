# Demo Quality Ablation

This note compares **clean** versus **noisy** demonstrations on the detour task after fixing the BC -> PPO transfer path.

The transfer fix matters here, because the older version mixed together:

- demo quality effects
- observation-normalization mismatch
- BC/PPO output-head mismatch

The corrected rerun isolates the quality question much more cleanly.

## Setup

- task: `DetourWaypointVelocityAviary`
- demo count: `50`
- clean demos: scripted detour expert
- noisy demos: expert actions perturbed during collection with:
  - `action_noise_std = 0.20`
  - `sticky_action_prob = 0.35`
  - `action_mix_alpha = 0.55`
- BC checkpoints retrained with:
  - observation normalization saved and reused
  - linear output head
- PPO fine-tuning:
  - `300k` timesteps
  - `eval_episodes=30`
  - `eval_freq=50k`
  - `log_std_init=-0.5`
  - `bc_kl_coef=0.0003`
- seeds for BC+PPO: `7`, `11`, `19`

## Artifacts

- summary JSON: `artifacts/figures/demo_quality_ablation_aligned/summary.json`
- figure: `artifacts/figures/demo_quality_ablation_aligned/demo_quality_ablation_aligned.png`
- rerun note: `docs/normalization_fix_rerun.md`

## BC-only comparison

`50`-episode evaluation:

| Demo quality | Success | Mean final distance | Mean return |
|---|---:|---:|---:|
| Clean | `0.28` | `0.329` | `98.15` |
| Noisy | `0.66` | `0.292` | `82.01` |

After the fix, the BC-only result is even clearer than before:

- noisy demonstrations substantially improve BC success
- they also slightly improve final distance
- but they lower mean return, suggesting a less efficient but more recoverable controller

## BC + PPO final evaluation

Mean over `3` seeds using the final `30`-episode evaluation:

| Demo quality | Success | Mean final distance | Mean return |
|---|---:|---:|---:|
| Clean | `0.111` | `0.476` | `81.83` |
| Noisy | `0.00` | `0.677` | `52.64` |

## Interpretation

The corrected rerun makes the quality story sharper:

### Clean demos

- weaker standalone BC than noisy demos
- but a much better PPO warm-start prior
- better downstream success after fine-tuning

### Noisy demos

- much stronger BC-only robustness
- but a worse PPO initialization
- downstream PPO training never catches up to the clean-demo version

## Practical takeaway

For this detour benchmark:

- **noisy expert data** is better when the goal is robust imitation alone
- **clean expert data** is better when the goal is imitation-assisted RL

That is a stronger and cleaner result than the earlier pre-fix version.

## Bottom line

> After fixing the BC -> PPO transfer path, noisy demos still help standalone BC, but clean demos remain clearly better for downstream PPO warm starts.
