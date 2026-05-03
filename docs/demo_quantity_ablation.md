# Demo Quantity Ablation

This note summarizes how the number of demonstrations affects detour-task performance **after fixing the BC -> PPO transfer path**.

That fix matters, because the old quantity story was partially entangled with a broken BC initialization path.

## Setup

- task: `DetourWaypointVelocityAviary`
- expert: scripted detour expert
- demo counts: `10`, `50`, `200`
- BC checkpoints retrained with:
  - observation normalization saved and reused
  - linear output head
- PPO setting for fine-tuning:
  - `300k` timesteps
  - `eval_episodes=30`
  - `eval_freq=50k`
  - `log_std_init=-0.5`
  - `bc_kl_coef=0.0003`
- seeds for BC+PPO: `7`, `11`, `19`

## Artifacts

- summary JSON: `artifacts/figures/demo_quantity_ablation_aligned/summary.json`
- figure: `artifacts/figures/demo_quantity_ablation_aligned/demo_quantity_ablation_aligned.png`
- rerun note: `docs/normalization_fix_rerun.md`

## BC-only evaluation

`50`-episode evaluation:

| Demo episodes | Success | Mean final distance | Mean return |
|---|---:|---:|---:|
| `10` | `0.02` | `0.747` | `50.08` |
| `50` | `0.28` | `0.329` | `98.15` |
| `200` | `0.94` | `0.075` | `71.38` |

The post-fix BC-only story is very clear:

- `10` demos are far too sparse
- `50` demos already produce a usable BC policy
- `200` demos make BC itself almost sufficient

## BC + PPO final evaluation

Mean over `3` seeds using the final `30`-episode evaluation:

| Demo episodes | Success | Mean final distance | Mean return |
|---|---:|---:|---:|
| `10` | `0.00` | `0.704` | `22.50` |
| `50` | `0.111` | `0.476` | `81.83` |
| `200` | `0.10` | `0.426` | `104.19` |

## Interpretation

After the fix, the quantity story changes a little.

### `10` demos

- still too weak for BC
- still too weak for a useful PPO warm start

### `50` demos

- a strong middle regime
- BC is useful but not dominant
- PPO clearly benefits from the prior

### `200` demos

- no longer looks like a clear regression regime
- BC+PPO becomes competitive again once initialization is correct
- BC-only is already so strong that RL is less essential

So the older “`50` demos is the only real sweet spot” story becomes softer.

The more accurate corrected version is:

- **too few demos**: weak prior
- **moderate demos**: strong IL -> RL regime
- **many demos**: BC itself becomes very strong, so PPO becomes less necessary rather than obviously harmful

## Practical takeaway

For this detour benchmark, the most interesting IL -> RL regime is still the middle one, but the high-data regime is healthier than it first appeared once the transfer bug is fixed.

## Bottom line

> After correcting the BC initialization path, `10` demos remain insufficient, `50` demos still provide a strong warm-start regime, and `200` demos become competitive again because BC itself is already close to solving the task.
