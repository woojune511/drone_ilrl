# Detour Quick 100k Results

## Purpose

This quick sweep checks whether the accelerated parallel PPO setup gives a useful early signal before committing to longer `300k` runs.

## Protocol

- task: `detour`
- BC checkpoint: `artifacts/checkpoints/bc/bc_20260608_132020/checkpoint.pt`
- demonstration dataset: `artifacts/datasets/detour_clean50_current_expert_20260608_131955.npz`
- seeds: `7`, `11`, `19`
- total timesteps: `100,000`
- eval frequency: `25,000`
- eval episodes: `10`
- parallel envs: `4`
- rollout steps per env: `256`
- exploration init: `log_std_init=-0.5`
- BC regularization: `bc_kl_coef=0.0003`

## Artifacts

- summary: `artifacts/figures/detour_quick_100k_clean50_compare/experiment_summary.json`
- success curve: `artifacts/figures/detour_quick_100k_clean50_compare/success_rate_vs_steps.png`
- final-distance curve: `artifacts/figures/detour_quick_100k_clean50_compare/final_distance_vs_steps.png`
- return curve: `artifacts/figures/detour_quick_100k_clean50_compare/return_vs_steps.png`
- trajectory comparison: `artifacts/figures/detour_quick_100k_clean50_compare/trajectory_comparison.png`

## Final 3-seed Mean

| Method | Success | Mean final distance | Mean return |
|---|---:|---:|---:|
| PPO scratch | `0.000` | `1.101` | `11.43` |
| BC + PPO | `0.033` | `0.749` | `31.27` |

## Interpretation

The quick sweep does not show robust task solving at `100k` steps. Scratch PPO stays at zero success for all three seeds. BC-initialized PPO reaches nonzero success in one seed and improves mean final distance and return, but the result is still high variance.

The defensible claim from this run is:

> Under a `100k`-step budget, BC initialization improves proximity-to-goal and return on average, while scratch PPO remains at zero success.

This is not yet strong enough to claim that BC+PPO reliably solves the detour task. The next meaningful step is a longer `300k` run, preferably with the same accelerated `N_ENVS=4`, `N_STEPS=256` setup and at least `3` seeds.
