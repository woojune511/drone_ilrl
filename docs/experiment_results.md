# Drone IL -> RL Experiment Results

## Final takeaway

The most trustworthy result in this repo now comes from the **detour navigation task after fixing the BC -> PPO transfer path**.

The original transfer had two implementation mismatches:

1. BC was trained on normalized observations, but PPO originally consumed raw observations.
2. BC originally used a `Tanh` output head, while PPO used a linear Gaussian mean head.

Those issues are documented in:

- `docs/normalization_fix_rerun.md`

After fixing them, the cleanest matched comparison still shows the same high-level story:

> On the detour-constrained task, BC-initialized PPO learns materially better behavior than scratch PPO under the same `300k`-step budget.

## Repo components

- Base waypoint environment: `src/ilrl_lab/envs/waypoint_vel_aviary.py`
- Harder detour variant: `src/ilrl_lab/envs/detour_vel_aviary.py`
- Scripted experts: `src/ilrl_lab/experts/velocity.py`
- BC model: `src/ilrl_lab/bc.py`
- Shared PPO utilities: `src/ilrl_lab/ppo_training.py`
- Scratch PPO entrypoint: `scripts/train_ppo.py`
- BC-initialized PPO entrypoint: `scripts/fine_tune_ppo_from_bc.py`
- Alignment sanity check: `scripts/check_bc_ppo_alignment.py`
- Plotting: `scripts/plot_experiment_results.py`

## Phase 1: Waypoint baseline

The original waypoint task validated the end-to-end pipeline but was still fairly PPO-friendly:

- full-state observation
- dense reward
- high-level velocity action
- a single goal without a true path-planning bottleneck

That setting was useful, but it did not separate IL from scratch RL sharply enough.

## Phase 2: Detour task

The detour task adds a real exploration bottleneck:

- start sampled on the left side of the workspace
- goal sampled on the right side
- a blocking wall prevents direct flight
- only an upper corridor is open

This turns the problem into a staged navigation task:

1. align with the corridor
2. pass the wall
3. return to the final goal and stabilize

The task became much more informative once three things were in place:

- a corridor-aware scripted expert
- detour-aware reward shaping (`v2`)
- less noisy evaluation (`30` episodes during training, `50` for BC-only checks)

## Transfer fix

The original BC -> PPO pipeline was not functionally exact at initialization.

Fixes applied:

- PPO training/evaluation now uses the BC checkpoint's observation normalization stats.
- BC fine-tuning now uses a **linear output head** so the copied BC actor matches PPO's action head.
- PPO initialization now rejects old tanh-headed BC checkpoints.

Sanity check:

- `artifacts/checks/bc_ppo_alignment_clean50_fixed.json`

Result:

- `mean_action_l2_diff = 0.0`
- `mean_cosine_similarity = 1.0`

So after the fix, BC and PPO really do start from the same policy.

## Final matched comparison

### Protocol

- task: `detour`
- reward: detour-aware shaping `v2`
- total timesteps: `300,000`
- seeds: `7`, `11`, `19`
- evaluation every `50,000` steps
- `30` evaluation episodes per checkpoint
- scratch PPO:
  - `log_std_init=-0.5`
  - same observation normalization as BC+PPO
- BC+PPO:
  - same PPO settings
  - BC actor initialization
  - `bc_kl_coef=0.0003`

### Artifacts

- summary JSON: `artifacts/figures/detour_aligned_matched_compare/experiment_summary.json`
- success curve: `artifacts/figures/detour_aligned_matched_compare/success_rate_vs_steps.png`
- return curve: `artifacts/figures/detour_aligned_matched_compare/return_vs_steps.png`
- distance curve: `artifacts/figures/detour_aligned_matched_compare/final_distance_vs_steps.png`
- trajectory comparison: `artifacts/figures/detour_aligned_matched_compare/trajectory_comparison.png`

### Final 30-episode mean

| Method | Success | Mean final distance | Mean return |
|---|---:|---:|---:|
| PPO scratch | `0.00` | `0.732` | `21.15` |
| BC + PPO | `0.111` | `0.476` | `81.83` |

### Learning-curve interpretation

The updated curves show a cleaner version of the same story:

- scratch PPO remains near zero success across all three seeds
- BC+PPO starts improving distance earlier
- BC+PPO is the only method that reaches nonzero mean success by the end of the matched budget

This comparison is more trustworthy than the older one because the transfer path is now behaviorally aligned at step 0.

## Quantity ablation after the fix

Artifacts:

- summary JSON: `artifacts/figures/demo_quantity_ablation_aligned/summary.json`
- figure: `artifacts/figures/demo_quantity_ablation_aligned/demo_quantity_ablation_aligned.png`

### BC-only

| Demo episodes | Success | Mean final distance | Mean return |
|---|---:|---:|---:|
| `10` | `0.02` | `0.747` | `50.08` |
| `50` | `0.28` | `0.329` | `98.15` |
| `200` | `0.94` | `0.075` | `71.38` |

### BC + PPO final 3-seed mean

| Demo episodes | Success | Mean final distance | Mean return |
|---|---:|---:|---:|
| `10` | `0.00` | `0.704` | `22.50` |
| `50` | `0.111` | `0.476` | `81.83` |
| `200` | `0.10` | `0.426` | `104.19` |

### Updated interpretation

- `10` demos are still too weak to form a useful prior.
- `50` demos remain a strong IL -> RL regime.
- `200` demos become competitive again after the transfer fix, because BC itself is now very strong and the PPO initialization is no longer distorted.

So the old “50 is clearly the only sweet spot” story becomes softer after the fix. The more accurate version is:

> moderate and large clean datasets both help, but they help in different ways: `50` demos provide a useful warm start while `200` demos make BC itself nearly sufficient.

## Quality ablation after the fix

Artifacts:

- summary JSON: `artifacts/figures/demo_quality_ablation_aligned/summary.json`
- figure: `artifacts/figures/demo_quality_ablation_aligned/demo_quality_ablation_aligned.png`

### BC-only

| Demo quality | Success | Mean final distance | Mean return |
|---|---:|---:|---:|
| Clean `50` | `0.28` | `0.329` | `98.15` |
| Noisy `50` | `0.66` | `0.292` | `82.01` |

### BC + PPO final 3-seed mean

| Demo quality | Success | Mean final distance | Mean return |
|---|---:|---:|---:|
| Clean `50` | `0.111` | `0.476` | `81.83` |
| Noisy `50` | `0.00` | `0.677` | `52.64` |

### Updated interpretation

This part of the story survives the transfer fix very clearly:

- noisy demonstrations make **BC-only** much stronger
- but noisy demonstrations make **BC-initialized PPO** worse

That suggests noisy demos inject recovery-friendly behavior for pure cloning, while clean demos provide a sharper and more useful RL warm-start prior.

## Bottom line

There are three results that matter most now.

1. The BC -> PPO transfer path originally had a real implementation defect, and fixing it mattered.
2. After the fix, the main detour result still holds:
   - BC-initialized PPO outperforms scratch PPO under a matched `300k`-step budget.
3. The more nuanced ablation story also survives:
   - noisy demos help BC-only robustness,
   - clean demos are better for downstream RL warm starts,
   - larger clean datasets make BC itself increasingly sufficient.
