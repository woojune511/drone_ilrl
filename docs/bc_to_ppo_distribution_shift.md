# BC to PPO Distribution Shift Analysis

## Purpose

This note documents the main technical problem currently limiting the portfolio project:

> BC produces a useful detour policy, but PPO fine-tuning can move the policy away from that useful behavior and reduce success.

For the portfolio story, this is important because the project should not only report a result. It should also show:

- what AI/ML failure mode was identified
- what hypotheses were tested
- what evidence supported or rejected each hypothesis
- what engineering changes were made to make the diagnosis measurable

## Problem

The BC -> PPO transition has a distribution-shift problem.

BC is trained on expert states. PPO then collects on-policy rollouts from its own policy. Once PPO actions deviate from the BC policy, the drone visits states that are less represented in the demonstration data. In those states:

- the copied BC prior may not give reliable recovery actions
- the PPO value function may be inaccurate
- policy-gradient updates can optimize dense reward locally while damaging the closed-loop route
- small action changes can compound over time into collisions, missed settling, or failed final approach

The practical symptom is:

- early checkpoints often preserve BC-like behavior
- later PPO checkpoints sometimes have lower success even when return or stage progress remains nontrivial
- `bc_probe_action_l2` increases after actor unfreezing, indicating drift from the BC prior

## Methods Already Tried

### 1. Exact BC actor initialization

BC actor weights are copied into PPO's actor network.

Purpose:

- avoid random-policy exploration at the start of PPO
- make PPO start from the demonstrated detour behavior

Evidence:

- the alignment check showed exact functional matching after the transfer fix:
  - action L2 difference: `0.0`
  - cosine similarity: `1.0`

Remaining issue:

- exact initialization only fixes step 0
- it does not prevent later PPO updates from drifting away

### 2. Fixed BC observation normalization

PPO uses the BC checkpoint's `obs_mean` and `obs_std`.

Purpose:

- make PPO consume observations in the same scale that BC saw during training
- remove a confound where PPO and BC were acting on different input distributions

Evidence:

- after this fix, BC and PPO initialization became exactly aligned
- this turned the BC -> PPO comparison into a valid experiment instead of a transfer-bug artifact

Remaining issue:

- input-scale alignment does not solve on-policy state-distribution shift during PPO training

### 3. Linear BC output head

BC was changed to use a linear output head compatible with PPO's Gaussian mean head.

Purpose:

- make copied BC actor weights represent the same action function inside PPO

Evidence:

- post-fix BC/PPO alignment is exact at initialization

Remaining issue:

- head compatibility fixes initialization, not fine-tuning stability

### 4. BC-KL regularization

PPO loss includes a KL-style penalty between PPO action means and BC action means.

Current ablation values:

- `bc_kl_coef = 0.003`
- `bc_kl_coef = 0.01`

Purpose:

- slow down policy drift away from BC
- keep PPO fine-tuning near the useful demonstration prior

Evidence:

- during freeze periods, `bc_probe_action_l2 = 0.0` and cosine similarity is `1.0`
- after unfreezing, `bc_probe_action_l2` can increase substantially

Remaining issue:

- the current BC prior is checked on a fixed expert-state probe
- staying close on expert states does not guarantee robust behavior on PPO-induced off-distribution states
- high KL does not fully prevent final success collapse in some seeds

### 5. Actor freeze warm-start

The actor can be frozen for an initial number of PPO environment steps.

Current ablation values:

- `freeze_actor_steps = 25,000`
- `freeze_actor_steps = 50,000`

Purpose:

- let the value function and rollout statistics adapt before actor updates begin
- preserve the BC controller during early PPO rollouts

Evidence:

- while frozen, BC drift metrics remain exact:
  - `bc_probe_action_l2 = 0.0`
  - `bc_probe_action_cosine = 1.0`
- `freeze50k` preserves the BC-like policy better than `freeze25k`

Remaining issue:

- freezing prevents damage but does not improve the policy
- once the actor is unfrozen, PPO can still degrade the controller

### 6. Lower PPO exploration variance

The Gaussian policy's initial log standard deviation is reduced.

Current ablation values:

- `log_std_init = -1.0`
- `log_std_init = -1.5`

Purpose:

- reduce destructive exploration around a useful BC policy
- avoid pushing the drone into unrecoverable off-distribution states early

Evidence:

- lower `policy_std_mean` is visible in diagnostics
- however, very low std can still fail after unfreezing

Remaining issue:

- low exploration protects the prior but may reduce useful correction learning
- it does not address biased or harmful policy-gradient updates

### 7. Expert-state augmentation with expert relabeling

The expert BC loss now supports local state augmentation around expert trajectory states.

Implementation:

- perturb expert observations in position and velocity space
- recompute `rel_goal` after position noise
- reject physically invalid detour-wall states
- relabel the perturbed states with the scripted expert instead of reusing stale dataset actions
- apply the augmented samples only through the auxiliary expert BC loss during PPO updates

Purpose:

- reduce BC -> PPO distribution shift around the expert manifold
- teach local recovery actions near demonstrated states
- give PPO a supervised anchor on nearby off-trajectory states without collecting human data

Current ablation on 2026-06-09:

- output root: `artifacts/checkpoints/ppo_expert_bc_aug_ablation_20260609_124357`
- diagnostics: `artifacts/analysis/ppo_expert_bc_aug_ablation_20260609_124357`
- seeds: `7, 19`
- PPO steps: `50k`
- common setup: `log_std_init=-1.0`, `bc_kl_coef=0.003`, `freeze_actor_steps=25k`, `expert_bc_loss_coef=1.0`
- augmentation copies: `2`

| Variant | Checkpoint | Success | Final distance | Collision | BC probe L2 |
|---|---|---:|---:|---:|---:|
| `expertbc_aug003_coef1_freeze25k` | best | `0.60` | `0.193` | `0.00` | `0.164` |
| `expertbc_aug003_coef1_freeze25k` | final | `0.25` | `0.354` | `0.05` | `0.172` |
| `expertbc_aug005_coef1_freeze25k` | best | `0.70` | `0.210` | `0.05` | `0.231` |
| `expertbc_aug005_coef1_freeze25k` | final | `0.80` | `0.129` | `0.00` | `0.231` |

Per-run final results for the stronger augmentation:

| Variant | Seed | Final success | Final distance | Final collision |
|---|---:|---:|---:|---:|
| `expertbc_aug005_coef1_freeze25k` | `7` | `0.90` | `0.101` | `0.00` |
| `expertbc_aug005_coef1_freeze25k` | `19` | `0.70` | `0.157` | `0.00` |

10-episode online-eval interpretation:

- `position_noise_std=0.05` and `velocity_noise_std=0.05` produced the strongest current result
- unlike earlier strong-regularization runs, the final checkpoint improved beyond the BC-only reference on the 10-episode online evaluation protocol
- the improvement came with larger but controlled BC probe drift (`~0.23` L2), suggesting that some deviation from BC is necessary
- `aug003` improved at intermediate checkpoints but still degraded by the final checkpoint, so augmentation strength matters

50-episode re-evaluation for `expertbc_aug005_coef1_freeze25k` final checkpoints:

| Policy | Seed | Success | Final distance | Mean return |
|---|---:|---:|---:|---:|
| `clean50 BC-only` | `20000` | `0.54` | `0.269` | `84.67` |
| `aug005 final PPO` | `7` | `0.82` | `0.119` | `80.23` |
| `aug005 final PPO` | `11` | `0.78` | `0.163` | `79.15` |
| `aug005 final PPO` | `19` | `0.64` | `0.150` | `107.76` |
| `aug005 final PPO` | `23` | `0.84` | `0.156` | `74.04` |
| `aug005 final PPO` | `31` | `0.56` | `0.206` | `107.37` |
| `aug005 final PPO mean` | `7,11,19,23,31` | `0.728` | `0.159` | `89.71` |
| `aug005 validation-best PPO mean` | `7,11,19,23,31` | `0.792` | `0.153` | `81.00` |

50-episode artifacts:

- `artifacts/evals/aug005_final_seed7_50eps_seed20000_20260609.json`
- `artifacts/evals/aug005_final_seed19_50eps_seed20000_20260609.json`
- `artifacts/evals/aug005_more_final_seed11_50eps_seed20000_20260609.json`
- `artifacts/evals/aug005_more_final_seed23_50eps_seed20000_20260609.json`
- `artifacts/evals/aug005_more_final_seed31_50eps_seed20000_20260609.json`
- `artifacts/evals/aug005_best_seed7_50eps_seed20000_20260609.json`
- `artifacts/evals/aug005_best_seed11_50eps_seed20000_20260609.json`
- `artifacts/evals/aug005_best_seed19_50eps_seed20000_20260609.json`
- `artifacts/evals/aug005_best_seed23_50eps_seed20000_20260609.json`
- `artifacts/evals/aug005_best_seed31_50eps_seed20000_20260609.json`

Updated conclusion:

- this is the first setting where PPO fine-tuning clearly improves beyond the standalone BC policy under the same `50`-episode evaluation seed
- supervised local recovery around expert states is the most defensible current explanation
- across `5` PPO seeds, final success improved from the BC-only `0.54` reference to `0.728` on average
- validation-best checkpoint selection improved the `5`-seed mean further to `0.792`
- the remaining risk is late-training variance: seed `31` final only reached `0.56`, while its validation-best checkpoint reached `0.82`

Final comparison artifacts:

- `artifacts/analysis/portfolio_final_20260609/headline_50ep_comparison.csv`
- `artifacts/analysis/portfolio_final_20260609/headline_50ep_success_rate.png`
- `artifacts/analysis/portfolio_final_20260609/headline_50ep_mean_final_distance.png`
- `artifacts/analysis/portfolio_final_20260609/online_diagnostics_comparison.csv`

Seed `31` rollout diagnosis:

- output directory: `artifacts/analysis/aug005_seed31_final_rollouts_20260609/`
- success rate: `0.56`
- validation-best success rate: `0.82`
- collision rate: `0.00`
- reached exit stage rate: `1.00`
- reached goal stage rate: `1.00`
- failures: `22 / 50`
- all failures reached the goal stage
- failure mean minimum distance: `0.105m`
- failure mean final distance: `0.358m`
- failure mean final speed: `0.076m/s`

Interpretation:

- seed `31` final does not mostly fail because it cannot discover or traverse the detour
- it reaches the goal stage and avoids collisions
- the weak point is final approach/settling: the policy often gets close, then fails to remain inside the success region with the required terminal behavior
- validation-best checkpoint selection substantially recovers this seed, so the next fixes should target late-training drift and the terminal controller/reward condition, not broad exploration

## Current Strong-Regularization Ablation

Status on 2026-06-09:

- output root: `artifacts/checkpoints/ppo_bc_strong_regularization_ablation_metrics_rerun_20260609_092402`
- total planned runs: `6`
- completed runs: `6`
- diagnostics output: `artifacts/analysis/ppo_bc_strong_regularization_ablation_metrics_rerun_20260609_092402`

### Per-Run Results

| Variant | Seed | Best step | Best success | Best final distance | Final success | Final distance | Final collision | Final BC probe L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `low_std_kl3e3_freeze50k` | `7` | `12.5k` | `0.6` | `0.272` | `0.6` | `0.251` | `0.0` | `0.090` |
| `low_std_kl3e3_freeze50k` | `19` | `12.5k` | `0.2` | `0.442` | `0.1` | `0.397` | `0.0` | `0.069` |
| `low_std_kl1e2_freeze25k` | `7` | `12.5k` | `0.6` | `0.272` | `0.0` | `0.580` | `0.3` | `0.341` |
| `low_std_kl1e2_freeze25k` | `19` | `37.5k` | `0.2` | `0.395` | `0.0` | `0.549` | `0.2` | `0.402` |
| `very_low_std_kl1e2_freeze25k` | `7` | `12.5k` | `0.6` | `0.272` | `0.0` | `0.654` | `0.6` | `0.266` |
| `very_low_std_kl1e2_freeze25k` | `19` | `12.5k` | `0.2` | `0.442` | `0.0` | `0.735` | `0.9` | `0.288` |

### Variant Means

| Variant | Checkpoint | Success | Final distance | Collision | BC probe L2 |
|---|---|---:|---:|---:|---:|
| `low_std_kl3e3_freeze50k` | best | `0.40` | `0.357` | `0.15` | `0.000` |
| `low_std_kl3e3_freeze50k` | final | `0.35` | `0.324` | `0.00` | `0.080` |
| `low_std_kl1e2_freeze25k` | best | `0.40` | `0.334` | `0.15` | `0.114` |
| `low_std_kl1e2_freeze25k` | final | `0.00` | `0.564` | `0.25` | `0.371` |
| `very_low_std_kl1e2_freeze25k` | best | `0.40` | `0.357` | `0.15` | `0.000` |
| `very_low_std_kl1e2_freeze25k` | final | `0.00` | `0.694` | `0.75` | `0.277` |

### Best Checkpoint 50-Episode Re-Evaluation

The online evaluations above used `10` episodes, so the selected best checkpoints were re-evaluated with `50` episodes.

Artifacts:

- output directory: `artifacts/evals/strong_reg_best_50eps_20260609`
- summary CSV: `artifacts/evals/strong_reg_best_50eps_20260609/best_50eps_summary.csv`
- BC-only reference: `artifacts/evals/bc_clean50_50eps_seed20000_20260609.json`

BC-only with the same `50`-episode protocol:

| Policy | Success | Mean final distance | Mean return |
|---|---:|---:|---:|
| `clean50 BC-only` | `0.54` | `0.269` | `84.67` |

| Variant | Seed | Best checkpoint success | Mean final distance | Mean return |
|---|---:|---:|---:|---:|
| `low_std_kl3e3_freeze50k` | `7` | `0.54` | `0.269` | `84.67` |
| `low_std_kl3e3_freeze50k` | `19` | `0.54` | `0.269` | `84.67` |
| `low_std_kl1e2_freeze25k` | `7` | `0.54` | `0.269` | `84.67` |
| `low_std_kl1e2_freeze25k` | `19` | `0.26` | `0.443` | `74.31` |
| `very_low_std_kl1e2_freeze25k` | `7` | `0.54` | `0.269` | `84.67` |
| `very_low_std_kl1e2_freeze25k` | `19` | `0.54` | `0.269` | `84.67` |

Variant means from the `50`-episode best-checkpoint re-evaluation:

| Variant | Best success | Final distance | Mean return |
|---|---:|---:|---:|
| `low_std_kl3e3_freeze50k` | `0.54` | `0.269` | `84.67` |
| `low_std_kl1e2_freeze25k` | `0.40` | `0.356` | `79.49` |
| `very_low_std_kl1e2_freeze25k` | `0.54` | `0.269` | `84.67` |

This re-evaluation changes the interpretation slightly:

- most best checkpoints are effectively BC-preserving early checkpoints
- their `50`-episode performance is close to standalone BC behavior
- the worse `low_std_kl1e2_freeze25k` seed `19` best checkpoint was selected later (`37.5k`) after policy drift had already begun
- best-checkpoint selection can recover a usable policy, but it does not show that PPO fine-tuning reliably improves beyond the BC prior

The direct BC-only comparison is the key point:

> The best PPO checkpoints do not currently beat BC-only under the same `50`-episode protocol. They mostly recover the BC policy.

### Interpretation

The strongest current signal is:

> `freeze50k` is more stable than `freeze25k`, but it mostly preserves the BC behavior rather than producing clear PPO improvement.

The `freeze25k` variants show the failure mode clearly:

- early checkpoints can match BC-level success
- after actor unfreezing, final success can fall to `0.0`
- BC probe drift rises to roughly `0.27-0.40`
- collision can increase substantially

The very-low-exploration variant was not better:

- `log_std_init = -1.5` reduced policy standard deviation
- final success still fell to `0.0`
- final collision increased to `0.75` on average
- its strong best-checkpoint re-evaluation mostly reflects early BC preservation, not successful low-std PPO improvement

This supports the hypothesis that the main issue is not initialization anymore. The main issue is **policy degradation during PPO fine-tuning**.

## Metrics Added for Diagnosis

Evaluation metrics:

- `success_rate`
- `mean_final_distance`
- `mean_min_distance`
- `position_only_success_rate`
- `collision_rate`
- `reached_exit_stage_rate`
- `reached_goal_stage_rate`
- `mean_final_speed`

Policy distribution metrics:

- `policy_log_std_mean`
- `policy_log_std_min`
- `policy_log_std_max`
- `policy_std_mean`

BC drift probe metrics:

- `bc_probe_action_l2`
- `bc_probe_action_cosine`
- `bc_probe_max_abs_action_diff`
- `bc_probe_action_saturation_rate`

PPO training diagnostics:

- `train/approx_kl`
- `train/entropy_loss`
- `train/policy_gradient_loss`
- `train/value_loss`
- `train/loss`
- `train/explained_variance`
- `train/clip_fraction`
- `train/bc_kl_loss`
- `train/bc_kl_loss_weighted`
- `train/std`
- `train/log_std_mean`
- `train/log_std_min`
- `train/log_std_max`

These metrics are plotted by:

- `scripts/plot_training_diagnostics.py`

## Current Technical Conclusion

The project has moved through three distinct technical stages:

1. **Original pipeline problem**
   - BC -> PPO transfer was not exactly aligned.
   - Fixed with observation normalization and linear BC output head.

2. **Exploration bottleneck problem**
   - Scratch PPO struggles on the detour task.
   - BC initialization gives useful early behavior and better proximity-to-goal.

3. **Fine-tuning stability problem**
   - PPO can damage a useful BC controller.
   - Stronger KL, actor freezing, and lower exploration reduce some damage but do not fully solve the distribution shift.
   - Expert-state augmentation with scripted relabeling provides local recovery supervision and improves beyond BC-only.

For portfolio framing, the honest story is:

> The project successfully built an IL -> RL pipeline, diagnosed BC-to-PPO distribution shift, and showed that local expert-state augmentation can turn BC initialization from a preserved prior into an improved policy. The remaining research issue is not detour discovery; it is late-training drift and final-approach stability.

## Next Candidate Interventions

The next interventions should target PPO's destructive late-update phase and final settling more directly.

1. **Make validation-best checkpoint selection explicit**
   - This is now the official portfolio protocol.
   - Final checkpoints are still reported as a robustness diagnostic.
   - Current `aug005` result: final success `0.728`, validation-best success `0.792`.

2. **Lower-drift PPO update**
   - Test lower actor learning rate, for example `3e-4 -> 1e-4`.
   - Add `target_kl` so PPO updates stop when policy movement becomes too large.
   - Goal: make final checkpoints closer to validation-best checkpoints.

3. **Adaptive BC-KL coefficient**
   - Increase KL when `bc_probe_action_l2` crosses a threshold.
   - This directly uses the measured drift signal.

4. **Final-approach augmentation**
   - Oversample goal-near expert states.
   - Perturb final-approach states and relabel with the scripted expert.
   - This targets the seed `31` failure mode directly.

5. **Separate success/settling reward audit**
   - Some policies reach the goal stage but fail strict success.
   - The final stabilization condition may need a more targeted reward or curriculum.

## Interview Framing

A concise interview explanation:

> I first fixed the BC-to-PPO transfer so the initialized PPO actor exactly matched the BC policy. After that, I found that initialization was not the main bottleneck anymore: PPO fine-tuning could move the policy away from useful BC behavior. KL regularization and actor freezing mostly preserved BC, but did not improve it. The key improvement was to add local recovery supervision by perturbing expert trajectory states and relabeling them with the scripted expert during PPO updates. Under the shared 50-episode evaluation protocol, this improved success from BC-only `0.54` to `0.792` with validation-best checkpoint selection across five PPO seeds.
