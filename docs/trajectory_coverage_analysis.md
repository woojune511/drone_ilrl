# Trajectory Coverage Analysis

This analysis looks beyond start/goal sampling and asks whether the **actual expert trajectories** help explain the `10 / 50 / 200` demo ablation.

## Artifacts

- script: `scripts/analyze_trajectory_coverage.py`
- summary JSON: `artifacts/figures/trajectory_coverage_analysis/summary.json`
- expert overlay: `artifacts/figures/trajectory_coverage_analysis/expert_xy_overlay.png`
- stage heatmaps: `artifacts/figures/trajectory_coverage_analysis/stage_heatmaps.png`
- diversity metrics: `artifacts/figures/trajectory_coverage_analysis/path_diversity_metrics.png`
- BC vs expert overlay: `artifacts/figures/trajectory_coverage_analysis/bc_vs_expert_overlay.png`

## What was checked

Four things were inspected.

1. **Expert trajectory XY coverage**
2. **Stage-wise visitation heatmaps**
3. **Trajectory diversity metrics**
4. **BC rollout vs expert rollout overlays on the same eval seeds**

## Main finding

The most important result is that the expert trajectories are **very consistent in shape** across all demo quantities.

That means the low-demo failure is less about the expert using wildly different path styles, and more about the policy simply seeing too few examples of the same detour template across randomized starts and goals.

## 1. Expert trajectory XY coverage

The XY overlays show that all expert datasets use essentially the same detour template:

- rise toward the upper corridor
- pass through the same gap
- descend and approach the goal from the right side

As the dataset grows from `10` to `200` demos, the cloud becomes denser, but it does **not** suddenly discover qualitatively different paths.

This supports the idea that the expert policy is stable and that the main change is **coverage density**, not strategy diversity.

## 2. Stage-wise visitation heatmaps

The stage heatmaps show the same pattern more clearly:

- `entry` stage mass stays concentrated near the same approach band
- `exit` stage mass stays tightly concentrated around the corridor
- `goal` stage mass broadens a bit with more data, especially after passing the wall

So the additional data mostly improves how thoroughly the policy sees:

- different entry alignments
- slightly different post-corridor recoveries
- more goal-approach states

## 3. Trajectory diversity metrics

The diversity metrics are surprisingly stable:

| Demo episodes | Mean path length | Pairwise path distance | Final approach heading std |
|---|---:|---:|---:|
| `10` | `1.606` | `0.0577` | `0.277` |
| `50` | `1.635` | `0.0639` | `0.277` |
| `200` | `1.625` | `0.0677` | `0.272` |

Interpretation:

- path length barely changes
- corridor behavior barely changes
- final approach heading variation barely changes
- pairwise path distance increases only a little

So the expert trajectories are not becoming dramatically more diverse with more data. They are mostly becoming **more densely sampled** versions of the same path family.

## 4. BC rollout vs expert rollout overlays

This is the most intuitive figure.

Using the same evaluation seeds:

- expert rollouts succeed in all three quantities
- BC-only rollouts for `10` and `50` still visibly diverge from expert behavior
- BC-only rollouts for `200` align much more closely and succeed on the sampled seeds

The `5`-seed quick overlay summary was:

| Demo episodes | Expert success | BC-only success |
|---|---:|---:|
| `10` | `1.0` | `0.0` |
| `50` | `1.0` | `0.0` |
| `200` | `1.0` | `1.0` |

That matches the larger `50`-episode BC-only evaluation trend very well.

## What this adds to the earlier coverage analysis

Earlier start/goal coverage analysis suggested:

- `10 demos` were coverage-limited
- `200 demos` were probably not failing because of poor data coverage

Trajectory-level analysis strengthens that interpretation:

1. The expert is already high-quality in all three settings.
2. The expert path family is stable across demo counts.
3. More data mostly increases **density of examples**, not path-style diversity.
4. Therefore:
   - `10 demos` fail because the policy sees too few examples of the detour template across randomized conditions.
   - `200 demos` do not fail because of bad expert data; BC-only already solves the task there.

## Bottom line

Trajectory-level inspection makes the story more precise:

> The low-demo regime is best explained by sparse coverage of a stable expert behavior pattern, not by poor expert quality. The high-demo regime is not coverage-limited at all; by that point BC already solves the task, so weaker BC+PPO results are more likely due to PPO perturbing an already-strong BC policy.
