# Data Coverage Analysis

This note answers a practical question from the demo-quantity ablation:

> Were the results mainly caused by poor expert data quality, or by insufficient state / goal coverage?

## Artifacts

- script: `scripts/analyze_demo_quantity_coverage.py`
- summary JSON: `artifacts/figures/demo_quantity_coverage/coverage_summary.json`
- figure: `artifacts/figures/demo_quantity_coverage/coverage_overview.png`

## Headline answer

The short answer is:

- **yes**, low-demo performance is strongly consistent with **coverage limitations**
- **no**, the high-demo regime is **not** well explained by poor data quality or poor coverage

In other words:

- `10 demos` looks like a coverage problem
- `200 demos` looks more like a regime where BC is already strong enough that PPO has less room to help

## Key observations

### 1. Expert quality was already high at every demo count

All three detour expert datasets had:

- expert success rate: `1.0`
- expert mean final distance around `0.07`

So the expert itself was not low-quality in the obvious sense.

### 2. Coverage improves substantially from `10 -> 50 -> 200`

Using the same evaluation start/goal distribution as the BC-only evaluation set, nearest-neighbor coverage in normalized start-goal space was:

| Demo episodes | Mean NN distance | P90 NN distance | Max NN distance |
|---|---:|---:|---:|
| `10` | `2.26` | `2.89` | `3.42` |
| `50` | `1.62` | `2.11` | `2.27` |
| `200` | `1.23` | `1.61` | `1.66` |

This is the clearest sign that `10 demos` simply do not cover the randomized detour distribution very well.

### 3. BC-only failure modes change with data quantity

BC-only detour evaluation over `50` episodes after the transfer-fix rerun:

| Demo episodes | Success | Mean final distance |
|---|---:|---:|
| `10` | `0.02` | `0.747` |
| `50` | `0.28` | `0.329` |
| `200` | `0.94` | `0.075` |

Stage-level failure analysis from the earlier rollout inspection still helps explain the trend:

- `10 demos`
  - `6/50` stuck before exit
  - `9/50` reached exit only
  - `35/50` reached goal stage but failed to settle
  - `0/50` success in the original pre-fix BC-only check

- `50 demos`
  - `0/50` stuck before exit
  - `1/50` reached exit only
  - many rollouts reached the goal stage but failed to settle consistently

- `200 demos`
  - BC-only becomes close to fully reliable

So the jump from `10` to `50` mainly improves **navigational coverage**, while the jump from `50` to `200` improves **closed-loop stability and final settling** enough that BC alone starts solving the task.

## What this means for the ablation result

The aligned demo-quantity ablation gave this BC+PPO final `30`-episode mean:

| Demo episodes | BC+PPO success | Mean final distance |
|---|---:|---:|
| `10` | `0.00` | `0.704` |
| `50` | `0.111` | `0.476` |
| `200` | `0.10` | `0.426` |

Coverage analysis explains why this is **not** a single simple story.

### `10 demos`

This regime really does look coverage-limited.

- sparse start/goal support
- weak BC-only generalization
- BC+PPO can recover some behavior, but the prior is thin

### `50 demos`

This is the strongest **IL -> RL warm-start regime**.

- coverage is much better
- BC alone is useful but not dominant
- PPO still has something meaningful to improve

### `200 demos`

This regime is different.

- coverage is strong
- BC-only is already close to solving the task
- therefore the remaining BC+PPO gap is not well explained by bad demonstrations

The more likely explanation is that PPO fine-tuning has less headroom to improve an already-strong BC policy.

## Bottom line

If we separate the regimes, the answer becomes much cleaner:

1. **Low-demo weakness is plausibly a coverage problem.**
2. **High-demo behavior is not plausibly a coverage problem, because BC-only is already very strong there.**
3. The current task appears to have a "middle regime" where imitation prior is good enough to help PPO, but not so complete that PPO becomes unnecessary.

That makes the `50-demo` result more meaningful, not less:

> It is the point where coverage is sufficient to provide a useful prior, but RL still has meaningful work left to do.
