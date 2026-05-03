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
- `200 demos` looks more like a PPO fine-tuning problem on top of an already-strong BC policy

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

BC-only detour evaluation over `50` episodes:

| Demo episodes | Success | Mean final distance |
|---|---:|---:|
| `10` | `0.00` | `0.818` |
| `50` | `0.04` | `0.481` |
| `200` | `1.00` | `0.052` |

Stage-level failure analysis makes this even clearer:

- `10 demos`
  - `6/50` stuck before exit
  - `9/50` reached exit only
  - `35/50` reached goal stage but failed to settle
  - `0/50` success

- `50 demos`
  - `0/50` stuck before exit
  - `1/50` reached exit only
  - `47/50` reached goal stage but failed to settle
  - `2/50` success

- `200 demos`
  - `50/50` success

So the jump from `10` to `50` mainly improves **navigational coverage**, while the jump from `50` to `200` improves **closed-loop stability and final settling** enough that BC alone starts solving the task.

## What this means for the ablation result

The demo-quantity ablation gave this conservative BC+PPO best-checkpoint result:

| Demo episodes | BC+PPO success | Mean final distance |
|---|---:|---:|
| `10` | `0.11` | `0.578` |
| `50` | `0.27` | `0.406` |
| `200` | `0.17` | `0.439` |

At first glance, that can look strange because `200 demos` is not best.

Coverage analysis explains why this is **not** a single simple story.

### `10 demos`

This regime really does look coverage-limited.

- sparse start/goal support
- weak BC-only generalization
- BC+PPO can recover some behavior, but the prior is thin

### `50 demos`

This is the strongest **IL -> RL warm-start regime**.

- coverage is much better
- BC alone is still weak
- PPO still has something meaningful to improve

### `200 demos`

This regime is different.

- coverage is strong
- BC-only already solves the task
- therefore weaker BC+PPO performance is not well explained by bad demonstrations

The more likely explanation is that PPO fine-tuning is disturbing an already-good BC policy.

## Bottom line

If we separate the regimes, the answer becomes much cleaner:

1. **Low-demo weakness is plausibly a coverage problem.**
2. **High-demo underperformance of BC+PPO is not plausibly a coverage problem, because BC-only already succeeds perfectly there.**
3. The current task appears to have a “middle regime” where imitation prior is good enough to help PPO, but not so complete that PPO becomes unnecessary.

That makes the `50-demo` result more meaningful, not less:

> It is the point where coverage is sufficient to provide a useful prior, but RL still has meaningful work left to do.
