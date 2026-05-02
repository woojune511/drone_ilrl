# Drone IL -> RL Experiment Results

## Main takeaway

The updated main experiment compared:

1. `PPO from scratch`
2. `BC-pretrained actor + PPO fine-tuning`

on the current waypoint-reaching drone task, using a larger `300k`-step budget and `5` seeds.

With this longer horizon, the picture became much more balanced than the earlier `100k` run:

- `BC + PPO` still gives a clearly better **early learning prior**
- by `300k` steps, **both methods are very close**
- `PPO scratch` is slightly better on final success mean
- `BC + PPO` is slightly better on final distance mean and final return mean
- overall, this now supports a more nuanced claim:
  - imitation pretraining improves early behavior
  - longer RL fine-tuning reduces the final gap substantially

## Task

Environment: [src/ilrl_lab/envs/waypoint_vel_aviary.py](C:/Users/gjlee/Desktop/drone_ilrl/src/ilrl_lab/envs/waypoint_vel_aviary.py:24)

- single drone
- randomized start near origin
- randomized 3D goal inside bounded workspace
- `18D` state observation
- `4D` high-level velocity action
- success requires:
  - distance to goal `< 0.10m`
  - speed `< 0.15`

This is a fully observed waypoint-reaching control task, not image-based navigation.

## Expert and BC Setup

Expert:

- [src/ilrl_lab/experts/velocity.py](C:/Users/gjlee/Desktop/drone_ilrl/src/ilrl_lab/experts/velocity.py:6)
- PD-style velocity expert using relative goal and current velocity

Expert dataset:

- [artifacts/datasets/waypoint_expert_20260501_164808.npz](C:/Users/gjlee/Desktop/drone_ilrl/artifacts/datasets/waypoint_expert_20260501_164808.npz)
- [artifacts/datasets/waypoint_expert_20260501_164808_summary.json](C:/Users/gjlee/Desktop/drone_ilrl/artifacts/datasets/waypoint_expert_20260501_164808_summary.json)
- `50` episodes
- `10,557` transitions
- expert success rate: `0.84`

BC model:

- [src/ilrl_lab/bc.py](C:/Users/gjlee/Desktop/drone_ilrl/src/ilrl_lab/bc.py:10)
- MLP: `18 -> 256 -> 256 -> 4`
- checkpoint: [artifacts/checkpoints/bc/bc_20260501_164820/checkpoint.pt](C:/Users/gjlee/Desktop/drone_ilrl/artifacts/checkpoints/bc/bc_20260501_164820/checkpoint.pt)
- training metrics: [artifacts/checkpoints/bc/bc_20260501_164820/metrics.json](C:/Users/gjlee/Desktop/drone_ilrl/artifacts/checkpoints/bc/bc_20260501_164820/metrics.json)

Standalone BC eval:

- [artifacts/evals/bc_eval_20260501_164837.json](C:/Users/gjlee/Desktop/drone_ilrl/artifacts/evals/bc_eval_20260501_164837.json)
- success rate: `0.50`
- mean final distance: `0.130`

## PPO Experiment Design

Shared PPO utilities:

- [src/ilrl_lab/ppo_training.py](C:/Users/gjlee/Desktop/drone_ilrl/src/ilrl_lab/ppo_training.py:1)

Training scripts:

- scratch PPO: [scripts/train_ppo.py](C:/Users/gjlee/Desktop/drone_ilrl/scripts/train_ppo.py:1)
- BC-init PPO: [scripts/fine_tune_ppo_from_bc.py](C:/Users/gjlee/Desktop/drone_ilrl/scripts/fine_tune_ppo_from_bc.py:1)

BC -> PPO initialization:

- BC hidden layers copied into PPO actor
- BC output layer copied into PPO action mean head
- critic and action std left PPO-initialized

Protocol:

- seeds: `7, 11, 19, 23, 29`
- budget: `300,000` env steps
- eval every `10,000` steps
- `20` eval episodes per checkpoint

## Main 300k Results

Summary and plots:

- summary JSON: [artifacts/figures/ppo_vs_bc_init_300k/experiment_summary.json](C:/Users/gjlee/Desktop/drone_ilrl/artifacts/figures/ppo_vs_bc_init_300k/experiment_summary.json)
- success curve: [artifacts/figures/ppo_vs_bc_init_300k/success_rate_vs_steps.png](C:/Users/gjlee/Desktop/drone_ilrl/artifacts/figures/ppo_vs_bc_init_300k/success_rate_vs_steps.png)
- return curve: [artifacts/figures/ppo_vs_bc_init_300k/return_vs_steps.png](C:/Users/gjlee/Desktop/drone_ilrl/artifacts/figures/ppo_vs_bc_init_300k/return_vs_steps.png)
- distance curve: [artifacts/figures/ppo_vs_bc_init_300k/final_distance_vs_steps.png](C:/Users/gjlee/Desktop/drone_ilrl/artifacts/figures/ppo_vs_bc_init_300k/final_distance_vs_steps.png)
- trajectory comparison: [artifacts/figures/ppo_vs_bc_init_300k/trajectory_comparison.png](C:/Users/gjlee/Desktop/drone_ilrl/artifacts/figures/ppo_vs_bc_init_300k/trajectory_comparison.png)

### Final metrics at 300k

| Method | Final success mean | Final distance mean | Final return mean | Success AUC |
|---|---:|---:|---:|---:|
| PPO scratch | `0.300` | `0.202` | `96.32` | `47050` |
| BC + PPO | `0.280` | `0.211` | `97.82` | `46900` |

### Early phase at 10k

| Method | Success | Final distance | Return |
|---|---:|---:|---:|
| PPO scratch | `0.000` | `0.754` | `30.22` |
| BC + PPO | `0.020` | `0.487` | `54.53` |

### Mid-training around 100k

| Method | Success | Final distance | Return |
|---|---:|---:|---:|
| PPO scratch | `0.140` | `0.479` | `75.60` |
| BC + PPO | `0.080` | `0.447` | `81.74` |

### End of training at 300k

| Method | Success | Final distance | Return |
|---|---:|---:|---:|
| PPO scratch | `0.270` | `0.197` | `96.98` |
| BC + PPO | `0.220` | `0.223` | `101.93` |

## Seed-level final results

### PPO scratch

| Seed | Success | Final distance | Return |
|---|---:|---:|---:|
| `7` | `0.10` | `0.286` | `102.70` |
| `11` | `0.15` | `0.199` | `105.77` |
| `19` | `0.45` | `0.196` | `83.42` |
| `23` | `0.15` | `0.207` | `113.82` |
| `29` | `0.65` | `0.123` | `75.88` |

### BC + PPO

| Seed | Success | Final distance | Return |
|---|---:|---:|---:|
| `7` | `0.30` | `0.247` | `87.43` |
| `11` | `0.45` | `0.191` | `95.72` |
| `19` | `0.20` | `0.198` | `102.66` |
| `23` | `0.25` | `0.216` | `98.99` |
| `29` | `0.20` | `0.213` | `106.72` |

## Interpretation

There are three important patterns here.

1. **BC initialization helps a lot at the beginning.**  
   At `10k` steps, `BC + PPO` has much better return and distance-to-goal than scratch PPO.

2. **Scratch PPO catches up strongly with enough horizon.**  
   By `300k`, the gap is mostly gone, and scratch slightly edges out BC-init on final success mean.

3. **BC + PPO looks more stable on some metrics, but not decisively better.**  
   BC-init has slightly higher final return and more consistent final distance spread, but it does not win clearly on the main terminal success metric.

So the honest takeaway is not:

> BC initialization always wins

but rather:

> BC initialization improves early control quality and short-horizon efficiency, while longer PPO fine-tuning allows scratch PPO to largely close the gap on this task.

## Preliminary 100k result

For context, the earlier `100k / 3 seeds` run had made BC-init look weaker on final success.  
That result is still useful, but it now reads as a **too-short horizon** snapshot rather than the main conclusion.

The longer `300k` run is the better portfolio result.

## Portfolio framing

A good, truthful project description would be:

> Built an imitation-learning-to-reinforcement-learning pipeline for goal-conditioned drone waypoint control. Behavior cloning provided a strong warm start for PPO, improving early reward and distance-to-goal metrics, while longer-horizon RL fine-tuning closed most of the performance gap on the final success metric across 5 random seeds.

If you want a stronger one-line resume bullet later, the next best move is to add:

1. a harder task variant, or
2. noisier demonstrations, or
3. partial observability

because those are the settings where an imitation prior is more likely to separate clearly from scratch RL.
