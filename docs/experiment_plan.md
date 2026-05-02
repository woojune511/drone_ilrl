# Drone Navigation IL->RL Experiment Plan

## Goal

Build a portfolio-ready experiment that supports the claim:

**Imitation learning pretraining improved sample efficiency for drone navigation reinforcement learning.**

The goal is not to perfectly reproduce the old AirSim + Unity + SPiRL project. The goal is to produce a clean, measurable, reproducible experiment with a strong engineering story:

- there was a concrete challenge
- a specific intervention was introduced
- a quantitative metric improved

## Why This Version

The original project had too many coupled difficulties at once:

- RGB-only observation with weak visual distinction between states
- human keyboard teleoperation with noisy, inconsistent actions
- continuous control with inertia and drift
- hierarchical latent-skill learning on top of already difficult RL tuning

That made it hard to tell whether failure came from perception, demonstrations, action noise, representation learning, or RL instability.

This version simplifies the setup so the main claim can be tested clearly:

1. First prove that IL helps RL in a controlled waypoint navigation environment.
2. Then add difficulty through ablations.
3. Only after that consider a more SPiRL-like extension.

## Current Repo Assets

The current repository already contains:

- a custom waypoint environment in `src/ilrl_lab/envs/waypoint_vel_aviary.py`
- an expert policy in `src/ilrl_lab/experts/velocity.py`
- expert rollout collection in `scripts/collect_expert_rollouts.py`
- behavior cloning training in `scripts/train_bc.py`
- BC evaluation in `scripts/evaluate_bc.py`

This means the missing piece is mainly the RL comparison layer.

## Core Experimental Claim

The main statement to support is:

**A policy initialized from imitation learning reaches useful navigation performance with fewer environment interactions than PPO trained from scratch.**

## Main Experimental Setup

### Environment

Use the existing state-based waypoint environment:

- observation: 18D state from `WaypointVelocityAviary`
- action: 4D velocity command
- reward: existing distance/speed-based reward
- success: existing goal tolerance + low-speed termination condition

### Methods to Compare

Compare these three methods:

1. **PPO from scratch**
2. **BC only**
3. **BC pretrained + PPO fine-tuning**

Interpretation:

- `PPO from scratch` is the baseline
- `BC only` shows how far imitation gets without RL
- `BC + PPO` tests whether IL improves downstream RL sample efficiency

## Main Metrics

Use the following metrics in the final report:

- `success_rate`
- `mean_final_distance_to_goal`
- `mean_episode_return`
- `mean_episode_length`
- `steps_to_70pct_success`
- `steps_to_80pct_success`
- `area_under_learning_curve`

Recommended primary headline metric:

**Environment steps required to reach 80% success rate**

This is easy to explain in a portfolio and directly supports the sample-efficiency claim.

## Evaluation Protocol

- train each RL method with **5 random seeds**
- evaluate every **10k environment steps**
- run **20 evaluation episodes** at each checkpoint
- total training budget: start with **200k to 300k environment steps**

For each evaluation point, record:

- success rate
- mean return
- mean final distance
- mean episode length

## Main Experiment

### Objective

Test whether BC initialization helps PPO learn faster than scratch PPO.

### Procedure

1. Collect expert rollouts
2. Train BC on the collected dataset
3. Train PPO from scratch
4. Initialize PPO from the BC policy
5. Fine-tune with RL
6. Compare learning curves and target-success sample efficiency

### Suggested Initial Settings

- expert dataset size: **50 episodes**
- BC epochs: **10 to 20**
- RL seeds: **5**
- RL budget: **200k or 300k steps**

### Expected Outcome

- `BC only` should perform reasonably well early, but may plateau
- `PPO from scratch` may eventually improve but should learn more slowly
- `BC + PPO` should achieve strong early performance and faster convergence

## Ablation 1: Demonstration Quantity

### Question

How much demonstration data is needed before IL meaningfully helps RL?

### Compare

- `10 demo episodes`
- `50 demo episodes`
- `200 demo episodes`

### Expected Story

- with too little demo data, BC initialization may be unstable
- with moderate or large demo data, BC warm-start should help PPO more consistently

## Ablation 2: Demonstration Quality

### Question

How sensitive is downstream learning to noisy demonstrations?

### Compare

- `clean scripted expert`
- `noisy expert`

### How to Create Noise

Possible noise mechanisms:

- add Gaussian noise to actions
- randomly repeat the previous action
- inject delay into control updates
- slightly perturb yaw or velocity commands

### Why This Matters

This ablation connects directly to the original project, where human keyboard teleoperation likely produced inconsistent action labels.

### Expected Story

- cleaner expert data should improve BC performance
- higher-quality demos should make PPO fine-tuning more sample-efficient

## Optional Ablation 3: Partial Observability

### Question

How does reduced observation quality affect IL and RL?

### Compare

- full state observation
- reduced observation
- optional stacked observations or proxy image observation

### Purpose

This creates a bridge toward the original vision-based project without jumping immediately into the hardest setup.

## Recommended Experimental Order

### Phase 1: Pilot

Run a short pilot to verify the pipeline:

1. collect `20 to 50` expert episodes
2. train BC briefly
3. run one short PPO-from-scratch training
4. run one short BC+PPO training
5. confirm that the curves are meaningfully different

### Phase 2: Main Result

Run the full comparison:

1. fixed dataset
2. 5 seeds
3. scratch PPO vs BC+PPO
4. full evaluation logs
5. final comparison plots and tables

### Phase 3: Ablations

Add:

- demo quantity
- demo quality

### Phase 4: Stretch Goal

If the main result is strong, explore:

- partial observability
- vision-based observation
- SPiRL-inspired latent skill extension

## Artifacts to Save

Save all outputs under `artifacts/` in a consistent structure.

Recommended outputs:

- datasets
- BC checkpoints
- PPO checkpoints
- evaluation summaries
- CSV/JSON logs
- learning curve figures
- final summary table

## Minimum Viable Portfolio Result

The minimum result worth shipping is:

1. collect expert demonstrations
2. train BC
3. compare `PPO from scratch` vs `BC + PPO`
4. show learning curves
5. report `steps to 80% success`

If this works, it is already enough for a resume or portfolio bullet.

## Suggested Portfolio Narrative

### Challenge

Pure reinforcement learning for drone navigation was sample-inefficient, and noisy demonstrations reduced the effectiveness of imitation-based initialization.

### Approach

Built a waypoint-navigation training pipeline using expert demonstrations, behavior cloning pretraining, and PPO fine-tuning in a PyBullet drone simulator.

### Result

Showed that imitation-initialized policies reached target navigation success rates in fewer environment steps than PPO trained from scratch.

## Example Resume Bullet Template

- Designed and evaluated a drone navigation learning pipeline combining behavior cloning and PPO fine-tuning; demonstrated improved sample efficiency over scratch RL by reducing the environment interactions needed to reach target success rates.

## Missing Implementation Tasks in This Repo

To execute this plan, the main additions needed are:

- `scripts/train_ppo.py`
- `scripts/fine_tune_ppo_from_bc.py`
- `scripts/evaluate_policy.py`
- a shared metric logging format
- a plotting script for learning curves

## Final Recommendation

Do not start by reproducing SPiRL exactly.

Start with the clearest measurable claim:

**BC pretraining improves RL sample efficiency in drone waypoint navigation.**

Once that result is stable and quantitative, extend toward harder observation spaces or more structured skill-learning methods.
