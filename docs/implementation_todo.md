# Drone Navigation IL->RL Implementation TODO

## Purpose

This document turns the high-level experiment plan into an execution checklist for this repository.

Primary goal:

**Show that BC pretraining improves RL sample efficiency over PPO from scratch in the waypoint drone environment.**

## Success Criteria

The minimum shippable result is complete when all of the following are true:

- expert demonstrations can be collected reproducibly
- a BC policy can be trained and evaluated
- a PPO baseline can be trained from scratch
- a BC-initialized PPO policy can be fine-tuned
- both RL methods are evaluated on the same protocol
- learning curves are plotted
- a final table reports sample-efficiency metrics

## Phase 0: Baseline Validation

### Goal

Confirm the current repo is stable enough to build on.

### Tasks

- [x] Create `.venv`
- [x] Install dependencies
- [x] Run `scripts/check_env.py`
- [x] Run `pytest`

### Notes

Environment setup has already been completed.

## Phase 1: Expert Data Pipeline

### Goal

Produce reproducible demonstration datasets for BC training.

### Existing Files

- `scripts/collect_expert_rollouts.py`
- `src/ilrl_lab/envs/waypoint_vel_aviary.py`
- `src/ilrl_lab/experts/velocity.py`

### Tasks

- [ ] Run a pilot collection with `20` episodes
- [ ] Inspect generated dataset summary JSON
- [ ] Run a medium collection with `50` episodes
- [ ] Optionally run a larger collection with `200` episodes for ablations
- [ ] Record dataset file names in experiment notes

### Deliverables

- `artifacts/datasets/waypoint_expert_*.npz`
- `artifacts/datasets/waypoint_expert_*_summary.json`

### Acceptance Criteria

- dataset saves without errors
- summary includes transitions, success rate, mean return, and mean final distance
- at least one medium-sized dataset is ready for BC

### Suggested Commands

```powershell
. .\.venv\Scripts\Activate.ps1
python scripts\collect_expert_rollouts.py --episodes 20
python scripts\collect_expert_rollouts.py --episodes 50
```

## Phase 2: Behavior Cloning Baseline

### Goal

Train a BC policy and establish its standalone performance.

### Existing Files

- `scripts/train_bc.py`
- `scripts/evaluate_bc.py`
- `src/ilrl_lab/bc.py`

### Tasks

- [ ] Train BC on the pilot or main dataset
- [ ] Save the best checkpoint
- [ ] Evaluate the trained BC policy
- [ ] Record BC success rate and final distance

### Deliverables

- `artifacts/checkpoints/bc/.../checkpoint.pt`
- `artifacts/checkpoints/bc/.../metrics.json`
- `artifacts/evals/bc_eval_*.json`

### Acceptance Criteria

- BC training runs to completion
- BC evaluation runs to completion
- BC policy performs above trivial/random behavior

### Suggested Commands

```powershell
. .\.venv\Scripts\Activate.ps1
python scripts\train_bc.py --epochs 15
python scripts\evaluate_bc.py --episodes 20
```

## Phase 3: PPO From Scratch

### Goal

Create the RL baseline used for comparison.

### New File To Add

- `scripts/train_ppo.py`

### Responsibilities of `train_ppo.py`

- create the waypoint environment
- train PPO from scratch
- evaluate periodically during training
- save checkpoints
- save structured logs to JSON or CSV

### Tasks

- [ ] Implement CLI arguments for training budget, seed, eval frequency, and output directory
- [ ] Instantiate `stable-baselines3` PPO with reproducible seeds
- [ ] Add evaluation callback or periodic manual evaluation
- [ ] Save training curve data
- [ ] Save final checkpoint
- [ ] Run one short pilot training
- [ ] Run the full baseline for 5 seeds

### Deliverables

- `artifacts/checkpoints/ppo_scratch/...`
- `artifacts/logs/ppo_scratch/...`
- `artifacts/evals/ppo_scratch/...`

### Acceptance Criteria

- scratch PPO training completes without runtime errors
- evaluation runs at regular intervals
- metrics are saved in machine-readable form

## Phase 4: BC-Initialized PPO Fine-Tuning

### Goal

Test whether imitation-based initialization improves PPO sample efficiency.

### New File To Add

- `scripts/fine_tune_ppo_from_bc.py`

### Responsibilities of `fine_tune_ppo_from_bc.py`

- load BC checkpoint
- initialize PPO policy weights from BC where compatible
- train PPO with the same evaluation protocol as scratch PPO
- save checkpoints and logs in a separate output path

### Tasks

- [ ] Define how BC weights map into PPO policy layers
- [ ] Implement BC checkpoint loading
- [ ] Copy compatible layers into PPO policy network
- [ ] Log whether initialization succeeded and which layers were copied
- [ ] Run one short pilot
- [ ] Run the full experiment for 5 seeds

### Deliverables

- `artifacts/checkpoints/ppo_bc_init/...`
- `artifacts/logs/ppo_bc_init/...`
- `artifacts/evals/ppo_bc_init/...`

### Acceptance Criteria

- BC weights load correctly
- PPO training starts from the BC initialization
- logs clearly distinguish scratch vs BC-init runs

## Phase 5: Shared Evaluation Utilities

### Goal

Ensure fair comparisons between methods.

### New File To Add

- `scripts/evaluate_policy.py`

### Responsibilities

- evaluate PPO checkpoints
- optionally evaluate BC checkpoints through a common interface
- run a fixed number of episodes
- save success rate, mean return, episode length, and final distance

### Tasks

- [ ] Implement a common evaluation loop
- [ ] Support checkpoint path input
- [ ] Save JSON summaries
- [ ] Reuse the same evaluation settings across all methods

### Acceptance Criteria

- both scratch PPO and BC-init PPO can be evaluated with the same script
- output format is identical across methods

## Phase 6: Logging and Experiment Schema

### Goal

Make every run reproducible and easy to compare.

### Recommended Schema

For every training run, save:

- method name
- seed
- dataset path if applicable
- total environment steps
- eval step
- success rate
- mean return
- mean episode length
- mean final distance
- checkpoint path

### Tasks

- [ ] Define a run directory layout under `artifacts/`
- [ ] Standardize JSON fields
- [ ] Save config metadata with each run
- [ ] Make sure scratch and BC-init formats match

### Acceptance Criteria

- all experiments can be aggregated without manual cleanup

## Phase 7: Plotting and Comparison

### Goal

Turn raw logs into portfolio-ready evidence.

### New File To Add

- `scripts/plot_learning_curves.py`

### Plots to Generate

- success rate vs environment steps
- mean final distance vs environment steps
- optional mean return vs environment steps

### Final Table Columns

- `method`
- `mean_success_rate`
- `mean_final_distance`
- `steps_to_70pct_success`
- `steps_to_80pct_success`
- `auc_success_curve`

### Tasks

- [ ] Load all seed logs
- [ ] Aggregate across seeds
- [ ] Plot mean curve
- [ ] Add variance band or min/max band
- [ ] Export PNG figures
- [ ] Export summary CSV or JSON

### Acceptance Criteria

- at least one plot clearly shows scratch PPO vs BC-init PPO
- final metrics table can be quoted in a portfolio or resume

## Phase 8: Main Experiment Run Order

### Step 1: Pilot

- [ ] collect `20 to 50` demos
- [ ] train BC quickly
- [ ] run one short scratch PPO training
- [ ] run one short BC-init PPO training
- [ ] verify the pipeline end to end

### Step 2: Main Comparison

- [ ] lock dataset choice
- [ ] run scratch PPO for `5` seeds
- [ ] run BC-init PPO for `5` seeds
- [ ] evaluate all runs
- [ ] plot results

### Step 3: BC Standalone Context

- [ ] report BC-only evaluation as a reference point

### Step 4: Final Summary

- [ ] write short conclusion with numbers
- [ ] draft resume bullet
- [ ] draft portfolio challenge/solution/result paragraph

## Phase 9: Ablations

Only start these after the main comparison is working.

### Ablation A: Demo Quantity

- [ ] `10` demo episodes
- [ ] `50` demo episodes
- [ ] `200` demo episodes

Question:

How much demonstration data is needed before BC initialization reliably helps RL?

### Ablation B: Demo Quality

- [ ] implement noisy demonstration collection or noisy action replay
- [ ] compare clean vs noisy demos

Question:

How sensitive is downstream RL improvement to demonstration quality?

### Ablation C: Reduced Observation Quality

- [ ] define reduced-observation environment variant
- [ ] compare full-state vs reduced-state settings

Question:

Does IL help more when the control problem becomes harder or more partially observed?

## Suggested File Creation Order

Implement in this order:

1. `scripts/train_ppo.py`
2. `scripts/evaluate_policy.py`
3. `scripts/fine_tune_ppo_from_bc.py`
4. `scripts/plot_learning_curves.py`

Reason:

- first establish the scratch RL baseline
- then make evaluation reusable
- then add BC initialization
- then produce figures

## Immediate Next Actions

If continuing right away, the best next sequence is:

1. run expert data collection
2. train and evaluate BC
3. implement `scripts/train_ppo.py`
4. run a short PPO pilot
5. implement BC-to-PPO initialization

## Portfolio Output Checklist

Before considering the project complete, make sure the following exist:

- [ ] one clear learning-curve figure
- [ ] one clear comparison table
- [ ] one sentence stating the improvement numerically
- [ ] one short explanation of the challenge
- [ ] one short explanation of the intervention

## Example Final Result Template

Use a result sentence like this once the numbers are available:

`BC-initialized PPO reached 80% waypoint success in X% fewer environment steps than scratch PPO, demonstrating improved sample efficiency in drone navigation training.`
