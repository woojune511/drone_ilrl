# Submission File Manifest

This project is intended to present a compact IL -> RL pipeline for a detour navigation task. The files below are the recommended submission set.

## Core Narrative

- `README.md`: entry point, headline result, reproduction commands.
- `docs/portfolio_one_page.md`: one-page interview/portfolio summary.
- `docs/portfolio_summary.md`: expanded result summary and framing.
- `docs/interview_qa.md`: prepared answers for likely technical interview questions.
- `docs/action_space_roadmap.md`: next-step plan for a deployment-oriented drone action interface.
- `docs/detour_planar_bc_baseline.md`: first clean BC baseline for the planar action variant.
- `docs/bc_to_ppo_distribution_shift.md`: technical diagnosis of BC -> PPO distribution shift and mitigation attempts.

## Core Implementation

- `src/ilrl_lab/envs/detour_vel_aviary.py`: detour wall environment.
- `src/ilrl_lab/envs/detour_planar_vel_aviary.py`: follow-up detour variant with body-frame planar velocity, yaw-rate, and altitude hold.
- `src/ilrl_lab/envs/waypoint_vel_aviary.py`: velocity waypoint base environment.
- `src/ilrl_lab/experts/velocity.py`: scripted velocity experts.
- `src/ilrl_lab/ppo_training.py`: PPO, BC initialization, BC-KL regularization, and expert-state BC loss.
- `scripts/train_bc.py`: behavior cloning training.
- `scripts/fine_tune_ppo_from_bc.py`: BC-initialized PPO fine-tuning with expert-state augmentation.
- `scripts/evaluate_policy.py`: policy evaluation.
- `scripts/plot_training_diagnostics.py`: training diagnostic plots.
- `scripts/build_portfolio_results.py`: final portfolio result tables and figures.
- `scripts/analyze_policy_rollouts.py`: rollout-level failure analysis.
- `scripts/run_detour_expert_bc_aug_ablation.sh`: final expert-state augmentation ablation command script.

## Final Result Artifacts

- `artifacts/analysis/portfolio_final_20260609/headline_50ep_comparison.csv`
- `artifacts/analysis/portfolio_final_20260609/headline_50ep_success_rate.png`
- `artifacts/analysis/portfolio_final_20260609/headline_50ep_mean_final_distance.png`
- `artifacts/analysis/portfolio_final_20260609/headline_50ep_mean_episode_return.png`
- `artifacts/analysis/portfolio_final_20260609/online_diagnostics_comparison.csv`
- `artifacts/analysis/portfolio_final_20260609/online_diagnostics_success_rate.png`
- `artifacts/analysis/portfolio_final_20260609/online_diagnostics_mean_final_distance.png`
- `artifacts/analysis/portfolio_final_20260609/online_diagnostics_collision_rate.png`
- `artifacts/analysis/portfolio_final_20260609/online_diagnostics_bc_probe_action_l2.png`
- `artifacts/analysis/portfolio_final_20260609/manifest.json`
- `artifacts/analysis/aug005_seed31_final_rollouts_20260609/summary.json`
- `artifacts/analysis/aug005_seed31_final_rollouts_20260609/episode_rollouts.csv`
- `artifacts/analysis/aug005_seed31_final_rollouts_20260609/failure_trajectories.png`
- `artifacts/analysis/aug005_seed31_final_rollouts_20260609/failure_distributions.png`

## Exclude From Submission

- `artifacts/checkpoints/`: large model checkpoints and TensorBoard logs.
- `artifacts/evals/`: intermediate per-run evaluation JSON files.
- `artifacts/datasets/`: generated expert datasets unless the reviewer explicitly asks for raw data.
- `artifacts/analysis/aug005_seed31_final_rollouts_20260609/episode_rollouts.json`: redundant full rollout JSON.
- `.venv/`, `.uv-cache/`, `.ruff_cache/`, `.pytest_cache/`, `__pycache__/`: local environment/cache files.

## Headline Numbers

- BC-only: 54.0% success over 50 evaluation episodes.
- BC + PPO expert-state augmentation, final checkpoint mean over 5 seeds: 72.8% success.
- BC + PPO expert-state augmentation, validation-best checkpoint mean over 5 seeds: 79.2% success.
