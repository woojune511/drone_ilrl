from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

from ilrl_lab.bc import load_bc_checkpoint, predict_action
from ilrl_lab.envs import DetourWaypointVelocityAviary, WaypointVelocityAviary
from ilrl_lab.experts import detour_waypoint_velocity_expert, waypoint_velocity_expert
from ilrl_lab.ppo_training import FixedObservationNormalization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PPO scratch vs BC+PPO experiment results.")
    parser.add_argument("--scratch-dir", type=Path, required=True, help="Root directory for scratch PPO runs.")
    parser.add_argument("--bc-init-dir", type=Path, required=True, help="Root directory for BC-init PPO runs.")
    parser.add_argument("--bc-checkpoint", type=Path, required=True, help="BC checkpoint for standalone rollout visualization.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "figures",
        help="Directory where figures and summaries are saved.",
    )
    parser.add_argument("--trajectory-seed", type=int, default=123)
    parser.add_argument(
        "--task-variant",
        choices=["waypoint", "detour"],
        default="waypoint",
        help="Environment/expert variant used for trajectory rollouts.",
    )
    parser.add_argument(
        "--total-timesteps-filter",
        type=int,
        default=None,
        help="Only include runs whose summary.json matches this total timestep count.",
    )
    parser.add_argument(
        "--dedupe-by-seed",
        action="store_true",
        help="Keep only the latest run for each seed when aggregating results.",
    )
    return parser.parse_args()


def make_env(task_variant: str):
    if task_variant == "detour":
        return DetourWaypointVelocityAviary(gui=False)
    if task_variant == "waypoint":
        return WaypointVelocityAviary(gui=False)
    raise ValueError(f"Unsupported task variant: {task_variant}")


def expert_policy(task_variant: str):
    if task_variant == "detour":
        return detour_waypoint_velocity_expert
    if task_variant == "waypoint":
        return waypoint_velocity_expert
    raise ValueError(f"Unsupported task variant: {task_variant}")


def discover_runs(root: Path, total_timesteps_filter: int | None, dedupe_by_seed: bool) -> list[dict]:
    runs: list[dict] = []
    deduped_runs: dict[int, dict] = {}
    for summary_path in sorted(root.glob("**/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if total_timesteps_filter is not None and payload.get("total_timesteps") != total_timesteps_filter:
            continue
        payload["_summary_path"] = str(summary_path)
        if dedupe_by_seed:
            deduped_runs[int(payload["seed"])] = payload
        else:
            runs.append(payload)
    if dedupe_by_seed:
        runs = [deduped_runs[seed] for seed in sorted(deduped_runs)]
    if not runs:
        raise FileNotFoundError(f"No summary.json files found under {root}")
    return runs


def load_eval_history(run_summary: dict) -> list[dict]:
    history_path = Path(run_summary["eval_history_path"])
    return json.loads(history_path.read_text(encoding="utf-8"))


def aggregate_histories(histories: list[list[dict]]) -> dict[int, dict[str, float]]:
    timesteps = sorted({record["timesteps"] for history in histories for record in history})
    aggregated: dict[int, dict[str, float]] = {}
    for step in timesteps:
        records = [record for history in histories for record in history if record["timesteps"] == step]
        aggregated[step] = {
            "success_mean": float(np.mean([record["success_rate"] for record in records])),
            "success_std": float(np.std([record["success_rate"] for record in records])),
            "return_mean": float(np.mean([record["mean_episode_return"] for record in records])),
            "return_std": float(np.std([record["mean_episode_return"] for record in records])),
            "distance_mean": float(np.mean([record["mean_final_distance"] for record in records])),
            "distance_std": float(np.std([record["mean_final_distance"] for record in records])),
        }
    return aggregated


def first_threshold_step(history: list[dict], threshold: float) -> float | None:
    for record in history:
        if record["success_rate"] >= threshold:
            return float(record["timesteps"])
    return None


def summarize_method(name: str, runs: list[dict]) -> dict:
    histories = [load_eval_history(run) for run in runs]
    aggregated = aggregate_histories(histories)
    final_successes = [run["final_eval"]["success_rate"] for run in runs]
    final_distances = [run["final_eval"]["mean_final_distance"] for run in runs]
    final_returns = [run["final_eval"]["mean_episode_return"] for run in runs]
    aucs = []
    threshold_70 = []
    threshold_80 = []
    for history in histories:
        xs = np.asarray([record["timesteps"] for record in history], dtype=np.float32)
        ys = np.asarray([record["success_rate"] for record in history], dtype=np.float32)
        aucs.append(float(np.trapz(ys, xs)))
        step_70 = first_threshold_step(history, 0.7)
        step_80 = first_threshold_step(history, 0.8)
        if step_70 is not None:
            threshold_70.append(step_70)
        if step_80 is not None:
            threshold_80.append(step_80)

    return {
        "name": name,
        "num_runs": len(runs),
        "aggregated_history": aggregated,
        "final_success_mean": float(np.mean(final_successes)),
        "final_success_std": float(np.std(final_successes)),
        "final_distance_mean": float(np.mean(final_distances)),
        "final_distance_std": float(np.std(final_distances)),
        "final_return_mean": float(np.mean(final_returns)),
        "final_return_std": float(np.std(final_returns)),
        "auc_success_mean": float(np.mean(aucs)),
        "auc_success_std": float(np.std(aucs)),
        "steps_to_70_success_mean": float(np.mean(threshold_70)) if threshold_70 else None,
        "steps_to_80_success_mean": float(np.mean(threshold_80)) if threshold_80 else None,
        "run_dirs": [run["run_dir"] for run in runs],
    }


def plot_metric(output_path: Path, scratch: dict, bc_init: dict, key_mean: str, key_std: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 5))
    for label, method, color in (
        ("PPO scratch", scratch, "#1f77b4"),
        ("BC + PPO", bc_init, "#d62728"),
    ):
        steps = np.array(sorted(method["aggregated_history"].keys()), dtype=np.float32)
        means = np.array([method["aggregated_history"][int(step)][key_mean] for step in steps], dtype=np.float32)
        stds = np.array([method["aggregated_history"][int(step)][key_std] for step in steps], dtype=np.float32)
        plt.plot(steps, means, label=label, color=color)
        plt.fill_between(steps, means - stds, means + stds, alpha=0.2, color=color)
    plt.xlabel("Environment Steps")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def rollout_expert(seed: int, task_variant: str) -> dict:
    env = make_env(task_variant)
    policy = expert_policy(task_variant)
    obs, info = env.reset(seed=seed)
    goal = np.asarray(info["goal"], dtype=np.float32)
    positions = [np.asarray(info["position"], dtype=np.float32)]
    episode_return = 0.0
    steps = 0
    while True:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        positions.append(np.asarray(info["position"], dtype=np.float32))
        episode_return += float(reward)
        steps += 1
        if terminated or truncated:
            env.close()
            return {
                "label": "Expert",
                "goal": goal.astype(float).tolist(),
                "positions": np.stack(positions).astype(float).tolist(),
                "success": bool(info["success"]),
                "return": float(episode_return),
                "steps": int(steps),
            }


def rollout_bc(checkpoint: Path, seed: int, task_variant: str) -> dict:
    device = torch.device("cpu")
    model, obs_mean, obs_std, _ = load_bc_checkpoint(checkpoint, device)
    env = make_env(task_variant)
    obs, info = env.reset(seed=seed)
    goal = np.asarray(info["goal"], dtype=np.float32)
    positions = [np.asarray(info["position"], dtype=np.float32)]
    episode_return = 0.0
    steps = 0
    while True:
        action = predict_action(model, obs, obs_mean, obs_std, device)
        obs, reward, terminated, truncated, info = env.step(action)
        positions.append(np.asarray(info["position"], dtype=np.float32))
        episode_return += float(reward)
        steps += 1
        if terminated or truncated:
            env.close()
            return {
                "label": "BC",
                "goal": goal.astype(float).tolist(),
                "positions": np.stack(positions).astype(float).tolist(),
                "success": bool(info["success"]),
                "return": float(episode_return),
                "steps": int(steps),
            }


def rollout_ppo(model_path: Path, seed: int, label: str, task_variant: str) -> dict:
    env = make_env(task_variant)
    model = PPO.load(model_path)
    obs, info = env.reset(seed=seed)
    goal = np.asarray(info["goal"], dtype=np.float32)
    positions = [np.asarray(info["position"], dtype=np.float32)]
    episode_return = 0.0
    steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        positions.append(np.asarray(info["position"], dtype=np.float32))
        episode_return += float(reward)
        steps += 1
        if terminated or truncated:
            env.close()
            return {
                "label": label,
                "goal": goal.astype(float).tolist(),
                "positions": np.stack(positions).astype(float).tolist(),
                "success": bool(info["success"]),
                "return": float(episode_return),
                "steps": int(steps),
            }


def rollout_ppo_with_obs_norm(
    model_path: Path,
    seed: int,
    label: str,
    task_variant: str,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
) -> dict:
    env = make_env(task_variant)
    env = FixedObservationNormalization(env, obs_mean=obs_mean, obs_std=obs_std)
    model = PPO.load(model_path)
    obs, info = env.reset(seed=seed)
    goal = np.asarray(info["goal"], dtype=np.float32)
    positions = [np.asarray(info["position"], dtype=np.float32)]
    episode_return = 0.0
    steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        positions.append(np.asarray(info["position"], dtype=np.float32))
        episode_return += float(reward)
        steps += 1
        if terminated or truncated:
            env.close()
            return {
                "label": label,
                "goal": goal.astype(float).tolist(),
                "positions": np.stack(positions).astype(float).tolist(),
                "success": bool(info["success"]),
                "return": float(episode_return),
                "steps": int(steps),
            }


def obs_norm_from_run_summary(run_summary: dict, fallback_bc_checkpoint: Path | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    if run_summary.get("uses_bc_obs_normalization"):
        init_info = run_summary.get("initialization", {})
        obs_mean_json = init_info.get("obs_mean")
        obs_std_json = init_info.get("obs_std")
        if obs_mean_json is not None and obs_std_json is not None:
            return (
                np.asarray(json.loads(obs_mean_json), dtype=np.float32),
                np.asarray(json.loads(obs_std_json), dtype=np.float32),
            )
    checkpoint_path = run_summary.get("obs_norm_bc_checkpoint")
    if checkpoint_path is None and fallback_bc_checkpoint is not None:
        checkpoint_path = str(fallback_bc_checkpoint)
    if checkpoint_path is None:
        return None
    _, obs_mean, obs_std, _ = load_bc_checkpoint(Path(checkpoint_path), torch.device("cpu"))
    return obs_mean, obs_std


def plot_trajectories(output_path: Path, rollouts: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    projections = [
        (axes[0], 0, 1, "X", "Y", "XY Trajectory"),
        (axes[1], 0, 2, "X", "Z", "XZ Trajectory"),
    ]
    colors = {
        "Expert": "#2ca02c",
        "BC": "#9467bd",
        "PPO scratch": "#1f77b4",
        "BC + PPO": "#d62728",
    }
    for ax, ix, iy, xlabel, ylabel, title in projections:
        for rollout in rollouts:
            positions = np.asarray(rollout["positions"], dtype=np.float32)
            goal = np.asarray(rollout["goal"], dtype=np.float32)
            color = colors.get(rollout["label"], None)
            ax.plot(positions[:, ix], positions[:, iy], label=rollout["label"], color=color)
            ax.scatter(positions[0, ix], positions[0, iy], marker="o", color=color, s=20)
            ax.scatter(goal[ix], goal[iy], marker="x", color=color, s=45)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scratch_runs = discover_runs(args.scratch_dir, args.total_timesteps_filter, args.dedupe_by_seed)
    bc_init_runs = discover_runs(args.bc_init_dir, args.total_timesteps_filter, args.dedupe_by_seed)
    scratch_summary = summarize_method("ppo_scratch", scratch_runs)
    bc_init_summary = summarize_method("ppo_bc_init", bc_init_runs)

    summary_payload = {
        "scratch": scratch_summary,
        "bc_init": bc_init_summary,
    }
    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    plot_metric(
        args.output_dir / "success_rate_vs_steps.png",
        scratch_summary,
        bc_init_summary,
        key_mean="success_mean",
        key_std="success_std",
        ylabel="Success Rate",
    )
    plot_metric(
        args.output_dir / "return_vs_steps.png",
        scratch_summary,
        bc_init_summary,
        key_mean="return_mean",
        key_std="return_std",
        ylabel="Mean Episode Return",
    )
    plot_metric(
        args.output_dir / "final_distance_vs_steps.png",
        scratch_summary,
        bc_init_summary,
        key_mean="distance_mean",
        key_std="distance_std",
        ylabel="Mean Final Distance",
    )

    scratch_best = Path(scratch_runs[0]["best_model_path"])
    bc_init_best = Path(bc_init_runs[0]["best_model_path"])
    scratch_obs_norm = obs_norm_from_run_summary(scratch_runs[0], fallback_bc_checkpoint=args.bc_checkpoint)
    bc_init_obs_norm = obs_norm_from_run_summary(bc_init_runs[0], fallback_bc_checkpoint=args.bc_checkpoint)
    rollouts = [
        rollout_expert(args.trajectory_seed, args.task_variant),
        rollout_bc(args.bc_checkpoint, args.trajectory_seed, args.task_variant),
        rollout_ppo_with_obs_norm(
            scratch_best,
            args.trajectory_seed,
            "PPO scratch",
            args.task_variant,
            obs_mean=scratch_obs_norm[0],
            obs_std=scratch_obs_norm[1],
        )
        if scratch_obs_norm is not None
        else rollout_ppo(scratch_best, args.trajectory_seed, "PPO scratch", args.task_variant),
        rollout_ppo_with_obs_norm(
            bc_init_best,
            args.trajectory_seed,
            "BC + PPO",
            args.task_variant,
            obs_mean=bc_init_obs_norm[0],
            obs_std=bc_init_obs_norm[1],
        )
        if bc_init_obs_norm is not None
        else rollout_ppo(bc_init_best, args.trajectory_seed, "BC + PPO", args.task_variant),
    ]
    (args.output_dir / "trajectory_rollouts.json").write_text(json.dumps(rollouts, indent=2), encoding="utf-8")
    plot_trajectories(args.output_dir / "trajectory_comparison.png", rollouts)

    print(f"Saved experiment figures to {args.output_dir}")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
