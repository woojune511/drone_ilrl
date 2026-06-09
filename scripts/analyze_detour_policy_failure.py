from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

from ilrl_lab.bc import load_bc_checkpoint, predict_action
from ilrl_lab.envs import DetourWaypointVelocityAviary
from ilrl_lab.experts import detour_waypoint_velocity_expert
from ilrl_lab.ppo_training import FixedObservationNormalization


STAGE_RANK = {"entry": 0, "exit": 1, "goal": 2}


@dataclass
class PolicySpec:
    label: str
    method: str
    seed: int | None
    predictor: Callable[[np.ndarray], np.ndarray]
    obs_mean: np.ndarray | None = None
    obs_std: np.ndarray | None = None
    checkpoint_path: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose why detour policies fail by sweeping success tolerance and trajectory stages."
    )
    parser.add_argument("--scratch-dir", type=Path, default=Path("artifacts/checkpoints/ppo_scratch/detour"))
    parser.add_argument("--bc-init-dir", type=Path, default=Path("artifacts/checkpoints/ppo_bc_init/detour"))
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analysis/detour_policy_failure"))
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--trajectory-seed", type=int, default=30000)
    parser.add_argument("--total-timesteps-filter", type=int, default=300000)
    parser.add_argument("--n-envs-filter", type=int, default=4)
    parser.add_argument("--checkpoint-selector", choices=["final", "best"], default="final")
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=[0.10, 0.20, 0.30, 0.50],
        help="Goal tolerances to evaluate with the default speed threshold.",
    )
    parser.add_argument("--success-speed-threshold", type=float, default=0.15)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_latest_runs(root: Path, total_timesteps: int, n_envs: int) -> list[dict[str, Any]]:
    by_seed: dict[int, tuple[float, dict[str, Any]]] = {}
    for summary_path in root.glob("**/summary.json"):
        payload = load_json(summary_path)
        if payload.get("total_timesteps") != total_timesteps:
            continue
        if payload.get("n_envs") != n_envs:
            continue
        seed = int(payload["seed"])
        payload["_summary_path"] = str(summary_path)
        mtime = summary_path.stat().st_mtime
        if seed not in by_seed or mtime > by_seed[seed][0]:
            by_seed[seed] = (mtime, payload)
    if not by_seed:
        raise FileNotFoundError(f"No matching summary.json files found under {root}")
    return [by_seed[seed][1] for seed in sorted(by_seed)]


def obs_norm_from_run_summary(summary: dict[str, Any], fallback_bc_checkpoint: Path | None) -> tuple[np.ndarray, np.ndarray] | None:
    if summary.get("uses_bc_obs_normalization"):
        init_info = summary.get("initialization", {})
        obs_mean_json = init_info.get("obs_mean")
        obs_std_json = init_info.get("obs_std")
        if obs_mean_json is not None and obs_std_json is not None:
            return (
                np.asarray(json.loads(obs_mean_json), dtype=np.float32),
                np.asarray(json.loads(obs_std_json), dtype=np.float32),
            )

    checkpoint_path = summary.get("obs_norm_bc_checkpoint")
    if checkpoint_path is None and fallback_bc_checkpoint is not None:
        checkpoint_path = str(fallback_bc_checkpoint)
    if checkpoint_path is None:
        return None
    _, obs_mean, obs_std, _ = load_bc_checkpoint(Path(checkpoint_path), torch.device("cpu"))
    return obs_mean, obs_std


def build_policy_specs(args: argparse.Namespace) -> list[PolicySpec]:
    specs: list[PolicySpec] = [
        PolicySpec(
            label="Expert",
            method="expert",
            seed=None,
            predictor=lambda obs: detour_waypoint_velocity_expert(obs),
        )
    ]

    device = torch.device("cpu")
    bc_model, bc_obs_mean, bc_obs_std, _ = load_bc_checkpoint(args.bc_checkpoint, device)
    specs.append(
        PolicySpec(
            label="BC",
            method="bc",
            seed=None,
            predictor=lambda obs: predict_action(bc_model, obs, bc_obs_mean, bc_obs_std, device),
            checkpoint_path=str(args.bc_checkpoint),
        )
    )

    for method, root in (("scratch", args.scratch_dir), ("bc_init", args.bc_init_dir)):
        for summary in discover_latest_runs(root, args.total_timesteps_filter, args.n_envs_filter):
            checkpoint_key = "final_model_path" if args.checkpoint_selector == "final" else "best_model_path"
            checkpoint_path = Path(summary[checkpoint_key])
            model = PPO.load(checkpoint_path, device="cpu")
            obs_norm = obs_norm_from_run_summary(summary, args.bc_checkpoint)
            label = f"{method}_seed{summary['seed']}"
            specs.append(
                PolicySpec(
                    label=label,
                    method=method,
                    seed=int(summary["seed"]),
                    predictor=lambda obs, model=model: model.predict(obs, deterministic=True)[0],
                    obs_mean=None if obs_norm is None else obs_norm[0],
                    obs_std=None if obs_norm is None else obs_norm[1],
                    checkpoint_path=str(checkpoint_path),
                )
            )
    return specs


def make_env(goal_tolerance: float, speed_threshold: float, spec: PolicySpec):
    env = DetourWaypointVelocityAviary(
        gui=False,
        goal_tolerance=goal_tolerance,
        success_speed_threshold=speed_threshold,
    )
    if spec.obs_mean is not None and spec.obs_std is not None:
        return FixedObservationNormalization(env, obs_mean=spec.obs_mean, obs_std=spec.obs_std)
    return env


def run_episode(
    spec: PolicySpec,
    seed: int,
    goal_tolerance: float,
    speed_threshold: float,
    keep_trajectory: bool = False,
) -> dict[str, Any]:
    env = make_env(goal_tolerance, speed_threshold, spec)
    try:
        return run_episode_with_env(spec, env, seed, keep_trajectory=keep_trajectory)
    finally:
        env.close()


def run_episode_with_env(
    spec: PolicySpec,
    env,
    seed: int,
    keep_trajectory: bool = False,
) -> dict[str, Any]:
    obs, info = env.reset(seed=seed)
    positions = [np.asarray(info["position"], dtype=np.float32)]
    distances = [float(info["distance_to_goal"])]
    speeds = [float(info["speed"])]
    stages = [str(info.get("detour_stage", "entry"))]
    returns = 0.0
    steps = 0
    terminated = False
    truncated = False

    while True:
        action = spec.predictor(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        returns += float(reward)
        steps += 1
        positions.append(np.asarray(info["position"], dtype=np.float32))
        distances.append(float(info["distance_to_goal"]))
        speeds.append(float(info["speed"]))
        stages.append(str(info.get("detour_stage", "entry")))
        if terminated or truncated:
            break

    positions_arr = np.stack(positions)
    distances_arr = np.asarray(distances, dtype=np.float32)
    speeds_arr = np.asarray(speeds, dtype=np.float32)
    stage_ranks = [STAGE_RANK.get(stage, 0) for stage in stages]
    max_stage_rank = max(stage_ranks) if stage_ranks else 0

    result: dict[str, Any] = {
        "label": spec.label,
        "method": spec.method,
        "policy_seed": spec.seed,
        "episode_seed": int(seed),
        "success": bool(info["success"]),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "collision": bool(info.get("collision", False)),
        "episode_return": float(returns),
        "episode_length": int(steps),
        "final_distance": float(distances_arr[-1]),
        "min_distance": float(distances_arr.min()),
        "final_speed": float(speeds_arr[-1]),
        "min_speed": float(speeds_arr.min()),
        "max_stage": max(STAGE_RANK, key=lambda key: STAGE_RANK[key] if STAGE_RANK[key] == max_stage_rank else -1),
        "reached_exit_stage": bool(max_stage_rank >= STAGE_RANK["exit"]),
        "reached_goal_stage": bool(max_stage_rank >= STAGE_RANK["goal"]),
        "max_x": float(positions_arr[:, 0].max()),
        "max_y": float(positions_arr[:, 1].max()),
        "min_y": float(positions_arr[:, 1].min()),
        "goal": np.asarray(info["goal"], dtype=float).tolist(),
    }
    if keep_trajectory:
        result["positions"] = positions_arr.astype(float).tolist()
        result["distances"] = distances_arr.astype(float).tolist()
        result["speeds"] = speeds_arr.astype(float).tolist()
        result["stages"] = stages
    return result


def evaluate_policy_at_tolerance(
    spec: PolicySpec,
    episodes: int,
    seed: int,
    tolerance: float,
    speed_threshold: float,
) -> dict[str, Any]:
    env = make_env(tolerance, speed_threshold, spec)
    try:
        records = [
            run_episode_with_env(
                spec=spec,
                env=env,
                seed=seed + episode_idx,
                keep_trajectory=False,
            )
            for episode_idx in range(episodes)
        ]
    finally:
        env.close()
    return {
        "label": spec.label,
        "method": spec.method,
        "policy_seed": spec.seed,
        "goal_tolerance": float(tolerance),
        "speed_threshold": float(speed_threshold),
        "episodes": int(episodes),
        "success_rate": float(np.mean([r["success"] for r in records])),
        "position_only_success_rate": float(np.mean([r["min_distance"] < tolerance for r in records])),
        "mean_return": float(np.mean([r["episode_return"] for r in records])),
        "mean_length": float(np.mean([r["episode_length"] for r in records])),
        "mean_final_distance": float(np.mean([r["final_distance"] for r in records])),
        "mean_min_distance": float(np.mean([r["min_distance"] for r in records])),
        "mean_final_speed": float(np.mean([r["final_speed"] for r in records])),
        "collision_rate": float(np.mean([r["collision"] for r in records])),
        "reached_exit_stage_rate": float(np.mean([r["reached_exit_stage"] for r in records])),
        "reached_goal_stage_rate": float(np.mean([r["reached_goal_stage"] for r in records])),
    }


def aggregate_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["method"], float(row["goal_tolerance"])), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    metric_keys = [
        "success_rate",
        "position_only_success_rate",
        "mean_return",
        "mean_length",
        "mean_final_distance",
        "mean_min_distance",
        "mean_final_speed",
        "collision_rate",
        "reached_exit_stage_rate",
        "reached_goal_stage_rate",
    ]
    for (method, tolerance), records in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        aggregate = {
            "method": method,
            "goal_tolerance": tolerance,
            "num_policies": len(records),
            "labels": [record["label"] for record in records],
        }
        for key in metric_keys:
            values = [float(record[key]) for record in records]
            aggregate[f"{key}_mean"] = float(np.mean(values))
            aggregate[f"{key}_std"] = float(np.std(values))
        aggregate_rows.append(aggregate)
    return aggregate_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_tolerance_sweep(path: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    methods = ["expert", "bc", "scratch", "bc_init"]
    colors = {
        "expert": "#2ca02c",
        "bc": "#9467bd",
        "scratch": "#1f77b4",
        "bc_init": "#d62728",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ("success_rate_mean", "Success Rate"),
        ("position_only_success_rate_mean", "Position-only Rate"),
        ("mean_min_distance_mean", "Mean Min Distance"),
    ]
    for ax, (key, ylabel) in zip(axes, metrics, strict=True):
        for method in methods:
            rows = [row for row in aggregate_rows if row["method"] == method]
            if not rows:
                continue
            rows = sorted(rows, key=lambda row: row["goal_tolerance"])
            xs = np.asarray([row["goal_tolerance"] for row in rows], dtype=np.float32)
            ys = np.asarray([row[key] for row in rows], dtype=np.float32)
            ax.plot(xs, ys, marker="o", label=method, color=colors.get(method))
        ax.set_xlabel("Goal tolerance (m)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_representative_trajectories(path: Path, rollouts: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {
        "Expert": "#2ca02c",
        "BC": "#9467bd",
        "scratch_seed7": "#1f77b4",
        "bc_init_seed7": "#d62728",
    }
    for ax, ix, iy, xlabel, ylabel, title in [
        (axes[0], 0, 1, "X", "Y", "XY Trajectory"),
        (axes[1], 0, 2, "X", "Z", "XZ Trajectory"),
    ]:
        for rollout in rollouts:
            positions = np.asarray(rollout["positions"], dtype=np.float32)
            goal = np.asarray(rollout["goal"], dtype=np.float32)
            label = rollout["label"]
            color = colors.get(label)
            ax.plot(positions[:, ix], positions[:, iy], label=label, color=color)
            ax.scatter(positions[0, ix], positions[0, iy], marker="o", color=color, s=20)
            ax.scatter(goal[ix], goal[iy], marker="x", color=color, s=50)
        ax.axvline(0.0, color="#666666", linewidth=0.8, alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_policy_specs(args)

    tolerance_rows: list[dict[str, Any]] = []
    for spec in specs:
        for tolerance in args.tolerances:
            print(f"Evaluating {spec.label} at tolerance={tolerance:.2f}", flush=True)
            tolerance_rows.append(
                evaluate_policy_at_tolerance(
                    spec=spec,
                    episodes=args.episodes,
                    seed=args.seed,
                    tolerance=tolerance,
                    speed_threshold=args.success_speed_threshold,
                )
            )

    aggregate_rows = aggregate_by_method(tolerance_rows)
    write_csv(args.output_dir / "tolerance_sweep_by_policy.csv", tolerance_rows)
    write_csv(args.output_dir / "tolerance_sweep_by_method.csv", aggregate_rows)

    representative_specs = [
        spec
        for spec in specs
        if spec.label in {"Expert", "BC", "scratch_seed7", "bc_init_seed7"}
    ]
    trajectory_rollouts = [
        run_episode(
            spec=spec,
            seed=args.trajectory_seed,
            goal_tolerance=min(args.tolerances),
            speed_threshold=args.success_speed_threshold,
            keep_trajectory=True,
        )
        for spec in representative_specs
    ]
    plot_tolerance_sweep(args.output_dir / "tolerance_sweep.png", aggregate_rows)
    plot_representative_trajectories(args.output_dir / "representative_trajectories.png", trajectory_rollouts)

    summary = {
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "trajectory_seed": int(args.trajectory_seed),
        "tolerances": [float(tolerance) for tolerance in args.tolerances],
        "success_speed_threshold": float(args.success_speed_threshold),
        "policy_count": len(specs),
        "policies": [
            {
                "label": spec.label,
                "method": spec.method,
                "seed": spec.seed,
                "checkpoint_path": spec.checkpoint_path,
                "uses_observation_normalization": spec.obs_mean is not None,
            }
            for spec in specs
        ],
        "by_policy": tolerance_rows,
        "by_method": aggregate_rows,
        "representative_trajectories": trajectory_rollouts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "by_method": aggregate_rows}, indent=2))


if __name__ == "__main__":
    main()
