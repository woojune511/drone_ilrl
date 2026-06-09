from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from ilrl_lab.envs import DetourWaypointVelocityAviary
from ilrl_lab.ppo_training import FixedObservationNormalization


STAGE_RANK = {"entry": 0, "exit": 1, "goal": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze deterministic PPO rollouts episode by episode.")
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--checkpoint-selector", choices=["best", "final"], default="final")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot-first-failures", type=int, default=6)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def obs_norm_from_summary(summary: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    if not summary.get("uses_bc_obs_normalization"):
        return None
    init_info = summary.get("initialization", {})
    obs_mean_json = init_info.get("obs_mean")
    obs_std_json = init_info.get("obs_std")
    if obs_mean_json is None or obs_std_json is None:
        return None
    return (
        np.asarray(json.loads(obs_mean_json), dtype=np.float32),
        np.asarray(json.loads(obs_std_json), dtype=np.float32),
    )


def make_env(summary: dict[str, Any]):
    env = DetourWaypointVelocityAviary(gui=False)
    obs_norm = obs_norm_from_summary(summary)
    if obs_norm is None:
        return env
    obs_mean, obs_std = obs_norm
    return FixedObservationNormalization(env, obs_mean=obs_mean, obs_std=obs_std)


def stage_name(max_rank: int) -> str:
    for name, rank in STAGE_RANK.items():
        if rank == max_rank:
            return name
    return "entry"


def run_episode(model: PPO, env, seed: int, keep_trajectory: bool) -> dict[str, Any]:
    obs, info = env.reset(seed=seed)
    goal = np.asarray(info["goal"], dtype=np.float32)
    positions = [np.asarray(info["position"], dtype=np.float32)]
    distances = [float(info["distance_to_goal"])]
    speeds = [float(info.get("speed", 0.0))]
    stages = [str(info.get("detour_stage", "entry"))]
    collided = bool(info.get("collision", False))
    episode_return = 0.0
    episode_length = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        episode_length += 1
        positions.append(np.asarray(info["position"], dtype=np.float32))
        distances.append(float(info["distance_to_goal"]))
        speeds.append(float(info.get("speed", 0.0)))
        stages.append(str(info.get("detour_stage", "entry")))
        collided = collided or bool(info.get("collision", False))
        if terminated or truncated:
            break

    positions_arr = np.stack(positions).astype(np.float32)
    distances_arr = np.asarray(distances, dtype=np.float32)
    speeds_arr = np.asarray(speeds, dtype=np.float32)
    max_stage_rank = max(STAGE_RANK.get(stage, 0) for stage in stages)
    record: dict[str, Any] = {
        "episode_seed": int(seed),
        "success": bool(info["success"]),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "collision": bool(collided),
        "episode_return": float(episode_return),
        "episode_length": int(episode_length),
        "final_distance": float(distances_arr[-1]),
        "min_distance": float(distances_arr.min()),
        "final_speed": float(speeds_arr[-1]),
        "min_speed": float(speeds_arr.min()),
        "max_stage": stage_name(max_stage_rank),
        "reached_exit_stage": bool(max_stage_rank >= STAGE_RANK["exit"]),
        "reached_goal_stage": bool(max_stage_rank >= STAGE_RANK["goal"]),
        "final_x": float(positions_arr[-1, 0]),
        "final_y": float(positions_arr[-1, 1]),
        "final_z": float(positions_arr[-1, 2]),
        "goal_x": float(goal[0]),
        "goal_y": float(goal[1]),
        "goal_z": float(goal[2]),
    }
    if keep_trajectory:
        record["positions"] = positions_arr.astype(float).tolist()
        record["distances"] = distances_arr.astype(float).tolist()
        record["speeds"] = speeds_arr.astype(float).tolist()
        record["stages"] = stages
    return record


def summarize(records: list[dict[str, Any]], summary: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    failures = [record for record in records if not record["success"]]
    return {
        "run_summary": summary["run_dir"],
        "checkpoint_path": str(checkpoint_path),
        "train_seed": int(summary["seed"]),
        "episodes": len(records),
        "success_rate": float(np.mean([record["success"] for record in records])),
        "mean_final_distance": float(np.mean([record["final_distance"] for record in records])),
        "mean_min_distance": float(np.mean([record["min_distance"] for record in records])),
        "mean_final_speed": float(np.mean([record["final_speed"] for record in records])),
        "collision_rate": float(np.mean([record["collision"] for record in records])),
        "reached_exit_stage_rate": float(np.mean([record["reached_exit_stage"] for record in records])),
        "reached_goal_stage_rate": float(np.mean([record["reached_goal_stage"] for record in records])),
        "failure_count": len(failures),
        "failure_reached_goal_stage_rate": (
            float(np.mean([record["reached_goal_stage"] for record in failures])) if failures else 0.0
        ),
        "failure_mean_min_distance": (
            float(np.mean([record["min_distance"] for record in failures])) if failures else 0.0
        ),
        "failure_mean_final_distance": (
            float(np.mean([record["final_distance"] for record in failures])) if failures else 0.0
        ),
        "failure_mean_final_speed": (
            float(np.mean([record["final_speed"] for record in failures])) if failures else 0.0
        ),
        "failure_collision_rate": (
            float(np.mean([record["collision"] for record in failures])) if failures else 0.0
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    scalar_rows = [
        {key: value for key, value in row.items() if key not in {"positions", "distances", "speeds", "stages"}}
        for row in rows
    ]
    if not scalar_rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scalar_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scalar_rows)


def plot_failure_trajectories(path: Path, records: list[dict[str, Any]], limit: int) -> None:
    failures = [record for record in records if not record["success"] and "positions" in record][:limit]
    if not failures:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for record in failures:
        positions = np.asarray(record["positions"], dtype=np.float32)
        label = f"seed {record['episode_seed']}"
        axes[0].plot(positions[:, 0], positions[:, 1], label=label, alpha=0.85)
        axes[1].plot(positions[:, 0], positions[:, 2], label=label, alpha=0.85)
        axes[0].scatter(record["goal_x"], record["goal_y"], marker="x", s=35)
        axes[1].scatter(record["goal_x"], record["goal_z"], marker="x", s=35)
    axes[0].axvline(0.0, color="#777777", linewidth=0.8, alpha=0.6)
    axes[1].axvline(0.0, color="#777777", linewidth=0.8, alpha=0.6)
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].set_title("Failure XY trajectories")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Z")
    axes[1].set_title("Failure XZ trajectories")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_failure_distributions(path: Path, records: list[dict[str, Any]]) -> None:
    failures = [record for record in records if not record["success"]]
    if not failures:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].hist([record["min_distance"] for record in failures], bins=12, color="#4c78a8")
    axes[0].set_title("Failure min distance")
    axes[1].hist([record["final_distance"] for record in failures], bins=12, color="#f58518")
    axes[1].set_title("Failure final distance")
    axes[2].hist([record["final_speed"] for record in failures], bins=12, color="#54a24b")
    axes[2].set_title("Failure final speed")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = load_json(args.run_summary)
    checkpoint_key = "best_model_path" if args.checkpoint_selector == "best" else "final_model_path"
    checkpoint_path = Path(summary[checkpoint_key])
    model = PPO.load(checkpoint_path, device="cpu")
    env = make_env(summary)
    try:
        records = [
            run_episode(
                model=model,
                env=env,
                seed=args.seed + episode_idx,
                keep_trajectory=True,
            )
            for episode_idx in range(args.episodes)
        ]
    finally:
        env.close()

    aggregate = summarize(records, summary, checkpoint_path)
    write_csv(args.output_dir / "episode_rollouts.csv", records)
    (args.output_dir / "episode_rollouts.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    plot_failure_trajectories(args.output_dir / "failure_trajectories.png", records, args.plot_first_failures)
    plot_failure_distributions(args.output_dir / "failure_distributions.png", records)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
