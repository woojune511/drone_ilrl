from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ilrl_lab.envs import WaypointVelocityAviary
from ilrl_lab.experts import waypoint_velocity_expert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect expert drone rollouts with a simple velocity-command expert."
    )
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to collect.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument("--gui", action="store_true", help="Run PyBullet with GUI enabled.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "datasets",
        help="Directory where the rollout dataset will be saved.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional per-episode step cap. Defaults to env episode length.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = WaypointVelocityAviary(gui=args.gui)

    obs_buffer: list[np.ndarray] = []
    act_buffer: list[np.ndarray] = []
    next_obs_buffer: list[np.ndarray] = []
    reward_buffer: list[float] = []
    done_buffer: list[bool] = []
    episode_start_buffer: list[bool] = []
    episode_id_buffer: list[int] = []

    episode_lengths: list[int] = []
    episode_returns: list[float] = []
    episode_successes: list[bool] = []
    episode_final_distances: list[float] = []
    episode_goals: list[np.ndarray] = []
    episode_initial_positions: list[np.ndarray] = []

    for episode_idx in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode_idx)
        episode_return = 0.0
        episode_steps = 0
        episode_initial_positions.append(np.asarray(info["position"], dtype=np.float32))
        episode_goals.append(np.asarray(info["goal"], dtype=np.float32))

        while True:
            action = waypoint_velocity_expert(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            obs_buffer.append(obs.astype(np.float32))
            act_buffer.append(action.astype(np.float32))
            next_obs_buffer.append(next_obs.astype(np.float32))
            reward_buffer.append(float(reward))
            done_buffer.append(bool(terminated or truncated))
            episode_start_buffer.append(episode_steps == 0)
            episode_id_buffer.append(episode_idx)

            obs = next_obs
            episode_return += float(reward)
            episode_steps += 1

            if terminated or truncated:
                episode_lengths.append(episode_steps)
                episode_returns.append(episode_return)
                episode_successes.append(bool(info["success"]))
                episode_final_distances.append(float(info["distance_to_goal"]))
                break

            if args.max_steps is not None and episode_steps >= args.max_steps:
                episode_lengths.append(episode_steps)
                episode_returns.append(episode_return)
                episode_successes.append(False)
                episode_final_distances.append(float(info["distance_to_goal"]))
                break

    env.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"waypoint_expert_{timestamp}"
    dataset_path = args.output_dir / f"{stem}.npz"
    summary_path = args.output_dir / f"{stem}_summary.json"

    np.savez_compressed(
        dataset_path,
        obs=np.stack(obs_buffer),
        acts=np.stack(act_buffer),
        next_obs=np.stack(next_obs_buffer),
        rewards=np.asarray(reward_buffer, dtype=np.float32),
        dones=np.asarray(done_buffer, dtype=bool),
        episode_starts=np.asarray(episode_start_buffer, dtype=bool),
        episode_ids=np.asarray(episode_id_buffer, dtype=np.int32),
        episode_lengths=np.asarray(episode_lengths, dtype=np.int32),
        episode_returns=np.asarray(episode_returns, dtype=np.float32),
        episode_successes=np.asarray(episode_successes, dtype=bool),
        episode_final_distances=np.asarray(episode_final_distances, dtype=np.float32),
        episode_goals=np.stack(episode_goals),
        episode_initial_positions=np.stack(episode_initial_positions),
    )

    success_rate = float(np.mean(episode_successes)) if episode_successes else 0.0
    mean_return = float(np.mean(episode_returns)) if episode_returns else 0.0
    mean_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    mean_final_distance = float(np.mean(episode_final_distances)) if episode_final_distances else 0.0

    summary = {
        "dataset_path": str(dataset_path),
        "episodes": args.episodes,
        "transitions": int(len(obs_buffer)),
        "success_rate": success_rate,
        "mean_episode_return": mean_return,
        "mean_episode_length": mean_length,
        "mean_final_distance": mean_final_distance,
        "seed": args.seed,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved dataset to {dataset_path}")
    print(f"Saved summary to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
