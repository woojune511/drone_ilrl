from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ilrl_lab.bc import load_bc_checkpoint, predict_action
from ilrl_lab.envs import WaypointVelocityAviary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a BC checkpoint in the waypoint environment.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a BC checkpoint. Defaults to the latest checkpoint under artifacts/checkpoints/bc.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts") / "checkpoints" / "bc",
        help="Directory used when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "evals",
        help="Where evaluation summaries are saved.",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


def latest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.glob("**/checkpoint.pt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")
    return candidates[-1]


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint if args.checkpoint is not None else latest_checkpoint(args.checkpoint_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, obs_mean, obs_std, metadata = load_bc_checkpoint(checkpoint_path, device)

    env = WaypointVelocityAviary(gui=args.gui)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_successes: list[bool] = []
    episode_final_distances: list[float] = []

    for episode_idx in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode_idx)
        episode_return = 0.0
        episode_steps = 0

        while True:
            action = predict_action(model, obs, obs_mean, obs_std, device)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            episode_steps += 1

            if terminated or truncated:
                episode_returns.append(episode_return)
                episode_lengths.append(episode_steps)
                episode_successes.append(bool(info["success"]))
                episode_final_distances.append(float(info["distance_to_goal"]))
                break

    env.close()

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": metadata.get("dataset_path"),
        "episodes": args.episodes,
        "success_rate": float(np.mean(episode_successes)) if episode_successes else 0.0,
        "mean_episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "mean_final_distance": float(np.mean(episode_final_distances)) if episode_final_distances else 0.0,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"bc_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved evaluation summary to {output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
