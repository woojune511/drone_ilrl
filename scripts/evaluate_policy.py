from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO

from ilrl_lab.bc import load_bc_checkpoint, predict_action
from ilrl_lab.envs import (
    DetourPlanarLocalObsAviary,
    DetourPlanarVelocityAviary,
    DetourWaypointVelocityAviary,
    WaypointVelocityAviary,
)
from ilrl_lab.ppo_training import FixedObservationNormalization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a BC or PPO checkpoint with a shared protocol."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Policy checkpoint to evaluate. PPO uses .zip; BC uses checkpoint.pt.",
    )
    parser.add_argument(
        "--policy-type",
        choices=["auto", "bc", "ppo"],
        default="auto",
        help="Policy loader to use. auto infers from checkpoint suffix.",
    )
    parser.add_argument(
        "--run-summary",
        type=Path,
        default=None,
        help="Optional PPO run summary.json. Used to infer task, checkpoint, and obs normalization.",
    )
    parser.add_argument(
        "--checkpoint-selector",
        choices=["best", "final"],
        default="best",
        help="Which PPO checkpoint from --run-summary to evaluate when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--obs-norm-bc-checkpoint",
        type=Path,
        default=None,
        help="Optional BC checkpoint whose obs_mean/obs_std should normalize PPO observations.",
    )
    parser.add_argument(
        "--task-variant",
        choices=["waypoint", "detour", "detour_planar", "detour_planar_local"],
        default=None,
        help="Environment variant. Defaults to --run-summary task_variant or waypoint.",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic PPO actions. BC evaluation remains deterministic.",
    )
    parser.add_argument(
        "--include-episodes",
        action="store_true",
        help="Include per-episode records in the output JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Exact output JSON path. Defaults to artifacts/evals/policy_eval_*.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "evals",
        help="Output directory used when --output is omitted.",
    )
    return parser.parse_args()


def make_env(task_variant: str, gui: bool):
    if task_variant == "detour_planar_local":
        return DetourPlanarLocalObsAviary(gui=gui)
    if task_variant == "detour_planar":
        return DetourPlanarVelocityAviary(gui=gui)
    if task_variant == "detour":
        return DetourWaypointVelocityAviary(gui=gui)
    if task_variant == "waypoint":
        return WaypointVelocityAviary(gui=gui)
    raise ValueError(f"Unsupported task variant: {task_variant}")


def load_run_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_task_variant(args: argparse.Namespace, run_summary: dict[str, Any]) -> str:
    if args.task_variant is not None:
        return args.task_variant
    return str(run_summary.get("task_variant", "waypoint"))


def resolve_checkpoint(args: argparse.Namespace, run_summary: dict[str, Any]) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    if not run_summary:
        raise ValueError("Either --checkpoint or --run-summary is required.")

    summary_key = "best_model_path" if args.checkpoint_selector == "best" else "final_model_path"
    checkpoint = run_summary.get(summary_key)
    if checkpoint is None:
        raise ValueError(f"--run-summary does not contain {summary_key}.")
    return Path(checkpoint)


def infer_policy_type(policy_type: str, checkpoint: Path) -> str:
    if policy_type != "auto":
        return policy_type
    if checkpoint.suffix == ".zip":
        return "ppo"
    if checkpoint.suffix == ".pt":
        return "bc"
    raise ValueError(
        f"Could not infer policy type from {checkpoint}. Pass --policy-type bc or --policy-type ppo."
    )


def obs_norm_from_run_summary(
    run_summary: dict[str, Any],
    fallback_bc_checkpoint: Path | None,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    if run_summary.get("uses_bc_obs_normalization"):
        init_info = run_summary.get("initialization", {})
        obs_mean_json = init_info.get("obs_mean")
        obs_std_json = init_info.get("obs_std")
        if obs_mean_json is not None and obs_std_json is not None:
            return (
                np.asarray(json.loads(obs_mean_json), dtype=np.float32),
                np.asarray(json.loads(obs_std_json), dtype=np.float32),
                "run_summary_initialization",
            )

    checkpoint_path = run_summary.get("obs_norm_bc_checkpoint")
    if checkpoint_path is None and fallback_bc_checkpoint is not None:
        checkpoint_path = str(fallback_bc_checkpoint)
    if checkpoint_path is None:
        return None

    _, obs_mean, obs_std, _ = load_bc_checkpoint(Path(checkpoint_path), torch.device("cpu"))
    return obs_mean, obs_std, str(checkpoint_path)


def output_path(args: argparse.Namespace, policy_type: str, task_variant: str) -> Path:
    if args.output is not None:
        return args.output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return args.output_dir / f"policy_eval_{policy_type}_{task_variant}_{timestamp}.json"


def evaluate_bc(
    checkpoint: Path,
    task_variant: str,
    episodes: int,
    seed: int,
    gui: bool,
    include_episodes: bool,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, obs_mean, obs_std, metadata = load_bc_checkpoint(checkpoint, device)
    env = make_env(task_variant, gui=gui)
    try:
        return evaluate_loop(
            env=env,
            episodes=episodes,
            seed=seed,
            include_episodes=include_episodes,
            action_fn=lambda obs: predict_action(model, obs, obs_mean, obs_std, device),
            extra={
                "policy_type": "bc",
                "checkpoint_path": str(checkpoint),
                "dataset_path": metadata.get("dataset_path"),
                "uses_observation_normalization": True,
                "obs_norm_source": str(checkpoint),
            },
        )
    finally:
        env.close()


def evaluate_ppo(
    checkpoint: Path,
    task_variant: str,
    episodes: int,
    seed: int,
    gui: bool,
    stochastic: bool,
    include_episodes: bool,
    obs_norm: tuple[np.ndarray, np.ndarray, str] | None,
) -> dict[str, Any]:
    env = make_env(task_variant, gui=gui)
    obs_norm_source = None
    if obs_norm is not None:
        obs_mean, obs_std, obs_norm_source = obs_norm
        env = FixedObservationNormalization(env, obs_mean=obs_mean, obs_std=obs_std)

    model = PPO.load(checkpoint)
    try:
        return evaluate_loop(
            env=env,
            episodes=episodes,
            seed=seed,
            include_episodes=include_episodes,
            action_fn=lambda obs: model.predict(obs, deterministic=not stochastic)[0],
            extra={
                "policy_type": "ppo",
                "checkpoint_path": str(checkpoint),
                "uses_observation_normalization": obs_norm is not None,
                "obs_norm_source": obs_norm_source,
                "stochastic": bool(stochastic),
            },
        )
    finally:
        env.close()


def evaluate_loop(
    env,
    episodes: int,
    seed: int,
    include_episodes: bool,
    action_fn,
    extra: dict[str, Any],
) -> dict[str, Any]:
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_successes: list[bool] = []
    episode_final_distances: list[float] = []
    episode_records: list[dict[str, Any]] = []

    for episode_idx in range(episodes):
        episode_seed = seed + episode_idx
        obs, _ = env.reset(seed=episode_seed)
        episode_return = 0.0
        episode_steps = 0

        while True:
            action = action_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            episode_steps += 1

            if terminated or truncated:
                success = bool(info["success"])
                final_distance = float(info["distance_to_goal"])
                episode_returns.append(episode_return)
                episode_lengths.append(episode_steps)
                episode_successes.append(success)
                episode_final_distances.append(final_distance)
                if include_episodes:
                    episode_records.append(
                        {
                            "episode": episode_idx,
                            "seed": episode_seed,
                            "success": success,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "episode_return": float(episode_return),
                            "episode_length": int(episode_steps),
                            "final_distance": final_distance,
                        }
                    )
                break

    summary = {
        **extra,
        "episodes": int(episodes),
        "seed": int(seed),
        "success_rate": float(np.mean(episode_successes)) if episode_successes else 0.0,
        "mean_episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "mean_final_distance": float(np.mean(episode_final_distances)) if episode_final_distances else 0.0,
    }
    if include_episodes:
        summary["episode_records"] = episode_records
    return summary


def main() -> None:
    args = parse_args()
    run_summary = load_run_summary(args.run_summary)
    task_variant = resolve_task_variant(args, run_summary)
    checkpoint = resolve_checkpoint(args, run_summary)
    policy_type = infer_policy_type(args.policy_type, checkpoint)

    if policy_type == "bc":
        summary = evaluate_bc(
            checkpoint=checkpoint,
            task_variant=task_variant,
            episodes=args.episodes,
            seed=args.seed,
            gui=args.gui,
            include_episodes=args.include_episodes,
        )
    elif policy_type == "ppo":
        obs_norm = obs_norm_from_run_summary(run_summary, args.obs_norm_bc_checkpoint)
        summary = evaluate_ppo(
            checkpoint=checkpoint,
            task_variant=task_variant,
            episodes=args.episodes,
            seed=args.seed,
            gui=args.gui,
            stochastic=args.stochastic,
            include_episodes=args.include_episodes,
            obs_norm=obs_norm,
        )
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}")

    summary["task_variant"] = task_variant
    summary["run_summary_path"] = None if args.run_summary is None else str(args.run_summary)
    summary["evaluated_at"] = datetime.now().isoformat(timespec="seconds")

    path = output_path(args, policy_type, task_variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved evaluation summary to {path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
