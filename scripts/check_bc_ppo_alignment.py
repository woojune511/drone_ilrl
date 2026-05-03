from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ilrl_lab.bc import load_bc_checkpoint, normalize_obs, predict_action
from ilrl_lab.envs import DetourWaypointVelocityAviary, WaypointVelocityAviary
from ilrl_lab.ppo_training import build_ppo_model, initialize_actor_from_bc


class Args:
    def __init__(self, run_dir: Path, log_std_init: float) -> None:
        self.learning_rate = 3e-4
        self.n_steps = 1024
        self.batch_size = 256
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.ent_coef = 0.0
        self.vf_coef = 0.5
        self.log_std_init = log_std_init
        self.run_dir = run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check BC->PPO initialization alignment on sampled states.")
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--task-variant", choices=["waypoint", "detour"], default="detour")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=70000)
    parser.add_argument("--max-states", type=int, default=512)
    parser.add_argument("--log-std-init", type=float, default=-0.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "checks" / "bc_ppo_alignment.json",
    )
    return parser.parse_args()


def make_env(task_variant: str):
    if task_variant == "detour":
        return DetourWaypointVelocityAviary(gui=False)
    return WaypointVelocityAviary(gui=False)


def sample_states(task_variant: str, episodes: int, seed: int, max_states: int) -> np.ndarray:
    env = make_env(task_variant)
    states: list[np.ndarray] = []
    for idx in range(episodes):
        obs, _ = env.reset(seed=seed + idx)
        while True:
            states.append(np.asarray(obs, dtype=np.float32))
            action = env.action_space.sample().astype(np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated or len(states) >= max_states:
                break
        if len(states) >= max_states:
            break
    env.close()
    return np.stack(states[:max_states])


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    bc_model, obs_mean, obs_std, _ = load_bc_checkpoint(args.bc_checkpoint, device)
    sampled_states = sample_states(args.task_variant, args.episodes, args.seed, args.max_states)
    normalized_states = normalize_obs(sampled_states, obs_mean, obs_std).astype(np.float32)

    dummy_run_dir = args.output.parent / "_alignment_tmp"
    dummy_run_dir.mkdir(parents=True, exist_ok=True)
    ppo_args = Args(run_dir=dummy_run_dir, log_std_init=args.log_std_init)
    from ilrl_lab.ppo_training import build_training_env

    env = build_training_env(
        gui=False,
        seed=args.seed,
        task_variant=args.task_variant,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )
    model = build_ppo_model(env, seed=args.seed, args=ppo_args)
    initialize_actor_from_bc(model, args.bc_checkpoint, device, bc_kl_coef=0.0)

    bc_actions = np.stack([predict_action(bc_model, obs, obs_mean, obs_std, device) for obs in sampled_states])
    ppo_actions = []
    for obs in normalized_states:
        action, _ = model.predict(obs, deterministic=True)
        ppo_actions.append(np.asarray(action, dtype=np.float32))
    ppo_actions = np.stack(ppo_actions)

    deltas = np.linalg.norm(ppo_actions - bc_actions, axis=1)
    cosine = np.sum(ppo_actions * bc_actions, axis=1) / (
        np.linalg.norm(ppo_actions, axis=1) * np.linalg.norm(bc_actions, axis=1) + 1e-8
    )
    summary = {
        "bc_checkpoint": str(args.bc_checkpoint),
        "task_variant": args.task_variant,
        "num_states": int(len(sampled_states)),
        "mean_action_l2_diff": float(np.mean(deltas)),
        "p90_action_l2_diff": float(np.percentile(deltas, 90)),
        "max_action_l2_diff": float(np.max(deltas)),
        "mean_cosine_similarity": float(np.mean(cosine)),
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
