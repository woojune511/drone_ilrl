from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ilrl_lab.bc import load_bc_checkpoint
from ilrl_lab.envs import WaypointVelocityAviary


@dataclass
class EvalRecord:
    timesteps: int
    episodes: int
    success_rate: float
    mean_episode_return: float
    mean_episode_length: float
    mean_final_distance: float


def evaluate_model(
    model: PPO,
    episodes: int,
    seed: int,
    return_trajectories: bool = False,
) -> tuple[EvalRecord, list[dict[str, Any]]]:
    env = WaypointVelocityAviary(gui=False)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_successes: list[bool] = []
    episode_final_distances: list[float] = []
    trajectories: list[dict[str, Any]] = []

    for episode_idx in range(episodes):
        obs, info = env.reset(seed=seed + episode_idx)
        episode_return = 0.0
        episode_steps = 0
        positions = [np.asarray(info["position"], dtype=np.float32)]
        goal = np.asarray(info["goal"], dtype=np.float32)

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            episode_steps += 1
            positions.append(np.asarray(info["position"], dtype=np.float32))

            if terminated or truncated:
                episode_returns.append(episode_return)
                episode_lengths.append(episode_steps)
                episode_successes.append(bool(info["success"]))
                episode_final_distances.append(float(info["distance_to_goal"]))
                if return_trajectories:
                    trajectories.append(
                        {
                            "seed": seed + episode_idx,
                            "success": bool(info["success"]),
                            "goal": goal.astype(float).tolist(),
                            "positions": np.stack(positions).astype(float).tolist(),
                            "episode_return": float(episode_return),
                            "episode_length": int(episode_steps),
                        }
                    )
                break

    env.close()
    record = EvalRecord(
        timesteps=0,
        episodes=episodes,
        success_rate=float(np.mean(episode_successes)) if episode_successes else 0.0,
        mean_episode_return=float(np.mean(episode_returns)) if episode_returns else 0.0,
        mean_episode_length=float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        mean_final_distance=float(np.mean(episode_final_distances)) if episode_final_distances else 0.0,
    )
    return record, trajectories


class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_freq: int,
        eval_episodes: int,
        eval_seed: int,
        run_dir: Path,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.eval_seed = int(eval_seed)
        self.run_dir = run_dir
        self.eval_history: list[EvalRecord] = []
        self.best_success_rate = -float("inf")
        self.history_path = self.run_dir / "eval_history.json"
        self.best_model_path = self.run_dir / "best_model.zip"

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True

        record, _ = evaluate_model(self.model, self.eval_episodes, self.eval_seed)
        record.timesteps = int(self.num_timesteps)
        self.eval_history.append(record)
        self._save_history()

        print(json.dumps(asdict(record)))
        if record.success_rate > self.best_success_rate:
            self.best_success_rate = record.success_rate
            self.model.save(self.best_model_path)
        return True

    def _save_history(self) -> None:
        payload = [asdict(record) for record in self.eval_history]
        self.history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_training_env(gui: bool, seed: int) -> DummyVecEnv:
    def make_env():
        env = WaypointVelocityAviary(gui=gui)
        env.reset(seed=seed)
        return env

    return DummyVecEnv([make_env])


def build_ppo_model(env: DummyVecEnv, seed: int, args: Any) -> PPO:
    policy_kwargs = {
        "activation_fn": torch.nn.ReLU,
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
    }
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
        tensorboard_log=str(args.run_dir / "tensorboard"),
    )


def initialize_actor_from_bc(model: PPO, bc_checkpoint: Path, device: torch.device) -> dict[str, str]:
    bc_model, _, _, metadata = load_bc_checkpoint(bc_checkpoint, device)
    bc_state = bc_model.state_dict()
    ppo_state = model.policy.state_dict()

    mapping = {
        "network.0.weight": "mlp_extractor.policy_net.0.weight",
        "network.0.bias": "mlp_extractor.policy_net.0.bias",
        "network.2.weight": "mlp_extractor.policy_net.2.weight",
        "network.2.bias": "mlp_extractor.policy_net.2.bias",
        "network.4.weight": "action_net.weight",
        "network.4.bias": "action_net.bias",
    }
    for bc_key, ppo_key in mapping.items():
        ppo_state[ppo_key] = bc_state[bc_key].clone()

    model.policy.load_state_dict(ppo_state)
    return {
        "bc_checkpoint": str(bc_checkpoint),
        "bc_dataset_path": str(metadata.get("dataset_path")),
        "copied_layers": json.dumps(mapping),
    }
