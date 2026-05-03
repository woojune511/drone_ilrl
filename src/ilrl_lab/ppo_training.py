from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import DummyVecEnv

from ilrl_lab.bc import BehaviorCloningPolicy, load_bc_checkpoint
from ilrl_lab.envs import DetourWaypointVelocityAviary, WaypointVelocityAviary


class BCKLRegularizedPPO(PPO):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bc_kl_coef = 0.0
        self.bc_prior_model: BehaviorCloningPolicy | None = None
        self.bc_prior_obs_mean: torch.Tensor | None = None
        self.bc_prior_obs_std: torch.Tensor | None = None

    def set_bc_kl_prior(
        self,
        bc_model: BehaviorCloningPolicy,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        coef: float,
    ) -> None:
        self.bc_kl_coef = float(coef)
        self.bc_prior_model = bc_model.to(self.device)
        self.bc_prior_model.eval()
        for param in self.bc_prior_model.parameters():
            param.requires_grad = False
        self.bc_prior_obs_mean = torch.as_tensor(obs_mean, dtype=torch.float32, device=self.device)
        self.bc_prior_obs_std = torch.as_tensor(obs_std, dtype=torch.float32, device=self.device)

    def _compute_bc_kl_loss(self, observations: torch.Tensor) -> torch.Tensor | None:
        if (
            self.bc_kl_coef <= 0.0
            or self.bc_prior_model is None
            or self.bc_prior_obs_mean is None
            or self.bc_prior_obs_std is None
        ):
            return None

        with torch.no_grad():
            normalized_obs = (observations - self.bc_prior_obs_mean) / self.bc_prior_obs_std
            bc_mean_actions = self.bc_prior_model(normalized_obs)

        features = self.policy.extract_features(observations)
        if self.policy.share_features_extractor:
            latent_pi, _ = self.policy.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
        current_mean_actions = self.policy.action_net(latent_pi)
        current_std = torch.exp(self.policy.log_std).detach().view(1, -1)
        return 0.5 * (((current_mean_actions - bc_mean_actions) ** 2) / (current_std.pow(2) + 1e-8)).mean()

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses, bc_kl_losses = [], [], []
        clip_fractions = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                bc_kl_loss = self._compute_bc_kl_loss(rollout_data.observations)
                if bc_kl_loss is not None:
                    loss = loss + self.bc_kl_coef * bc_kl_loss
                    bc_kl_losses.append(bc_kl_loss.item())

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if bc_kl_losses:
            self.logger.record("train/bc_kl_loss", np.mean(bc_kl_losses))
            self.logger.record("train/bc_kl_coef", self.bc_kl_coef)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


def make_env_instance(task_variant: str, gui: bool):
    if task_variant == "detour":
        return DetourWaypointVelocityAviary(gui=gui)
    if task_variant == "waypoint":
        return WaypointVelocityAviary(gui=gui)
    raise ValueError(f"Unsupported task variant: {task_variant}")


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
    task_variant: str = "waypoint",
    return_trajectories: bool = False,
) -> tuple[EvalRecord, list[dict[str, Any]]]:
    env = make_env_instance(task_variant, gui=False)
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
        task_variant: str = "waypoint",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.eval_seed = int(eval_seed)
        self.run_dir = run_dir
        self.task_variant = task_variant
        self.eval_history: list[EvalRecord] = []
        self.best_success_rate = -float("inf")
        self.history_path = self.run_dir / "eval_history.json"
        self.best_model_path = self.run_dir / "best_model.zip"

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True

        record, _ = evaluate_model(
            self.model,
            self.eval_episodes,
            self.eval_seed,
            task_variant=self.task_variant,
        )
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


class FreezeActorCallback(BaseCallback):
    def __init__(self, freeze_steps: int, freeze_mode: str = "all", verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.freeze_steps = int(freeze_steps)
        self.freeze_mode = freeze_mode
        self._is_frozen = False
        self._tracked_names: list[str] = []

    def _on_training_start(self) -> None:
        if self.freeze_steps <= 0:
            return
        self._set_frozen(True)

    def _on_step(self) -> bool:
        if self.freeze_steps > 0 and self._is_frozen and self.num_timesteps >= self.freeze_steps:
            self._set_frozen(False)
        return True

    def _set_frozen(self, frozen: bool) -> None:
        tracked_names: list[str] = []
        for name, param in self.model.policy.named_parameters():
            should_track = False
            if "mlp_extractor.policy_net" in name:
                should_track = True
            elif self.freeze_mode == "all" and "action_net" in name:
                should_track = True
            if should_track:
                param.requires_grad = not frozen
                tracked_names.append(name)
        self._tracked_names = tracked_names
        self._is_frozen = frozen
        state = "Frozen" if frozen else "Unfroze"
        print(
            json.dumps(
                {
                    "callback": "FreezeActorCallback",
                    "state": state,
                    "timesteps": int(self.num_timesteps),
                    "freeze_mode": self.freeze_mode,
                }
            )
        )


def build_training_env(gui: bool, seed: int, task_variant: str = "waypoint") -> DummyVecEnv:
    def make_env():
        env = make_env_instance(task_variant, gui=gui)
        env.reset(seed=seed)
        return env

    return DummyVecEnv([make_env])


def build_ppo_model(env: DummyVecEnv, seed: int, args: Any) -> PPO:
    policy_kwargs = {
        "activation_fn": torch.nn.ReLU,
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
    }
    if hasattr(args, "log_std_init"):
        policy_kwargs["log_std_init"] = args.log_std_init
    return BCKLRegularizedPPO(
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


def initialize_actor_from_bc(
    model: PPO,
    bc_checkpoint: Path,
    device: torch.device,
    bc_kl_coef: float = 0.0,
) -> dict[str, str]:
    bc_model, obs_mean, obs_std, metadata = load_bc_checkpoint(bc_checkpoint, device)
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
    if bc_kl_coef > 0.0 and isinstance(model, BCKLRegularizedPPO):
        model.set_bc_kl_prior(bc_model, obs_mean, obs_std, coef=bc_kl_coef)
    return {
        "bc_checkpoint": str(bc_checkpoint),
        "bc_dataset_path": str(metadata.get("dataset_path")),
        "copied_layers": json.dumps(mapping),
        "bc_kl_coef": str(float(bc_kl_coef)),
    }


def build_bc_fine_tune_callback(
    eval_freq: int,
    eval_episodes: int,
    eval_seed: int,
    run_dir: Path,
    task_variant: str,
    freeze_actor_steps: int,
    freeze_actor_mode: str,
) -> tuple[BaseCallback, PeriodicEvalCallback]:
    callbacks: list[BaseCallback] = []
    eval_callback = PeriodicEvalCallback(
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        eval_seed=eval_seed,
        run_dir=run_dir,
        task_variant=task_variant,
    )
    if freeze_actor_steps > 0:
        callbacks.append(
            FreezeActorCallback(
                freeze_steps=freeze_actor_steps,
                freeze_mode=freeze_actor_mode,
            )
        )
    callbacks.append(eval_callback)
    if len(callbacks) == 1:
        return callbacks[0], eval_callback
    return CallbackList(callbacks), eval_callback
