from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ilrl_lab.bc import BehaviorCloningPolicy, load_bc_checkpoint
from ilrl_lab.envs import (
    DetourPlanarLocalObsAviary,
    DetourPlanarRaycastAviary,
    DetourPlanarVelocityAviary,
    DetourWaypointVelocityAviary,
    WaypointVelocityAviary,
)
from ilrl_lab.experts import (
    detour_planar_local_obs_expert,
    detour_planar_velocity_expert,
    detour_waypoint_velocity_expert,
    waypoint_velocity_expert,
)


class FixedObservationNormalization(gym.ObservationWrapper):
    """Apply fixed observation normalization statistics to an env."""

    def __init__(self, env: gym.Env, obs_mean: np.ndarray, obs_std: np.ndarray) -> None:
        super().__init__(env)
        self.obs_mean = np.asarray(obs_mean, dtype=np.float32)
        self.obs_std = np.asarray(obs_std, dtype=np.float32)
        shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=shape,
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(observation, dtype=np.float32) - self.obs_mean) / self.obs_std
        return normalized.astype(np.float32)


class BCKLRegularizedPPO(PPO):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bc_kl_coef = 0.0
        self.bc_prior_model: BehaviorCloningPolicy | None = None
        self.bc_prior_expects_normalized_obs = True
        self.expert_bc_loss_coef = 0.0
        self.expert_bc_batch_size = 256
        self.expert_bc_observations: torch.Tensor | None = None
        self.expert_bc_actions: torch.Tensor | None = None

    def set_bc_kl_prior(
        self,
        bc_model: BehaviorCloningPolicy,
        coef: float,
        expects_normalized_obs: bool = True,
    ) -> None:
        self.bc_kl_coef = float(coef)
        self.bc_prior_model = bc_model.to(self.device)
        self.bc_prior_model.eval()
        for param in self.bc_prior_model.parameters():
            param.requires_grad = False
        self.bc_prior_expects_normalized_obs = expects_normalized_obs

    def set_expert_bc_dataset(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        coef: float,
        batch_size: int = 256,
        task_variant: str = "waypoint",
        augment_copies: int = 0,
        position_noise_std: float = 0.0,
        velocity_noise_std: float = 0.0,
        rpy_noise_std: float = 0.0,
        angular_velocity_noise_std: float = 0.0,
        augment_seed: int = 0,
    ) -> None:
        expert_observations, expert_actions = build_expert_bc_training_arrays(
            observations=observations,
            actions=actions,
            task_variant=task_variant,
            augment_copies=augment_copies,
            position_noise_std=position_noise_std,
            velocity_noise_std=velocity_noise_std,
            rpy_noise_std=rpy_noise_std,
            angular_velocity_noise_std=angular_velocity_noise_std,
            seed=augment_seed,
        )
        normalized_observations = ((expert_observations - obs_mean) / obs_std).astype(np.float32)
        self.expert_bc_observations = torch.as_tensor(
            normalized_observations,
            dtype=torch.float32,
            device=self.device,
        )
        self.expert_bc_actions = torch.as_tensor(
            expert_actions,
            dtype=torch.float32,
            device=self.device,
        )
        self.expert_bc_loss_coef = float(coef)
        self.expert_bc_batch_size = int(batch_size)

    def _policy_mean_actions(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.policy.extract_features(observations)
        if self.policy.share_features_extractor:
            latent_pi, _ = self.policy.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
        return self.policy.action_net(latent_pi)

    def _compute_bc_kl_loss(self, observations: torch.Tensor) -> torch.Tensor | None:
        if self.bc_kl_coef <= 0.0 or self.bc_prior_model is None:
            return None

        with torch.no_grad():
            if not self.bc_prior_expects_normalized_obs:
                raise RuntimeError("BC KL prior expects raw observations, which is no longer supported.")
            bc_mean_actions = self.bc_prior_model(observations)

        current_mean_actions = self._policy_mean_actions(observations)
        current_std = torch.exp(self.policy.log_std).detach().view(1, -1)
        return 0.5 * (((current_mean_actions - bc_mean_actions) ** 2) / (current_std.pow(2) + 1e-8)).mean()

    def _compute_expert_bc_loss(self) -> torch.Tensor | None:
        if (
            self.expert_bc_loss_coef <= 0.0
            or self.expert_bc_observations is None
            or self.expert_bc_actions is None
        ):
            return None

        num_samples = int(self.expert_bc_observations.shape[0])
        batch_size = min(self.expert_bc_batch_size, num_samples)
        indices = torch.randint(num_samples, (batch_size,), device=self.device)
        observations = self.expert_bc_observations[indices]
        expert_actions = self.expert_bc_actions[indices]
        current_mean_actions = self._policy_mean_actions(observations)
        return F.mse_loss(current_mean_actions, expert_actions)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses, bc_kl_losses, expert_bc_losses = [], [], [], []
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

                expert_bc_loss = self._compute_expert_bc_loss()
                if expert_bc_loss is not None:
                    loss = loss + self.expert_bc_loss_coef * expert_bc_loss
                    expert_bc_losses.append(expert_bc_loss.item())

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
            bc_kl_loss_mean = float(np.mean(bc_kl_losses))
            self.logger.record("train/bc_kl_loss", bc_kl_loss_mean)
            self.logger.record("train/bc_kl_loss_weighted", self.bc_kl_coef * bc_kl_loss_mean)
            self.logger.record("train/bc_kl_coef", self.bc_kl_coef)
        if expert_bc_losses:
            expert_bc_loss_mean = float(np.mean(expert_bc_losses))
            self.logger.record("train/expert_bc_loss", expert_bc_loss_mean)
            self.logger.record("train/expert_bc_loss_weighted", self.expert_bc_loss_coef * expert_bc_loss_mean)
            self.logger.record("train/expert_bc_loss_coef", self.expert_bc_loss_coef)
        if hasattr(self.policy, "log_std"):
            log_std = self.policy.log_std.detach()
            std = torch.exp(log_std)
            self.logger.record("train/std", std.mean().item())
            self.logger.record("train/log_std_mean", log_std.mean().item())
            self.logger.record("train/log_std_min", log_std.min().item())
            self.logger.record("train/log_std_max", log_std.max().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


def make_env_instance(task_variant: str, gui: bool):
    if task_variant == "detour_planar_raycast":
        return DetourPlanarRaycastAviary(gui=gui)
    if task_variant == "detour_planar_local":
        return DetourPlanarLocalObsAviary(gui=gui)
    if task_variant == "detour_planar":
        return DetourPlanarVelocityAviary(gui=gui)
    if task_variant == "detour":
        return DetourWaypointVelocityAviary(gui=gui)
    if task_variant == "waypoint":
        return WaypointVelocityAviary(gui=gui)
    raise ValueError(f"Unsupported task variant: {task_variant}")


def expert_for_task_variant(task_variant: str):
    if task_variant == "detour_planar_raycast":
        raise ValueError("detour_planar_raycast requires an env-bound privileged expert.")
    if task_variant == "detour_planar_local":
        return detour_planar_local_obs_expert
    if task_variant == "detour_planar":
        return detour_planar_velocity_expert
    if task_variant == "detour":
        return detour_waypoint_velocity_expert
    if task_variant == "waypoint":
        return waypoint_velocity_expert
    raise ValueError(f"Unsupported task variant: {task_variant}")


def expert_action_for_env(env: Any, task_variant: str, observation: np.ndarray) -> np.ndarray:
    if hasattr(env, "privileged_expert_action"):
        return env.privileged_expert_action()
    return expert_for_task_variant(task_variant)(observation)


@dataclass
class EvalRecord:
    timesteps: int
    episodes: int
    success_rate: float
    mean_episode_return: float
    mean_episode_length: float
    mean_final_distance: float
    mean_min_distance: float = 0.0
    position_only_success_rate: float = 0.0
    collision_rate: float = 0.0
    reached_exit_stage_rate: float = 0.0
    reached_goal_stage_rate: float = 0.0
    mean_final_speed: float = 0.0
    policy_log_std_mean: float | None = None
    policy_log_std_min: float | None = None
    policy_log_std_max: float | None = None
    policy_std_mean: float | None = None
    bc_probe_action_l2: float | None = None
    bc_probe_action_cosine: float | None = None
    bc_probe_max_abs_action_diff: float | None = None
    bc_probe_action_saturation_rate: float | None = None


def evaluate_model(
    model: PPO,
    episodes: int,
    seed: int,
    task_variant: str = "waypoint",
    obs_mean: np.ndarray | None = None,
    obs_std: np.ndarray | None = None,
    return_trajectories: bool = False,
    bc_probe: dict[str, Any] | None = None,
) -> tuple[EvalRecord, list[dict[str, Any]]]:
    env = make_env_instance(task_variant, gui=False)
    goal_tolerance = float(getattr(env, "goal_tolerance", 0.10))
    if obs_mean is not None and obs_std is not None:
        env = FixedObservationNormalization(env, obs_mean=obs_mean, obs_std=obs_std)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_successes: list[bool] = []
    episode_final_distances: list[float] = []
    episode_min_distances: list[float] = []
    episode_position_only_successes: list[bool] = []
    episode_collisions: list[bool] = []
    episode_reached_exit_stage: list[bool] = []
    episode_reached_goal_stage: list[bool] = []
    episode_final_speeds: list[float] = []
    trajectories: list[dict[str, Any]] = []

    for episode_idx in range(episodes):
        obs, info = env.reset(seed=seed + episode_idx)
        episode_return = 0.0
        episode_steps = 0
        positions = [np.asarray(info["position"], dtype=np.float32)]
        goal = np.asarray(info["goal"], dtype=np.float32)
        distances = [float(info["distance_to_goal"])]
        max_stage_rank = _detour_stage_rank(str(info.get("detour_stage", "entry")))
        collided = bool(info.get("collision", False))

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            episode_steps += 1
            positions.append(np.asarray(info["position"], dtype=np.float32))
            distances.append(float(info["distance_to_goal"]))
            max_stage_rank = max(max_stage_rank, _detour_stage_rank(str(info.get("detour_stage", "entry"))))
            collided = collided or bool(info.get("collision", False))

            if terminated or truncated:
                min_distance = float(np.min(distances))
                episode_returns.append(episode_return)
                episode_lengths.append(episode_steps)
                episode_successes.append(bool(info["success"]))
                episode_final_distances.append(float(distances[-1]))
                episode_min_distances.append(min_distance)
                episode_position_only_successes.append(min_distance < goal_tolerance)
                episode_collisions.append(collided)
                episode_reached_exit_stage.append(max_stage_rank >= _detour_stage_rank("exit"))
                episode_reached_goal_stage.append(max_stage_rank >= _detour_stage_rank("goal"))
                episode_final_speeds.append(float(info.get("speed", 0.0)))
                if return_trajectories:
                    trajectories.append(
                        {
                            "seed": seed + episode_idx,
                            "success": bool(info["success"]),
                            "goal": goal.astype(float).tolist(),
                            "positions": np.stack(positions).astype(float).tolist(),
                            "episode_return": float(episode_return),
                            "episode_length": int(episode_steps),
                            "min_distance": min_distance,
                            "final_distance": float(distances[-1]),
                            "final_speed": float(info.get("speed", 0.0)),
                            "collision": collided,
                        }
                    )
                break

    env.close()
    policy_stats = policy_distribution_stats(model)
    probe_stats = compute_bc_probe_metrics(model, bc_probe) if bc_probe is not None else {}
    record = EvalRecord(
        timesteps=0,
        episodes=episodes,
        success_rate=float(np.mean(episode_successes)) if episode_successes else 0.0,
        mean_episode_return=float(np.mean(episode_returns)) if episode_returns else 0.0,
        mean_episode_length=float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        mean_final_distance=float(np.mean(episode_final_distances)) if episode_final_distances else 0.0,
        mean_min_distance=float(np.mean(episode_min_distances)) if episode_min_distances else 0.0,
        position_only_success_rate=(
            float(np.mean(episode_position_only_successes)) if episode_position_only_successes else 0.0
        ),
        collision_rate=float(np.mean(episode_collisions)) if episode_collisions else 0.0,
        reached_exit_stage_rate=(
            float(np.mean(episode_reached_exit_stage)) if episode_reached_exit_stage else 0.0
        ),
        reached_goal_stage_rate=(
            float(np.mean(episode_reached_goal_stage)) if episode_reached_goal_stage else 0.0
        ),
        mean_final_speed=float(np.mean(episode_final_speeds)) if episode_final_speeds else 0.0,
        **policy_stats,
        **probe_stats,
    )
    return record, trajectories


def _detour_stage_rank(stage: str) -> int:
    return {"entry": 0, "exit": 1, "goal": 2}.get(stage, 0)


def policy_distribution_stats(model: PPO) -> dict[str, float] | dict[str, None]:
    if not hasattr(model.policy, "log_std"):
        return {
            "policy_log_std_mean": None,
            "policy_log_std_min": None,
            "policy_log_std_max": None,
            "policy_std_mean": None,
        }
    log_std = model.policy.log_std.detach().cpu()
    return {
        "policy_log_std_mean": float(log_std.mean().item()),
        "policy_log_std_min": float(log_std.min().item()),
        "policy_log_std_max": float(log_std.max().item()),
        "policy_std_mean": float(torch.exp(log_std).mean().item()),
    }


def ppo_mean_actions(model: PPO, observations: np.ndarray) -> np.ndarray:
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=model.device)
    with torch.no_grad():
        features = model.policy.extract_features(obs_tensor)
        if model.policy.share_features_extractor:
            latent_pi, _ = model.policy.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = model.policy.mlp_extractor.forward_actor(pi_features)
        actions = model.policy.action_net(latent_pi)
    return actions.detach().cpu().numpy().astype(np.float32)


def compute_bc_probe_metrics(model: PPO, bc_probe: dict[str, Any]) -> dict[str, float]:
    observations = np.asarray(bc_probe["observations"], dtype=np.float32)
    bc_actions = np.asarray(bc_probe["bc_actions"], dtype=np.float32)
    ppo_actions = ppo_mean_actions(model, observations)
    diffs = ppo_actions - bc_actions
    l2 = np.linalg.norm(diffs, axis=1)
    numerator = np.sum(ppo_actions * bc_actions, axis=1)
    denominator = np.linalg.norm(ppo_actions, axis=1) * np.linalg.norm(bc_actions, axis=1) + 1e-8
    cosine = numerator / denominator
    return {
        "bc_probe_action_l2": float(np.mean(l2)),
        "bc_probe_action_cosine": float(np.mean(cosine)),
        "bc_probe_max_abs_action_diff": float(np.max(np.abs(diffs))),
        "bc_probe_action_saturation_rate": float(np.mean(np.abs(ppo_actions) > 0.95)),
    }


def build_bc_probe(
    task_variant: str,
    bc_checkpoint: Path,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    seed: int,
    episodes: int = 6,
    max_steps_per_episode: int = 180,
    stride: int = 10,
) -> dict[str, Any]:
    bc_model, _, _, _ = load_bc_checkpoint(bc_checkpoint, torch.device("cpu"))
    bc_model.eval()
    env = make_env_instance(task_variant, gui=False)
    observations: list[np.ndarray] = []
    try:
        for episode_idx in range(episodes):
            obs, _ = env.reset(seed=seed + episode_idx)
            for step_idx in range(max_steps_per_episode):
                if step_idx % stride == 0:
                    observations.append(np.asarray(obs, dtype=np.float32).copy())
                obs, _, terminated, truncated, _ = env.step(expert_action_for_env(env, task_variant, obs))
                if terminated or truncated:
                    break
    finally:
        env.close()

    raw_obs = np.stack(observations).astype(np.float32)
    normalized_obs = ((raw_obs - obs_mean) / obs_std).astype(np.float32)
    with torch.no_grad():
        bc_actions = bc_model(torch.as_tensor(normalized_obs, dtype=torch.float32)).cpu().numpy()
    return {
        "observations": normalized_obs,
        "bc_actions": bc_actions.astype(np.float32),
        "num_states": int(len(normalized_obs)),
        "seed": int(seed),
        "episodes": int(episodes),
        "max_steps_per_episode": int(max_steps_per_episode),
        "stride": int(stride),
    }


def build_expert_bc_training_arrays(
    observations: np.ndarray,
    actions: np.ndarray,
    task_variant: str,
    augment_copies: int = 0,
    position_noise_std: float = 0.0,
    velocity_noise_std: float = 0.0,
    rpy_noise_std: float = 0.0,
    angular_velocity_noise_std: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if augment_copies <= 0:
        return observations.copy(), actions.copy()
    if task_variant in {"detour_planar_local", "detour_planar_raycast"}:
        raise ValueError(f"Expert-state augmentation is not implemented for {task_variant}'s observation layout.")

    rng = np.random.default_rng(seed)
    expert = expert_for_task_variant(task_variant)
    obs_batches = [observations.copy()]
    action_batches = [actions.copy()]

    for _ in range(int(augment_copies)):
        augmented_obs = observations.copy()
        if position_noise_std > 0.0:
            augmented_obs[:, 0:3] += rng.normal(0.0, position_noise_std, size=augmented_obs[:, 0:3].shape)
        if rpy_noise_std > 0.0:
            augmented_obs[:, 3:6] += rng.normal(0.0, rpy_noise_std, size=augmented_obs[:, 3:6].shape)
        if velocity_noise_std > 0.0:
            augmented_obs[:, 6:9] += rng.normal(0.0, velocity_noise_std, size=augmented_obs[:, 6:9].shape)
        if angular_velocity_noise_std > 0.0:
            augmented_obs[:, 9:12] += rng.normal(
                0.0,
                angular_velocity_noise_std,
                size=augmented_obs[:, 9:12].shape,
            )

        augmented_obs[:, 0] = np.clip(augmented_obs[:, 0], -0.75, 0.75)
        augmented_obs[:, 1] = np.clip(augmented_obs[:, 1], -0.75, 0.75)
        augmented_obs[:, 2] = np.clip(augmented_obs[:, 2], 0.10, 1.10)
        augmented_obs[:, 15:18] = augmented_obs[:, 12:15] - augmented_obs[:, 0:3]

        valid_mask = _valid_augmented_observation_mask(augmented_obs, task_variant)
        augmented_obs = augmented_obs[valid_mask]
        if len(augmented_obs) == 0:
            continue
        augmented_actions = np.stack([expert(obs) for obs in augmented_obs]).astype(np.float32)
        obs_batches.append(augmented_obs.astype(np.float32))
        action_batches.append(augmented_actions)

    return np.concatenate(obs_batches, axis=0), np.concatenate(action_batches, axis=0)


def _valid_augmented_observation_mask(observations: np.ndarray, task_variant: str) -> np.ndarray:
    pos = observations[:, 0:3]
    mask = (
        (np.abs(pos[:, 0]) <= 0.75)
        & (np.abs(pos[:, 1]) <= 0.75)
        & (pos[:, 2] >= 0.10)
        & (pos[:, 2] <= 1.10)
    )
    if task_variant not in {"detour", "detour_planar", "detour_planar_local", "detour_planar_raycast"}:
        return mask

    wall_x = 0.0
    wall_half_thickness = 0.05
    gap_center_y = 0.50
    gap_half_width = 0.18
    wall_z_center = 0.55
    wall_half_height = 0.55
    inside_wall_x = np.abs(pos[:, 0] - wall_x) <= wall_half_thickness
    outside_gap_y = np.abs(pos[:, 1] - gap_center_y) > gap_half_width
    inside_wall_z = np.abs(pos[:, 2] - wall_z_center) <= wall_half_height
    inside_blocking_wall = inside_wall_x & outside_gap_y & inside_wall_z
    return mask & ~inside_blocking_wall


class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_freq: int,
        eval_episodes: int,
        eval_seed: int,
        run_dir: Path,
        task_variant: str = "waypoint",
        obs_mean: np.ndarray | None = None,
        obs_std: np.ndarray | None = None,
        bc_probe: dict[str, Any] | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.eval_seed = int(eval_seed)
        self.run_dir = run_dir
        self.task_variant = task_variant
        self.obs_mean = None if obs_mean is None else np.asarray(obs_mean, dtype=np.float32)
        self.obs_std = None if obs_std is None else np.asarray(obs_std, dtype=np.float32)
        self.bc_probe = bc_probe
        self.eval_history: list[EvalRecord] = []
        self.best_success_rate = -float("inf")
        self.best_final_distance = float("inf")
        self.best_episode_return = -float("inf")
        self.best_record: EvalRecord | None = None
        self.history_path = self.run_dir / "eval_history.json"
        self.best_model_path = self.run_dir / "best_model.zip"
        self.best_record_path = self.run_dir / "best_eval.json"
        self.next_eval_timestep = self.eval_freq if self.eval_freq > 0 else 0

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps < self.next_eval_timestep:
            return True

        record, _ = evaluate_model(
            self.model,
            self.eval_episodes,
            self.eval_seed,
            task_variant=self.task_variant,
            obs_mean=self.obs_mean,
            obs_std=self.obs_std,
            bc_probe=self.bc_probe,
        )
        record.timesteps = int(self.num_timesteps)
        self.eval_history.append(record)
        self._save_history()

        print(json.dumps(asdict(record)))
        if self._is_better_record(record):
            self.best_success_rate = record.success_rate
            self.best_final_distance = record.mean_final_distance
            self.best_episode_return = record.mean_episode_return
            self.best_record = record
            self.model.save(self.best_model_path)
            self.best_record_path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        self.next_eval_timestep += self.eval_freq
        return True

    def _is_better_record(self, record: EvalRecord) -> bool:
        eps = 1e-9
        if record.success_rate > self.best_success_rate + eps:
            return True
        if record.success_rate < self.best_success_rate - eps:
            return False
        if record.mean_final_distance < self.best_final_distance - eps:
            return True
        if record.mean_final_distance > self.best_final_distance + eps:
            return False
        return record.mean_episode_return > self.best_episode_return + eps

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


def build_training_env(
    gui: bool,
    seed: int,
    task_variant: str = "waypoint",
    obs_mean: np.ndarray | None = None,
    obs_std: np.ndarray | None = None,
    n_envs: int = 1,
) -> DummyVecEnv | SubprocVecEnv:
    n_envs = int(n_envs)
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")

    def make_env(rank: int = 0):
        env_seed = seed + rank
        env = make_env_instance(task_variant, gui=gui)
        if obs_mean is not None and obs_std is not None:
            env = FixedObservationNormalization(env, obs_mean=obs_mean, obs_std=obs_std)
        env.reset(seed=env_seed)
        return env

    env_fns = [lambda rank=rank: make_env(rank) for rank in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    if gui:
        raise ValueError("Parallel PPO training requires --gui to be disabled.")
    return SubprocVecEnv(env_fns, start_method="spawn")


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
    if bool(metadata.get("squash_output", True)):
        raise ValueError(
            "BC checkpoint uses a tanh-squashed output head. Retrain BC with linear output before PPO initialization."
        )
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
        model.set_bc_kl_prior(bc_model, coef=bc_kl_coef, expects_normalized_obs=True)
    return {
        "bc_checkpoint": str(bc_checkpoint),
        "bc_dataset_path": str(metadata.get("dataset_path")),
        "copied_layers": json.dumps(mapping),
        "bc_kl_coef": str(float(bc_kl_coef)),
        "obs_mean": json.dumps(obs_mean.astype(float).tolist()),
        "obs_std": json.dumps(obs_std.astype(float).tolist()),
    }


def build_bc_fine_tune_callback(
    eval_freq: int,
    eval_episodes: int,
    eval_seed: int,
    run_dir: Path,
    task_variant: str,
    obs_mean: np.ndarray | None,
    obs_std: np.ndarray | None,
    freeze_actor_steps: int,
    freeze_actor_mode: str,
    bc_probe: dict[str, Any] | None = None,
) -> tuple[BaseCallback, PeriodicEvalCallback]:
    callbacks: list[BaseCallback] = []
    eval_callback = PeriodicEvalCallback(
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        eval_seed=eval_seed,
        run_dir=run_dir,
        task_variant=task_variant,
        obs_mean=obs_mean,
        obs_std=obs_std,
        bc_probe=bc_probe,
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
