from __future__ import annotations

import numpy as np

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


def test_waypoint_env_expert_smoke() -> None:
    env = WaypointVelocityAviary(gui=False, episode_len_sec=2.0)
    obs, info = env.reset(seed=0)

    assert obs.shape == env.observation_space.shape
    assert "goal" in info
    assert "distance_to_goal" in info

    for _ in range(8):
        action = waypoint_velocity_expert(obs)
        assert action.shape == env.action_space.shape
        assert np.all(action <= env.action_space.high + 1e-6)
        assert np.all(action >= env.action_space.low - 1e-6)

        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        if terminated or truncated:
            break

    env.close()


def test_detour_waypoint_env_expert_smoke() -> None:
    env = DetourWaypointVelocityAviary(gui=False, episode_len_sec=2.0)
    obs, info = env.reset(seed=0)

    assert obs.shape == env.observation_space.shape
    assert info["task_variant"] == "detour_waypoint"
    assert "collision" in info

    for _ in range(8):
        action = detour_waypoint_velocity_expert(obs)
        assert action.shape == env.action_space.shape
        assert np.all(action <= env.action_space.high + 1e-6)
        assert np.all(action >= env.action_space.low - 1e-6)

        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        if terminated or truncated:
            break

    env.close()


def test_detour_planar_env_expert_smoke() -> None:
    env = DetourPlanarVelocityAviary(gui=False, episode_len_sec=2.0)
    obs, info = env.reset(seed=0)

    assert obs.shape == env.observation_space.shape
    assert env.action_space.shape == (3,)
    assert info["task_variant"] == "detour_planar_velocity"
    assert "hold_altitude" in info
    assert "planar_body_velocity_cmd" in info
    assert "yaw_rate_cmd" in info

    for _ in range(8):
        action = detour_planar_velocity_expert(obs)
        assert action.shape == env.action_space.shape
        assert np.all(action <= env.action_space.high + 1e-6)
        assert np.all(action >= env.action_space.low - 1e-6)

        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        if terminated or truncated:
            break

    env.close()


def test_detour_planar_local_obs_env_expert_smoke() -> None:
    env = DetourPlanarLocalObsAviary(gui=False, episode_len_sec=2.0)
    obs, info = env.reset(seed=0)

    assert obs.shape == env.observation_space.shape
    assert env.observation_space.shape == (15,)
    assert env.action_space.shape == (3,)
    assert info["task_variant"] == "detour_planar_local_obs"
    assert info["observation_variant"] == "local_detour_target"

    for _ in range(8):
        action = detour_planar_local_obs_expert(obs)
        assert action.shape == env.action_space.shape
        assert np.all(action <= env.action_space.high + 1e-6)
        assert np.all(action >= env.action_space.low - 1e-6)

        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        if terminated or truncated:
            break

    env.close()


def test_detour_planar_raycast_env_privileged_expert_smoke() -> None:
    env = DetourPlanarRaycastAviary(gui=False, episode_len_sec=2.0)
    obs, info = env.reset(seed=0)

    assert obs.shape == env.observation_space.shape
    assert env.observation_space.shape == (33,)
    assert env.action_space.shape == (3,)
    assert info["task_variant"] == "detour_planar_raycast"
    assert info["observation_variant"] == "raycast_depth"
    assert "raycast_min_distance_norm" in info

    for _ in range(8):
        action = env.privileged_expert_action()
        assert action.shape == env.action_space.shape
        assert np.all(action <= env.action_space.high + 1e-6)
        assert np.all(action >= env.action_space.low - 1e-6)

        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert np.all(obs[-env.ray_count :] >= -1e-6)
        assert np.all(obs[-env.ray_count :] <= 1.0 + 1e-6)
        assert isinstance(reward, float)
        if terminated or truncated:
            break

    env.close()
