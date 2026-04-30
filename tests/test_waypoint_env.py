from __future__ import annotations

import numpy as np

from ilrl_lab.envs import WaypointVelocityAviary
from ilrl_lab.experts import waypoint_velocity_expert


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
