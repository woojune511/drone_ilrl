from __future__ import annotations

import numpy as np


def waypoint_velocity_expert(
    observation: np.ndarray,
    kp: float = 1.25,
    kd: float = 0.35,
    max_speed: float = 1.0,
    stop_radius: float = 0.05,
) -> np.ndarray:
    """A simple PD-style expert over relative goal displacement.

    Observation layout:
    pos(3), rpy(3), vel(3), ang_vel(3), goal(3), rel_goal(3)
    """

    obs = np.asarray(observation, dtype=np.float32)
    vel = obs[6:9]
    rel_goal = obs[15:18]
    distance = float(np.linalg.norm(rel_goal))

    if distance < stop_radius:
        return np.zeros(4, dtype=np.float32)

    desired_velocity = kp * rel_goal - kd * vel
    speed = float(np.linalg.norm(desired_velocity))
    if speed < 1e-6:
        return np.zeros(4, dtype=np.float32)

    direction = desired_velocity / speed
    speed_scale = np.clip(speed / max_speed, 0.0, 1.0)
    action = np.concatenate([direction, np.array([speed_scale], dtype=np.float32)])
    return np.clip(action, -1.0, 1.0).astype(np.float32)
