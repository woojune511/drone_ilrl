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


def detour_waypoint_velocity_expert(
    observation: np.ndarray,
    kp: float = 1.8,
    kd: float = 0.2,
    max_speed: float = 0.25,
    stop_radius: float = 0.05,
    corridor_center_y: float = 0.50,
    corridor_z: float = 0.60,
    wall_x: float = 0.0,
    entry_x: float = -0.18,
    exit_x: float = 0.22,
    corridor_y_tolerance: float = 0.10,
    entry_margin_x: float = 0.03,
    exit_margin_x: float = 0.03,
    intermediate_stop_radius: float = 0.015,
) -> np.ndarray:
    """A simple obstacle-aware expert for the fixed detour task variant."""

    obs = np.asarray(observation, dtype=np.float32)
    pos = obs[0:3]
    vel = obs[6:9]
    goal = obs[12:15]

    target = goal.copy()
    detour_height = max(float(goal[2]), corridor_z)
    corridor_aligned = abs(float(pos[1]) - corridor_center_y) <= corridor_y_tolerance
    using_intermediate_target = False

    if goal[0] > wall_x:
        if pos[0] < wall_x - 0.05:
            if not corridor_aligned:
                target = np.array([entry_x, corridor_center_y, detour_height], dtype=np.float32)
                using_intermediate_target = True
            else:
                target = np.array([exit_x, corridor_center_y, detour_height], dtype=np.float32)
                using_intermediate_target = True
        elif pos[0] < exit_x - exit_margin_x:
            target = np.array([exit_x, corridor_center_y, detour_height], dtype=np.float32)
            using_intermediate_target = True

    if using_intermediate_target and target[0] <= entry_x + 1e-6:
        if pos[0] >= entry_x - entry_margin_x and corridor_aligned:
            target = np.array([exit_x, corridor_center_y, detour_height], dtype=np.float32)
    elif using_intermediate_target and target[0] >= exit_x - 1e-6:
        if pos[0] >= exit_x - exit_margin_x:
            target = goal.copy()
            using_intermediate_target = False

    rel_goal = target - pos
    distance = float(np.linalg.norm(rel_goal))
    active_stop_radius = intermediate_stop_radius if using_intermediate_target else stop_radius
    if distance < active_stop_radius:
        return np.zeros(4, dtype=np.float32)

    desired_velocity = kp * rel_goal - kd * vel
    speed = float(np.linalg.norm(desired_velocity))
    if speed < 1e-6:
        return np.zeros(4, dtype=np.float32)

    direction = desired_velocity / speed
    speed_scale = np.clip(speed / max_speed, 0.0, 1.0)
    action = np.concatenate([direction, np.array([speed_scale], dtype=np.float32)])
    return np.clip(action, -1.0, 1.0).astype(np.float32)
