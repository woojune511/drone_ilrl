from __future__ import annotations

import numpy as np


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


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


def detour_planar_velocity_expert(
    observation: np.ndarray,
    kp: float = 1.8,
    kd: float = 0.25,
    yaw_kp: float = 1.5,
    max_planar_speed: float = 0.35,
    max_yaw_rate: float = np.pi / 3.0,
    stop_radius: float = 0.05,
    corridor_center_y: float = 0.50,
    wall_x: float = 0.0,
    entry_x: float = -0.18,
    exit_x: float = 0.22,
    corridor_y_tolerance: float = 0.10,
    exit_margin_x: float = 0.03,
    intermediate_stop_radius: float = 0.015,
) -> np.ndarray:
    """Expert for the deployment-oriented 3D planar velocity action space.

    Returns normalized body-frame vx, body-frame vy, and yaw-rate.
    Altitude is controlled by the environment, so the expert plans in xy.
    """

    obs = np.asarray(observation, dtype=np.float32)
    pos = obs[0:3]
    yaw = float(obs[5])
    vel_xy = obs[6:8]
    goal = obs[12:15]

    target_xy = goal[0:2].copy()
    using_intermediate_target = False
    corridor_aligned = abs(float(pos[1]) - corridor_center_y) <= corridor_y_tolerance

    if goal[0] > wall_x:
        if pos[0] < wall_x - 0.05:
            if not corridor_aligned:
                target_xy = np.array([entry_x, corridor_center_y], dtype=np.float32)
                using_intermediate_target = True
            else:
                target_xy = np.array([exit_x, corridor_center_y], dtype=np.float32)
                using_intermediate_target = True
        elif pos[0] < exit_x - exit_margin_x:
            target_xy = np.array([exit_x, corridor_center_y], dtype=np.float32)
            using_intermediate_target = True

    rel_xy = target_xy - pos[0:2]
    distance = float(np.linalg.norm(rel_xy))
    active_stop_radius = intermediate_stop_radius if using_intermediate_target else stop_radius
    if distance < active_stop_radius:
        return np.zeros(3, dtype=np.float32)

    desired_world_xy = kp * rel_xy - kd * vel_xy
    speed = float(np.linalg.norm(desired_world_xy))
    if speed > max_planar_speed:
        desired_world_xy = desired_world_xy / speed * max_planar_speed
        speed = max_planar_speed
    if speed < 1e-6:
        return np.zeros(3, dtype=np.float32)

    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    body_vx = cos_yaw * desired_world_xy[0] + sin_yaw * desired_world_xy[1]
    body_vy = -sin_yaw * desired_world_xy[0] + cos_yaw * desired_world_xy[1]

    desired_heading = float(np.arctan2(desired_world_xy[1], desired_world_xy[0]))
    yaw_error = _wrap_angle(desired_heading - yaw)
    yaw_rate = np.clip(yaw_kp * yaw_error, -max_yaw_rate, max_yaw_rate)

    action = np.array(
        [
            body_vx / max_planar_speed,
            body_vy / max_planar_speed,
            yaw_rate / max_yaw_rate,
        ],
        dtype=np.float32,
    )
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def detour_planar_local_obs_expert(
    observation: np.ndarray,
    kp: float = 1.8,
    kd: float = 0.25,
    yaw_kp: float = 1.5,
    max_planar_speed: float = 0.35,
    max_yaw_rate: float = np.pi / 3.0,
    stop_radius: float = 0.05,
) -> np.ndarray:
    """Expert for DetourPlanarLocalObsAviary's 15D local observation."""

    obs = np.asarray(observation, dtype=np.float32)
    body_vel_xy = obs[0:2]
    rel_detour_target_body = obs[9:12]
    target_xy_body = rel_detour_target_body[0:2]
    distance = float(np.linalg.norm(rel_detour_target_body))

    if distance < stop_radius:
        return np.zeros(3, dtype=np.float32)

    desired_body_xy = kp * target_xy_body - kd * body_vel_xy
    speed = float(np.linalg.norm(desired_body_xy))
    if speed > max_planar_speed:
        desired_body_xy = desired_body_xy / speed * max_planar_speed
        speed = max_planar_speed
    if speed < 1e-6:
        return np.zeros(3, dtype=np.float32)

    yaw_error = float(np.arctan2(desired_body_xy[1], desired_body_xy[0]))
    yaw_rate = np.clip(yaw_kp * yaw_error, -max_yaw_rate, max_yaw_rate)
    action = np.array(
        [
            desired_body_xy[0] / max_planar_speed,
            desired_body_xy[1] / max_planar_speed,
            yaw_rate / max_yaw_rate,
        ],
        dtype=np.float32,
    )
    return np.clip(action, -1.0, 1.0).astype(np.float32)
