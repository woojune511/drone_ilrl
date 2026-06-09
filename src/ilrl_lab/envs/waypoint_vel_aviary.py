from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    ActionType,
    DroneModel,
    ObservationType,
    Physics,
)


@dataclass
class WorkspaceBounds:
    xy_limit: float = 0.75
    z_min: float = 0.20
    z_max: float = 1.10


class WaypointVelocityAviary(BaseRLAviary):
    """Single-drone waypoint task with goal-aware observations and velocity actions.

    The environment exposes a compact 18D observation:
    position (3), roll/pitch/yaw (3), linear velocity (3), angular velocity (3),
    goal position (3), and relative goal vector (3).

    The action is a 4D velocity command:
    normalized xyz direction (3) plus a speed scale in [0, 1].
    Low-level motor control is still handled by the integrated PID controller
    inside ``gym-pybullet-drones``.
    """

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui: bool = False,
        record: bool = False,
        workspace: WorkspaceBounds | None = None,
        episode_len_sec: float = 8.0,
        goal_tolerance: float = 0.10,
        success_speed_threshold: float = 0.15,
        min_goal_distance: float = 0.35,
        progress_reward_scale: float = 0.75,
        near_goal_radius_multiplier: float = 2.0,
        near_goal_position_bonus: float = 0.2,
        near_goal_speed_penalty: float = 0.1,
        near_goal_settle_bonus: float = 0.2,
        target_pos: np.ndarray | None = None,
        initial_xyzs: np.ndarray | None = None,
    ) -> None:
        self.workspace = workspace or WorkspaceBounds()
        self.EPISODE_LEN_SEC = float(episode_len_sec)
        self.goal_tolerance = float(goal_tolerance)
        self.success_speed_threshold = float(success_speed_threshold)
        self.min_goal_distance = float(min_goal_distance)
        self.progress_reward_scale = float(progress_reward_scale)
        self.near_goal_radius_multiplier = float(near_goal_radius_multiplier)
        self.near_goal_position_bonus = float(near_goal_position_bonus)
        self.near_goal_speed_penalty = float(near_goal_speed_penalty)
        self.near_goal_settle_bonus = float(near_goal_settle_bonus)
        self._prev_distance_to_goal: float | None = None

        self._rng = np.random.default_rng()
        self._fixed_target_pos = None if target_pos is None else np.asarray(target_pos, dtype=np.float32)
        self._fixed_initial_xyz = None if initial_xyzs is None else np.asarray(initial_xyzs, dtype=np.float32)
        self.target_pos = (
            np.array([0.5, 0.0, 0.8], dtype=np.float32)
            if self._fixed_target_pos is None
            else self._fixed_target_pos.astype(np.float32).copy()
        )

        init_xyzs = (
            np.array([[0.0, 0.0, 0.3]], dtype=np.float32)
            if self._fixed_initial_xyz is None
            else np.asarray(initial_xyzs, dtype=np.float32).reshape(1, 3)
        )

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=init_xyzs,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=ObservationType.KIN,
            act=ActionType.VEL,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._sample_start_and_goal()
        obs, info = super().reset(seed=seed, options=options)
        self._prev_distance_to_goal = float(np.linalg.norm(self.target_pos - obs[0:3]))
        return obs, {**info, **self._build_info(success=False)}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(4)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs, reward, terminated, truncated, info = super().step(action[np.newaxis, :])
        info = {**info, **self._build_info(success=bool(terminated))}
        self._prev_distance_to_goal = float(info["distance_to_goal"])
        return obs, reward, terminated, truncated, info

    def _actionSpace(self):
        return spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _observationSpace(self):
        lo = np.full(18, -np.inf, dtype=np.float32)
        hi = np.full(18, np.inf, dtype=np.float32)
        lo[2] = 0.0
        lo[14] = 0.0
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        ang_vel = state[13:16]
        rel_goal = self.target_pos - pos
        obs = np.concatenate([pos, rpy, vel, ang_vel, self.target_pos, rel_goal])
        return obs.astype(np.float32)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        distance = np.linalg.norm(self.target_pos - state[0:3])
        speed = np.linalg.norm(state[10:13])
        reward = 1.0 - np.tanh(2.0 * distance) - 0.02 * speed

        if self._prev_distance_to_goal is not None:
            progress = self._prev_distance_to_goal - float(distance)
            reward += self.progress_reward_scale * progress

        near_goal_radius = self.goal_tolerance * self.near_goal_radius_multiplier
        if distance < near_goal_radius:
            proximity = 1.0 - float(distance / near_goal_radius)
            reward += self.near_goal_position_bonus * proximity
            reward -= self.near_goal_speed_penalty * proximity * float(speed)
            if speed < self.success_speed_threshold:
                speed_ratio = 1.0 - float(speed / self.success_speed_threshold)
                reward += self.near_goal_settle_bonus * proximity * max(speed_ratio, 0.0)

        if distance < self.goal_tolerance and speed < self.success_speed_threshold:
            reward += 2.0
        return float(reward)

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        distance = np.linalg.norm(self.target_pos - state[0:3])
        speed = np.linalg.norm(state[10:13])
        return bool(distance < self.goal_tolerance and speed < self.success_speed_threshold)

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        x, y, z = state[0:3]
        roll, pitch = state[7], state[8]

        if abs(x) > self.workspace.xy_limit + 0.35 or abs(y) > self.workspace.xy_limit + 0.35:
            return True
        if z < 0.05 or z > self.workspace.z_max + 0.35:
            return True
        if abs(roll) > 0.75 or abs(pitch) > 0.75:
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        return {}

    def _sample_start_and_goal(self) -> None:
        if self._fixed_initial_xyz is not None:
            init_xyz = self._fixed_initial_xyz.reshape(1, 3).astype(np.float32)
        else:
            init_xyz = np.array([[0.0, 0.0, 0.3]], dtype=np.float32)
            init_xyz[0, 0] = self._rng.uniform(-0.15, 0.15)
            init_xyz[0, 1] = self._rng.uniform(-0.15, 0.15)
            init_xyz[0, 2] = self._rng.uniform(0.20, 0.45)

        if self._fixed_target_pos is not None:
            goal = self._fixed_target_pos.astype(np.float32).copy()
        else:
            goal = self._sample_goal_far_from(init_xyz[0])

        self.INIT_XYZS = init_xyz
        self.INIT_RPYS = np.zeros((1, 3), dtype=np.float32)
        self.target_pos = goal

    def _sample_goal_far_from(self, start_xyz: np.ndarray) -> np.ndarray:
        for _ in range(256):
            goal = np.array(
                [
                    self._rng.uniform(-self.workspace.xy_limit, self.workspace.xy_limit),
                    self._rng.uniform(-self.workspace.xy_limit, self.workspace.xy_limit),
                    self._rng.uniform(self.workspace.z_min, self.workspace.z_max),
                ],
                dtype=np.float32,
            )
            if np.linalg.norm(goal - start_xyz) >= self.min_goal_distance:
                return goal
        return np.array([0.5, 0.0, 0.8], dtype=np.float32)

    def _build_info(self, success: bool) -> dict[str, float | bool | list[float]]:
        state = self._getDroneStateVector(0)
        distance = float(np.linalg.norm(self.target_pos - state[0:3]))
        speed = float(np.linalg.norm(state[10:13]))
        return {
            "success": success,
            "distance_to_goal": distance,
            "speed": speed,
            "goal": self.target_pos.astype(float).tolist(),
            "position": state[0:3].astype(float).tolist(),
        }
