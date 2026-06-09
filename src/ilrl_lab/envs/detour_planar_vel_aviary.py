from __future__ import annotations

import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary

from ilrl_lab.envs.detour_vel_aviary import DetourWaypointVelocityAviary


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class DetourPlanarVelocityAviary(DetourWaypointVelocityAviary):
    """Detour task with a deployment-oriented planar velocity interface.

    The policy-facing action is 3D:
    body-frame vx, body-frame vy, yaw-rate.

    Altitude is held by the low-level PID controller, and command changes are
    rate-limited before being sent to the controller. This keeps navigation
    learning separate from altitude stabilization.
    """

    def __init__(
        self,
        *args,
        hold_altitude: float = 0.55,
        max_planar_speed: float = 0.35,
        max_yaw_rate: float = np.pi / 3.0,
        max_planar_accel: float = 0.80,
        max_yaw_accel: float = np.pi,
        max_vertical_speed: float = 0.25,
        altitude_kp: float = 1.2,
        altitude_kd: float = 0.25,
        **kwargs,
    ) -> None:
        self.hold_altitude = float(hold_altitude)
        self.max_planar_speed = float(max_planar_speed)
        self.max_yaw_rate = float(max_yaw_rate)
        self.max_planar_accel = float(max_planar_accel)
        self.max_yaw_accel = float(max_yaw_accel)
        self.max_vertical_speed = float(max_vertical_speed)
        self.altitude_kp = float(altitude_kp)
        self.altitude_kd = float(altitude_kd)
        self._last_body_velocity_cmd = np.zeros(2, dtype=np.float32)
        self._last_yaw_rate_cmd = 0.0
        self._last_policy_action = np.zeros(3, dtype=np.float32)
        super().__init__(*args, **kwargs)

    def reset(self, seed: int | None = None, options: dict | None = None):
        self._last_body_velocity_cmd = np.zeros(2, dtype=np.float32)
        self._last_yaw_rate_cmd = 0.0
        self._last_policy_action = np.zeros(3, dtype=np.float32)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(3)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._last_policy_action = action.astype(np.float32).copy()

        obs, reward, terminated, truncated, info = BaseRLAviary.step(self, action[np.newaxis, :])
        info = {**info, **self._build_info(success=bool(terminated))}
        self._prev_distance_to_goal = float(info["distance_to_goal"])

        pos = np.asarray(self._getDroneStateVector(0)[0:3], dtype=np.float32)
        stage, target = self._detour_navigation_target(pos)
        self._prev_detour_stage = stage
        self._prev_detour_distance = float(np.linalg.norm(target - pos))
        info["detour_stage"] = stage
        info["detour_target"] = target.astype(float).tolist()
        return obs, reward, terminated, truncated, info

    def _actionSpace(self):
        return spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _sample_start_and_goal(self) -> None:
        if self._fixed_initial_xyz is not None:
            init_xyz = self._fixed_initial_xyz.reshape(1, 3).astype(np.float32)
            init_xyz[0, 2] = self.hold_altitude
        else:
            init_xyz = np.array([[0.0, 0.0, self.hold_altitude]], dtype=np.float32)
            init_xyz[0, 0] = self._rng.uniform(-0.65, -0.45)
            init_xyz[0, 1] = self._rng.uniform(-0.10, 0.10)

        if self._fixed_target_pos is not None:
            goal = self._fixed_target_pos.astype(np.float32).copy()
            goal[2] = self.hold_altitude
        else:
            goal = np.array(
                [
                    self._rng.uniform(0.45, 0.65),
                    self._rng.uniform(-0.10, 0.10),
                    self.hold_altitude,
                ],
                dtype=np.float32,
            )

        self.INIT_XYZS = init_xyz
        self.INIT_RPYS = np.zeros((1, 3), dtype=np.float32)
        self.target_pos = goal

    def _preprocessAction(self, action):
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES, 4))
        dt = float(self.CTRL_TIMESTEP)

        for k in range(action.shape[0]):
            target = np.asarray(action[k, :], dtype=np.float32)
            desired_body_velocity = target[0:2] * self.max_planar_speed
            desired_yaw_rate = float(target[2] * self.max_yaw_rate)

            max_delta_v = self.max_planar_accel * dt
            velocity_delta = np.clip(
                desired_body_velocity - self._last_body_velocity_cmd,
                -max_delta_v,
                max_delta_v,
            )
            body_velocity_cmd = self._last_body_velocity_cmd + velocity_delta

            max_delta_yaw_rate = self.max_yaw_accel * dt
            yaw_rate_cmd = float(
                np.clip(
                    desired_yaw_rate - self._last_yaw_rate_cmd,
                    -max_delta_yaw_rate,
                    max_delta_yaw_rate,
                )
                + self._last_yaw_rate_cmd
            )

            state = self._getDroneStateVector(k)
            yaw = float(state[9])
            cos_yaw = float(np.cos(yaw))
            sin_yaw = float(np.sin(yaw))
            target_vel = np.array(
                [
                    cos_yaw * body_velocity_cmd[0] - sin_yaw * body_velocity_cmd[1],
                    sin_yaw * body_velocity_cmd[0] + cos_yaw * body_velocity_cmd[1],
                    self._altitude_hold_velocity(state),
                ],
                dtype=np.float32,
            )
            target_yaw = _wrap_angle(yaw + yaw_rate_cmd * dt)

            rpm_k, _, _ = self.ctrl[k].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=np.array([state[0], state[1], self.hold_altitude], dtype=np.float32),
                target_rpy=np.array([0.0, 0.0, target_yaw], dtype=np.float32),
                target_vel=target_vel,
                target_rpy_rates=np.array([0.0, 0.0, yaw_rate_cmd], dtype=np.float32),
            )
            rpm[k, :] = rpm_k

            self._last_body_velocity_cmd = body_velocity_cmd.astype(np.float32)
            self._last_yaw_rate_cmd = yaw_rate_cmd

        return rpm

    def _altitude_hold_velocity(self, state: np.ndarray) -> float:
        z_error = self.hold_altitude - float(state[2])
        z_velocity = float(state[12])
        command = self.altitude_kp * z_error - self.altitude_kd * z_velocity
        return float(np.clip(command, -self.max_vertical_speed, self.max_vertical_speed))

    def _build_info(self, success: bool) -> dict[str, float | bool | list[float]]:
        info = super()._build_info(success)
        info["task_variant"] = "detour_planar_velocity"
        info["hold_altitude"] = self.hold_altitude
        info["planar_body_velocity_cmd"] = self._last_body_velocity_cmd.astype(float).tolist()
        info["yaw_rate_cmd"] = float(self._last_yaw_rate_cmd)
        info["policy_action"] = self._last_policy_action.astype(float).tolist()
        return info

    def _has_obstacle_collision(self) -> bool:
        if not self._detour_obstacle_ids:
            return False
        contacts = p.getContactPoints(bodyA=int(self.DRONE_IDS[0]), physicsClientId=self.CLIENT)
        obstacle_ids = set(self._detour_obstacle_ids)
        for contact in contacts:
            if int(contact[2]) in obstacle_ids:
                return True
        return False

    def privileged_observation(self) -> np.ndarray:
        """Return the original full-state observation for privileged teachers."""

        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        ang_vel = state[13:16]
        rel_goal = self.target_pos - pos
        obs = np.concatenate([pos, rpy, vel, ang_vel, self.target_pos, rel_goal])
        return obs.astype(np.float32)

    def privileged_expert_action(self) -> np.ndarray:
        from ilrl_lab.experts import detour_planar_velocity_expert

        return detour_planar_velocity_expert(self.privileged_observation())


class DetourPlanarLocalObsAviary(DetourPlanarVelocityAviary):
    """Planar detour task with local, route-conditioned observations.

    This variant removes absolute position and absolute goal from the policy
    observation. It keeps a local detour target vector, which corresponds to a
    simple upstream route planner rather than raw perception.

    Observation layout:
    body_vel_xy(2), altitude_error(1), z_vel(1), sin_yaw/cos_yaw(2),
    rel_goal_body_xyz(3), rel_detour_target_body_xyz(3), previous_action(3).
    """

    def _observationSpace(self):
        return spaces.Box(
            low=np.full(15, -np.inf, dtype=np.float32),
            high=np.full(15, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3].astype(np.float32)
        yaw = float(state[9])
        vel = state[10:13].astype(np.float32)

        stage, detour_target = self._detour_navigation_target(pos)
        _ = stage

        body_vel_xy = self._world_xy_to_body_xy(vel[0:2], yaw)
        rel_goal_body = self._world_xyz_to_body_xyz(self.target_pos - pos, yaw)
        rel_detour_target_body = self._world_xyz_to_body_xyz(detour_target - pos, yaw)
        obs = np.concatenate(
            [
                body_vel_xy,
                np.array([self.hold_altitude - pos[2], vel[2]], dtype=np.float32),
                np.array([np.sin(yaw), np.cos(yaw)], dtype=np.float32),
                rel_goal_body,
                rel_detour_target_body,
                self._last_policy_action.astype(np.float32),
            ]
        )
        return obs.astype(np.float32)

    def _build_info(self, success: bool) -> dict[str, float | bool | list[float]]:
        info = super()._build_info(success)
        info["task_variant"] = "detour_planar_local_obs"
        info["observation_variant"] = "local_detour_target"
        return info

    @staticmethod
    def _world_xy_to_body_xy(vector_xy: np.ndarray, yaw: float) -> np.ndarray:
        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))
        return np.array(
            [
                cos_yaw * vector_xy[0] + sin_yaw * vector_xy[1],
                -sin_yaw * vector_xy[0] + cos_yaw * vector_xy[1],
            ],
            dtype=np.float32,
        )

    @classmethod
    def _world_xyz_to_body_xyz(cls, vector_xyz: np.ndarray, yaw: float) -> np.ndarray:
        body_xy = cls._world_xy_to_body_xy(vector_xyz[0:2], yaw)
        return np.array([body_xy[0], body_xy[1], vector_xyz[2]], dtype=np.float32)


class DetourPlanarRaycastAviary(DetourPlanarVelocityAviary):
    """Planar detour task with low-dimensional depth-ray observations.

    This variant removes the local detour target from the policy observation.
    A privileged scripted expert can still be used to label demonstrations.
    """

    def __init__(
        self,
        *args,
        ray_count: int = 21,
        ray_fov: float = np.pi,
        ray_max_distance: float = 1.5,
        ray_start_offset: float = 0.08,
        **kwargs,
    ) -> None:
        self.ray_count = int(ray_count)
        self.ray_fov = float(ray_fov)
        self.ray_max_distance = float(ray_max_distance)
        self.ray_start_offset = float(ray_start_offset)
        super().__init__(*args, **kwargs)

    def _observationSpace(self):
        obs_dim = 12 + self.ray_count
        return spaces.Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float32),
            high=np.full(obs_dim, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3].astype(np.float32)
        yaw = float(state[9])
        vel = state[10:13].astype(np.float32)

        body_vel_xy = DetourPlanarLocalObsAviary._world_xy_to_body_xy(vel[0:2], yaw)
        rel_goal_body = DetourPlanarLocalObsAviary._world_xyz_to_body_xyz(self.target_pos - pos, yaw)
        rays = self._raycast_distances(pos=pos, yaw=yaw)
        obs = np.concatenate(
            [
                body_vel_xy,
                np.array([self.hold_altitude - pos[2], vel[2]], dtype=np.float32),
                np.array([np.sin(yaw), np.cos(yaw)], dtype=np.float32),
                rel_goal_body,
                self._last_policy_action.astype(np.float32),
                rays,
            ]
        )
        return obs.astype(np.float32)

    def _raycast_distances(self, pos: np.ndarray, yaw: float) -> np.ndarray:
        angles = np.linspace(-0.5 * self.ray_fov, 0.5 * self.ray_fov, self.ray_count)
        from_positions = []
        to_positions = []
        for angle in angles:
            theta = yaw + float(angle)
            direction = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float32)
            start = pos + self.ray_start_offset * direction
            end = start + self.ray_max_distance * direction
            start[2] = self.hold_altitude
            end[2] = self.hold_altitude
            from_positions.append(start.astype(float).tolist())
            to_positions.append(end.astype(float).tolist())

        results = p.rayTestBatch(from_positions, to_positions, physicsClientId=self.CLIENT)
        distances = []
        drone_id = int(self.DRONE_IDS[0])
        for result in results:
            hit_body_id = int(result[0])
            hit_fraction = float(result[2])
            if hit_body_id < 0 or hit_body_id == drone_id:
                distances.append(1.0)
            else:
                distances.append(float(np.clip(hit_fraction, 0.0, 1.0)))
        return np.asarray(distances, dtype=np.float32)

    def _build_info(self, success: bool) -> dict[str, float | bool | list[float]]:
        info = super()._build_info(success)
        state = self._getDroneStateVector(0)
        rays = self._raycast_distances(pos=state[0:3].astype(np.float32), yaw=float(state[9]))
        info["task_variant"] = "detour_planar_raycast"
        info["observation_variant"] = "raycast_depth"
        info["raycast_min_distance_norm"] = float(np.min(rays))
        info["raycast_distances_norm"] = rays.astype(float).tolist()
        return info
