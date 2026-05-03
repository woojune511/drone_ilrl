from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p

from ilrl_lab.envs.waypoint_vel_aviary import WaypointVelocityAviary, WorkspaceBounds


@dataclass
class DetourLayout:
    wall_x: float = 0.0
    gap_center_y: float = 0.50
    gap_half_width: float = 0.18
    wall_half_thickness: float = 0.05
    wall_half_height: float = 0.55
    wall_z_center: float = 0.55
    visual_rgba: tuple[float, float, float, float] = (0.75, 0.25, 0.25, 1.0)


class DetourWaypointVelocityAviary(WaypointVelocityAviary):
    """Waypoint task variant with a blocking wall and a single upper corridor.

    The drone starts on the left side of the workspace and the goal is sampled
    on the right side. A central wall blocks the direct route, leaving only a
    narrow upper passage. This creates a simple but meaningful detour task.
    """

    def __init__(
        self,
        *args,
        layout: DetourLayout | None = None,
        collision_penalty: float = 2.0,
        corridor_z: float = 0.60,
        entry_x: float = -0.18,
        exit_x: float = 0.22,
        corridor_y_tolerance: float = 0.10,
        exit_margin_x: float = 0.03,
        detour_target_reward_scale: float = 0.18,
        detour_progress_reward_scale: float = 0.45,
        detour_stage_transition_bonus: float = 0.15,
        **kwargs,
    ) -> None:
        kwargs.setdefault("episode_len_sec", 14.0)
        self.layout = layout or DetourLayout()
        self.collision_penalty = float(collision_penalty)
        self.corridor_z = float(corridor_z)
        self.entry_x = float(entry_x)
        self.exit_x = float(exit_x)
        self.corridor_y_tolerance = float(corridor_y_tolerance)
        self.exit_margin_x = float(exit_margin_x)
        self.detour_target_reward_scale = float(detour_target_reward_scale)
        self.detour_progress_reward_scale = float(detour_progress_reward_scale)
        self.detour_stage_transition_bonus = float(detour_stage_transition_bonus)
        self._detour_obstacle_ids: list[int] = []
        self._prev_detour_stage: str | None = None
        self._prev_detour_distance: float | None = None
        super().__init__(*args, **kwargs)

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        pos = np.asarray(obs[0:3], dtype=np.float32)
        stage, target = self._detour_navigation_target(pos)
        self._prev_detour_stage = stage
        self._prev_detour_distance = float(np.linalg.norm(target - pos))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        pos = np.asarray(obs[0:3], dtype=np.float32)
        stage, target = self._detour_navigation_target(pos)
        self._prev_detour_stage = stage
        self._prev_detour_distance = float(np.linalg.norm(target - pos))
        info["detour_stage"] = stage
        info["detour_target"] = target.astype(float).tolist()
        return obs, reward, terminated, truncated, info

    def _addObstacles(self):
        self._detour_obstacle_ids = []

        y_min = -self.workspace.xy_limit
        y_max = self.workspace.xy_limit
        gap_low = self.layout.gap_center_y - self.layout.gap_half_width
        gap_high = self.layout.gap_center_y + self.layout.gap_half_width

        segments = []
        if gap_low > y_min:
            lower_half_y = 0.5 * (gap_low - y_min)
            lower_center_y = y_min + lower_half_y
            segments.append((lower_center_y, lower_half_y))
        if gap_high < y_max:
            upper_half_y = 0.5 * (y_max - gap_high)
            upper_center_y = gap_high + upper_half_y
            segments.append((upper_center_y, upper_half_y))

        for center_y, half_y in segments:
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[
                    self.layout.wall_half_thickness,
                    half_y,
                    self.layout.wall_half_height,
                ],
                physicsClientId=self.CLIENT,
            )
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[
                    self.layout.wall_half_thickness,
                    half_y,
                    self.layout.wall_half_height,
                ],
                rgbaColor=self.layout.visual_rgba,
                physicsClientId=self.CLIENT,
            )
            body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[
                    self.layout.wall_x,
                    center_y,
                    self.layout.wall_z_center,
                ],
                physicsClientId=self.CLIENT,
            )
            self._detour_obstacle_ids.append(int(body_id))

    def _sample_start_and_goal(self) -> None:
        if self._fixed_initial_xyz is not None:
            init_xyz = self._fixed_initial_xyz.reshape(1, 3).astype(np.float32)
        else:
            init_xyz = np.array([[0.0, 0.0, 0.3]], dtype=np.float32)
            init_xyz[0, 0] = self._rng.uniform(-0.65, -0.45)
            init_xyz[0, 1] = self._rng.uniform(-0.10, 0.10)
            init_xyz[0, 2] = self._rng.uniform(0.24, 0.40)

        if self._fixed_target_pos is not None:
            goal = self._fixed_target_pos.astype(np.float32).copy()
        else:
            goal = np.array(
                [
                    self._rng.uniform(0.45, 0.65),
                    self._rng.uniform(-0.10, 0.10),
                    self._rng.uniform(0.30, 0.85),
                ],
                dtype=np.float32,
            )

        self.INIT_XYZS = init_xyz
        self.INIT_RPYS = np.zeros((1, 3), dtype=np.float32)
        self.target_pos = goal

    def _computeReward(self):
        reward = super()._computeReward()
        state = self._getDroneStateVector(0)
        pos = np.asarray(state[0:3], dtype=np.float32)
        stage, target = self._detour_navigation_target(pos)
        detour_distance = float(np.linalg.norm(target - pos))

        if stage != "goal":
            reward += self.detour_target_reward_scale * (1.0 - np.tanh(2.0 * detour_distance))
            if self._prev_detour_stage == stage and self._prev_detour_distance is not None:
                reward += self.detour_progress_reward_scale * (self._prev_detour_distance - detour_distance)
            elif self._prev_detour_stage is not None and self._prev_detour_stage != stage:
                reward += self.detour_stage_transition_bonus

        if self._has_obstacle_collision():
            reward -= self.collision_penalty
        return float(reward)

    def _computeTruncated(self):
        if self._has_obstacle_collision():
            return True
        return super()._computeTruncated()

    def _build_info(self, success: bool) -> dict[str, float | bool | list[float]]:
        info = super()._build_info(success)
        info["collision"] = self._has_obstacle_collision()
        info["task_variant"] = "detour_waypoint"
        pos = np.asarray(self._getDroneStateVector(0)[0:3], dtype=np.float32)
        stage, target = self._detour_navigation_target(pos)
        info["detour_stage"] = stage
        info["detour_target"] = target.astype(float).tolist()
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

    def _detour_navigation_target(self, pos: np.ndarray) -> tuple[str, np.ndarray]:
        goal = self.target_pos.astype(np.float32).copy()
        if goal[0] <= self.layout.wall_x:
            return "goal", goal

        detour_height = max(float(goal[2]), self.corridor_z)
        corridor_aligned = abs(float(pos[1]) - self.layout.gap_center_y) <= self.corridor_y_tolerance
        entry_target = np.array([self.entry_x, self.layout.gap_center_y, detour_height], dtype=np.float32)
        exit_target = np.array([self.exit_x, self.layout.gap_center_y, detour_height], dtype=np.float32)

        if pos[0] < self.layout.wall_x - self.layout.wall_half_thickness:
            if not corridor_aligned:
                return "entry", entry_target
            if pos[0] < self.exit_x - self.exit_margin_x:
                return "exit", exit_target
        elif pos[0] < self.exit_x - self.exit_margin_x:
            return "exit", exit_target
        return "goal", goal
