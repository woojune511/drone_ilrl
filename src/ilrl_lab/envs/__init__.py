"""Custom environments for IL/RL drone experiments."""

from ilrl_lab.envs.detour_vel_aviary import DetourWaypointVelocityAviary
from ilrl_lab.envs.waypoint_vel_aviary import WaypointVelocityAviary

__all__ = ["WaypointVelocityAviary", "DetourWaypointVelocityAviary"]
