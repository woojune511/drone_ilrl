"""Expert policies used to generate demonstrations."""

from ilrl_lab.experts.velocity import detour_waypoint_velocity_expert, waypoint_velocity_expert

__all__ = ["waypoint_velocity_expert", "detour_waypoint_velocity_expert"]
