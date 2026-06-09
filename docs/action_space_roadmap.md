# Action Space Roadmap

This note defines the next experimental direction after the state-based detour portfolio result.

## Motivation

The earlier AirSim-style setup used depth-only observations and keyboard-like discrete commands. Reproducing that setup exactly is less important than asking what action interface would be realistic for an actual drone navigation policy.

The current portfolio environment uses a high-level 4D velocity command:

```text
direction xyz + speed scale
```

That is stable for studying BC -> PPO transfer, but it is not the cleanest deployment interface because altitude, yaw, and translation are all exposed together.

## Recommended Deployment-Oriented Interface

The next interface is:

```text
action = [vx_body, vy_body, yaw_rate]
```

with:

- body-frame planar velocity
- yaw-rate control
- low-level altitude hold
- command rate limits
- bounded planar speed and yaw rate

This keeps RL focused on navigation while leaving fast stabilization and altitude control to the lower-level controller.

## Implemented Variant

`DetourPlanarVelocityAviary` adds this interface as a separate task variant:

```bash
--task-variant detour_planar
```

The policy-facing action space is 3D:

```text
vx_body_norm in [-1, 1]
vy_body_norm in [-1, 1]
yaw_rate_norm in [-1, 1]
```

The environment maps this to PID targets:

- max planar speed: `0.35 m/s`
- max yaw rate: `pi / 3 rad/s`
- altitude hold: `0.55 m`
- planar acceleration limit: `0.80 m/s^2`
- yaw acceleration limit: `pi rad/s^2`

The original `detour` task remains unchanged for reproducing the portfolio result.

## First Observation Reduction

`DetourPlanarLocalObsAviary` keeps the same action interface and detour task, but removes absolute position and absolute goal from the policy observation:

```bash
--task-variant detour_planar_local
```

Observation layout:

```text
body_vel_xy(2)
altitude_error(1)
z_vel(1)
sin_yaw, cos_yaw(2)
rel_goal_body_xyz(3)
rel_detour_target_body_xyz(3)
previous_action(3)
```

This is not depth-only perception. It is a route-conditioned local tracking interface: a simple upstream planner still provides the local detour target. The purpose is to remove global privileged coordinates before moving to harder perception.

## Raycast Observation Variant

`DetourPlanarRaycastAviary` removes the local detour target and replaces it with low-dimensional horizontal ray distances:

```bash
--task-variant detour_planar_raycast
```

Observation layout:

```text
body_vel_xy(2)
altitude_error(1)
z_vel(1)
sin_yaw, cos_yaw(2)
rel_goal_body_xyz(3)
previous_action(3)
raycast_distances(21)
```

The teacher remains privileged: demonstrations are labeled by the full-state scripted expert, while the student policy only observes raycast distances and goal-relative local state.

Initial result:

- expert success: `1.0`
- BC success: `0.84`
- BC mean final distance: `0.245m`

This is the first follow-up setting with meaningful headroom beyond BC.

## Experimental Order

Change one axis at a time:

1. Keep current state observation and detour task.
2. Switch only the action interface to `detour_planar`.
3. Collect clean scripted demonstrations with `detour_planar_velocity_expert`.
4. Train BC and confirm that clean imitation works.
5. Switch only the observation interface to `detour_planar_local`.
6. Train BC and confirm that local route-conditioned imitation works.
7. Replace local route-conditioned state with `detour_planar_raycast`.
8. Inspect BC failure modes.
9. Fine-tune with PPO or compare AWAC/IQL only after raycast BC failure modes are understood.
10. Only after that, expand task complexity or inject human-like data noise.

Avoid changing action space, observation space, task difficulty, and data quality in the same experiment. If learning fails, the cause becomes ambiguous.

## Why Not Keyboard Discrete Actions First?

Discrete keyboard actions are useful for analyzing the old AirSim prototype, but they are not the most realistic control interface for deployment. A real drone stack is more likely to expose a bounded velocity or body-rate offboard API. For this project, constrained body-frame velocity is a better next step because it is both realistic and still simple enough for controlled experiments.
