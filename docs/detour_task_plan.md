# Detour Task Extension Plan

## Why add a harder task

The current waypoint task is useful for validating the IL -> RL pipeline, but it is still relatively easy for PPO:

- full-state observation
- dense reward
- high-level velocity action
- single-goal reaching with no true path-planning requirement

That makes it likely that scratch PPO will eventually catch up to BC-initialized PPO, even if imitation helps early learning.

To make the imitation prior matter more, the next task should require a **non-greedy navigation behavior**:

- the agent should not be able to succeed by flying approximately straight toward the goal
- a useful prior from demonstrations should reduce the RL exploration burden

## Chosen first extension

The first harder task variant will be:

**DetourWaypointVelocityAviary**

Core idea:

- the drone starts on the left side of the workspace
- the goal is sampled on the right side
- a static vertical wall blocks the direct path
- only an upper corridor is left open

This forces a detour:

1. move upward in `y`
2. pass through the corridor
3. move back toward the final goal

## Why this variant first

This is the smallest safe extension because it:

- keeps the same action space
- keeps the same low-level drone dynamics
- keeps the same reward family
- adds just enough structure to require path choice
- is still easy to explain and debug

It should also be easier to support with a scripted expert than a fully random obstacle field.

## Intended environment behavior

### Start-goal structure

- sample start on the left side of the map
- sample goal on the right side of the map
- keep `y` near the wall centerline so direct flight is usually blocked

### Obstacles

- replace the inherited default decorative Bullet obstacles
- add a custom static wall using PyBullet collision shapes
- leave a clear corridor on the upper side of the wall

### Reward and termination

Keep the current dense goal-reaching reward as the base signal, but add:

- collision penalty
- truncation on obstacle collision

This keeps training stable while making obstacle avoidance matter.

## Expert design

The current `waypoint_velocity_expert` is not obstacle-aware, so it will not be reliable in the detour task.

The first detour expert will be a simple two-stage planner:

1. if the wall blocks the direct route, aim for an intermediate waypoint near the corridor
2. once past the wall, aim for the final goal

Then reuse the same PD-style velocity controller to convert the active target into a velocity action.

This gives a clean, scripted expert without introducing a full planner.

## Observation choice

For the first implementation, keep the same `18D` observation:

- position
- orientation
- linear velocity
- angular velocity
- goal position
- relative goal vector

The obstacle layout is fixed in the world, so the policy can learn obstacle-aware behavior from position alone.

This keeps the experiment focused on navigation structure rather than new perception complexity.

## Expected effect on experiments

Compared with the original waypoint task, this variant should:

- make scratch PPO exploration harder
- make BC initialization more valuable early
- create a stronger difference in sample efficiency

This is the intended bridge between the simple state-based benchmark and a more navigation-like task.

## Implementation scope for first pass

### Environment

- add `DetourWaypointVelocityAviary`
- custom wall and corridor obstacle layout
- left-to-right constrained start/goal sampling
- collision-aware truncation and reward penalty

### Expert

- add `detour_waypoint_velocity_expert`
- use an intermediate corridor waypoint before the final goal when needed

### Tests

- add a smoke test for the new environment
- verify the new expert produces valid actions

## Follow-up after implementation

Once the first detour variant is stable, the next experiment should be:

1. collect demonstrations with the detour expert
2. train BC on the detour dataset
3. compare `PPO scratch` vs `BC + PPO` again
4. check whether the sample-efficiency gap widens relative to the original task
