# Imitation Learning Failure Modes from the Old Drone Project

## Context

This note summarizes why the original setup was difficult from an imitation-learning perspective.

Original setup:

- Observation: camera image only
- Action: forward / backward / left / right / clockwise turn / counterclockwise turn
- Dynamics: motion persisted with near-constant velocity unless actively corrected, with some inertia
- Expert data: collected manually with keyboard teleoperation
- Learning style: imitation learning / behavior cloning

The main point is that several hard factors were combined at once:

- partial observability from camera-only state
- delayed and inertial dynamics
- noisy human teleoperation
- behavior-cloning distribution shift

## Problems and Possible Fixes

### 1. Partial observability from camera-only state

Problem:
The image does not directly reveal velocity, angular velocity, acceleration, or latent motion state. The same frame can correspond to very different underlying dynamics.

Why it hurts:
The policy may see visually similar observations that require different actions.

Possible fixes:

- stack multiple frames
- use a recurrent policy such as LSTM or GRU
- add optical flow
- add IMU, velocity, or pose estimates
- use a state estimator

### 2. Inertial dynamics make action effects delayed

Problem:
Actions do not produce immediate stop-and-go motion. Motion continues after the command, so an action changes future state instead of only current state.

Why it hurts:
The action label is weakly aligned with the immediate visual frame.

Possible fixes:

- use a low-level stabilizer or PID controller
- redefine high-level action as target velocity or target waypoint
- use fixed action hold duration
- log command duration explicitly

### 3. Same image can have multiple valid actions

Problem:
In human demonstrations, the same scene may lead to different corrections depending on timing and preference.

Why it hurts:
Behavior cloning tends to average those modes and may output a weak or ambiguous action.

Possible fixes:

- redesign the action space to be more structured
- use mixture policies
- use diffusion-style action models
- use action chunking
- use DAgger or interactive relabeling

### 4. Keyboard input is coarse and quantized

Problem:
Keyboard control is binary and does not support smooth continuous control.

Why it hurts:
The demonstrations contain abrupt and noisy control labels.

Possible fixes:

- use joystick or gamepad input
- use scripted expert control
- aggregate keypresses into fixed macro-actions
- record action duration precisely

### 5. Human reaction delay

Problem:
The pilot reacts to a previous frame, not the exact current frame.

Why it hurts:
Observation and action labels are misaligned in time.

Possible fixes:

- record timestamps precisely
- align observation-action pairs offline
- run teleoperation with a fixed-rate control loop
- use hindsight relabeling if applicable

### 6. Sensor-action-physics desynchronization

Problem:
Rendering, input polling, and physics stepping may run at different rates.

Why it hurts:
Even small time misalignment can degrade imitation performance.

Possible fixes:

- synchronize simulator, action, and observation on a fixed timestep
- log exact timestamps
- resample data onto a single control frequency

### 7. Covariate shift from offline demonstrations

Problem:
The expert mostly visits safe states, but the learned policy inevitably drifts into unseen states.

Why it hurts:
Small mistakes compound and the policy has no recovery behavior for off-distribution states.

Possible fixes:

- use DAgger
- collect intervention data
- add perturbations during collection
- oversample near-failure and recovery states

### 8. Recovery behavior is underrepresented

Problem:
Human demos often contain mostly successful flight, not bad states and recovery maneuvers.

Why it hurts:
The policy learns how to stay good, but not how to recover when things go wrong.

Possible fixes:

- collect recovery-focused demonstrations
- inject disturbances and record expert corrections
- weight recovery segments more during training

### 9. Action space is not control-friendly

Problem:
Discrete directional movement plus yaw commands is intuitive for humans but not ideal for learning drone control under coupled dynamics.

Why it hurts:
The policy must learn a harder mapping from perception to low-level correction commands.

Possible fixes:

- use target velocity commands
- use body-frame velocity plus yaw rate
- use local waypoint actions
- separate high-level planning from low-level control

### 10. Goal definition is too vague

Problem:
"Fly safely without crashing" is intuitive but underspecified.

Why it hurts:
It is hard to define success, compare methods, or explain performance clearly.

Possible fixes:

- define corridor following
- define waypoint reaching
- define obstacle avoidance with forward progress
- define hover stability or trajectory tracking

### 11. Pure BC is weak for long-horizon control

Problem:
Behavior cloning only imitates local action labels and does not optimize long-term performance directly.

Why it hurts:
Delayed dynamics and compounding errors are especially hard in long rollouts.

Possible fixes:

- use BC for pretraining
- fine-tune with RL
- consider offline RL
- use hybrid IL + RL pipelines

### 12. Rare but important danger states are imbalanced

Problem:
Most frames are easy cruising states. The truly important near-collision frames are rare.

Why it hurts:
The model may do well on average but fail exactly where performance matters most.

Possible fixes:

- oversample hard cases
- reweight losses by state difficulty
- mine near-failure segments
- use a curriculum

### 13. Limited visual field and ambiguous depth

Problem:
A single RGB camera has limited field of view and weak depth cues.

Why it hurts:
Obstacle avoidance and timing become harder.

Possible fixes:

- use depth or RGB-D
- use stereo vision
- use multiple cameras
- widen FOV
- use segmentation or semantic cues

### 14. Overfitting to simulator visuals

Problem:
The policy may exploit textures, lighting, or simulator-specific artifacts instead of learning robust flight behavior.

Why it hurts:
Generalization across maps or conditions becomes poor.

Possible fixes:

- use domain randomization
- vary lighting and textures
- vary obstacle layouts
- hold out test maps

### 15. Single-frame observations miss motion cues

Problem:
Instantaneous images do not contain explicit temporal information.

Why it hurts:
The model may react too late or misjudge momentum.

Possible fixes:

- stack recent frames
- use temporal CNNs
- use recurrent architectures

### 16. Control frequency mismatch

Problem:
Human control timing may not match the simulator control frequency.

Why it hurts:
The same keypress may correspond to inconsistent real effect durations.

Possible fixes:

- enforce a fixed control rate
- hold actions for a fixed number of simulator steps
- resample or smooth the recorded action stream

### 17. Human anticipation is hidden latent state

Problem:
Humans act based on an internal prediction of the future, not only on the visible frame.

Why it hurts:
The dataset does not capture the internal belief used to choose the action.

Possible fixes:

- add temporal context
- include richer state
- use scripted experts where the control logic is explicit

### 18. Low reproducibility of human-collected datasets

Problem:
Manual collection quality varies by session, fatigue, attention, and operator skill.

Why it hurts:
Experiments become noisy and hard to compare fairly.

Possible fixes:

- fix seeds and evaluation maps
- automate data collection with scripted experts
- separate train / validation / test environments

### 19. Weak or missing metrics

Problem:
"It flies well" is not a quantitative evaluation.

Why it hurts:
It is difficult to diagnose problems or present convincing results.

Possible fixes:

- success rate
- collision-free episode rate
- mean survival time
- forward progress
- minimum obstacle clearance
- trajectory smoothness
- intervention rate

### 20. Data quality issues in logged trajectories

Problem:
The dataset may contain accidental key presses, stuck behavior, duplicate frames, logging lag, or uninformative segments.

Why it hurts:
The model learns noise instead of useful control structure.

Possible fixes:

- replay and visualize episodes
- filter out low-quality segments
- detect outliers
- score episode quality before training

## Good Interview Framing

A concise way to explain the old project's difficulty:

> The setup combined camera-only partial observability, delayed inertial dynamics, noisy keyboard teleoperation, and behavior-cloning covariate shift. That made both expert data collection and policy learning much harder than a standard imitation-learning benchmark.

## If Rebuilding the Project Today

A better modern setup would likely be:

- high-level action: target velocity or waypoint
- low-level control: PID or stabilizing controller
- observations: image plus short temporal context, or image plus velocity / IMU
- expert data: scripted or semi-automated expert instead of pure keyboard teleoperation
- training: BC baseline first, then RL fine-tuning
- evaluation: explicit success, collision, and trajectory metrics
