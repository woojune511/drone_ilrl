# Interview Q&A

This file is a practical speaking guide for explaining the project without overstating the result.

## 30-Second Pitch

I built a PyBullet drone navigation benchmark to study when imitation learning helps PPO. The first waypoint task was too easy, so I redesigned it as a detour task where a wall blocks the direct path and the drone must use a corridor. BC gave a useful prior, but naive PPO fine-tuning suffered from BC -> PPO distribution shift and could degrade the policy. The final method added expert-state augmentation with scripted relabeling as local recovery supervision during PPO, improving `50`-episode success from a BC-only `0.54` to `0.792 +/- 0.104` with validation-best checkpoints across five PPO seeds.

## 2-Minute Pitch

The project started from a practical problem I had seen before: it is hard to tell whether IL -> RL is actually working when the environment, demonstrations, and evaluation protocol are noisy. I moved to a lightweight PyBullet setup and created a detour-constrained waypoint task. The drone starts on the left, the goal is on the right, and a wall blocks the greedy straight-line path except for a narrow corridor.

The first important lesson was task design. In the simple waypoint task, scratch PPO could learn quickly, so imitation pretraining did not have much room to help. The detour task created a real exploration and closed-loop control bottleneck.

The second lesson was transfer stability. After making BC-to-PPO initialization exact, I found that PPO did not automatically improve the BC policy. KL regularization and actor freezing mostly preserved BC behavior, but did not reliably improve it. The main failure mode was distribution shift: once PPO changed the policy, it visited states slightly off the expert trajectory where BC had weak recovery behavior.

The final method addressed that directly. I perturbed expert trajectory states, recomputed relative-goal features, rejected invalid wall states, and relabeled each augmented state with the scripted expert. During PPO updates, this auxiliary expert-state loss acted as local recovery supervision around the demonstration manifold. Under a shared `50`-episode deterministic protocol, BC-only reached `0.54` success, while the expert-state augmented BC+PPO variant reached `0.728 +/- 0.109` for final checkpoints and `0.792 +/- 0.104` for validation-best checkpoints across five PPO seeds.

I still report the final checkpoint separately because late PPO updates can degrade final approach behavior. That is part of the technical conclusion, not something I hide.

## Core Technical Questions

### Why was the original waypoint task not enough?

It had full-state observation, dense reward, high-level velocity control, and no real path-planning bottleneck. Scratch PPO could make progress without demonstrations, so it was a weak benchmark for proving that imitation helped.

### What made the detour task harder?

The direct path from start to goal is blocked by a wall. The drone has to first move toward the corridor, which can be temporarily non-greedy with respect to the goal direction. That makes exploration and closed-loop recovery more important than in the simple waypoint task.

### What exactly failed in naive BC -> PPO?

The initial BC policy was useful, but PPO updates changed the policy distribution. Once the drone visited states slightly outside the expert trajectory distribution, the policy could produce worse recovery actions. This showed up as success dropping or final approach becoming unstable even when the drone still reached the corridor and goal stage.

### What did KL regularization and actor freezing do?

They helped preserve BC-like behavior, but mostly failed to improve beyond BC. They were useful diagnostics because they showed that preserving the prior is not the same as learning better closed-loop behavior.

### What does relabeling mean here?

For each augmented state, I recomputed the expert action with the scripted expert instead of reusing the original demonstration action.

For example:

```text
original data:       (state s, expert action a)
after augmentation:  noisy state s'
wrong label choice:  (s', a)
used label choice:   (s', expert(s'))
```

This matters because a perturbed off-trajectory state may require a corrective action that differs from the original action.

### Is this still reinforcement learning if a scripted expert is used during PPO?

The policy is still updated with PPO from on-policy rollouts, but the final method is not pure RL. It is BC-initialized PPO with an auxiliary supervised recovery loss on augmented expert states. The claim is not "PPO alone solved it." The claim is that diagnosing distribution shift and adding local expert recovery supervision made PPO fine-tuning useful instead of destructive.

### Is validation-best checkpoint selection cherry-picking?

It would be if the final checkpoint were hidden. Here both are reported. Validation-best is selected by a fixed protocol: highest success rate, then lowest final distance, then highest return. Final checkpoints are reported separately as a robustness diagnostic. The gap between final and best is evidence of late-training drift.

### What did PPO contribute if there is still an expert loss?

PPO provided online interaction and policy improvement pressure, but it needed a safer local region around the demonstration manifold. The expert-state loss made the fine-tuning less brittle by teaching recovery actions near expert states. The evidence is that KL/freeze preserved BC but did not improve it, while expert-state augmentation improved success and final distance on average.

### Why not use DAgger, AWAC, or IQL?

Those are good next steps. DAgger would collect corrective labels on the learner's visited states, which is a cleaner version of the local relabeling idea. AWAC or IQL could make better use of expert data with offline-RL-style objectives. I kept the current scope smaller to isolate the BC -> PPO distribution-shift issue first.

### What is the main remaining weakness?

Late-training final approach and settling stability. The weak seed still reached the corridor and goal stage without collisions, but the final checkpoint failed the success condition more often than the validation-best checkpoint. That points to fine-tuning stability, not detour discovery or obstacle avoidance.

## Short Defensive Answers

### "Did PPO actually improve anything?"

Naive PPO did not. PPO became useful only after I added expert-state recovery supervision. The strongest claim is BC-only `0.54` success versus BC+PPO expert-state augmentation `0.792 +/- 0.104` validation-best success across five seeds.

### "Are you overclaiming the result?"

No. I explicitly report final checkpoints, validation-best checkpoints, and the dependency on a scripted expert. The contribution is the diagnosis and targeted mitigation of BC -> PPO distribution shift, not a claim that PPO alone solved the task.

### "Why should I care about this project?"

It shows the full loop: design a task where IL should matter, build a scripted expert to control data quality, identify why naive IL -> RL transfer fails, add diagnostics, test interventions, and frame the final result with clear limitations.

### "What would you do next?"

I would replace local scripted relabeling with DAgger-style learner-state aggregation, then compare PPO+expert-state loss against AWAC/IQL-style fine-tuning. I would also add a final-approach curriculum or explicit settling objective because failure analysis shows that remaining failures are near-goal stability issues.

## Best Resume Bullet

- Diagnosed BC-to-PPO distribution shift in a detour-constrained PyBullet drone navigation task and improved BC-initialized PPO over standalone BC by adding expert-state augmentation with scripted relabeling, raising validation-best success from `0.54` to `0.792 +/- 0.104` across `5` PPO seeds.

