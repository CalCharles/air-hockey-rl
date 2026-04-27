# Real-world episode lifecycle

End-to-end flow of a single policy episode on the real UR5, from collection through replay ingestion and artifact output.

Primary code: [`async_td3_real.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real.py) (`collector_process`).
Helpers: [`real_episode_buffers.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/helper/real_episode_buffers.py), [`real_stop_state.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/helper/real_stop_state.py), [`real_motion_rewards.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/helper/real_motion_rewards.py), [`episode_artifacts.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/helper/episode_artifacts.py), [`real_warm_start.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/helper/real_warm_start.py).

## Overview

```
soft_reset / prime paddle
        |
        v
  [step loop] ──> action = actor(obs) + noise / primitive
        |              |
        |         env.step(action)
        |              |
        |         compute task + motion reward
        |         classify stop event
        |         append to episode trajectory + episode_rows
        |              |
        |         if done (termination | truncation | stop):
        |              break
        |
        v
  [end-of-episode processing]
        |
        ├── truncate if readiness-fail
        ├── route to success/failure replay
        ├── run learner iterations
        ├── save artifacts (HDF5, GIF, video, latency)
        ├── update rolling metrics + TensorBoard
        |
        v
  [reset path]
        |
        ├── run ResetPolicyFSM (or hard_reset_with_pause)
        ├── soft_reset + prime paddle history
        └── enter next episode with transition hold
```

## Step loop

Each step:

1. **Action selection:** deterministic actor output + Gaussian noise, or primitive exploration override.
2. **Environment step:** `env.step(action)` returns `(obs, reward, termination, truncation, info)`.
3. **Task reward:** environment reward (base + shaping + survival bonus).
4. **Motion reward:** computed via `_compute_motion_reward_components` using `MotionRewardState` to track paddle/puck history across steps. See [reward-shaping.md](../../training/reward-shaping.md).
5. **Stop classification:** `_classify_stop_event` checks for protective stops, controller disconnects, and legacy e-stop signals.
6. **Trajectory append:** observation, action, rewards, done flags are appended to `EpisodeTrajectory`; raw sensor/timing data is appended to `episode_rows` for HDF5.

## Stop event classification

**Code:** [`real_stop_state.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/helper/real_stop_state.py)

`_classify_stop_event` probes multiple sources in priority order:

1. `step_info` dict keys (`protective_stop`, `controller_connected`)
2. Legacy `estop` key in `step_info`
3. `simulator.robot_command_readiness()` function
4. `simulator.rcv.isProtectiveStopped()` (RTDE interface)
5. `simulator.vals` array (oldest fallback)

Returns a `StopEventState` frozen dataclass with:
- `protective_stop` / `controller_disconnected` -- cause flags
- `active` -- whether any stop condition was detected
- `episode_end_type` / `episode_end_reason` -- structured labels for logging
- `artifact_label` -- label used in HDF5/artifact file naming

## Episode truncation for readiness-fail

**Code:** [`real_episode_buffers.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/helper/real_episode_buffers.py)

If the robot's command readiness fails mid-episode (e.g., brief communication drop that recovers), the episode is truncated at the first failure step:

1. `_truncate_episode_trajectory_inplace` keeps only transitions up to `readiness_first_fail_step_idx + 1`.
2. The final transition's `dones` and `bootstrap_terminals` are set to 1.0 (terminal).
3. `episode_rows`, images, and latency lists are truncated to match.
4. The cutoff row's `stop_flags` are updated to mark the readiness-fail.

This prevents post-failure garbage transitions from entering the replay buffer.

## Replay routing

After optional truncation, the episode is routed to success or failure replay via the same quantile-based logic as simulation. See [replay-and-episodes.md](../../training/replay-and-episodes.md) for the partitioning algorithm.

## Learner updates

After each collected episode, the learner runs one or more critic + actor update iterations from the shared replay. If the actor weights changed, the collector syncs its actor copy and enters a **transition hold** period (configurable steps where the actor output is blended or suppressed) to smooth the policy transition on hardware.

## Episode artifacts

**Code:** [`episode_artifacts.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/helper/episode_artifacts.py)

Four artifact types are written per episode:

### Split HDF5

`save_split_episode_hdf5` writes per-episode trajectory data in a structured HDF5 format. Each row contains:
- Timing (`cur_time`)
- Robot state (pose, speed, force, acceleration, desired pose)
- Puck state (position)
- Actions (policy output, realized displacement)
- Rewards (task, motion, components)
- Stop flags

Optional datasets (timing breakdown, stop_flags) are included when all rows contain them.

### GIF (joint side-by-side)

`generate_episode_gif` renders one GIF per episode where each frame is the **Box2D projection of the HDF5 trajectory on the left** and the matching **real-world camera frame (`train_img`) on the right**. Building blocks are pre-existing:

- Box2D projection: `RealTrajectoryRenderer.render_frame` ([`visualize_real_trajectory.py`](../../../../scripts/smooth_policy/visualize_demo/visualize_real_trajectory.py))
- Real-world camera frames: `train_img` dataset stored in the split HDF5
- Side-by-side stitcher: `_side_by_side` ([`replay_real_in_sim.py`](../../../../scripts/smooth_policy/visualize_demo/replay_real_in_sim.py))

The actual stitching loop lives in `_create_joint_trajectory_gif` inside [`episode_artifacts.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/helper/episode_artifacts.py). When `train_img` is absent, the function falls back to the Box2D-only `create_trajectory_gif` path.

### Camera video

`generate_episode_camera_video` writes the collected camera frames as a standalone MP4 (with codec fallback from H.264 to MJPEG). This is redundant with the right panel of the joint GIF but kept as a separate full-resolution artifact.

### Latency profile

`_write_latency_profile_episode` generates a matplotlib figure showing per-step timing breakdown (puck detection, model inference, block sleep, other).

## Warm start

**Code:** [`real_warm_start.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/helper/real_warm_start.py)

Before live collection begins, the replay buffer can be seeded from previously saved HDF5 episodes:

1. `_list_warm_start_hdf5_files` discovers files across configured input directories, interleaving from multiple sources.
2. For each file, `_load_warm_start_episode` reads the split HDF5 data, reconstructs `state_info` dicts from the stored arrays, and rebuilds transitions.
3. `_recompute_warm_start_rewards` replays the environment reward and motion reward logic on the reconstructed states, ensuring rewards match the current reward configuration (not the original training run's).
4. An e-stop penalty of `-5.0` is applied to the motion reward at the first stop event in each episode.
5. Episodes are routed to success/failure partitions using the same quantile threshold logic as live episodes.

This allows training to start with meaningful replay data from prior real-world sessions, even if the reward function has changed.

## Rolling metrics

**Code:** [`real_collector_metrics.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/helper/real_collector_metrics.py)

A rolling window of the last 50 episodes tracks:
- Average task reward
- Average motion reward
- Average episode length
- E-stop episode count

These are written to TensorBoard under the `rolling50/` prefix and shared with the learner via `stats` dict for console logging.

## Related docs

- [Reward shaping](../../training/reward-shaping.md) -- motion reward component details
- [Replay and episodes](../../training/replay-and-episodes.md) -- buffer types and partitioning
- [Reset FSM](reset-fsm.md) -- how the puck is reset between episodes
- [Async replay semantics](td3-async-replay.md) -- `dones` column conventions
