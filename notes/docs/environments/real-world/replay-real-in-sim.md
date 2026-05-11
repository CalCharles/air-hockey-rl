# Replaying Real Trajectories in the Box2D Sim

Tool for sim-to-real system identification: take a recorded real-robot episode,
replay the exact actions inside the Box2D simulator, and render the two
trajectories side-by-side so the dynamics gap is visible frame-by-frame.

**Script:** `scripts/visualization/replay_real_in_sim.py`

## Quick start

```bash
/home/pearl/miniconda3/envs/air/bin/python \
    scripts/visualization/replay_real_in_sim.py
# → writes ./sim_vs_real_trajectory_data451.gif
```

Key flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--episode` | `real_runs/online_run/episode_hdf5/100-200/trajectory_data451.hdf5` | Real HDF5 episode (split schema) |
| `--config` | `…/new_juggle/sysid_best_params_hist2.yaml` | Sim YAML |
| `--output` | `./sim_vs_real_<stem>.gif` | GIF destination |
| `--enable-noise` | off | Use config noise/delay/termination verbatim; default is clean deterministic replay |
| `--start-frame` | `0` | Index in the real episode where the comparison begins. Sim is reset to this frame's state and the replay starts here (use to skip episode warm-up). |
| `--puck-vel-fit` | off | Estimate initial puck velocity by fitting `airhockey.sims.real.velocity_estimator.fit_velocity_from_positions` over a small window around `--start-frame` (default ±5 frames = 11 samples) instead of a two-point finite difference. More robust on noisy/occluded puck tracks. |
| `--puck-vel-half-window` | `5` | Half-window (frames) for the velocity fit. |
| `--max-steps` / `--fps` / `--frame-width` | — | Standard controls |

## What it does

1. **Load real HDF5** via `load_split_trajectory_data`. Extracts `pose_xy`,
   `speed_xy`, `desired_xy`, `puck_xy`, `cur_time` from the canonical 35-field
   layout. `pose` and `desired_pose` are already in table frame — no frame
   conversion needed.

2. **Reconstruct normalized actions**:
   ```
   actions = clip((desired_pose − pose) / move_lims, −1, 1)
   ```
   with `move_lims = (0.26, 0.12)` — identical on real and sim. This recovers
   the *exact* `[-1, 1]` action the policy produced at each step, pre-pipeline.
   Same inversion pattern as `async_td3_real_reset_policy.py:318-320`.

3. **Build the sim env** from the YAML. Unless `--enable-noise`, the script
   overrides these keys to make replay deterministic:
   - `simulator_params`: `puck_noise`, `enable_random_occlusions`,
     `enable_observation_delay`, `enable_action_delay`,
     `enable_action_force_attenuation` → `False`
   - `air_hockey`: all `terminate_on_*` flags → `False` (so the replay runs the
     full recorded length regardless of goal/out-of-bounds events)

4. **Seed the sim state** via `env.reset_from_state(state0)`. The 8-vector
   format is the one in `airhockey_base.py:906`
   (`AirHockeyBaseEnv.create_world_objects_from_state`):
   ```
   [paddle_x, paddle_y, paddle_vx, paddle_vy,
    puck_x,   puck_y,   puck_vx,   puck_vy]
   ```
   **Beware:** a second definition in `airhockey_simple_tasks.py:537` has the
   *opposite* order. Only the base-env one is active for the juggle task.

   Initial values:
   - **paddle pos** = `pose_xy[0]` (real's first logged paddle position)
   - **paddle vel** = `speed_xy[0]` (directly from the HDF5 `speed` dataset — the
     robot controller logs this)
   - **puck pos** = `puck_xy[0]`
   - **puck vel** — *not* stored in the HDF5. Two options:
     - **Default (two-point finite difference)**: `(puck_xy[s+1] − puck_xy[s]) / dt0`
       where `s = --start-frame` and `dt0` comes from `cur_time` (~0.05 s, with a
       0.05 s fallback for degenerate timestamps). Cheap but noise-sensitive.
     - **`--puck-vel-fit`**: gravity-linear LSQ fit via
       `airhockey.sims.real.velocity_estimator.fit_velocity_from_positions` over
       `[s − h, s + h]` (default `h = 5`, so 11 samples). Returns `v_at_times[k]`
       where `k = s − lo` — the smoothed velocity *exactly at* the start frame.
       Default gravity is `(0, 0)` (flat real table). More robust to occlusions
       and pose noise; logs SNR + n_valid for sanity.
     Small seeding error here bleeds off quickly as sim dynamics take over either
     way; the fit option mainly helps when the trajectory near `start_frame` has
     measurement noise or a missing sample.

5. **Replay loop** — with a subtle timing fix. Row `i` in the HDF5 is written
   *after* `env.step()` (confirmed in `collector_process_modular` /
   `_build_split_episode_row` in `extras/async_td3_real.py` and
   `helper/real_td3_runtime.py`, plus `airhockey_base.py:784`), so `pose[i] = T_{i+1}` (post-step) and
   `actions[i] = a_{i+1}` (the action that produced `pose[i]`). After resetting
   sim to `pose[0] = T_1`, the *next* action in the real timeline is `actions[1]`
   — **not** `actions[0]`, which already happened. Loop:
   ```python
   for i in range(n):
       render sim + real for step i
       if i < n - 1:
           env.step(actions[i + 1])
   ```
   At `i=0` sim matches real exactly (sanity check). From then on, any drift is
   pure dynamics gap.

6. **Render side-by-side.** Both panels use the Box2D rendering style at the
   same `render_size`:
   - **Sim panel** via `AirHockeyRenderer.get_frame()` (built-in target marker
     *disabled* — we draw our own).
   - **Real panel** via `RealTrajectoryRenderer.render_frame(... target=None)`
     with `paddle_input_frame='table'`.
   - **Sim ghost overlay on the real panel**: light-gray, alpha-blended circles
     at the sim's current paddle and puck positions with a thin dark outline.
     This makes the drift visible directly on a single panel.
   - **Consistent target marker** (orange cross+circle, matching
     `AirHockeyRenderer.draw_target_marker` exactly) drawn on *both* panels at
     `current_paddle_pos + action * move_lims`. Pixel coordinates come from a
     shared helper that mirrors
     `AirHockeyRenderer.world_xy_to_output_pixel` for `orientation='vertical'`,
     so both panels place the marker at matching pixels given matching physical
     positions.

7. **Postprocess and write GIF**: BGR→RGB, resize width to 160 (`td3_training.py`
   convention), label panels with `"REAL"` / `"SIM"` and step index,
   horizontal concat with a 3-px light-gray separator, `imageio.mimsave`.

## Interpreting the output

| What you see | Meaning |
|---|---|
| Step 0: real and sim match exactly, gray ghosts overlap real objects | Sanity check — sim was just reset to real's state |
| Paddle ghost offset from real paddle | PID / paddle-tracking gap |
| Puck ghost offset from real puck | Puck physics gap (restitution, damping, friction) |
| Divergence grows over time | Expected — small per-step errors compound |

Use this as the inner "evaluate candidate params" loop when tuning Box2D
parameters against real data.

## Related

- Recording sysid trajectories: [`teleop-system-id.md`](teleop-system-id.md)
- Real episode schema: [`episode-lifecycle.md`](episode-lifecycle.md)
- Real UR5 stack overview: [`overview.md`](overview.md)
- Box2D internals: [`../box2d/simulator-essentials.md`](../box2d/simulator-essentials.md)
- Obs/action spaces: [`../observation-action-spaces.md`](../observation-action-spaces.md)
