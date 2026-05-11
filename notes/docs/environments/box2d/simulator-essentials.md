# Box2D simulator essentials

Mirror of [`.cursor/rules/project-box2d-simulator.mdc`](../../../../.cursor/rules/project-box2d-simulator.mdc).

Primary simulator file: [`airhockey/sims/airhockey_box2d.py`](../../../../airhockey/sims/airhockey_box2d.py).

## Bound-related configs (important)

- **PID workspace clipping** (`use_pid=True` path), **inside Box2D only**:
  - `x_min_lim`, `x_max_lim`, `y_min`, `y_max`
  - `top_abs`, `bot_abs`, `max_bias_p`, `max_bias_m`
  - These become `lims` / `edge_lims` in [`airhockey_box2d.py`](../../../../airhockey/sims/airhockey_box2d.py); applied by `_clip_limits()` and `_clip_pid_target_to_workspace()`.
  - Keys `paddle_bounds` / `paddle_edge_bounds` may appear under `simulator_params` (the env copies them for API parity) but **this simulator does not read them**.
- **Per-step movement limits** (not global workspace):
  - `rmax_x`, `rmax_y` (stored as `move_lims`)
  - Applied in `_compute_pid_target_pos()` via `_get_edge()`.
- **Physical table limits**:
  - From `length`, `width` → `table_x_min/max`, `table_y_min/max`.
  - These are hard world walls in Box2D.

## Two layers: sim vs env (paddle limits)

1. **Box2D** (`air_hockey.simulator_params`): `x_min_lim` … `y_max` and `top_abs` … `max_bias_m` constrain targets inside the sim (PID / transition). See class docstring on `AirHockeyBox2D`.

2. **Base env** ([`airhockey_base.py`](../../../../airhockey/airhockey_base.py)): top-level `air_hockey.paddle_bounds` (rectangle `x_min`, `x_max`, `y_min`, `y_max`) and `air_hockey.paddle_edge_bounds` (`top_abs`, `bot_abs`, `max_bias_p`, `max_bias_m`) feed `get_clip_limits()` in `single_agent_step()` and **clip the policy action before** `simulator.get_transition()`.

Keep YAML values consistent across both places; otherwise pre-step action gating and in-sim clipping can disagree. If `paddle_bounds` / `paddle_edge_bounds` are omitted, the env falls back to table-wide defaults and loose edge defaults, while Box2D still uses its own `lims`.

## Practical maximum reach (default table)

Given default Box2D dimensions in [`airhockey/sims/airhockey_box2d.py`](../../../../airhockey/sims/airhockey_box2d.py):

- `length = 1.9304`, `width = 0.8636`, `paddle_radius = 0.0508`
- Table center-coordinate extents:
  - `x in [-0.9652, 0.9652]`
  - `y in [-0.4318, 0.4318]`
- Practical paddle-center extents (subtract radius from walls):
  - `x in [-0.9144, 0.9144]`
  - `y in [-0.3810, 0.3810]`

If configured bounds exceed these, reachable motion is still capped by table walls.

## Coordinate reminder

- Internal Box2D and base coordinates are converted through:
  - `base_coord_to_box2d()`
  - `convert_from_box2d_coords()`
- Keep frame conventions consistent when adding or diagnosing bounds logic.

## Delay toggles and shared delay value

- Box2D now supports independent toggles for:
  - `enable_action_delay`
  - `enable_observation_delay`
- Both features use the same `delay_seconds` value (clamped per step to `[0, time_per_step]`).
- Optional fluctuation:
  - `randomize_delay` enables per-step randomization of the realized delay.
  - `delay_relative_range` controls multiplicative half-width (for example `0.25` means `delay_seconds * [0.75, 1.25]` before clamping).
- Observation-delay snapshots can show stale paddle `acceleration`/`jerk` fields because those derivatives are refreshed after the full step; positions/velocities remain mid-step consistent.

### ⚠ Subtle side effect: `enable_observation_delay` changes `puck_history` sampling rate

`get_singleagent_transition` (`airhockey_box2d.py:1681`) splits each 20 Hz
env step into one or more sub-steps based on the breakpoints
`{0, t_obs, t_action, time_per_step}`. The paddle/puck history lists
are appended **inside** that sub-step loop (line 1830 / 1837), so the
number of history entries per env step depends on which delay toggles
are on:

| Config | Sub-steps / env step | Appends / env step | Effective sampling rate |
|---|---:|---:|---:|
| `enable_observation_delay: true`  (canonical baseline) | 2 | 2 | ~40 Hz |
| `enable_observation_delay: false` (e.g. zero-shot ablation) | 1 | 1 | 20 Hz |

The 20 Hz **env / control step** is unchanged by either setting (every
`step()` advances sim time by exactly `time_per_step = 1/20 s`). What
changes is the temporal *density* of the puck/paddle history that the
policy reads via `puck_history[-5:]` / `paddle_history[-5:]` (see
[`obs construction`](../observation-action-spaces.md#temporal-density-caveat)).
The real-world simulator (`air_hockey_real.py:1549`) does not have a
sub-step loop and appends exactly once per 20 Hz env step — so it
matches the `enable_observation_delay: false` density, **not** the
canonical training density.

## Paddle-puck contact caveat (important)

Empirically in this Box2D setup, paddle-puck outcomes are often less stable than expected from ideal rigid-body intuition:

- Contact behavior is highly sensitive to **relative velocity at impact** (especially along the collision normal).
- Changing `paddle_density` / `puck_density` can matter, but in many practical runs it has a weaker effect than impact timing and approach speed.
- Small differences in pre-contact motion (controller force limits, damping, jitter cadence, action lag, and step timing) can dominate the post-contact result.

Practical guidance:

- Treat density sweeps as a secondary knob for tuning contact behavior.
- First tune and compare pre-contact velocity profiles and impact timing, then use density for finer adjustment.

Evidence reference:

- Contact-scenario implementation and metrics collection: [`scripts/box2d_paddle_puck_contact_scenario.py`](../../../../scripts/box2d_paddle_puck_contact_scenario.py) (see parsing of `paddle:puck` sweep pairs, per-run `paddle_density` / `puck_density` recording, and pre/post-contact speed logging).

## Collision physics

All collision restitution is handled by a custom `CollisionForceListener` that **disables Box2D's built-in restitution** and re-applies it deterministically in `PostSolve`. This avoids jitter from Box2D's global `b2_velocityThreshold`.

### Bodies and restitution values

| Body | Shape | Restitution (fixture) |
|---|---|---|
| Puck | Circle (`puck_radius`) | `puck_restitution` (e.g. 1.09145) |
| Paddle | Circle (`paddle_radius`) | `paddle_restitution` (default 1.0) |
| Side walls (left/right) | `b2EdgeShape` | `side_wall_restitution` (e.g. 0.99) |
| End walls (top/bottom) | `b2EdgeShape` | `end_wall_restitution` (e.g. 0.70) |

Friction is 0.0 everywhere. Gravity is a downward `gravity` value (e.g. `-0.65 m/s²`) simulating table tilt.

### Puck ↔ wall

**PreSolve:** computes `incoming_speed` (puck's normal-component speed toward the wall), disables Box2D's restitution (`contact.restitution = 0.0`), and stores `{incoming_speed, normal_inward, restitution}` keyed by puck name.

**PostSolve:** reads the stored entry and applies a corrective impulse:
- If `incoming_speed >= puck_wall_restitution_threshold_speed` (default 0.25 m/s): `v_out = incoming_speed * restitution`
- If `incoming_speed < threshold`: enforces a minimum rebound of `puck_wall_min_rebound_speed_below_threshold` (default 0.1 m/s)

The impulse is `J = mass * (target_outgoing - current_outgoing)` applied along the inward normal.

### Puck ↔ paddle

**PreSolve:** computes relative approach speed of puck w.r.t. paddle along the contact normal. Uses `combined_e = max(puck_restitution, paddle_restitution)`. Disables Box2D restitution and stores state in `_pending_paddle_puck`.

**PostSolve:** enforces `v_rel_desired = e * approach_speed` via a momentum-conserving reduced-mass impulse:

```
j = (v_rel_desired - v_rel_post) * m_paddle * m_puck / (m_paddle + m_puck)
```

Applied as `+j` to puck and `-j` to paddle along the contact normal.

A `puck_restitution > 1.0` (e.g. 1.09145) means the puck leaves slightly faster than it arrived — simulating a springy puck.

### Paddle force model

The paddle is a **dynamic body** driven by a PID controller (`pid_kp`, `pid_kd`, `pid_ki`). Each step the policy outputs a target position; the PID computes a force that is applied to the paddle body. The paddle's velocity at contact time is the result of this accumulated force, so the collision outcome depends on PID tuning and the step timing. `paddle_damping` provides heavy deceleration between steps.

### Quick-reference with `pid_noise_constant_upper_half_custom_sim_params.yaml`

```
Puck-side wall:   e = 0.99,   threshold = 0.25 m/s,  min rebound = 0.1 m/s
Puck-end wall:    e = 0.70,   same threshold logic
Puck-paddle:      e = 1.09145  (superelastic — puck gains energy from springiness)
Puck damping:     0.25
Paddle damping:   17
Gravity:         -0.65 m/s²
PID gains:        Kp=5000, Kd=200, Ki=0
```

## Per-tier collision scaling and external parameter interface

The simulator supports per-speed-tier restitution multipliers that an external optimizer (or another RL agent) can update during training without restarting.

### Speed tiers

| Tier | Incoming speed range |
|---|---|
| `low` | < 0.25 m/s |
| `mid` | 0.25 – 0.75 m/s |
| `high` | ≥ 0.75 m/s |

Breakpoints are configurable. The effective restitution for each collision is `base_restitution * scale[tier]`.

### Simulator API

```python
# Push new scales at an episode boundary.
sim.set_collision_scales(
    wall_scales=[1.0, 1.0, 1.0],    # [low, mid, high]
    paddle_scales=[1.0, 1.0, 1.0],
    speed_breakpoints=(0.25, 0.75), # optional, m/s
)

# Read per-tier stats accumulated during the episode (resets counters).
stats = sim.get_episode_collision_stats()
# stats = {
#   "wall":   {"low": {"count", "mean_speed_in", "mean_speed_out"}, ...},
#   "paddle": {"low": ..., "mid": ..., "high": ...},
# }
```

### `CollisionParamManager`

[`airhockey/sims/collision_param_manager.py`](../../../../airhockey/sims/collision_param_manager.py) bridges the training loop and an external optimizer.

**Training loop integration:**

```python
from airhockey.sims.collision_param_manager import CollisionParamManager

manager = CollisionParamManager(
    status_path="runs/collision_status.json",
    params_path="runs/collision_params.json",
)
manager.attach_sim(sim)   # pushes current params to sim immediately

# --- end of each episode ---
stats = sim.get_episode_collision_stats()
manager.on_episode_end(
    episode=episode_idx,
    collision_stats=stats,
    episode_outcome={"total_reward": rew, "juggle_count": juggles},
)
```

**External optimizer (any process):**

```python
import json

# Read: what happened last episode?
status = json.load(open("runs/collision_status.json"))
# status["collision_stats"], status["current_params"], status["episode_outcome"]

# Write: push new params for the next episode.
json.dump(
    {"wall_scales": [0.95, 1.0, 1.05], "paddle_scales": [0.9, 1.0, 1.1]},
    open("runs/collision_params.json", "w"),
)
```

The manager polls `collision_params.json` on each `on_episode_end` call, consumes and deletes it, and calls `sim.set_collision_scales()` automatically. Writes to `collision_status.json` are atomic (temp-file + rename), so the external process never reads a partial file.

### Status file schema

```json
{
  "episode": 1042,
  "current_params": {
    "wall_scales": [1.0, 1.0, 1.0],
    "paddle_scales": [1.0, 1.0, 1.0],
    "speed_breakpoints": [0.25, 0.75]
  },
  "collision_stats": {
    "wall": {
      "low":  {"count": 12, "mean_speed_in": 0.14, "mean_speed_out": 0.10},
      "mid":  {"count": 34, "mean_speed_in": 0.48, "mean_speed_out": 0.47},
      "high": {"count":  8, "mean_speed_in": 0.91, "mean_speed_out": 0.95}
    },
    "paddle": {
      "low":  {"count":  3, "mean_speed_in": 0.11, "mean_speed_out": 0.12},
      "mid":  {"count": 21, "mean_speed_in": 0.41, "mean_speed_out": 0.44},
      "high": {"count":  9, "mean_speed_in": 0.88, "mean_speed_out": 0.98}
    }
  },
  "episode_outcome": {
    "total_reward": 12.4,
    "juggle_count": 7,
    "termination_reason": "puck_hit_bottom"
  }
}
```

## Sim-to-sim collision adaptation

`scripts/collision_adaptation/` implements a two-phase algorithm that tunes the learner sim's per-tier paddle restitution scales to match an oracle sim's collision behaviour.  This is the sim-to-sim proxy for the eventual real-to-sim workflow where real-world collision statistics replace the oracle rollouts.

### What is being fit

The simulator has three **per-tier restitution scale multipliers** for paddle-puck collisions — one per speed tier (see [Per-tier collision scaling](#per-tier-collision-scaling-and-external-parameter-interface) above). A scale > 1 means the puck bounces off faster than the base restitution; < 1 means it absorbs more energy. These three numbers `[low_scale, mid_scale, high_scale]` are the parameters the adaptation procedure fits.

The oracle simulator represents "ground truth" physics — either a hand-set target (sim-to-sim testing) or eventually real-world collision statistics (sim-to-real). The learner starts at `[1.0, 1.0, 1.0]` and the algorithm adjusts its scales until its outgoing puck speeds per tier match the oracle's.

### How the fitting loop works

Both oracle and learner use the **same trained TD3 policy** for rollouts. The policy runs deterministically (no exploration noise) and generates realistic paddle motions including varied approach angles and speeds. After `n_episodes` episodes the simulator's `get_episode_collision_stats()` returns per-tier `{count, mean_speed_in, mean_speed_out}` accumulated across all episodes.

For each speed tier `t`, the update rule is:

```
ratio_t  = oracle_mean_speed_out_t / learner_mean_speed_out_t
scale_t' = scale_t × (1 + lr × (ratio_t − 1))
scale_t' = clamp(scale_t', min_scale, max_scale)
```

This is **proportional control on the speed ratio**: if the oracle bounces the puck 20% faster than the learner (`ratio = 1.2`), the learner scale is increased by `lr × 0.2`. With `lr = 0.2` this takes a 20% step toward matching each iteration, giving geometric convergence in the noise-free case.

A tier is **skipped** if either simulator collected fewer than `min_count` (default 3) collisions for that tier. This prevents noisy updates from a single outlier event.

The convergence metric reported each iteration is `max(|ratio_t − 1|)` across non-skipped tiers — the worst-case fractional speed mismatch remaining.

Wall bounces are intentionally ignored; only paddle-puck collisions are used because the oracle's wall scales are fixed at 1.0 (same as learner) in this setup.

### Why the update rule is well-behaved

Outgoing puck speed is approximately linear in the restitution scale (doubling the scale roughly doubles the exit speed), so the ratio `oracle_out / learner_out` is a direct proxy for the scale ratio needed. The multiplicative form `scale × (1 + lr × (ratio − 1))` is equivalent to a gradient step on the squared log-speed error. It converges as long as `lr < 1`, with smaller `lr` giving more stability at the cost of slower convergence.

### Observed convergence behaviour (oracle = [0.7, 1.0, 1.2])

| Tier | Oracle | Learner start | Learner after 20 iters | Notes |
|------|--------|---------------|------------------------|-------|
| low  | 0.7    | 1.0           | ~0.69                  | Noisy — low-speed collisions are rare during normal juggling play; count often hits `min_count` threshold |
| mid  | 1.0    | 1.0           | ~0.96                  | Moderate noise — mid-speed collisions occur but signal is ~10–20 per rollout |
| high | 1.2    | 1.0           | ~1.18                  | Cleanest — high-speed collisions happen frequently; converges within ~9 iterations |

`max(|ratio_t − 1|)` starts around 0.35 and oscillates in the 0.05–0.20 range after ~10 iterations rather than reaching near-zero. This residual noise is dominated by rollout variance, not algorithm correctness.

**Known limitation — low-tier coverage:** The juggling policy keeps the puck moving fast, so slow paddle-puck collisions are rare. The `low` scale converges slowly and noisily. Remedies: increase `n_episodes`, lower `min_count`, or use a dedicated slow-puck data-collection phase to generate more low-tier collisions.

### Files

| File | Purpose |
|---|---|
| `scenarios.py` | 33 crafted collision scenarios spanning stationary, toward, retreat, lateral, diagonal, and extreme-speed paddle cases |
| `render_scenarios.py` | Phase 1: render 66 GIFs (33 scenarios × oracle + learner) for visual inspection |
| `rollout.py` | `rollout_episodes()` — runs n policy episodes and returns aggregate paddle tier stats using privileged Box2D velocities |
| `rollout_position_based.py` | `rollout_episodes_position_based()` — same interface but estimates velocities from noisy position histories (see below) |
| `adapt.py` | `compute_scale_updates()` — scale update rule + `max_abs_ratio_minus_one` convergence metric |
| `run_adaptation.py` | Phase 2: full adaptation loop CLI (privileged) |
| `run_adaptation_position_based.py` | Phase 2: full adaptation loop CLI (position-based) |

### Phase 1 — inspect before adapting

```bash
python scripts/collision_adaptation/render_scenarios.py \
    --config configs/new_juggle/sysid_best_params.yaml \
    --oracle-paddle-scales 0.7 1.0 1.2 \
    --output-dir runs/collision_adaptation \
    --fps 20
```

Outputs to `runs/collision_adaptation/inspect/`:
- 66 GIFs: `{oracle,learner}_scenario_{name}.gif`
- `scenarios.json` — pre/post puck speed for each scenario in both sims

Noise, occlusions, and action/observation delays are disabled for clean deterministic scenarios.

### Phase 2 — adaptation loop

```bash
python scripts/collision_adaptation/run_adaptation.py \
    --config configs/new_juggle/sysid_best_params.yaml \
    --model-path runs/td3/final/task_only/checkpoint_345000/model.pth \
    --oracle-paddle-scales 0.7 1.0 1.2 \
    --n-iterations 20 \
    --n-episodes 50 \
    --lr 0.2 \
    --output-dir runs/collision_adaptation
```

Outputs `runs/collision_adaptation/adaptation_history.json` with per-iteration scales, per-tier stats, and the convergence metric `max(|ratio_t − 1|)` for each iteration.

### Convergence expectations

- Oracle scales = `[1.0, 1.0, 1.0]` (identity): learner scales should stay near 1.0. Any drift indicates rollout noise, not a systematic error.
- Oracle scales = `[0.7, 1.0, 1.2]`: `high` converges cleanest (within ~9 iterations), `mid` follows, `low` is slowest due to sparse collision count. With 50 episodes the residual `max(|ratio_t − 1|)` typically plateaus around 0.05–0.20 rather than reaching zero — additional episodes or a decaying `lr` schedule would reduce this floor.

---

### Position-based variant

`rollout_position_based.py` / `run_adaptation_position_based.py` implement the same adaptation loop but **without privileged access to Box2D body velocities**. Instead, they estimate puck speeds from observed position trajectories — simulating what would be done on a real table where only camera-tracked positions are available.

#### How it works

1. During each episode, paddle-puck collision steps are detected by monitoring `get_collision_forces()` incrementally (checks `bodyA`/`bodyB` for `"paddle"` and `"puck"`).
2. After the episode ends, `env.simulator.puck_history` (a `[x, y, occluded]` list with observation noise and delay applied) is retrieved. The first 5 entries are spawn-time padding. **The step→index mapping depends on `enable_observation_delay`** (see [delay toggles side-effect](#-subtle-side-effect-enable_observation_delay-changes-puck_history-sampling-rate) above): with delay on (canonical baseline), each env step appends 2 entries, so step `k` maps to index `5 + 2*k` (and `puck_history` ends up at ~40 Hz density). With delay off, step `k` maps to `5 + k` (20 Hz). The position-based collision-window code below was originally written assuming the `5 + k` mapping; if you run it against the canonical baseline, double-check the windowing.
3. For each detected collision at step `col_idx`:
   - **Pre-collision window**: `positions[col_idx − W : col_idx]` — excludes `col_idx` itself because the position at that index is post-impact (physics runs before position is appended).
   - **Post-collision window**: `positions[col_idx : col_idx + W]` — the first entry is the first free-flight frame after impact.
   - Both windows are fitted with `fit_velocity_from_positions(gravity=(-0.65, 0.0))` from `airhockey/sims/real/velocity_estimator.py`.
   - Estimates with `snr < min_snr` (default 8.0) are discarded.
4. Pre-collision puck speed magnitude is used as the tier proxy (replaces approach speed from the privileged path).
5. Same `adapt.py` update rule is applied.

```bash
python scripts/collision_adaptation/run_adaptation_position_based.py \
    --config configs/new_juggle/sysid_best_params.yaml \
    --model-path runs/td3/final/task_only/checkpoint_350000/model.pth \
    --oracle-paddle-scales 0.7 1.0 1.2 \
    --n-iterations 50 \
    --n-episodes 100 \
    --lr 0.15 \
    --min-snr 8.0 \
    --output-dir runs/collision_adaptation_position_based
```

#### Experimental results (50 iters, 100 episodes, checkpoint_350000)

Two oracle configurations were tested:

**oracle = [0.7, 1.0, 1.2]** (subtle deviations)

| Tier | Oracle | Learner final | Expected | Verdict |
|------|--------|--------------|----------|---------|
| low  | 0.7    | 0.362        | ~0.7     | Overshot |
| mid  | 1.0    | 2.419        | ~1.0     | **Badly wrong** |
| high | 1.2    | 0.990        | ~1.2     | Stalled (62% skip rate) |

Mean collision counts: oracle high=20.6/iter vs learner high=4.5/iter — a 4× asymmetry that starves the high tier and causes cascading mid tier contamination.

**oracle = [0.5, 1.0, 2.0]** (large deviations)

| Tier | Oracle | Learner final | Expected | Verdict |
|------|--------|--------------|----------|---------|
| low  | 0.5    | 0.300        | ~0.5     | Hit min clamp (0.3), still overshot |
| mid  | 1.0    | 0.968        | ~1.0     | Correct ✓ |
| high | 2.0    | 1.853        | ~2.0     | Converging cleanly ✓ |

Mid ratio std = 0.073, high ratio std = 0.081 over last 20 iterations — stable signal.

#### Why the [0.7, 1.0, 1.2] case fails

The root cause is **distribution shift between oracle and learner**. With oracle high scale = 1.2, the oracle amplifies high-speed bounces, keeping the puck moving faster — generating ~4× more high-tier collisions than the learner. The learner's mid tier therefore fills with collisions that in the oracle would be in the high tier. This makes oracle mid-out consistently higher than learner mid-out (ratio ≈ 1.11 in the last 20 iters, not noise — std ≈ 0.04) and drives mid scale up to 2.4.

The [0.5, 1.0, 2.0] case avoids this because oracle mid = 1.0 keeps the mid-tier population symmetric; the strong high-tier signal (×2.0, 54 counts/iter) is sufficient to converge cleanly.

#### When to use each variant

| Situation | Recommendation |
|-----------|---------------|
| Sim-to-sim with known ground truth | Privileged (`run_adaptation.py`) — faster, more accurate |
| Large uniform deviations (factor ≥ 1.5×) | Position-based works for mid and high tiers |
| Subtle deviations across all tiers (< 30%) | Position-based unreliable — distribution shift contaminates mid tier |
| Real-world calibration | Position-based is the only option; accept that tier-crossing will require more episodes and potentially a decaying lr |

#### Known limitations

- **Tier proxy mismatch**: puck speed magnitude ≠ approach speed (which requires paddle velocity). If the policy's paddle-puck relative velocity differs systematically between oracle and learner, some collisions are misbucketed.
- **Low tier is always hard**: juggling generates very few slow collisions (9–23/iter at 100 episodes). Low-tier estimates are noisy regardless of the velocity estimation method.
- **Distribution shift**: any oracle configuration that substantially changes the puck speed distribution (by boosting or damping high-speed bounces) will contaminate the mid and high tier estimates through tier-boundary crossovers. This is a fundamental limitation of single-level bucketing.

## Puck observation sine y-warp (sim2sim perception error)

Edge-preserving sine warp on the puck's `y` (sideways) observation only. Models a partially-calibrated overhead tracker — corners anchored to the table side walls, interior reads bow off-true. Lives in `airhockey/observation_homography.py:apply_sine_y_warp_xy` and is plumbed through `airhockey/utils.py` via the `puck_obs_warp_fn` kwarg.

```
y_obs = y_true + A · sin(π · (y_true − y_left) / (y_right − y_left))
x_obs = x_true                                    # x is unchanged
```

- Edges preserved: `y_obs == y_true` at both side walls.
- Peak deviation `+A` at the midline (`y = 0`).
- Monotonic iff `|A| < (y_right − y_left) / π ≈ 0.275 m` at full table width. Enforced by `make_sine_y_warp_fn`.
- Paddle observations untouched. Physics untouched (collisions still resolve at the *true* puck position; only what gets written into the puck-history slots of the obs vector is warped).

**Config keys** (in `air_hockey.simulator_params`):

| Key | Default | Meaning |
|---|---:|---|
| `puck_obs_sine_warp_amplitude` | `0.0` | `A` in meters. `0.0` = warp disabled (no-op, `puck_obs_warp_fn` is `None`). |
| `puck_obs_sine_warp_y_left`  | `null` | Left edge of the warp domain. `null` defaults to `−width/2`. |
| `puck_obs_sine_warp_y_right` | `null` | Right edge. `null` defaults to `+width/2`. |

**Canonical sim2sim target using this warp**: `configs/new_juggle/sim2sim_combined_warp.yaml` (paddle50 + dynamics deltas + `A = 0.05 m`). See [`notes/scratch/experiments/2026-05-07_02-05_sim2sim-puck-obs-warp.md`](../../../scratch/experiments/2026-05-07_02-05_sim2sim-puck-obs-warp.md) for the rationale and the visualization at `/tmp/sine_warp_viz.png`.

The older `obs_position_homography` (3×3 perspective matrix applied to both paddle and puck) was removed in favor of this puck-only mechanism. See the experiment writeup for what was removed.

## Paddle boundary visualization utility

For Box2D boundary debugging and validation, use:

- [`scripts/box2d_boundary_validation/validate_paddle_bounds_gif.py`](../../../../scripts/box2d_boundary_validation/validate_paddle_bounds_gif.py)

This script renders a looped GIF of the paddle tracing the effective boundary and overlays the actual clipped boundary polygon (including corner cuts from edge-shaping).

### Parameters (brief)

- **Workspace bounds**
  - `--x-min`, `--x-max`, `--y-min`, `--y-max`: rectangle-style workspace limits to test.
- **Corner / edge shaping**
  - `--top-abs`: slope factor for right-side corner tapering versus `y`.
  - `--max-bias-m`, `--max-bias-p`: right-edge bias terms used in `x_max(y) = min(x_max, max_bias_m - top_abs*y, max_bias_p + top_abs*y)`.
  - `--bot-abs`: accepted for parity with config, but current clip math keeps `x_min` fixed and does not currently apply this term.
- **Coordinate-frame interpretation**
  - `--limits-frame raw_robot|centered`: whether input x-limits are raw-robot style (converted by `center_offset_constant`) or already centered.
- **Traversal controls**
  - `--loops`: number of perimeter loops.
  - `--steps-per-edge`: sampling density along each perimeter segment.
  - `--control-substeps`: inner control updates per waypoint.
  - `--action-scale`: per-update movement magnitude scaling.
  - `--position-tol`: tolerance used for boundary-touch and violation checks.
- **Rendering / outputs**
  - `--fps`: GIF framerate.
  - `--name`, `--output-dir`: output naming/path.
  - `--renderer-orientation`: renderer orientation (project rule defaults this to `vertical`).
  - `--config-path`: YAML source of `air_hockey.simulator_params`.

### Artifacts

- GIF: `runs/paddle_boundary_validation/<name>.gif`
- Summary JSON: `runs/paddle_boundary_validation/<name>.json`
