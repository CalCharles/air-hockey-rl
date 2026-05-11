# Decoupling observation delay from dynamics — design doc

**Status:** proposal, not yet implemented.
**Author context:** see [`notes/scratch/experiments/2026-05-05_17-38_zero-shot-sim2real-ablations.md`](experiments/2026-05-05_17-38_zero-shot-sim2real-ablations.md) for the failed `no_obs_delay` ablation that surfaced this.

## What's broken today

In `airhockey/sims/airhockey_box2d.py:get_singleagent_transition`, toggling `enable_observation_delay` changes the number of physics sub-steps inside an env step (1 vs 2), which silently changes:

- **Action smoothing** (`_filter_update`, `hist_len=2`): runs per sub-step. With delay on it ends up a no-op for the second sub-step; with delay off it becomes a permanent lag-1 50/50 averager. Net effect: policy action authority is ~75 % with delay on vs ~50 % with delay off — a 33 % swing from one boolean.
- **Puck/paddle history density**: appended per sub-step → 40 Hz with delay on, 20 Hz with delay off. The policy's `puck_history[-5:]` covers different real-time spans (~125 ms vs ~250 ms).
- **PID + force application cadence**: recomputed and re-applied per sub-step.
- **Noise / occlusion / action-attenuation sampling**: per sub-step.

The real-world simulator (`airhockey/sims/air_hockey_real.py`) appends history once per 20 Hz env step and applies smoothing once per env step. So the "delay off" path matches real for those mechanics, the "delay on" path doesn't — and the canonical recipe was trained with delay on. This is the silent root of the `no_obs_delay` training failure and a latent sim-to-real obs distribution gap.

## What we want

1. **Sim and real have identical dynamics structure**: 20 Hz control, action smoothing at 20 Hz, history at 20 Hz, one PID + force decision per env step.
2. **Physics integration is fine-grained for accuracy**: default 25 ms sub-steps (2 per env step). Sub-step granularity is independent of `observation_delay_seconds`.
3. **Observation delay is a pure observation-time lookup**: pluck the world state from `now − D` and feed it to the policy. Toggling `D` cannot affect dynamics. `D` is configurable to any value (≥ 0), with optional jitter.

## Design

### One env step (`get_singleagent_transition`)

```
1.  Compute PID target_pos from current paddle pos + filtered policy action
        (target = pos + action * move_lims)
2.  _filter_update — append (pos, target) to pose/dpose deques
        (still maxlen=2; runs ONCE per env step → matches real-world 20 Hz smoothing)
3.  Compute force from PID(filtered_target, pos, vel)
4.  Apply boundary / clip / scale / action-force-attenuation (all sampled ONCE)
5.  For each sub-step in N (default N=2, sub_dt = 0.025):
        - ApplyForceToCenter(force)        # re-applied; Box2D clears per Step
        - world.Step(sub_dt, 100, 100)
        - record snapshot at sub-step end-time into _obs_snapshot_buffer
6.  Append to puck_history / paddle_history ONCE (end-of-step state)
7.  Sample puck_noise / occlusion ONCE (applied to the obs at lookup time, not stored raw)
8.  Update acceleration / jerk / e-stop derivatives
9.  last_action = action
```

Key invariant: PID, smoothing, force computation, history append, noise/occlusion all fire **once per env step**, regardless of N. The sub-step loop is *only* there to advance Box2D physics in finer increments and to populate the snapshot buffer.

### Snapshot buffer

```python
self._obs_snapshot_buffer: deque  # maxlen ≈ ceil(max_delay / sub_dt) + 2
# entry: { "t": world_time, "puck_pos": (x,y), "puck_vel": (vx,vy),
#          "paddle_pos": (x,y), "paddle_vel": (vx,vy) }
```

Pushed once per sub-step (line 5 above). With default `sub_dt = 0.025`, snapshots land at world times 25, 50, 75, 100, … ms — so any delay that's a multiple of 25 ms hits a snapshot exactly.

### Obs delay lookup

```python
def get_delayed_obs_state(self, delay_seconds: float) -> dict:
    # 1. target_time = current_world_time - delay_seconds + per-step jitter
    # 2. Find bracketing snapshots S_lo, S_hi with S_lo.t <= target_time <= S_hi.t
    # 3. frac = (target_time - S_lo.t) / (S_hi.t - S_lo.t)
    # 4. Return state_info-shaped dict with positions = lerp(S_lo, S_hi, frac)
    #    and valid_flag = S_lo.valid AND S_hi.valid
    # 5. Apply puck_noise / occlusion to the result here, not at sim-step time
```

`delay_seconds = 0` returns the latest snapshot directly (no lerp). `delay_seconds = 0.025` with default `sub_dt = 0.025` hits a buffered snapshot exactly (frac = 0, no lerp). Linear interpolation only kicks in when delay falls between sub-step boundaries (e.g., jittered to 27 ms).

In `airhockey/airhockey_base.py:single_agent_step`, replace the `observation_state_info`-fallback branch with a single call to `simulator.get_delayed_obs_state(simulator.observation_delay_seconds_with_jitter())`.

### YAML config

```yaml
simulator_params:
  physics_substep_seconds: 0.025          # default; must divide time_per_step evenly
  observation_delay_seconds: 0.025        # 0 disables delay
  observation_delay_jitter_seconds: 0.00625  # ±value uniform; 0 disables jitter
```

These three replace **all** of the following legacy keys (which become deprecated shims that warn + translate for one release, then are deleted):

| Legacy key | Replacement |
|---|---|
| `enable_observation_delay` | `observation_delay_seconds > 0` |
| `enable_action_delay` | drop (never used in canonical recipe; reintroduce later as `action_delay_seconds` symmetrically with an action buffer if needed) |
| `delay_seconds` | `observation_delay_seconds` |
| `randomize_delay` + `delay_relative_range` | `observation_delay_jitter_seconds` |
| `enable_puck_delay_interpolation`, `puck_delay_interpolation_min/max` | drop (was a hack for the same problem) |

## Why 25 ms default sub-step

- 25 ms ÷ 50 ms env step = 2 sub-steps. Same physics-integration accuracy as the current canonical "delay on" path, so we don't regress puck/paddle dynamics.
- 25 ms snapshot grid lines up exactly with the canonical real-world latency target (25 ms ± 25 % jitter), so the typical use case never needs interpolation.
- Anyone wanting finer delay resolution sets `physics_substep_seconds: 0.0125` (4 sub-steps) — a single config knob, no other code changes.

## What changes for code consumers

- The simulator no longer exposes `observation_state_info` / `observation_puck_history` / `observation_paddle_history`. Replaced by `get_delayed_obs_state()`.
- `puck_history` and `paddle_history` are unambiguously 20 Hz (one append per env step). All downstream code that assumed `puck_history[5+k]` indexing for episode step `k` (e.g., `simulator-essentials.md` collision-window docs, `scripts/collision_adaptation/rollout_position_based.py`) becomes correct again — and matches real-world's history density.
- Action smoothing (`_filter_update`) is unambiguously a 20 Hz lag-1 averager. Same as real.
- `enable_observation_delay` no longer exists as a knob; ablation is `observation_delay_seconds: 0.0` vs `0.025`.

## Migration plan

1. **Land the simulator change** (one PR). Keep legacy YAML keys as deprecation shims that warn and translate. Smoke-test on `td3_zeroshot_baseline.yaml`-equivalent config to confirm training still launches and produces step logs.
2. **Trajectory-diff sanity check** (~30 min): seeded fixed-action rollout on legacy "delay on" sim vs new sim with `observation_delay_seconds: 0.025`. Compare `puck_history`, `paddle_history`, episode return. Expect small differences from the force-application change (PID computed once not twice per env step), large differences would indicate a bug.
3. **Retrain canonical** (`hist2_motion0_v3`, ~3.5 h on one GPU at 1 M steps). Eval on the same source sim. Compare peak / mean to v2 (169.72 / 145).
4. **Sim2real handoff** with v3 model on the user's other machine. Compare to v2 sim2real result.
5. **Re-run obs-delay ablation cleanly**: `observation_delay_seconds ∈ {0.0, 0.025, 0.050}` × 500 k steps. This is the experiment we wanted from day one — varies *only* obs lag, with no entangled side effects.
6. **Delete legacy YAML keys** after one release of deprecation warnings.

## Risks

- **Box2D under one big force application vs two small ones**: with sub-stepping the force is now re-applied identically at each sub-step instead of recomputed via PID. Paddle dynamics will differ slightly from legacy canonical (where PID self-corrected mid-step). Mitigation: trajectory-diff sanity check (step 2 above). Expected diff is small; if large, probably a bug.
- **Existing checkpoints (v1, v2) trained on entangled mechanics** may transfer worse to the simplified sim than a freshly-trained v3. Treat them as legacy.
- **Linear interp during paddle-puck contact**: rare (collision intervals are short relative to 25 ms snapshots) but possible. If sim2real shows a "wrong puck velocity right after a hit" artifact, revisit by triggering an extra sub-step around detected collisions. YAGNI until evidence.
- **Real-world simulator unchanged**: this design only touches `airhockey_box2d.py`. Real env continues to use its existing single-append-per-env-step semantics, which is what we're aligning sim to. No risk to real-world rollout code.

## File touch list

- `airhockey/sims/airhockey_box2d.py` — rewrite `get_singleagent_transition`, add `_obs_snapshot_buffer` + `get_delayed_obs_state`, simplify config dict, add deprecation shims for legacy keys.
- `airhockey/airhockey_base.py:899-906` — replace snapshot-fallback branch with single `get_delayed_obs_state` call.
- `notes/docs/environments/box2d/simulator-essentials.md` — rewrite the "Delay toggles" section, fix the `puck_history` indexing claims (now unambiguously 20 Hz).
- `notes/docs/environments/observation-action-spaces.md` — remove the "temporal density caveat" section (no longer applicable; sim and real both 20 Hz).
- `configs/new_juggle/sysid_best_params*.yaml` — replace legacy delay keys with the three new ones.
- Optional: `notes/docs/training/td3-configs.md` — note the v3 retrain.
