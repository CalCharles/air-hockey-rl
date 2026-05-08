# Sim2sim puck-observation sine y-warp — implementation, no rollouts yet

- **Date**: 2026-05-07 02:05 UTC
- **Status**: implementation landed; awaiting first rollout (zero-shot eval + residual fine-tune)
- **Configs**:
  - Target sim: `scripts/smooth_policy/amp_history/configs/new_juggle/sim2sim_combined_warp.yaml`
  - Source policy: `latest_model/hist2_motion0_v2/`

## Question

Build a sim2sim target where adaptation success is a credible proxy for real-world adaptation. Three perturbations need to be present together so that no single class of error dominates:

1. **Smaller paddle** — already in `sim2sim_combined.yaml` (paddle50, mass-preserved).
2. **Systematic puck observation error** — NEW: the policy reads puck position warped along the lateral axis. Models a partially-calibrated overhead tracker where the table corners are anchored but interior reads bow off-true.
3. **Different dynamics** — already in `sim2sim_combined.yaml` (`pid_kp 9000→7200`, action delay enabled, wider wall cone, etc.).

This file documents (2) — the new puck-y sine warp.

## Setup

### Coordinate convention

Verified in `airhockey/airhockey_base.py:213-216`. The Box2D env uses:
- `x` = lengthwise (long axis, `length=1.9304m`, `table_x_top/bot = ±0.9652`). The paddle moves into the upper half along `x`.
- `y` = sideways (short axis, `width=0.8636m`, `table_y_left/right = ±0.4318`). This is the **horizontal** axis from a viewer at the player end.

So "horizontal" = `y`. The warp is on `y`.

### Warp formula

For each puck observation (current frame + 4 history frames):

```
y_obs = y_true + A · sin(π · (y_true − y_left) / (y_right − y_left))
x_obs = x_true
```

- Edge-preserving: `sin(0) = sin(π) = 0`, so `y_obs == y_true` at both side walls.
- Peak displacement `+A` at the midline (`y = 0`).
- **Monotonic iff `|A| < (y_right − y_left) / π ≈ 0.275 m`** at full table width. Enforced by an assertion in `make_sine_y_warp_fn`.
- Paddle observations untouched. Physics untouched (collisions still resolve at the *true* puck position; only what gets written into the obs vector is warped).

### Default amplitude — `A = 0.05 m`

At the midpoint, the policy reads the puck about one paddle50-diameter (5cm) off from where it really is. At the side walls, perception matches truth. Velocity proxy `obs[27:29] − obs[15:17]` is also indirectly perturbed because the local stretch factor varies from `1 − Aπ/W ≈ 0.82` to `1 + Aπ/W ≈ 1.18` across the table.

Picked because it's "meaningful enough that the policy must adapt, small enough that the task isn't crippling." Not yet tuned against zero-shot drop — pick by feel after first rollout.

## Implementation

All changes are gated by `puck_obs_sine_warp_amplitude` (default `0.0` = no-op). Existing configs that don't set this key produce byte-identical observations to before.

### Files modified

| File | Change |
|---|---|
| `airhockey/observation_homography.py` | Added `apply_sine_y_warp_xy(x, y, A, y_left, y_right)` and `make_sine_y_warp_fn(A, y_left, y_right)`. Factory returns `None` when `A==0` (no-op convention) and raises if `A` would break monotonicity. |
| `airhockey/utils.py` | Added `_maybe_apply_puck_warp` and `_apply_puck_warp_history`. Threaded a `puck_obs_warp_fn` kwarg through `get_observation_by_type`. Applied at every puck call site (12 total across 9 obs types — paddle sites untouched). |
| `airhockey/sims/airhockey_box2d.py` | Added 3 new `simulator_params` keys: `puck_obs_sine_warp_amplitude` (float, default `0.0`), `puck_obs_sine_warp_y_left` / `_y_right` (float, default `None` → table side walls). Init builds `self.puck_obs_warp_fn`. |
| `airhockey/airhockey_base.py` | `_get_observation_by_type_with_position_homography` reads `simulator.puck_obs_warp_fn` and passes it as a kwarg. |

### Files created

| File | Purpose |
|---|---|
| `scripts/smooth_policy/amp_history/configs/new_juggle/sim2sim_combined_warp.yaml` | Clone of `sim2sim_combined.yaml` + `puck_obs_sine_warp_amplitude: 0.05`. The full three-perturbation sim2sim target. |

### Removed — observation homography (deprecated as part of this change)

The older `obs_position_homography` mechanism in box2d was a more complex, less-targeted way to perturb obs (3×3 perspective matrix, applied to *both* paddle and puck). The sine warp supersedes it for the puck-only perception-error use case. Removed:

- `airhockey/sims/airhockey_box2d.py`: `enable_obs_position_homography`, `obs_position_homography_matrix`, `obs_position_homography_seed` defaults + init block.
- `airhockey/airhockey_base.py`: `position_homography` kwarg threading.
- `airhockey/utils.py`: `_maybe_warp_xy`, `_warp_history_xy`, all paddle/puck homography call sites.
- `airhockey/observation_homography.py`: `apply_plane_homography_xy`, `sample_near_identity_homography`, `pixel_homography_from_world_homography`.
- `scripts/smooth_policy/validate_obs_homography_gif.py`: deleted (purpose-specific renderer).
- `scripts/smooth_policy/amp_history/amp_training/td3/tests/test_observation_homography.py`: replaced with `test_sine_y_warp.py` covering the new helper.
- `scripts/smooth_policy/amp_history/configs/new_juggle/sim_real_world_adaptation.yaml`: removed the 3 homography keys (other physics keys preserved; this config is already labeled legacy).

## Verification (math + env wiring only — no rollouts yet)

1. **Math**: `apply_sine_y_warp_xy` returns identity at both edges (deviation `0.0e+00`); midpoint deviation matches `+A` exactly; x-coordinate unchanged; monotonicity guard fires at `A = W/π`.
2. **Env wiring**: built env from `sim2sim_combined.yaml` with warp on/off (same seed). Disabled = `puck_obs_warp_fn is None`. Enabled differs only in obs slots `[16, 19, 22, 25, 28]` (the y component of each of the 5 puck history entries). Paddle slots `[0:15]` byte-identical. After 5 simulation steps with the same actions, paddle current-pos identical, puck `dx = 0`, puck `dy = +0.0364` matches `0.05 · sin(π · (0.2075 − (−0.4318)) / 0.8636)` exactly.

## Conclusion

Implementation landed and verified end-to-end on the math and env-wiring side. **No conclusion yet on whether `A = 0.05` is the right perturbation strength** — that requires zero-shot eval against `latest_model/hist2_motion0_v2/`.

Predicted but untested: zero-shot mean drops from ~67.5 (paddle50 alone) to somewhere in the 50s. If it barely moves, bump `A` to `0.08` or `0.10`. If it crashes below 30, drop `A` to `0.03`. Aim for ~50.

## Next

1. Zero-shot eval: `latest_model/hist2_motion0_v2/` on `sim2sim_combined_warp.yaml`, n=50 episodes. Compare to paddle50 baseline (67.5).
2. If amplitude needs adjustment, repeat zero-shot at the new value before any fine-tuning.
3. Once `A` is settled, residual RL fine-tune on this target, single seed, ~100k–300k steps. Reuse the v27 / v30_explore_lite recipes from the residual-RL recipe doc — same big-gap regime as paddle50.
4. If adaptation works on the source-side, schedule a real-world residual fine-tune on the same source policy as a sanity check that this proxy generalized.
