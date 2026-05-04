# Wall-Collision System Identification

Fitting Box2D's `side_wall_restitution` against real-world side-wall puck bounces (the y± table boundaries in the table-frame convention — despite the parameter name, `side_wall_restitution` is applied to the `table_x_min/max` walls in `airhockey_box2d.py`, which map to the y± walls the puck actually hits).

## Motivation

The [puck grid search](puck-system-id.md) fits gravity and damping on non-colliding trajectories. It cannot constrain wall restitution — a separate dataset of wall-bounce segments is needed.

A preliminary Box2D replay over 11 curated wall-bounce segments (`sysid/wall_collision_fit.py`, output at `sysid/wall_collision/box2d_eval/summary.txt`) showed that sim wall rebounds retain ~99% of incoming speed (matching the configured `side_wall_restitution = 0.99`) while real-world clean bounces drop to roughly 65% — a ~34 pp speed-ratio gap. The sweep below asks whether closing that gap actually improves post-collision position prediction.

## Data

Curated wall-bounce segments under `sysid/wall_collision/trajectory_dataN/wall_<start>_<end>/`, sourced from teleop and online recordings. Each segment is a short window around a clean puck-wall bounce (no paddle contact in the bounce window).

**11 segments with detected collisions** (1 rejected — `td0_wall_655_700` had no clean bounce). Per-segment `real_ratio = |v_after| / |v_before|` ranges from 0.32 to 1.64 (values > 1 are noise/measurement artifacts).

## Method

Scripts: `sysid/wall_collision_fit.py` (detection + fit + replay) and `sysid/wall_collision_restitution_fit.py` (sweep over restitution).

1. **Collision detection**: find the frame of a sharp sign flip in the perpendicular puck-velocity component, near a wall, with no paddle contact.
2. **Pre-collision velocity fit**: LSQ fit of the damped kinematic model on the pre-collision window, using the sysid-best puck params `gx = -0.661, γ = 0.178`. Produces `v_fit_prev` — the model-predicted puck velocity one frame before the bounce.
3. **Box2D replay**: reset the sim to the real puck position one frame before collision with `v_fit_prev` as the initial velocity, then step Box2D forward replaying the reconstructed real action stream through the PID (paddle doesn't collide in this window, so paddle tracking is incidental).
4. **Metric**: mean Euclidean puck-position error over the first **5 post-bounce frames**. Longer windows get contaminated by secondary bounces and integration drift.
5. **Fit-set filter**: a segment enters the fit set only if `real_ratio < 1.0` (physical) **and** pre-collision fit residual < 2 cm. Yields 8 / 11 segments. The remaining 3 are evaluated but not used to pick the winner.

## Sweep results

Grid: `side_wall_restitution ∈ {0.45, 0.50, 0.55, ..., 0.95}` (11 values). The default `0.99` is not in the grid but is reported per-segment for comparison.

Output: `sysid/wall_collision/box2d_eval/restitution_fit/summary.txt` + `sweep_curve.png` + side-by-side GIFs per segment.

| restitution | fit-set mean err (cm) | all-seg mean err (cm) |
|---|---|---|
| 0.45 | 7.27 | 7.18 |
| 0.55 | 6.59 | 6.59 |
| 0.65 | 5.93 | 6.02 |
| 0.75 | 5.28 | 5.48 |
| 0.85 | 4.72 | 5.03 |
| 0.95 | **4.39** | **4.76** |
| **0.99 (default, outside grid)** | **4.29** | **4.68** |

### Key finding

**The sweep is monotonic up to 0.95 but never crosses the default 0.99.** On the 5-frame post-collision metric, `0.99` beats the grid's best (`0.95`) by ~0.10 cm on both the fit set and all segments.

So: the real puck loses ~35% of its speed in a bounce, but the sim's much higher 0.99 restitution produces **lower** 5-frame position error than any lower value. Two factors explain this:

1. **Damping absorbs the extra energy fast.** With `γ = 0.178`, the sim puck slows enough in 5 frames that the initial overshoot from high restitution doesn't accumulate much position error.
2. **Reducing restitution shifts the post-bounce speed too far low.** Lowering it toward the real ratio (~0.65) makes the sim puck move too slowly — the position error from undershooting is bigger than the error from overshooting at 0.99.

### Conclusion

The 5-frame position metric does **not** discriminate between the sweep's best (0.95) and the old default (0.99) — both produce ~4.3 cm fit-set mean error, with 0.99 edging out 0.95 by 0.10 cm. But the metric misses the underlying speed-ratio gap: real bounces lose ~35% of their speed while sim at 0.99 loses ~1%. That gap would compound on multi-bounce rallies and bias speed-thresholded rewards.

**Applied change — eye-balled system ID:** `side_wall_restitution` set from `0.99 → 0.925` in `sysid_best_params.yaml` (and the `hist3/hist4/hist5` siblings). This is not what the 5-frame position metric recommended. Rationale:

- The real-world speed ratio on clean bounces is ~0.65. The position-error metric is insensitive to this mismatch on short horizons, so the sweep validates both 0.99 and 0.95 equally, which is uninformative.
- Splitting the difference toward a lower value at least narrows the speed-ratio gap without meaningfully hurting the position metric (0.95 was 0.10 cm worse; 0.925 is expected to be between 0.10 and 0.20 cm worse — well within the ~3 cm per-segment variance).
- An exhaustive joint refit over (restitution, damping) would be the principled answer; this is a low-cost interim.

**Verification that the param is wired through**: the puck is spawned with `restitution = puck_restitution` (`airhockey_box2d.py:1169`), and the side walls are created as edge fixtures with their own `restitution = side_wall_restitution` (lines 768–769, 778). The custom `CollisionForceListener.PreSolve` reads `wall_fixture.restitution` into a pending record (line 194, 242), and `PostSolve` applies `target_outgoing = incoming_speed * restitution * scale` via an impulse (lines 337–366). Single-step probes at 0.99 / 0.925 / 0.70 produce post-bounce speed ratios of 0.98 / 0.92 / 0.69 respectively (the 0.01 reduction from the bare restitution value is the `puck_damping = 0.178` acting over the 0.05 s env step).

The speed-ratio mismatch would also matter on:
- multi-bounce rallies (compounding speed overestimate),
- tasks that key on puck speed directly (e.g. goal-speed thresholds), or
- longer prediction horizons than 5 frames.

The proper follow-up is a combined restitution + damping re-fit rather than restitution alone — the two parameters trade off along the position metric.

## Per-segment post-collision error

First-5-frames mean, cm. `delta = best − default` (negative means default is better).

| Segment | wall | real_ratio | default (0.99) | best (0.95) | delta | set |
|---|---|---|---|---|---|---|
| td0 / wall_1030_1080 | y+ | 0.65 | 5.36 | 5.57 | −0.21 | fit |
| td0 / wall_1190_1280 | y+ | 0.84 | 4.62 | 4.70 | −0.08 | fit |
| td0 / wall_200_300 | y+ | 0.64 | 7.82 | 8.24 | −0.42 | fit |
| td0 / wall_440_490 | y− | 0.50 | 2.38 | 2.13 | +0.25 | fit |
| td0 / wall_850_880 | y+ | 0.67 | 6.00 | 6.17 | −0.17 | fit |
| td457 / wall_10_45 | y+ | 1.19 | 2.05 | 2.14 | −0.10 | eval |
| td461 / wall_65_100 | y− | 1.64 | 3.92 | 3.74 | +0.17 | eval |
| td467 / wall_5_50 | y+ | 0.89 | 4.18 | 4.28 | −0.10 | fit |
| td478 / wall_0_30 | y+ | 0.32 | 2.15 | 2.20 | −0.05 | fit |
| td481 / wall_20_55 | y+ | 0.89 | 1.82 | 1.86 | −0.04 | fit |
| td486 / wall_65_90 | y+ | 1.41 | 11.22 | 11.36 | −0.14 | eval |

## Related

- Preliminary Box2D wall-replay (pre-sweep): `sysid/wall_collision_fit.py` · output at `sysid/wall_collision/box2d_eval/summary.txt`
- Puck physics (gravity/damping) sweep: [`puck-system-id.md`](puck-system-id.md)
- Paddle dynamics (PID + density) sweeps: [`teleop-system-id.md`](teleop-system-id.md)
- Box2D collision internals: [`../box2d/simulator-essentials.md`](../box2d/simulator-essentials.md)

## Scripts

| Script | Purpose |
|---|---|
| `sysid/wall_collision_fit.py` | Detect + fit + Box2D replay at the default restitution (two seeding variants: finite-diff vs fitted velocity). |
| `sysid/wall_collision_restitution_fit.py` | Sweep `side_wall_restitution` ∈ [0.45, 0.95] using the fitted-velocity variant only. Generates sweep curve, per-segment tables, side-by-side GIFs. |

## Open problems

- **Restitution + damping coupling**: the sweep fixes `γ = 0.178` and varies restitution alone. A joint re-fit over (restitution, γ) on the wall-bounce dataset might reveal a better (lower-restitution, higher-damping) combination that matches both the speed ratio and the position error.
- **Wall-parallel contact noise**: several segments have non-trivial tangential velocity at impact. The metric currently treats pure position error — adding rebound-angle error as a secondary metric might discriminate candidates the 5-frame position metric cannot.
- **Longer-horizon metric**: post-bounce errors grow past 5 frames. A metric over 10–20 frames (filtering out secondary bounces) would stress-test the parameter choice under the horizons that matter for reward shaping.
