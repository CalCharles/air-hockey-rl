# Multi-puck spawning (staggered juggle cycle)

How pucks are placed at reset when `num_pucks > 1` on any juggle task
(`multipuck_juggle*`, i.e. every subclass of `AirHockeyPuckJuggleEnv`).

Implementation: `_sample_staggered_multipuck_configurations()` /
`_create_world_objects_staggered_multipuck()` in
[`airhockey/airhockey_simple_tasks.py`](../../../airhockey/airhockey_simple_tasks.py).

---

## Why

Each puck used to be sampled independently from the single-puck spawn
distribution (top edge with `vel = (1, 0)` for the plain juggle env, or a random
upper-half position with a random-heading velocity for the linear-top family).
Independent sampling puts no constraint on *when* the pucks fall into the
paddle's workspace, so they routinely arrived within a few frames of each other
at opposite ends of the table — physically impossible to cover, and the episode
ended on the first miss.

## What happens now

All pucks are placed on **one shared ballistic trajectory**, at different phases
of it, so that their arrival times are evenly spaced.

- The **reach line** `x_reach` is the far edge of the paddle's workspace,
  `paddle_x_min` by default (override with `multipuck_stagger_reach_x`). A puck
  "arrives" when it crosses this line moving down-table.
- One **cycle** is a puck launched upward from `x_reach` and falling back to it.
  Its launch speed is chosen so the apex sits a sampled fraction
  (`multipuck_stagger_apex_frac_{min,max}`) of the way from the reach line to
  the top edge, so the cycle length `T` is re-randomized every reset.
- Puck `i` of `n` is placed at the point of that cycle with time-to-arrival
  `(i + 1) · T / n`. So `puck_0` is the one falling in soonest, and
  `puck_{n-1}` was just launched up from the reach line.
- Arrival gaps are therefore `T / n` — the largest achievable even spacing for a
  cycle of length `T`.

Concretely (matching the requested behavior):

| Pucks | State at reset |
|-------|----------------|
| 2 | one falling in, one rising |
| 3 | one falling in, two rising (one near the apex, one just launched) |
| n | one per `T/n` slot; roughly the top half of the pucks are still rising |

Every puck travels the same x corridor, just at a different time, so they are
also given **separate y lanes**: the usable width is cut into `n` lanes
(shuffled, one per puck) with `multipuck_stagger_min_y_separation_m` of guard
between neighbours, and each puck's y is sampled inside its lane (retried to
clear the paddle). Lateral velocity is uniform in
`±multipuck_stagger_lateral_speed_max`, further capped so a puck cannot drift
out of its lane before it arrives — without that cap, the long-flight pucks in
`n ≥ 4` setups collide mid-air and the stagger collapses.

The trajectory model is the continuous version of what Box2D integrates —
constant down-table acceleration `|gravity|` (base +x points from the goal edge
toward the paddle) with the puck's linear damping:

```
v(t)  = v_inf + (v0 - v_inf)·e^(-d·t)          v_inf = a/d
x(t)  = x_reach + v_inf·t + (v0 - v_inf)(1 - e^(-d·t))/d
```

Apex, launch speed for a target apex, and cycle time `T` are solved from this by
bisection. Measured arrival gaps track the model closely (see below).

## Config knobs (`air_hockey:` block)

| Key | Default | Meaning |
|-----|---------|---------|
| `multipuck_stagger` | `true` | Set `false` to restore independent per-puck sampling |
| `multipuck_stagger_apex_frac_min` | `0.65` | Lower bound on apex height, as a fraction of reach-line → top-edge distance |
| `multipuck_stagger_apex_frac_max` | `0.95` | Upper bound (kept < 1 so pucks do not touch the top wall) |
| `multipuck_stagger_reach_x` | `null` | Explicit reach line; defaults to `paddle_x_min` |
| `multipuck_stagger_lateral_speed_max` | `0.1` | Lateral (y) speed range, m/s (capped further by the lane-drift budget) |
| `multipuck_stagger_min_y_separation_m` | `null` | Guard between y lanes; defaults to 4 puck radii, shrunk automatically if the table is too narrow |
| `multipuck_stagger_phase_jitter` | `0.0` | Random jitter per puck, as a fraction of one `T/n` slot |

Single-puck behaviour is untouched: the staggered path only runs when
`num_pucks > 1`, and reset RNG draws for `num_pucks == 1` are bit-identical to
before this change (verified by hashing 8 seeded reset+rollout sequences on the
canonical hist2 config).

## Measured stagger

Rolled out in Box2D on canonical `sysid_best_params_hist2.yaml` physics
(`gravity = -0.661`, `puck_damping = 0.178`), zero action, arrival = puck
crossing `x_reach = -0.102` moving down-table:

| Pucks | Arrival times (s), seed 0 | Gaps (s) |
|-------|---------------------------|----------|
| 2 | 1.50, 2.95 | 1.45 |
| 3 | 1.00, 2.00, 2.95 | 1.00, 0.95 |
| 4 | 0.75, 1.50, 2.25, 2.95 | 0.75, 0.75, 0.70 |
| 5 | 0.60, 1.20, 1.80, 2.40, 2.95 | 0.60, 0.60, 0.60, 0.55 |

The worst per-episode deviation of any gap from that episode's mean gap is 2 %
(n=2), 4 % (n=3), 6 % (n=4), 8 % (n=5), 9 % (n=6) over 20 seeds each, and 12 %
(n=8) over 10 seeds — mostly the 20 Hz sampling of the crossing time. Beyond ~8 pucks the lanes get
thinner than a puck diameter and mid-flight collisions take over (n=12 is
already ragged); the spawn logic degrades rather than raising.

Regression coverage:
[`scripts/td3/tests/test_multipuck_staggered_spawn.py`](../../../scripts/td3/tests/test_multipuck_staggered_spawn.py).

## Known limitation (pre-existing, not addressed here)

`AirHockeyBaseEnv.has_finished()` inspects `state_info['pucks'][0]` only, so
`terminate_on_puck_hit_bottom` / `terminate_on_puck_pass_paddle` /
`terminate_on_puck_stop` all key off `puck_0`. With `n > 1` the episode ends when
the *first-arriving* puck is missed and ignores misses on the others. Reward
(`AirHockeyPuckJuggleUpperHalfReward` and friends) does average over all pucks.
