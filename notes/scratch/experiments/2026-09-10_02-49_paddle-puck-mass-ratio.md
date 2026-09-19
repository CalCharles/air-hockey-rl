# Do Box2D paddle–puck collisions depend on the paddle / puck masses? (real weights: paddle 58 g, puck 13 g)

- **Date**: 2026-09-10 02:49 UTC
- **Status**: done (measurement + analysis only — no config or env change made)
- **Configs**: `configs/new_juggle/sysid_best_params_hist2.yaml` (canonical, unmodified)
- **Code touched**: none. Probe scripts were throwaway (scratchpad), reproduced below.
- **Related**: [`notes/docs/environments/box2d/simulator-essentials.md`](../../docs/environments/box2d/simulator-essentials.md) ·
  [`notes/docs/environments/real-world/teleop-system-id.md`](../../docs/environments/real-world/teleop-system-id.md) ·
  [`notes/docs/environments/real-world/sysid-pipeline.md`](../../docs/environments/real-world/sysid-pipeline.md)

## Question

Real hardware weights: **paddle 58 g, puck 13 g** (ratio **4.46**). Does the Box2D
collision model actually care about these numbers, or is the outcome fixed by
restitution alone?

## Answer: yes, the collision is a first-order function of the mass **ratio**

`CollisionForceListener` (`airhockey/sims/airhockey_box2d.py:372-422`) disables Box2D's
own restitution and enforces the **relative** normal velocity
`v_rel_out = e · v_approach` with a momentum-conserving reduced-mass impulse
`j = Δv_rel · m_pad·m_puck/(m_pad+m_puck)`. Because only the *relative* speed is pinned,
how that relative speed is split between the two bodies is set entirely by the mass
ratio. For a paddle at speed `v` hitting a resting puck head-on:

```
v_puck_out = (1 + e) · m_pad / (m_pad + m_puck) · v
v_pad_out  = v - (m_puck/m_pad) · (v_puck_out - 0)      (recoil)
```

Measured in the actual sim (paddle spawned at 1.0 m/s into a resting puck, damping and
gravity disabled on both bodies so only the contact acts, `e = puck_restitution = 1.09145`):

| paddle_density | puck_density | m_pad (kg) | m_puck (kg) | m_pad/m_puck | puck out (m/s) | paddle out (m/s) |
|---|---|---|---|---|---|---|
| 3000 | 3000  | 24.322 |  9.501 |   2.560 | **1.5040** | 0.4125 |
| 3000 | 1721  | 24.322 |  5.450 |   4.463 | **1.7086** | 0.6171 |
| 3000 | 250   | 24.322 |  0.792 |  30.720 | 2.0255 | 0.9341 |
| 3000 | 30    | 24.322 |  0.095 | 256.0   | 2.0833 | 0.9919 |
| 3000 | 30000 | 24.322 | 95.008 |   0.256 | 0.4263 | −0.6652 |

The measured column matches the closed form `(1+e)·m_pad/(m_pad+m_puck)` to 4 decimals in
every row, so the mechanism is exactly the reduced-mass split — nothing else in the env
(damping, PID, step timing) contributes to the launch speed at the moment of contact.

**Only the ratio matters, not the absolute masses.** Scaling both densities by ×0.1 /
×10 / ×100 leaves `puck_out = 1.50396 m/s` bit-identical.

## Puck–wall collisions do **not** depend on the puck mass

Same listener, different branch (`airhockey_box2d.py:330-370`). The wall is a static body
(infinite mass) and the corrective impulse is `J = m_puck · (target_out − current_out)`, so
`Δv = J/m_puck` cancels the mass algebraically — the outgoing speed is `e · v_in` (or the
fixed min-rebound below the 0.25 m/s threshold) regardless of density. Measured, damping and
gravity disabled so only the contact acts:

| puck_density | m_puck (kg) | side wall, in 1.0 m/s | end wall, in 1.0 m/s | either wall, in 0.1 m/s |
|---|---|---|---|---|
| 30    |  0.0950 | 0.990000 | 0.700000 | 0.100000 |
| 250   |  0.7917 | 0.990000 | 0.700000 | 0.100000 |
| 3000  |  9.5008 | 0.990000 | 0.700000 | 0.100000 |
| 30000 | 95.0077 | 0.990000 | 0.700000 | 0.100000 |

Identical to 6 decimals across a 1000× mass range, in both the restitution branch
(`v_in ≥ 0.25 m/s`) and the min-rebound branch. Nothing else reintroduces mass: Box2D
`linearDamping` is a velocity decay (`v *= 1/(1+dt·damping)`) and gravity is an acceleration.

**So the paddle collision is the only place puck mass enters the dynamics at all.** Two
practical consequences:

- The wall restitutions already fitted by [`sysid-pipeline.md`](../../docs/environments/real-world/sysid-pipeline.md)
  (`side 0.90`, canonical `0.99`) are *invariant* to any future change of `puck_density` —
  a mass-ratio refit cannot invalidate them, and vice versa.
- Conversely, the wall data carries **no information about the puck mass**, which is why the
  `(e, mass ratio)` degeneracy noted below can only be broken with paddle-collision segments.

One further consumer, inert under the canonical config: `max_puck_vel` is derived as
`(m_pad/m_puck) · max_paddle_vel` (`airhockey_box2d.py:704-710`) and used only as an
observation-space bound for the velocity-carrying obs types (`airhockey_base.py:304-305`).
The active `history` obs type carries positions and flags only, so it never sees it.

## Where the sim currently sits vs. the real weights

Masses are `density · π · radius²` (`airhockey_box2d.py:700-701`), and the canonical
config uses equal densities with unequal radii, so the ratio is purely geometric:

```
m_pad/m_puck = (0.0508 / 0.03175)² = 2.560     (canonical sim)
58 g / 13 g                        = 4.462     (real hardware)
```

To reproduce the hardware ratio with `paddle_density: 3000` fixed you would set
`puck_density: 1721` — measured effect on a head-on hit: **+13.6 %** puck launch speed
(1.504 → 1.709 × paddle speed), and paddle recoil rises 0.41 → 0.62 m/s.

## …but 58 g / 13 g is probably the wrong target anyway

The sim paddle is a free dynamic body: the current setup gives it a **59 % velocity loss**
on a 1 m/s head-on hit (1.0 → 0.41 m/s). On the real robot the paddle is bolted to a UR5
running a stiff position controller, so a 13 g puck barely perturbs it — the collision-
relevant inertia is the arm's **reflected inertia at the tool**, kilograms, not the 58 g
plastic head. In that limit `m_pad/m_puck → ∞` and `v_puck_out → (1+e)·v = 2.09 v`,
i.e. **39 % higher than the sim delivers today**.

Two further reasons not to just paste 58/13 into the config:

1. **`paddle_density` is not a mass parameter here — it is the PID plant inertia.** The
   paddle is driven with `ApplyForceToCenter` (`airhockey_box2d.py:1491`), so `a = F/m`
   and the density directly sets tracking speed. Measured response to a 0.10 m step
   command (canonical Kp=9000, Kd=50, 0.05 s steps), displacement per step:

   | paddle_density | mass | step 1 | 2 | 3 | 4 | 5 | 6 |
   |---|---|---|---|---|---|---|---|
   | 1000 |  8.11 kg | 0.041 | 0.070 | 0.085 | 0.093 | 0.097 | 0.098 |
   | 3000 | 24.32 kg | 0.014 | 0.027 | 0.039 | 0.049 | 0.058 | 0.065 |
   | 9000 | 72.97 kg | 0.005 | 0.010 | 0.015 | 0.019 | 0.024 | 0.028 |

   `paddle_density: 3000` was fitted **jointly with Kp/Kd** against real teleop paddle
   trajectories ([`teleop-system-id.md`](../../docs/environments/real-world/teleop-system-id.md)),
   and it is what makes the sim paddle track like the real arm. Changing it to match a
   58 g head would invalidate that fit outright. (The doc's line "real paddle inertia is
   much closer to density=3000 than density=1000" is about *arm* inertia, in Box2D's
   abstract kg/m² units — it is not a claim about the paddle head's weight.)

2. **`puck_restitution = 1.09145` has never been fitted against real paddle collisions** —
   [`sysid-pipeline.md`](../../docs/environments/real-world/sysid-pipeline.md) explicitly
   lists paddle–puck restitution as not yet identified (only gravity, puck damping and
   wall restitution are). For head-on hits only the product
   `(1+e) · m_pad/(m_pad+m_puck)` is observable, so `e` and the mass ratio are degenerate:
   the superelastic `e > 1` is plausibly compensating for a paddle that recoils far more
   than the real arm does. Changing the ratio without refitting `e` moves the one
   quantity the sim has been implicitly tuned around.

## Recommendation

- Record the real weights (paddle 58 g, puck 13 g) as reference data — done in
  [`simulator-essentials.md`](../../docs/environments/box2d/simulator-essentials.md#body-masses-and-the-real-world-reference).
- **Do not** change `puck_density` / `paddle_density` on the strength of these weights
  alone. The right experiment is the missing sysid stage: harvest the paddle-collision
  segments the segmenter already produces (`scripts/sysid/segment_trajectories.py`) and
  fit `(e, m_pad/m_puck)` jointly against real paddle→puck exit speeds, with the paddle
  approach speed as the regressor. If the real arm behaves near-rigidly (expected), the
  fit should land at a large ratio and an `e` below 1.
- If a quick sensitivity check is wanted first: `puck_density: 1721` (real 58/13 ratio) or
  `puck_density: 250` (near-rigid paddle) are the two informative points; both raise the
  puck launch speed (+14 % / +35 %) and will change juggling behaviour materially.

## Reproduce

```python
# head-on hit, only densities vary
import numpy as np, yaml
from Box2D import b2Vec2
from airhockey.sims.airhockey_box2d import AirHockeyBox2D
params = yaml.safe_load(open('configs/new_juggle/sysid_best_params_hist2.yaml'))['air_hockey']['simulator_params']

def hit(paddle_density, puck_density, v=1.0):
    p = dict(params); p['paddle_density'] = paddle_density; p['puck_density'] = puck_density
    sim = AirHockeyBox2D(**p); sim.reset(0)
    sim.spawn_paddle((0., 0.), (0., 0.), 'paddle_ego'); sim.spawn_puck((0., 0.), (0., 0.), 'puck')
    pad, puck = sim.paddles['paddle_ego'], sim.pucks['puck']
    gap = sim.paddle_radius + sim.puck_radius
    pad.position = (0., 0.); pad.linearVelocity = b2Vec2(0., v); pad.linearDamping = 0.
    puck.position = (0., gap * 1.02); puck.linearVelocity = b2Vec2(0., 0.)
    puck.linearDamping = 0.; puck.gravityScale = 0.
    for _ in range(400):
        sim.world.Step(1/2000, 100, 100)
        if puck.linearVelocity[1] > 1e-6 and (puck.position[1] - pad.position[1]) > gap * 1.05:
            break
    return pad.mass, puck.mass, float(puck.linearVelocity[1]), float(pad.linearVelocity[1])
```
