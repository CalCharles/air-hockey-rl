# Known Issues

## 1. Paddle density too high in simulation

The paddle density in the Box2D / MuJoCo simulation is higher than the real-world
paddle by an estimated 150%–200%. This causes sim-to-real transfer gaps: the
simulated paddle has more inertia, so policies trained in sim apply too much force
and exhibit sluggish corrections when deployed on the real robot.

**Status:** Open — requires tuning the paddle body density parameter downward by
1.5x–2x to better match the physical paddle.

## 2. Asynchronous z-force clamping: too infrequent, too strong

The async z-force worker (`_async_z_force_worker` in
`airhockey/sims/air_hockey_real.py`) pushes the paddle onto the table surface via
periodic `forceMode` commands. The original settings (100 Hz, wrench_z = 1.0)
applied force too infrequently at too high a magnitude, causing jerky contact
behaviour and occasional bounce/slip.

**Fix applied:** increased loop rate from 100 Hz to 150 Hz (force magnitude kept at
original 1.0). This produces smoother, more continuous table contact. See edit in:

- `airhockey/sims/air_hockey_real.py` — default config `async_z_force_target_hz`
