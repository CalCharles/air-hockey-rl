# Known Issues

## 1. Asynchronous z-force clamping: too infrequent, too strong

The async z-force worker (`_async_z_force_worker` in
`airhockey/sims/air_hockey_real.py`) pushes the paddle onto the table surface via
periodic `forceMode` commands. The original settings (100 Hz, wrench_z = 1.0)
applied force too infrequently at too high a magnitude, causing jerky contact
behaviour and occasional bounce/slip.

**Fix applied:** increased loop rate from 100 Hz to 150 Hz (force magnitude kept at
original 1.0). This produces smoother, more continuous table contact. See edit in:

- `airhockey/sims/air_hockey_real.py` — default config `async_z_force_target_hz`
