# Visualization helpers

General trajectory-rendering utilities shared by the training stack and the real-robot stack.

| File | Role |
|------|------|
| `visualize_real_trajectory.py` | Render a single HDF5 trajectory (from real or sim) to a GIF. |
| `visualize_real_trajectory_split.py` | Split-and-render variant used when GIF length needs to be bounded per-episode. |
| `render_teleop_segments.py` | Render demonstration / teleop segments (used by `scripts/real/teleoperate.py`). |

Imported by: `airhockey/sims/air_hockey_real.py`, `scripts/real/teleoperate.py`, `scripts/td3/helper/{episode_artifacts.py, real_warm_start.py}`, `scripts/td3/extras/async_td3_real_reset_policy.py`.
