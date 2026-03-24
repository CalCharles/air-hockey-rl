# Real-world environment (UR5)

The real-hardware path mirrors the same environment abstractions as simulation, backed by the UR5 air-hockey table (vision, control, and safety constraints differ from Box2D).

## Code map

| Component | Location |
|-----------|----------|
| Simulator adapter (env-facing) | [`airhockey/sims/air_hockey_real.py`](../../../../airhockey/sims/air_hockey_real.py) |
| Control, camera, detection helpers | [`airhockey/sims/real/`](../../../../airhockey/sims/real) (imported by `air_hockey_real.py`) |
| Scripts (teleop, collection, etc.) | [`scripts/real/`](../../../../scripts/real) |

**Homography (rectified camera view and coordinates):** [`homography.md`](homography.md)

**Async TD3 (real collection, replay `dones`, legacy checkpoints):** [`td3-async-replay.md`](td3-async-replay.md)

The high-level env/task layering is the same as for Box2D; see [`../README.md`](../README.md).

## Operating the robot

Operational steps (touchpad program, e-stop awareness) live in the project [README](../../../../README.md) under **Running on the Physical UR5**. Prefer that section for day-of checklists; this doc stays focused on where code lives.

## Safety and control style

Real-robot constraints (smooth motion, e-stops) are summarized in [`../../repo/project-goal-and-safety.md`](../../repo/project-goal-and-safety.md).
