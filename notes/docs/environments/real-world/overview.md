# Real-world environment (UR5)

The real-hardware path mirrors the same environment abstractions as simulation, backed by the UR5 air-hockey table (vision, control, and safety constraints differ from Box2D).

## Code map

| Component | Location |
|-----------|----------|
| Simulator adapter (env-facing) | [`airhockey/sims/air_hockey_real.py`](../../../../airhockey/sims/air_hockey_real.py) |
| Control, camera, detection helpers | [`airhockey/sims/real/`](../../../../airhockey/sims/real) (imported by `air_hockey_real.py`) |
| Scripts (teleop, collection, etc.) | [`scripts/real/`](../../../../scripts/real) |

The high-level env/task layering is the same as for Box2D; see [`../README.md`](../README.md).

## Sub-docs

**Runtime behavior (live training/eval pipeline):**

| Doc | What it covers |
|-----|----------------|
| [`episode-lifecycle.md`](episode-lifecycle.md) | One episode end-to-end: collection, stop events, truncation, artifacts, warm start |
| [`reset-fsm.md`](reset-fsm.md) | Reset policy FSM that gets the puck to a valid state between episodes |
| [`td3-async-replay.md`](td3-async-replay.md) | Async TD3 replay semantics, `dones` convention, legacy two-column checkpoints, launch commands |

**Vision pipeline:**

| Doc | What it covers |
|-----|----------------|
| [`homography.md`](homography.md) | Camera → rectified table view → robot coordinates |

**System identification (real → sim parameter fitting):**

| Doc | What it covers |
|-----|----------------|
| [`puck-system-id.md`](puck-system-id.md) | Puck dynamics (gravity, damping) grid search |
| [`teleop-system-id.md`](teleop-system-id.md) | Paddle dynamics (PID, density) grid search across teleop categories |
| [`wall-collision-system-id.md`](wall-collision-system-id.md) | Wall restitution / collision tuning |

**Diagnostics & ops:**

| Doc | What it covers |
|-----|----------------|
| [`replay-real-in-sim.md`](replay-real-in-sim.md) | Replaying recorded real trajectories in Box2D for debugging |
| [`protective-stops.md`](protective-stops.md) | UR5 protective-stop causes, the 2026-04-17 incident, diagnostic checklist |
| [`paddle-clamping-coverage-gap.md`](paddle-clamping-coverage-gap.md) | Sync-only paddle clamping has a >2 s coverage gap during reset/idle phases (UR `forceMode` controller-side timeout); stopgaps and the threaded-worker fix |

## Operating the robot

Operational steps (touchpad program, e-stop awareness) live in the project [README](../../../../README.md) under **Running on the Physical UR5**. Prefer that section for day-of checklists; this doc stays focused on where code lives.

## Safety and control style

Real-robot constraints (smooth motion, e-stops) are summarized in [`../../repo/project-goal-and-safety.md`](../../repo/project-goal-and-safety.md).
