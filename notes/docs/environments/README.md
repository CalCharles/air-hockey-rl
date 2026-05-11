# Environments

Task wrappers and simulator backends for air hockey: shared env APIs, Box2D, and the real UR5 stack.

- **Architecture (layers and adapters):** this page below.
- **Box2D:** [`box2d/simulator-essentials.md`](box2d/simulator-essentials.md)
- **Robosuite (legacy, basic functionality restored 2026-05-01):** [`robosuite/overview.md`](robosuite/overview.md)
- **Real world:** [`real-world/overview.md`](real-world/overview.md) · [`real-world/homography.md`](real-world/homography.md) · [`real-world/td3-async-replay.md`](real-world/td3-async-replay.md) (async TD3 replay semantics and legacy buffers)

---

Mirror of [`.cursor/rules/project-environment-architecture.mdc`](../../../.cursor/rules/project-environment-architecture.mdc).

The [`airhockey`](../../../airhockey) package provides environment/task wrappers around lower-level simulators.

## High-level layers

- Base environment interface and shared behavior: [`airhockey/airhockey_base.py`](../../../airhockey/airhockey_base.py)
- Task-level wrappers and variants:
  - [`airhockey/airhockey_tasks`](../../../airhockey/airhockey_tasks)
  - [`airhockey/airhockey_simple_tasks.py`](../../../airhockey/airhockey_simple_tasks.py)
  - [`airhockey/airhockey_hierarchical_tasks.py`](../../../airhockey/airhockey_hierarchical_tasks.py)
- Reward-specific logic: [`airhockey/airhockey_rewards`](../../../airhockey/airhockey_rewards)

## Simulator backends

- Simulator adapters live in [`airhockey/sims`](../../../airhockey/sims)
- Real-world adapter: [`airhockey/sims/air_hockey_real.py`](../../../airhockey/sims/air_hockey_real.py) — details in [`real-world/overview.md`](real-world/overview.md)
- Box2D adapter: [`airhockey/sims/airhockey_box2d.py`](../../../airhockey/sims/airhockey_box2d.py) — details in [`box2d/simulator-essentials.md`](box2d/simulator-essentials.md)
- Shared simulator abstractions/utilities: [`airhockey/sims/airhockey_sim.py`](../../../airhockey/sims/airhockey_sim.py)

## Guidance for edits

- Keep env/task wrappers separated from low-level simulator-specific behavior.
- Place simulator-specific changes under `airhockey/sims` and keep task APIs stable where possible.
