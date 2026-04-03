# Documentation index

Formal documentation is grouped into four areas:

| Area | Path | Contents |
|------|------|----------|
| **Repository** | [`repo/`](repo/) | Top-level layout, project goal, real-robot safety |
| **Environments** | [`environments/`](environments/) | Env/task layers, Box2D simulator, real UR5 stack |
| **Training** | [`training/`](training/) | Training pipelines, entrypoints, module layout |
| **Exploration** | [`exploration/`](exploration/) | TD3 primitive exploration behavior and knobs |

**Quick links**

- Repo: [`repo/repository-structure.md`](repo/repository-structure.md) · [`repo/project-goal-and-safety.md`](repo/project-goal-and-safety.md)
- Environments: [`environments/README.md`](environments/README.md) · [`box2d/simulator-essentials.md`](environments/box2d/simulator-essentials.md) · [`real-world/overview.md`](environments/real-world/overview.md) · [`real-world/homography.md`](environments/real-world/homography.md) · [`real-world/td3-async-replay.md`](environments/real-world/td3-async-replay.md) · [`real-world/episode-lifecycle.md`](environments/real-world/episode-lifecycle.md) · [`real-world/reset-fsm.md`](environments/real-world/reset-fsm.md)
- Training: [`training/architecture.md`](training/architecture.md) · [`training/td3-algorithm.md`](training/td3-algorithm.md) · [`training/ppo-amp-discriminator.md`](training/ppo-amp-discriminator.md) · [`training/reward-shaping.md`](training/reward-shaping.md) · [`training/network-architecture.md`](training/network-architecture.md) · [`training/replay-and-episodes.md`](training/replay-and-episodes.md) · [`training/checkpointing.md`](training/checkpointing.md)
- Exploration: [`exploration/td3-primitives.md`](exploration/td3-primitives.md)
