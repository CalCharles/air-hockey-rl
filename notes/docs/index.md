# Documentation index

Formal documentation is grouped into four areas:

| Area | Path | Contents |
|------|------|----------|
| **Repository** | [`repo/`](repo/) | Top-level layout, project goal, real-robot safety |
| **Environments** | [`environments/`](environments/) | Env/task layers, obs/action spaces, Box2D simulator, real UR5 stack |
| **Training** | [`training/`](training/) | Training pipelines, entrypoints, module layout |
| **Exploration** | [`exploration/`](exploration/) | TD3 primitive exploration behavior and knobs |

- **Recent commands**: [`recent-commands.md`](recent-commands.md)

**Quick links**

- Repo: [`repo/repository-structure.md`](repo/repository-structure.md) · [`repo/project-goal-and-safety.md`](repo/project-goal-and-safety.md)
- Environments: [`environments/README.md`](environments/README.md) · [`environments/observation-action-spaces.md`](environments/observation-action-spaces.md) · [`box2d/simulator-essentials.md`](environments/box2d/simulator-essentials.md) · [`robosuite/overview.md`](environments/robosuite/overview.md) (legacy, broken) · [`real-world/overview.md`](environments/real-world/overview.md) · [`real-world/homography.md`](environments/real-world/homography.md) · [`real-world/td3-async-replay.md`](environments/real-world/td3-async-replay.md) · [`real-world/episode-lifecycle.md`](environments/real-world/episode-lifecycle.md) · [`real-world/reset-fsm.md`](environments/real-world/reset-fsm.md) · [`real-world/teleop-system-id.md`](environments/real-world/teleop-system-id.md) · [`real-world/puck-system-id.md`](environments/real-world/puck-system-id.md) · [`real-world/wall-collision-system-id.md`](environments/real-world/wall-collision-system-id.md) · [`real-world/replay-real-in-sim.md`](environments/real-world/replay-real-in-sim.md) · [`environments/velocity-estimation.md`](environments/velocity-estimation.md) · [`real-world/async-z-force-future-steps.md`](environments/real-world/async-z-force-future-steps.md) · [`real-world/protective-stops.md`](environments/real-world/protective-stops.md)
- Training: [`training/architecture.md`](training/architecture.md) · [`training/td3-algorithm.md`](training/td3-algorithm.md) · [`training/td3-configs.md`](training/td3-configs.md) · [`training/td3-real-world-configs.md`](training/td3-real-world-configs.md) · [`training/monitoring.md`](training/monitoring.md) · [`training/ppo-amp-discriminator.md`](training/ppo-amp-discriminator.md) · [`training/ppo-configs.md`](training/ppo-configs.md) · [`training/sim-env-configs.md`](training/sim-env-configs.md) · [`training/reward-shaping.md`](training/reward-shaping.md) · [`training/network-architecture.md`](training/network-architecture.md) · [`training/replay-and-episodes.md`](training/replay-and-episodes.md) · [`training/checkpointing.md`](training/checkpointing.md) · [`training/legacy-configs.md`](training/legacy-configs.md) · [`training/td3-exploration-ablations.md`](training/td3-exploration-ablations.md) · [`training/sim2sim.md`](training/sim2sim.md)
- Exploration: [`exploration/td3-primitives.md`](exploration/td3-primitives.md)
