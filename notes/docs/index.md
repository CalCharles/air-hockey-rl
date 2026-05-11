# Documentation index

Formal docs for the air-hockey TD3 stack — sim and real robot. Active code paths and conventions live in [`CLAUDE.md`](../../CLAUDE.md) at the repo root.

## Start here

- New to the project — read [`repo/project-goal-and-safety.md`](repo/project-goal-and-safety.md) and [`repo/repository-structure.md`](repo/repository-structure.md).
- New to the env — [`environments/README.md`](environments/README.md), then [`environments/observation-action-spaces.md`](environments/observation-action-spaces.md).
- New to training — [`training/README.md`](training/README.md), then [`training/architecture.md`](training/architecture.md) and [`training/td3-algorithm.md`](training/td3-algorithm.md).
- About to launch a real-robot run — [`environments/real-world/overview.md`](environments/real-world/overview.md) and [`environments/real-world/episode-lifecycle.md`](environments/real-world/episode-lifecycle.md).
- Planning a residual fine-tune — [`training/residual-rl-recipe.md`](training/residual-rl-recipe.md).

## Sections

| Area | Path | What lives here |
|------|------|-----------------|
| **Repo** | [`repo/`](repo/) | Project goal, real-robot safety, top-level layout |
| **Environments** | [`environments/`](environments/) | Env/task layers, obs/action spaces, Box2D + real UR5 |
| **Training** | [`training/`](training/) | TD3 architecture, configs, runtime behavior, recipes, ablations |
| **Exploration** | [`exploration/`](exploration/) | TD3 primitive exploration knobs |

## Logbooks (intentionally short-lived)

- [`recent-commands.md`](recent-commands.md) — append-only training-command log; newest at top.
- [`known-issues.md`](known-issues.md) — open bugs and applied workarounds.
