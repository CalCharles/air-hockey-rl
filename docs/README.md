# Repository reference

Research records: [Experiment logbook](experiment_logbook.md) · [Agent logbook](agent_logbook.md).

Last inspected: 2026-10-21. These are compact working notes for future coding sessions, created at the user's request. The existing detailed documentation remains in [`notes/docs/`](../notes/docs/index.md); this directory does not replace it.

Read in this order:

1. [Architecture and code map](architecture.md): environment, observations, learning, deployment, and system identification.
2. [Workflows](workflows.md): configuration selection, commands, artifacts, and relevant tests.
3. [Caveats and research context](caveats.md): documentation drift, recent investigations, and constraints that are easy to miss.
4. [Sustained juggling](../notes/docs/training/sustained-juggling.md): continuing-task fixes, transfer-aware recipes, running GPU campaign, and monitoring commands added after the initial repository review.
5. [Clean juggling iterations](juggling-iterations.md): branch, hypotheses, adaptive experiments, GIFs, and monitoring for the subsequent clean-sensor benchmark.

The project trains policies in Box2D for a physical UR5 air-hockey robot. Juggling is the central task; reaching, touching, puck velocity, and sparse puck-goal tasks provide other benchmarks. The main learning stack is TD3 with transformed Bellman targets, scalar-output critics, prioritized replay, and optional domain randomization, residual learning, CQL, or HER. RMA provides a separate adaptation baseline.

These notes were checked against source files and representative YAMLs. No training, robot operation, dependency installation, or test execution was performed for this documentation task. Historical experiment results below are attributed to existing writeups, not independently reproduced.

For future edits, start with [`CLAUDE.md`](../CLAUDE.md), then inspect the actual entrypoint and the selected YAML pair. Config generations and historical documentation differ; do not infer current behavior from a filename alone.
