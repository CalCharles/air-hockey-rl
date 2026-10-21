# Workflows and configuration

Run commands from the repository root. Dependencies are declared in [`pyproject.toml`](../pyproject.toml), pinned in `uv.lock`; an existing `.venv/` is present. Documented installation is `uv sync --extra train`; hardware support additionally has a `real` extra. Installation was not attempted during this review.

## Choose the configuration deliberately

Training takes an algorithm YAML via `--args-file`; its `config:` points to an environment YAML. The trainer loads the args file, then parses CLI overrides. Environment YAMLs put task and simulator settings beneath `air_hockey`. Paths are commonly repository-relative; launch from the root.

| Purpose | Starting point |
|---|---|
| Current identified hist2 environment | [`sysid_v2_hist2.yaml`](../configs/new_juggle/sysid_v2_hist2.yaml) |
| V2 task/method campaign | [`configs/td3/tasks_v2/README.md`](../configs/td3/tasks_v2/README.md), paired sim configs in `configs/new_juggle/tasks_v2/` |
| Older recommended source-sim recipe | [`td3_recommended_top50_hist2.yaml`](../configs/td3/td3_recommended_top50_hist2.yaml), references v1 physics |
| Documented transfer source with ±25% DR | [`td3_paramrand_pm25.yaml`](../configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml) |
| Big-gap residual transfer | [`phaseC_actor2_1M.yaml`](../configs/td3/sim2sim/warp075_p30_residual/phaseC_actor2_1M.yaml) |
| Real residual CQL / older baseline | `configs/td3_real_world/td3_residual_cql.yaml` / `td3_residual.yaml` |
| Hardware rollout geometry/history/task | `configs/real_configs/`, including task-specific `tasks/` |

Generated v2 task files explicitly say not to edit by hand. Inspect [`make_sysid_v2_configs.py`](../scripts/td3/extras/make_sysid_v2_configs.py) for generation and [`run_sysid_v2_campaign.py`](../scripts/td3/extras/run_sysid_v2_campaign.py) for orchestration. Review generator scope and output before regenerating.

Representative v2 juggle settings: one environment, 1M steps, batch 512, learning starts 20k, Q updates 25, actor updates 6, zero Q weight decay, flat replay enabled, checkpoint interval 25k. These are recipe values, not `Args` defaults; defaults differ substantially.

## Simulation commands

V2 fixed-physics juggling:

```bash
.venv/bin/python -m scripts.td3.td3_training \
  --args-file configs/td3/tasks_v2/juggle_sysid.yaml
```

Documented transfer-source DR recipe (older config family; inspect physics before reusing):

```bash
.venv/bin/python -m scripts.td3.td3_training_dr \
  --args-file configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml
```

For residual transfer, inspect `config`, `model_path`, output directory, seed, and source checkpoint compatibility before launching. Use the [residual recipe guide](../notes/docs/training/residual-rl-recipe.md), rather than copying historical absolute model paths.

Minimal environment construction without the TD3 trainer:

```python
import yaml
from airhockey import AirHockeyEnv

with open("configs/new_juggle/sysid_v2_hist2.yaml") as stream:
    config = yaml.safe_load(stream)
env = AirHockeyEnv(config["air_hockey"])
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

This example was checked against constructor/observation code, not executed during the review.

## Artifacts and verification

Training writes run-local configuration, models, checkpoints, TensorBoard data, and evaluation artifacts. Check [`td3_checkpointing.py`](../scripts/td3/helper/td3_checkpointing.py) and the [checkpoint guide](../notes/docs/training/checkpointing.md) for exact schemas and resume semantics. Actor weights and a complete training-state checkpoint are different artifacts. Real async resume also requires attention to replay restoration and `include_non_vital_training_state_fields`.

Relevant test commands, to select according to the change:

```bash
.venv/bin/python -m pytest scripts/td3/tests/ -q
.venv/bin/python -m pytest scripts/rma/tests/ -q
.venv/bin/python -m pytest sysid/paddle_pid/code/tests/ sysid/paddle_puck_collision/code/tests/ -q
.venv/bin/python -m pytest sysid/common/tests/ -q
```

The sysid pipeline smoke test needs the shared mouse dataset. Top-level `tests/` is ignored local scratch; the tracked TD3, RMA, and sysid test directories are better starting points. The GitHub workflow is manually dispatched and runs dependency setup on Linux/macOS; it does not currently invoke pytest.

Ignored artifacts include `runs/`, `real_runs/`, system-ID data/results, weights, and media. Referenced checkpoints or `/data2` datasets may be absent in another checkout. Do not assume a tracked directory means its large artifacts are available.
