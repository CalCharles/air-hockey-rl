# Architecture and code map

## Environment

[`AirHockeyEnv`](../airhockey/__init__.py) is a Python factory, not a single environment class or Gym registration. It dispatches on `cfg["task"]` and calls the selected task's `from_dict`. YAML environment files wrap this dictionary in `air_hockey`.

| Concern | Source to inspect |
|---|---|
| Shared construction, reset, step, termination, history | [`airhockey_base.py`](../airhockey/airhockey_base.py) |
| Juggle, touch, velocity, scoring task classes | [`airhockey_simple_tasks.py`](../airhockey/airhockey_simple_tasks.py) |
| Goal tasks, sparse goals, action spaces | [`airhockey_tasks/`](../airhockey/airhockey_tasks/) |
| Task rewards | [`airhockey_rewards/`](../airhockey/airhockey_rewards/) |
| Observation construction and puck observation warps | [`utils.py`](../airhockey/utils.py) |
| Physics, PID, smoothing, delays, occlusions | [`airhockey_box2d.py`](../airhockey/sims/airhockey_box2d.py) |
| Hardware environment | [`air_hockey_real.py`](../airhockey/sims/air_hockey_real.py) |
| Cameras, transforms, robot control, velocity estimates | [`sims/real/`](../airhockey/sims/real/) |
| Rendering | [`renderers/`](../airhockey/renderers/) |

The base environment selects `box2d`, `real`, or a legacy `robosuite` branch. The presence of a branch does not establish that its dependencies or backend are available.

For single-puck `obs_type: history`, the observation contains five paddle samples followed by five puck samples, each `[x, y, valid]`, oldest first: 30 values total. Latest paddle position is `[12:14]`; latest puck position is `[27:29]`. With `use_last_action_in_policy_state`, the trainer appends two previous-action values, producing a 32-value policy state. Goal and RMA wrappers change this contract. The five observation samples are distinct from `simulator_params.hist_len`, which controls PID target smoothing.

Actions are two normalized paddle displacement targets. Base defaults scale them by 0.26 m and 0.12 m, then the simulator handles control and workspace constraints. Check coordinate transforms before modifying geometry: the v2 hist2 YAML explicitly supplies robot-to-table offsets and robot-frame workspace limits.

## Learning

[`td3_training.py`](../scripts/td3/td3_training.py) owns `Args`, YAML/CLI parsing, environment construction, model setup, rollout, optimization, checkpointing, and evaluation scheduling. `SingleEnvVector` supplies a lightweight vector interface for one environment. Wrappers can patch module-level functions before invoking `_entrypoint()`; this coupling matters when refactoring.

| Variant | Entry point / implementation |
|---|---|
| Base simulation TD3 | [`td3_training.py`](../scripts/td3/td3_training.py) |
| Per-reset physics domain randomization and multi-env evaluation | [`td3_training_dr.py`](../scripts/td3/td3_training_dr.py) |
| Sparse goal training with hindsight relabeling | [`td3_training_her.py`](../scripts/td3/td3_training_her.py), [`td3_her.py`](../scripts/td3/helper/td3_her.py) |
| Residual actor over frozen source policy | [`residual_agent.py`](../scripts/td3/residual_agent.py), [`td3_residual.py`](../scripts/td3/helper/td3_residual.py) |
| Conservative Q-learning | [`td3_cql.py`](../scripts/td3/helper/td3_cql.py) |
| Batch experiments / GPU assignment | [`run_experiments.py`](../scripts/td3/run_experiments.py) |

Actor definitions live in [`agent.py`](../scripts/td3/agent.py) and [`deterministic_agent.py`](../scripts/td3/deterministic_agent.py). [`q_network.py`](../scripts/td3/helper/q_network.py) defines one scalar output per critic. The trainer creates an ensemble (`num_critics`, default 2), with corresponding targets and an optional target subset. “Single-head” does not mean one critic. Residual trunk depth also differs from a conventional count of plain MLP layers.

The helper directory separates CUDA-graph updates and Bellman transforms (`td3_graphed_update.py`), replay storage/sampling, episode collection, logging, and checkpointing. Read the [throughput design](../notes/docs/training/training-throughput.md) before changing the hot loop: CPU rollout actors, GPU learning, graph capture, and background evaluation interact.

## Real deployment

[`async_td3_real.py`](../scripts/td3/extras/async_td3_real.py) orchestrates collection and learning. [`real_td3_runtime.py`](../scripts/td3/helper/real_td3_runtime.py) holds shared arguments, checkpoint/runtime support, and learner logic; neighboring `real_*` helpers implement policy execution, reset handling, transition holds, buffers, metrics, and stop states.

Frozen evaluation uses [`async_td3_real_eval.py`](../scripts/td3/extras/async_td3_real_eval.py). Agent construction is dispatched through `real_eval_agents.py`; `real_task_eval_hooks.py` chooses metrics and between-episode behavior. Reaching tasks use paddle repositioning; puck tasks can use a learned puck reset FSM. Mouse baseline evaluation uses `async_td3_real_teleop_eval.py`. Consult the [evaluation pipeline](../notes/docs/training/real-world-eval-pipeline.md) before changing these interfaces.

## System identification and adaptation

[`sysid/`](../sysid/README.md) contains four fits: puck gravity/damping, side/end wall restitution, paddle PID gains, and paddle–puck contact response. `common/` handles segmentation, datasets, validation, and puck/wall orchestration. Each fit separates tracked `code/` from ignored local `data/` and `results/`; fitted output does not automatically modify canonical YAMLs.

[`scripts/rma/`](../scripts/rma/README.md) contains privileged-factor base training, supervised history-to-latent adaptation, paired evaluation, and a long-history policy control without privileged inputs. Goal-conditioned configurations can also use HER here. Compare the long-history control when assessing whether adaptation itself helps.
