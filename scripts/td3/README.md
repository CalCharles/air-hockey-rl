# TD3

The active training stack: TD3 with dual-head critics and transformed Bellman targets. See [`notes/docs/training/architecture.md`](../../notes/docs/training/architecture.md) for the full architecture writeup.

## Quick reference

| File | Role |
|------|------|
| `td3_training.py` | Sim TD3 trainer (CLI entrypoint). |
| `agent.py` | Stochastic TD3 actor. |
| `deterministic_agent.py` | Frozen / deployment actor. |
| `residual_agent.py` | Residual-head actor wrapping a frozen base. |
| `encoder.py` | Actor encoder. |
| `evaluate.py` | `evaluate_agent()` — sync episode rollouts. |
| `eval_utils.py` | Eval helpers. |
| `helper/` | Runtime support (replay, dual-head Q, exploration, real-world runners, checkpointing). |
| `extras/` | Real-world CLI entrypoints. |
| `tests/` | Pytest suite. |

## Canonical commands

Sim training:

```bash
.venv/bin/python -m scripts.td3.td3_training \
  --args-file configs/td3/td3_recommended_top50_hist2.yaml \
  --num-envs 1
```

Sim2sim residual fine-tune:

```bash
.venv/bin/python -m scripts.td3.td3_training \
  --args-file configs/td3/sim2sim/warp075_p30_residual/phaseC_actor2_1M.yaml \
  --num-envs 1
```

Real-robot residual training (see [`extras/async_td3_real.py`](extras/async_td3_real.py)):

```bash
python -m scripts.td3.extras.async_td3_real \
  --config configs/real_configs/rollout_config_residual.yaml \
  --args-file configs/td3_real_world/td3_residual.yaml \
  --model-path <training_state.pth> \
  --train-args <args.yaml> \
  --collector-device cpu --learner-device cuda:0 \
  --data-root-dir real_runs/online_run
```

See [`notes/docs/recent-commands.md`](../../notes/docs/recent-commands.md) for the full canonical command list.
