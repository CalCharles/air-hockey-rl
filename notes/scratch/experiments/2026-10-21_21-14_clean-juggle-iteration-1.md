# Clean juggling iteration 1 results

- **Date**: 2026-10-21 21:14 UTC
- **Status**: done (bounded iteration; not a claim of indefinite survival)
- **Run dirs**: `/scratch/cluster/mrudolph/documents/air-hockey-rl/runs/td3/juggle_iterations`
- **Selected trial**: `upper`; validation score `(0.125, 1005.625, 1011.9375)`


Clean Box2D; unchanged sysid-v2 controller and workspace. Validation seeds 20000–20015, 5,000-step cap.

| Trial | Mean steps | Mean juggles | Genuine cap completion | Start GIF | End/failure GIF |
|---|---:|---:|---:|---|---|
| upper | 1011.9 | 16.6 | 12.5% | [start](../../../runs/td3/juggle_iterations/iteration_01/upper/validation/clean_start.gif) | [end](../../../runs/td3/juggle_iterations/iteration_01/upper/validation/clean.gif) |
| survival | 1188.6 | 16.8 | 6.2% | [start](../../../runs/td3/juggle_iterations/iteration_01/survival/validation/clean_start.gif) | [end](../../../runs/td3/juggle_iterations/iteration_01/survival/validation/clean.gif) |
| memory | 234.6 | 3.1 | 0.0% | [start](../../../runs/td3/juggle_iterations/iteration_01/memory/validation/clean_start.gif) | [end](../../../runs/td3/juggle_iterations/iteration_01/memory/validation/clean.gif) |

Each trial has one training seed; table means are over 16 validation episodes. Rewards/memory differ as recorded in each run's args/config. Controller gains and limits are unchanged. Selection uses genuine cap completion and repeated-juggle duration. Independent final tests follow selection.
