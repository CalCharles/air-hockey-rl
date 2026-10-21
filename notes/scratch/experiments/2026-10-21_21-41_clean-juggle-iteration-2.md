# Clean juggling iteration 2 results

- **Date**: 2026-10-21 21:41 UTC
- **Status**: done (bounded iteration; not a claim of indefinite survival)
- **Run dirs**: `/scratch/cluster/mrudolph/documents/air-hockey-rl/runs/td3/juggle_iterations`
- **Selected trial**: `long_discount`; validation score `(1.0, 5000.0, 5000.0)`


Clean Box2D; unchanged sysid-v2 controller and workspace. Validation seeds 20000–20015, 5,000-step cap.

| Trial | Mean steps | Mean juggles | Genuine cap completion | Start GIF | End/failure GIF |
|---|---:|---:|---:|---|---|
| continue | 4397.1 | 56.4 | 87.5% | [start](../../../runs/td3/juggle_iterations/iteration_02/continue/validation/clean_start.gif) | [end](../../../runs/td3/juggle_iterations/iteration_02/continue/validation/clean.gif) |
| replicate | 4693.3 | 71.8 | 93.8% | [start](../../../runs/td3/juggle_iterations/iteration_02/replicate/validation/clean_start.gif) | [end](../../../runs/td3/juggle_iterations/iteration_02/replicate/validation/clean.gif) |
| long_discount | 5000.0 | 65.1 | 100.0% | [start](../../../runs/td3/juggle_iterations/iteration_02/long_discount/validation/clean_start.gif) | [end](../../../runs/td3/juggle_iterations/iteration_02/long_discount/validation/clean.gif) |

Each trial has one training seed; table means are over 16 validation episodes. Rewards/memory differ as recorded in each run's args/config. Controller gains and limits are unchanged. Selection uses genuine cap completion and repeated-juggle duration. Independent final tests follow selection.
