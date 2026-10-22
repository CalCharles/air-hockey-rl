- Status: done; bounded simulation experiments, not indefinite survival.
- Selected trial: source_g099; selection uses these validation starts.
- Each trial: one training seed, 1M steps; three-parameter physics DR.

# Original-sensing juggling

Canonical sysid-v2 sensing and controller unchanged; three-parameter physics DR enabled.
32 validation starts, seeds 100000–100031, 10,000-step cap.

| Trial | Mean steps | Mean juggles | Genuine cap completion | GIF |
|---|---:|---:|---:|---|
| source_g0995 | 179.2 | 1.0 | 0.0% | [rollout](../../../runs/td3/original_sensing/source_g0995/validation/training.gif) |
| clean_warm_g0995 | 282.7 | 2.2 | 0.0% | [rollout](../../../runs/td3/original_sensing/clean_warm_g0995/validation/training.gif) |
| source_g099 | 342.8 | 2.8 | 0.0% | [rollout](../../../runs/td3/original_sensing/source_g099/validation/training.gif) |
