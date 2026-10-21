# Bounded residual CQL comparison for sustained juggling

- **Date**: 2026-10-21 19:00 UTC
- **Status**: in-flight (first 100k validation snapshot)
- **Run dirs**: `runs/td3/sustained_20261021_residual/{res_g0975,res_g099,res_g0995}/seed0/`
- **Configs**: `configs/td3/sustained/res_g*.yaml`, `configs/new_juggle/sustained/juggle.yaml`
- **Follows**: [full-policy warm-start campaign](2026-10-21_18-50_sustained-juggle-warm-start.md)

## Question

Can bounded residual adaptation avoid the early survival degradation observed when updating the whole pretrained actor under the corrected observation timing?

## Setup

Based on the existing [residual recipe](../../docs/training/residual-rl-recipe.md), adapted to fixed-step collection. Source actor frozen, residual correction bounded by 0.15 per action component; five critics, CQL alpha 20, batches of 256, two critic and two actor updates per 32 steps. Fresh critics warm up from 10k steps; actor updates begin at 50k. Physics/sensor DR, measured controller, workspace, and one-sample-per-control-step history match the full-policy campaign. Three discounts, seed 0 only, 500k training steps each. This is not an exact replication of the historical residual recipe; replay, update schedule, target environment, and initialization context differ.

These jobs share the same three allocated A40s with the ongoing full-policy runs. Prior observed utilization was approximately 48% on the inspected GPU, with about 0.5 GB memory occupied. Existing allocations were reused, not new allocations requested. Main-session implementation only, no subagents; comparison jobs run in parallel. A GPU smoke test verified five-critic CQL graph updates, unchanged source-actor tensors, nonzero learned residual-head weights, checkpoint evaluation, and inference loading.

## First results

Eight paired validation episodes, 2,000-step cap, one training seed per recipe. Mean observed episode lengths:

| Actor | Nominal | Randomized | Camera 50 ms | Camera 100 ms |
|---|---:|---:|---:|---:|
| Unchanged source with the same observations | 243.2 | 357.0 | 190.4 | 107.4 |
| Residual gamma 0.975, 100k | 345.1 | 438.4 | 138.8 | 197.8 |
| Residual gamma 0.99, 100k | 314.0 | 183.0 | 169.0 | 168.4 |
| Residual gamma 0.995, 100k | 156.1 | 354.9 | 227.9 | 198.1 |

No episode reached the 2,000-step cap. The longest observed was 1,649 steps for gamma 0.975 under randomized conditions. There is no across-training-seed variance estimate. No recipe wins every scenario, and these small validation sets are not final tests.

Residual gamma 0.975 retains source-like action-change RMS (nominal about 0.549 versus source 0.547), unlike the smoother but less reliable early full-policy trials. A frozen base plus bounded correction does not guarantee smoother commands or avoidance of hardware protective stops. Simulator gains and limits are unchanged; no deployment has occurred.

## Current conclusion

The bounded residual approach preserves useful juggling better than the early unrestricted full-policy updates and shows some promising validation improvements. This does **not** establish reliable long-horizon juggling, indefinite survival, or zero-shot success on hardware.

## Next

Allow the bounded 500k-step comparison to finish. Automatic final testing uses 32 episodes per scenario, 10,000-step caps, and fresh seeds starting at 100000. Any checkpoint selected using these results needs another independent test set. Inspect both failure clips and control metrics before selecting a deployment candidate. Follow-up experiments and completed results belong in new dated files; this report is an immutable in-flight snapshot.
