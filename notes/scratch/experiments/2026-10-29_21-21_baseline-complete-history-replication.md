# Baseline completes; scratch history replication starts

Recorded 2026-10-29 16:21:30 -0500. The high-exploration feed-forward baseline completed its full 6M target without solving juggling. Its released allocation now runs a fresh second training seed of the stronger history recipe.

## Completed baseline

Root `runs/td3/noisy_iteration2_20261029_long`. Actual runner start 2026-10-29 12:16:46.679279 -0500; final completion 2026-10-29 16:17:37.672199 -0500 (durable status timestamps). The log reaches step5,999,999; exact training-loop finish time was not separately recorded. This run is **complete**, not interrupted. All checkpoints remain local.

| Frozen actor | Starts | Cap | Mean steps | Median | Physical juggles | Caps |
|---|---|---:|---:|---:|---:|---:|
| Final 6M | 200000–200031 | 10,000 | 890.84375 | 915 | 9.53125 | 0/32 |
| Validation-selected 2.2M | 400000–400031 | 10,000 | 1091.46875 | 718 | 9.875 | 0/32 |

These are different seed sets, so their mean difference is not a paired policy improvement. Both sets have been reused across methods and are not untouched acceptance starts. The selected evaluator completed successfully and retained2.2M as the best saved validation checkpoint through the full budget (one genuine5k validation cap, mean1681.125). Increasing this baseline's training to6M did not produce a reliably sustained policy. [Portable per-start results](../../../docs/juggling-baseline-6m.csv); raw `final_eval/results.json`, `selected_eval/results.json` and statuses under the root.

## Running scratch replication

Config [history seed1](../../../configs/td3/sustained/noisy_history_scratch_seed1.yaml), root `runs/td3/noisy_history_20261029_seed1`. Actual start from status: 2026-10-29T21:18:15.860119+00:00 (UTC). Allocation112935, step112935.6, launcher1317877; selector1319154. Confirmed old baseline launcher and step112935.5 had ended before launch. No job was interrupted. Seed0 history and survival learners remain live.

Only training seed0→1 and run name differ from the existing history config. Actor, critics and encoder start randomly; model_path null, history_warm_start_plain false, no residual or pretrained source. Reviewed simulator/config hashes match the preflight snapshot, the saved canonical environment passes, optional history sampling remains false and the full established±25% distribution remains intact. [Launch/provenance JSON](../../../docs/juggling-history-seed1-launch.json) records hashes, exact command and actual start. The process passed20k initial random-exploration steps and advanced beyond35k. No checkpoint-quality result yet; this is operational progress, not evidence that the architecture replicates successfully.

```sh
# Within an available existing allocation, same thread limits as the launch JSON:
.venv/bin/python -u -m scripts.td3.juggle_noisy_scratch --root NEW_ROOT --args-file configs/td3/sustained/noisy_history_scratch_seed1.yaml
.venv/bin/python -u -m scripts.td3.juggle_select --root NEW_ROOT --wait
```

The progress figure now says one curve per training run, allowing the second history seed. [Updated curves](../../../docs/juggling-noisy-progress.md). No new GIF for a mostly random35k policy; [E025](../../../docs/experiment_logbook.md#e025) contains the latest verified rollout clip from the earlier history candidate. No simulator implementation or sensing/control change and no new implementation tests; this is a seed-only configuration launch. Validation: baseline completion/results, actual Slurm/process checks, scratch/canonical/hash preflight, saved launch args, report generation, links/images, anchors/TOCs and whitespace.
