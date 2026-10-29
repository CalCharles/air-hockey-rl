# Survival interruption, longer-discount history launch and replication check

Recorded 2026-10-29 16:26:07 -0500. The survival recipe is intentionally interrupted after its saved 4M milestone; a new scratch history trial changes only gamma from .995 to .999. Both existing history seeds continue.

## Interrupted survival recipe

Root `runs/td3/noisy_survival_20261029_seed0`. All 40 saved validation checkpoints through4M have zero genuine caps; the best remains3M (963 mean steps). Latest4M validation mean461,0/8caps; prior3M shared benchmark717.65625,0/32caps. The additional million steps after the3M improvement did not improve the best validation. This is an adaptive resource decision, not proof the unfinished2M budget could never help.

Verified launcher1250064 and step112936.5 belonged to this run. Signaled only that step, confirmed launcher/step ended, preserved allocation112936 and every artifact, then marked durable status interrupted. Last logged step4,049,999; latest saved4M; exact last executed step unknown. Confirmation/status timestamp: 2026-10-29T21:23:39.865521+00:00 UTC. The planned6M target was not completed.

Existing selector1251615 recognized the interrupted state and completed normally, selecting3M. Reused seeds400000–400031,cap10000: mean811.25,median571,physical juggles7.59375,0/32caps,32failures. [Portable data](../../../docs/juggling-survival-selected.csv). Raw result/status remain under `selected_eval/`. No final6M actor exists.

## Running longer-discount history trial

Config [new history gamma .999](../../../configs/td3/sustained/noisy_history_long_discount_scratch.yaml), root `runs/td3/noisy_history_long_discount_20261029_seed0`. Actual start 2026-10-29T21:23:40.035965+00:00 UTC; allocation112936,step112936.6,launcher1320884,selector1321073. Passed20k initial random-exploration steps; quality not yet evaluated at recording. Six million steps planned, seed0.

Compared with the original history seed0 recipe, only gamma .995→.999 and the run name change. Reward remains the original upper-half reward, with the same noisy sensing, raw30-dimensional observations, 50-step policy memory, PID/smoothing, contacts, workspace and full established±25% DR. Actor, critics and encoder initialize randomly; no checkpoint, residual or encoder warm start. The aim is to test a longer value horizon while retaining the successful memory architecture. No effectiveness claim yet.

Prelaunch scratch/canonical assertions and previously reviewed simulator/config hashes pass. [Exact launch command and hashes](../../../docs/juggling-history-long-discount-launch.json). Actual saved launch args and environment were rechecked after startup; existing two history learners remained live throughout.

```sh
.venv/bin/python -u -m scripts.td3.juggle_noisy_scratch --root NEW_ROOT --args-file configs/td3/sustained/noisy_history_long_discount_scratch.yaml
.venv/bin/python -u -m scripts.td3.juggle_select --root NEW_ROOT --wait
```

## Completed seed1 first-checkpoint check

At100k, seed1's saved training state confirms62,500critic updates and48actor/72critic optimizer-state entries. Metadata modehistory, 50×8 latest-frame/action memory, no privileged actor. Frozen validation eight reused starts10000–10007,cap5000: mean64.25,median45.5,0.5physical juggles,0/8caps. This verifies learning executed, not useful policy quality or successful replication. Early weakness alone is insufficient to stop it, given seed0's delayed improvement. [Portable checkpoint/metrics evidence](../../../docs/juggling-history-seed1-first-checkpoint.json).

No new GIF for these mostly random/new or weak policies; [E025](../../../docs/experiment_logbook.md#e025) contains the latest verified rollout ending. No simulator implementation or sensing/control changes. Checks: actual process/Slurm termination and new launch, saved configs and scratch guards, source/config hashes, partial selection completion, optimizer-state inspection, saved-data report generation, relative links/images, anchors/TOCs and whitespace. No new implementation tests for a seed/gamma-only launch.
