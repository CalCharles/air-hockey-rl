# Stop the weak width trial and start a fresh deployable history policy

- **Date**: 2026-10-29 18:22 UTC
- **Status**: width trial intentionally interrupted; partial selected evaluation done; history trial running
- **Run dirs**: `runs/td3/noisy_iteration2_20261029_large_low_explore/`, `runs/td3/noisy_history_20261029_seed0/`
- **Configs**: [width trial](../../../configs/td3/sustained/noisy_large_low_explore.yaml), [new history trial](../../../configs/td3/sustained/noisy_history_scratch.yaml)

## Question
How should the next available GPU budget be spent? The wider network has been worse than the smaller reduced-exploration control throughout its first million steps, while E010 supports testing deployable memory.

## Completed partial width comparison

Eight reused validation starts 10000–10007, 5k cap, one training seed each:

| Checkpoint | Wider network mean survival | Smaller network mean survival |
|---|---:|---:|
| 100,000 | 44.000 | 59.250 |
| 200,000 | 72.625 | 128.125 |
| 300,000 | 86.625 | 129.500 |
| 400,000 | 108.250 | 154.625 |
| 500,000 | 134.750 | 233.125 |
| 600,000 | 86.500 | 308.375 |
| 700,000 | 165.000 | 289.125 |
| 800,000 | 180.750 | 591.875 |
| 900,000 | 204.000 | 478.500 |
| 1,000,000 | 152.625 | 894.125 |

All comparisons have zero capped completions. These are ten correlated checkpoints of one training seed, not ten independent replications. They justify an adaptive resource choice, not a universal conclusion that wide networks cannot improve with more training. The originally planned 6M width budget was not completed.

Stopped only Slurm step `112937.3`, after confirming it matched launcher PID 1214388 and the width-trial command. Verified the step and launcher were gone and its allocated GPU had no compute process. Did not cancel allocation 112937, touch either other learner or delete any checkpoint. Last log: 1,024,999; latest saved checkpoint: 1M; exact last executed step is unknown. Durable interruption record written 2026-10-29 13:18:44 -05:00; this is recording time, not an invented precise kernel stop timestamp.

The original waiting selection worker (PID 1222847) was stopped and its status directory preserved as `selected_eval_wait_interrupted`. A fresh selector evaluated the best saved validation checkpoint, 900k (204 validation mean steps), without resuming training. On 32 seed-400000 starts, cap 10k: mean survival **225.8125**, median 188.5, physical-trajectory juggles **1.9375**, 0/32 cap completions. [Raw selected results](../../../runs/td3/noisy_iteration2_20261029_large_low_explore/selected_eval/results.json). No final 6M actor exists; do not report this as a completed full run. Local rollout GIF is in the selected_eval directory; no new tracked GIF is needed for this weak partial policy.

## Fresh history trial

Actual start: 2026-10-29 13:18:44.949 -05:00, durable runner status. Existing allocation 112937, new step 112937.5, launcher PID 1241157. Selected-checkpoint waiter PID 1241813. New root `runs/td3/noisy_history_20261029_seed0/`. Six million steps planned, seed 0, same reduced-exploration recipe/learning rates as the small control. Add learned 50-step observation/action memory, no privileged actor input. This is the config prepared and CPU-smoked in E011; no smoke or regression weights reused.

The exact canonical noisy config plus established DR assertion passed before launch. Simulator diff against `94acd81932c59f3cd3b827d6f39a0283747a1beb` remains the same two preexisting files; no simulator modification in this task. Policy/encoder/critics initialize randomly. Initial collection reached 20k; CUDA compilation and learning startup were still in progress at the first log inspection. Before publication, the GPU learning loop advanced to about 65k; no checkpoint validation result available yet.

The selected-evaluator now treats `interrupted` as a terminal training state and evaluates whatever saved checkpoints are available, recording that training status. This prevents an intentional stop from leaving new waiting workers asleep indefinitely. Failed runs remain errors. Existing other waiting workers need no change because their runs continue normally.

## Validation and reproduction

All 21 targeted continuing-task tests pass, including a new mocked interruption test that fails if the worker keeps sleeping. Existing tests prove sensor-neutral policy memory and reject checkpoint/residual/privileged initialization. Updated progress report explicitly includes runner status so the width trial cannot be mistaken for running/completed.

```sh
# Inside the existing allocation, with its usual OMP/MKL thread limits:
.venv/bin/python -u -m scripts.td3.juggle_noisy_scratch --root <new-history-root> --args-file configs/td3/sustained/noisy_history_scratch.yaml
.venv/bin/python -u -m scripts.td3.juggle_select --root <new-history-root> --wait
```

## Conclusion and next
The adaptive search now has two continuing short-history controls and one fresh long-history trial. The width result is an interrupted negative pilot, not a completed six-million-step comparison. Preserve all artifacts. Continue checkpoint validation, final/selected benchmark separation, and new acceptance starts if a candidate reaches sustained survival. Goal remains unsolved.
