# Reassign the plateauing reduced-exploration pilot to a fresh survival-focused recipe

- **Date**: 2026-10-29 18:49 UTC
- **Status**: reduced-exploration pilot intentionally interrupted; partial selection completed; new survival trial running
- **Run dirs**: `runs/td3/noisy_iteration2_20261029_low_explore/`, `runs/td3/noisy_survival_20261029_seed0/`
- **Configs**: [survival environment](../../../configs/new_juggle/sustained/noisy_survival.yaml), [survival training](../../../configs/td3/sustained/noisy_survival_scratch.yaml)

## Decision and limits
The reduced-exploration pilot did not improve its best eight-start validation checkpoint after 1M through the fixed 2M comparison, and produced no caps. Its 2M shared-benchmark mean is 664.4375 versus the baseline's 1017.34375, but the paired interval includes zero. This is an adaptive resource choice to try a distinct reward/optimization candidate, not proof that continuing the original 6M low-exploration budget could not succeed. That budget was not completed.

Verified step 112936.3 belonged to launcher PID 1214383 and the low-exploration command, signaled only that step, then confirmed the step/launcher were gone and its GPU had no compute process. Preserved allocation 112936, all checkpoints and both other learners. Last logged step 2,074,999; latest saved checkpoint 2M; exact last executed step unknown. Interruption recording time: 2026-10-29T13:44:39.832949-05:00. Durable status is `interrupted`, not completed/failed. Original waiting worker PID 1222846 was stopped; its status directory was preserved as `selected_eval_wait_interrupted`.

## Partial selected evaluation
A new waiting selector (PID 1250063) recognized the interrupted state, selected the best saved validation checkpoint (1M), and completed its 32 seed-400000 starts, cap 10k. Mean survival 824.40625, median 544.5, physical-trajectory juggles 7.90625, 0/32 caps. These differ from the seed-300000 results because they are different starts; do not interpret that as new training improvement. [Raw selected results](../../../runs/td3/noisy_iteration2_20261029_low_explore/selected_eval/results.json). No final 6M actor exists.

## New trial
Actual start: 2026-10-29T13:44:40.128064-05:00. Existing allocation 112936, new step 112936.5, launcher PID 1250064. Selected-checkpoint waiter PID 1251615, attached only after root/status existed. Root `runs/td3/noisy_survival_20261029_seed0/`. Six million steps planned, seed 0; no actor, critic, encoder, residual or smoke weights loaded.

This launches the combined candidate prepared in E014: upper-half task retained, +1 nonterminal survival bonus, loss penalty 5, gamma .999, actor LR 1e-4, two actor updates per cycle, training-only action-delta penalty .005 and existing low exploration. It is not an isolated reward or optimizer ablation. Compare survival and genuine-juggle metrics, not raw reward totals across recipes. Capped stationary/sticky behavior is insufficient; repeated physical-trajectory juggles remain required.

Prelaunch environment comparison printed exactly three differences from original_sensing: `enable_survival_bonus: true`, `survival_bonus_per_step: 1.0`, `puck_loss_penalty: 5.0`. All sensing, history sampling, PID, action smoothing, geometry, contacts, workspace, terminations and DR remain identical. Re-read the two preexisting simulator/base-env diffs against `94acd81932c59f3cd3b827d6f39a0283747a1beb`; no simulator implementation changes in this task and the optional simulator history mode stays false. Prior 23-test validation already proved paired sensor/terminal equality and exact intended reward differences; it was not rerun for this operational launch.

Initial learning passed 20k on GPU. Before publication, the first 100k frozen-policy validation completed: 76.375 mean steps, 0.5 physical-trajectory juggles and 0/8 caps. The saved 100k training state confirms 62,500 critic updates and populated actor optimizer state (36 parameter entries). This is an early learning signal, not evidence of solved control. No new GIF for the initial random policy; the companion first-cap record supplies the substantive current rollout figures. Baseline and history trials continue.

## Reproduce and next

```sh
# Within an available existing allocation using the same single-thread limits:
.venv/bin/python -u -m scripts.td3.juggle_noisy_scratch --root <new-survival-root> --args-file configs/td3/sustained/noisy_survival_scratch.yaml
.venv/bin/python -u -m scripts.td3.juggle_select --root <new-survival-root> --wait
```

Continue milestone monitoring and preserve all pilots' artifacts/status histories. Assess the combined recipe using unchanged noisy-simulator survival and repeated juggling, followed by untouched acceptance starts if it becomes reliable. Goal remains unsolved.
