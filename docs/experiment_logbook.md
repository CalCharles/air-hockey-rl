# Experiment logbook

Scope: recent sustained-juggling studies on this branch, not a complete repository history. Backfilled on 2026-10-28 12:23:43-05:00 from Git author timestamps, existing experiment notes, saved reports and tracked GIFs. Historical headings are **recording milestones**, not run start/end times. Timezone: America/Chicago, explicit UTC offsets. All results below are historical, not freshly reproduced. [Agent logbook](agent_logbook.md).

Entries and contents are ordered newest first; IDs remain permanent.

## Table of contents

- [E017 — Reject a fixed observer after negative offline estimates](#e017)
- [E016 — Interrupt low-exploration pilot and launch survival-focused scratch trial](#e016)
- [E015 — First noisy capped rollout and fixed-2M benchmark](#e015)
- [E014 — Prepare survival-focused scratch candidate; not launched](#e014)
- [E013 — Interrupt width pilot and launch fresh policy-memory trial](#e013)
- [E012 — One-million-step exploration comparison and shared benchmark](#e012)
- [E011 — Long-history scratch path passes CPU smoke; full trial planned](#e011)
- [E010 — Offline state estimation supports a memory-policy trial](#e010)
- [E009 — Reduced exploration leads at the 500k checkpoint](#e009)
- [E008 — Validation-selected scratch checkpoint doubles paired survival](#e008)
- [E007 — Verify read-only physical-trajectory diagnostics](#e007)
- [E006 — Full learning-curve and failure-action analysis](#e006)
- [E005 — Scratch run completed; longer-training comparison launched](#e005)
- [E004 — Noisy-simulator training from scratch](#e004)
- [E003 — Original-sensing fine-tuning regression](#e003)
- [E002 — Clean-simulation iterations and failed transfer](#e002)
- [E001 — Initial sustained-juggling studies](#e001)

<a id="e017"></a>

## E017 — 2026-10-29 14:09:34 -0500 — Reject a fixed observer after negative offline estimates

One fixed nominal-physics puck observer estimates saved physical states worse than the E010 learned regressors, so it will not enter an active policy.

**Completed offline:** Same saved E010 trajectories and reused eight comparison seeds (16 frozen-actor episodes). No new rollout, policy/regressor training or parameter fit. Filter uses latest puck xy/flag, nominal mean physics, approximate wall reflection, 25 ms latency compensation and fixed noise settings; no sampled DR factors or true state as input. Position/velocity-proxy errors **5.1748 cm / 1.2656 m/s**, versus E010 memory regressor **4.1023 / 0.6587** and larger current-input regressor **4.2508 / 0.7438**. Different inputs and approximate impact/velocity modeling limit causal interpretation; one setting does not rule out the broader filter family.

![Negative fixed-observer comparison](juggling-filter-observability.png)

Figure generated 2026-10-29 from saved data. Paired dots represent eight reused environment seeds averaged across two actors; this is not a juggling-performance test. [Raw results](juggling-filter-observability.json), [methods and commands](../notes/scratch/experiments/2026-10-29_19-09_fixed-observer-negative.md), [analysis code](../scripts/analysis/juggle_filter_observability.py). Experimental observer remains isolated under scripts/analysis; no active trainer/evaluator uses it. All three RL jobs continue. [A018](agent_logbook.md#a018).

<a id="e016"></a>

## E016 — 2026-10-29 13:49:22 -0500 — Interrupt low-exploration pilot and launch survival-focused scratch trial

The reduced-exploration pilot is intentionally stopped after its 2M milestone and its GPU now trains the prepared combined survival/actor-stability candidate from scratch.

**Dated E005 status correction:** Reduced-exploration 6M budget is **interrupted**, not completed. Validation best remained at 1M (894.125); fixed-2M mean 392.0, no caps. Last logged step 2,074,999; latest saved 2M; actual final executed step unknown. Interruption recorded 2026-10-29T13:44:39.832949-05:00. Stopped only step 112936.3 and its old waiter, preserved all artifacts and both other learners. This adaptive resource decision does not prove later training could never improve.

**Partial selection completed:** Best saved checkpoint 1M, 32 seed-400000 starts, cap 10k: mean 824.40625, median 544.5, physical-trajectory juggles 7.90625, 0/32 caps. [Raw results](../runs/td3/noisy_iteration2_20261029_low_explore/selected_eval/results.json); status records interrupted training. These are different starts from the 300000 benchmark, not training progress. No full 6M actor.

**E014 now running:** Actual start 2026-10-29T13:44:40.128064-05:00; root `runs/td3/noisy_survival_20261029_seed0/`, allocation 112936, step .5, launcher 1250064, waiter 1251615. [Training config](../configs/td3/sustained/noisy_survival_scratch.yaml), [environment](../configs/new_juggle/sustained/noisy_survival.yaml). Fresh actor/critics, no reused weights; 6M planned. Exactly three allowed environment reward differences; all sensing/dynamics/control/DR unchanged. First 100k validation: 76.375 mean steps, 0.5 physical-trajectory juggles, 0/8 caps; an early result, not a conclusion. Judge survival plus repeated juggling, not raw reward or stationary capped behavior.

[Detailed decision, validation history and commands](../notes/scratch/experiments/2026-10-29_18-49_survival-candidate-start.md). No new learned-policy figure for this initial run; E015 contains current meaningful rollout GIFs. Baseline and history trials remain running. [A017](agent_logbook.md#a017).

<a id="e015"></a>

## E015 — 2026-10-29 13:49:22 -0500 — First noisy capped rollout and fixed-2M benchmark

The baseline first completes a 5k validation rollout, but its best demonstrated episode fails at 6,859 steps and every shared 10k benchmark episode still fails.

**Completed milestone:** At 1.9M, validation seed 10005 completes 5,000 steps with 59 physical-trajectory juggles; other seven starts fail (mean 1236.875). Extended that specifically selected seed to 20k: failed at **6,859** (`puck_passed_paddle`), **75** physical-trajectory juggles, about 5 min 43 s. All eight saved NPZ fields match the original prefix exactly through 5,000 steps. A selected favorable validation seed is not independent reliability evidence.

![Beginning of the long noisy rollout](assets/juggling/noisy-first-cap-start.gif)

![Final 20 seconds, ending in loss](assets/juggling/noisy-first-cap-end.gif)

New 2026-10-29 clips from the same episode: first 10 seconds and last 20 seconds, with an intentional time gap. Original 20 fps clips remain local; compact copies are 160×357, 10 fps, 64 colors with duration preserved. No clean sensing used.

**Shared 32-start benchmark (300000–300031, cap 10k):** baseline selected 1.9M 827.9375 mean steps; baseline fixed 2M **1017.34375** / 9.4375 physical juggles; reduced exploration fixed 2M **664.4375** / 6.0625; prior selected incumbent 816.9375 / 7.75. All **0/32** cap completions. Baseline 2M minus reduced exploration paired mean +352.90625, bootstrap 95% interval [−128.93, 832.44], wins 24/32; versus prior incumbent +200.40625, interval [−270.88, 701.06]. One training seed; shared starts reused. Best measured mean is not proven robust superiority.

Ongoing 2.2M validation: 1681.125 mean steps and 1/8 cap, now seed 10004 (39 physical-trajectory juggles). Still reused validation starts.

[Detailed results, uncertainty and reproduction](../notes/scratch/experiments/2026-10-29_18-49_first-noisy-cap-2m-results.md), [per-start CSV](juggling-2m-benchmark.csv), [curves including cap fraction](juggling-noisy-progress.md). Baseline continues toward 6M. [A016](agent_logbook.md#a016).

<a id="e014"></a>

## E014 — 2026-10-29 13:31:51 -0500 — Prepare survival-focused scratch candidate; not launched

A reward-focused scratch candidate is prepared for assessment after the current 2M milestones, with canonical dynamics, observations and DR still locked.

**Planned only:** [Environment config](../configs/new_juggle/sustained/noisy_survival.yaml) retains the original upper-half task and adds existing reward-only fields: survival bonus +1 on nonterminal steps and puck-loss penalty 5. [Training config](../configs/td3/sustained/noisy_survival_scratch.yaml): 6M planned, seed 0, gamma 0.999, actor LR 1e-4, two actor updates per cycle, training-only action-delta penalty 0.005, existing low-exploration recipe. This is a combined candidate, not an isolated causal ablation. The actor penalty does not filter or remap deployed actions. No policy weights loaded and no production trial started.

**Why:** Current runs show repeated checkpoint regressions and no cap completions. This candidate tests a longer-horizon survival objective plus more conservative actor optimization. E013 history trial remains running. At latest inspected validation: baseline 1.8M 495.875 mean steps, reduced exploration 1.7M 513.25, history 200k 93.0; each 0/8 caps.

**Guard/validation:** Only `enable_survival_bonus`, `survival_bonus_per_step`, `puck_loss_penalty` can differ at the top level; values are type/finite/nonnegative checked. Task, termination rules, sensing, history sampling, PID, smoothing, geometry, contact parameters, workspace and DR still require exact canonical equality. All 23 targeted tests pass. A paired run through actual puck-loss termination has bitwise-identical observations/terminal flags and exactly the expected reward differences. This is implementation validation, not a learned-policy result. No figure or checkpoint exists for the planned candidate.

**Future command, not run:** `.venv/bin/python -u -m scripts.td3.juggle_noisy_scratch --root <new-survival-root> --args-file configs/td3/sustained/noisy_survival_scratch.yaml`. [A015](agent_logbook.md#a015).

<a id="e013"></a>

## E013 — 2026-10-29 13:22:22 -0500 — Interrupt width pilot and launch fresh policy-memory trial

The width pilot remains weak through 1M, so its GPU now trains a fresh long-history policy while the two small-network controls continue.

**Dated E005 correction/status:** The planned 6M width trial is **intentionally interrupted**, not completed. It underperformed the smaller reduced-exploration actor at all ten matched 100k checkpoints through 1M (last means 152.625 versus 894.125). Correlated checkpoints, one training seed; this is a resource-allocation choice, not proof it could never catch up. Last logged step 1,024,999; latest saved checkpoint 1M; exact final executed step unknown. Interrupted status recorded 2026-10-29 13:18:44 -05:00. Stopped only step 112937.3, preserved allocation/checkpoints and both other learners.

**Partial selection completed:** Best saved validation checkpoint 900k (204 mean steps) evaluated on 32 seed-400000 benchmark starts, cap 10k: mean **225.8125**, median 188.5, physical-trajectory juggles **1.9375**, 0/32 caps. [Raw results](../runs/td3/noisy_iteration2_20261029_large_low_explore/selected_eval/results.json). No 6M final actor. Local GIF exists in that directory; no new tracked figure for this weak partial trial. [Progress curves and explicit runner statuses](juggling-noisy-progress.md).

**E011 full history trial now running:** Actual start 2026-10-29 13:18:44.949 -05:00; root `runs/td3/noisy_history_20261029_seed0/`, existing allocation 112937, step .5. [Config](../configs/td3/sustained/noisy_history_scratch.yaml), 6M planned, seed 0, same low-exploration recipe, learned 50-step policy memory. Random actor/critics/encoder, no privileged actor input or reused weights. Canonical noisy sensing/dynamics and DR verified unchanged. Initial 20k random collection completed; GPU learning then advanced to about 65k before publication, with no checkpoint evaluation available yet. [Full evidence, commands and limits](../notes/scratch/experiments/2026-10-29_18-22_width-stop-history-start.md), [A014](agent_logbook.md#a014).

<a id="e012"></a>

## E012 — 2026-10-29 13:03:57 -0500 — One-million-step exploration comparison and shared benchmark

Reduced exploration leads the matched 1M checkpoint validation but does not establish a better incumbent on the shared 32-start benchmark.

**Running E005 milestone:** At 1M, baseline mean survival 266.25 and physical-trajectory juggles 2.125 versus reduced exploration 894.125 and 6.625 on the same eight validation starts; both 0/8 5k-cap completions. Reduced exploration also leads at 800k and 900k, after a near tie at 600k. The larger network remains slower, 165 mean steps at 700k, no capped completions.

**Completed fixed-checkpoint benchmark:** New 1M actor: mean **695.375**, median 553, physical-trajectory juggles **6.09375**, 0/32 10k-cap completions. Existing selected 1.8M incumbent on the same seed-300000 starts: **816.9375**, median 612, juggles **7.75**, 0/32. New-minus-incumbent mean −121.5625, paired episode-bootstrap 95% interval [−469.88, 210.23], wins 16/32. One training seed; these shared benchmark starts are no longer untouched and training budgets differ. No reliable superiority claim. Retain E008 incumbent; keep current training running.

![Reduced exploration at 1M, shared benchmark](assets/juggling/noisy-low-explore-1m.gif)

New clip generated 2026-10-29, first episode's last 20 seconds, 10 fps/64 colors, duration preserved. [Detailed methods, commands and limitations](../notes/scratch/experiments/2026-10-29_18-03_exploration-1m-benchmark.md), [per-start CSV](juggling-exploration-1m-benchmark.csv), [refreshed curves](juggling-noisy-progress.md). Policy inputs remain canonical noisy observations; physical trajectories are diagnostic only. [A013](agent_logbook.md#a013).

<a id="e011"></a>

## E011 — 2026-10-29 12:50:01 -0500 — Long-history scratch path passes CPU smoke; full trial planned

A 96-step CPU smoke test validates fresh long-history TD3 training and evaluation under unchanged canonical sensing; sustained-control performance remains untested.

**Completed:** Actual start/end 2026-10-29 12:48:15–12:48:34 -05:00. Canonical noisy simulator plus established DR, random actor/critics/encoder, 50-step policy memory, no checkpoint or residual base. Eager CPU, 96 environment steps, three critic updates and actor updates confirmed in saved optimizer states. Checkpoint at 64, metadata/bundle reload and frozen-policy evaluation succeeded. Raw status/training/args/checkpoints live under `runs/td3/noisy_history_smoke_20261029/`.

**Validation/limitations:** All 20 targeted tests pass, including exact equality of canonical sensor/reward/termination traces with and without appended policy memory, plus scratch/privileged-mode rejection. This smoke does not validate CUDA compilation, learning quality or long survival. No meaningful learned-policy figure available from 96 steps.

**Planned, not launched:** [6M-step history config](../configs/td3/sustained/noisy_history_scratch.yaml), same low-exploration algorithm recipe and DR as E005, learned 50-step memory added inside policy. No E010 regression weights or smoke weights will be reused. Current three production trials continue uninterrupted. [Detailed setup and commands](../notes/scratch/experiments/2026-10-29_17-50_scratch-history-smoke.md), [A012](agent_logbook.md#a012).

<a id="e010"></a>

## E010 — 2026-10-29 12:45:50 -0500 — Offline state estimation supports a memory-policy trial

Twelve-step deployable history modestly improves held-out physical-state estimation over the current observation alone, without demonstrating improved juggling.

**Completed offline diagnostic:** 64 saved E008 actor episodes, split by seed into 48 training and 16 held-out episodes (eight distinct held-out environment seeds shared across two actors). Fresh supervised MLPs, 3,000 updates each, no actor/critic training or new rollouts. Width-64 current-input / width-128 current-input / width-64 memory position RMSE: 4.444 / 4.251 / 4.102 cm; prior-step mean velocity RMSE: 0.769 / 0.744 / 0.659 m/s. Memory improves velocity error 11.4% over the larger current-input control. Only one regressor training seed; this frozen-policy dataset does not establish control performance.

![Held-out state-estimation errors](juggling-observability.png)

Figure generated 2026-10-29 from saved trajectories. Paired dots represent eight held-out seeds, each averaging two frozen-actor episodes. [Detailed methods, limitations and command](../notes/scratch/experiments/2026-10-29_17-45_noisy-state-observability.md), [raw JSON](juggling-observability.json), [diagnostic script](../scripts/analysis/juggle_observability.py). No weights retained or authorized for initializing an actor; all future policies still start from scratch.

**Running E005 update:** At 600k, baseline mean survival 311.5 versus reduced exploration 308.375; both 0/8 cap completions. Thus E009's 500k lead is not yet robust. Continue the declared comparison; a from-scratch history architecture is a candidate if these trials fail. [A011](agent_logbook.md#a011).

<a id="e009"></a>

## E009 — 2026-10-29 12:39:35 -0500 — Reduced exploration leads at the 500k checkpoint

The reduced-exploration scratch trial has higher mean survival at 500k than the matched baseline, but neither completes the evaluation cap.

**Running E005 status:** Same eight reused validation starts (10000–10007), 5k cap. At 500k, baseline mean survival **149.75** and physical-trajectory juggle heuristic **0.625**; reduced exploration **233.125** and **1.75**. Both 0/8 cap completions. At 400k the ranking was reversed (183.0 versus 154.625), so this single checkpoint is encouraging but insufficient to establish a reliable advantage. The larger-network trial is slower and has only reached its 300k checkpoint: 86.625 mean steps, 0/8 cap completions. Compare capacity at matched steps later.

![Reduced-exploration 500k validation rollout](assets/juggling/noisy-low-explore-500k.gif)

Newly published 2026-10-29 from the first checkpoint-validation episode, 160×357, reduced to 10 fps/64 colors with duration preserved. Individual rollout only. [Updated learning curves](juggling-noisy-progress.md), [CSV including optional physical-trajectory counts](juggling-noisy-progress.csv). Original artifacts: `runs/td3/noisy_iteration2_20261029_low_explore/train/checkpoint_500000/survival_eval/`.

**Decision:** Continue the predeclared 6M-step comparisons. Exploration primitives anneal to the low floor at 500k; the current point does not yet measure extended learning at that floor. No policy, reward, sensing, physics, randomization or configuration changes. All three training processes and all three selected-evaluation waiting workers remain live. [Task A010](agent_logbook.md#a010).

<a id="e008"></a>

## E008 — 2026-10-29 12:31:39 -0500 — Validation-selected scratch checkpoint doubles paired survival

The validation-selected 1.8M scratch checkpoint outperforms the final actor on fresh paired starts, but both fail every 10k-cap episode.

**Completed:** Frozen policies, same 32 starts (300000–300031), canonical noisy sensing and established DR. Selected checkpoint: mean survival **816.94**, median 612, physical-trajectory juggle heuristic **7.75** per episode. Final actor: **415.31**, median 358.5, physical-trajectory juggles **3.84**. Both 0/32 cap completions. Selected actor wins 20/32 pairs; paired mean gain 401.625 steps, episode-bootstrap 95% interval [174.59, 660.10]. One training seed; no seed-robustness claim. Checkpoint selected on the earlier eight validation starts, not these fresh starts.

![Selected scratch-policy rollout](assets/juggling/noisy-scratch-selected.gif)

New simulation clip generated 2026-10-29, first paired episode tail, compacted to 10 fps/64 colors while preserving duration; not aggregate evidence. [Detailed note and commands](../notes/scratch/experiments/2026-10-29_17-31_scratch-paired-checkpoints.md), [per-start CSV](juggling-paired-checkpoints.csv). No training or simulator changes for this paired test. This is a stronger incumbent, not a solved policy.

**Next evaluation change:** Attached live selected-checkpoint evaluation workers to all three E005 trials. After training and final benchmark, selection uses validation-only genuine-cap fraction then mean survival; selected policy evaluated on seeds 400000–400031, 10k cap. New worker refuses overlap with checkpoint validation/final benchmark seeds and checks canonical config (allowing only trainer-saved top-level episode seed normalization). Cross-recipe winner still requires untouched acceptance seeds. [A009](agent_logbook.md#a009).

<a id="e007"></a>

## E007 — 2026-10-29 12:24:45 -0500 — Verify read-only physical-trajectory diagnostics

A two-start replay confirms that physical-trajectory logging leaves all historical observations and actions bitwise unchanged, while the observed juggle heuristic can disagree with the physical-trajectory heuristic.

**Experiment completed:** Reran the final E004 actor on seeds 200000 and 200001, training-distribution scenario, 10k cap, no GIF rendering. Mean survival 199.0 steps, no cap completions. Every prior trajectory field (`observation`, `next_observation`, `action`, `reward`, `terminated`, `truncated`) exactly matches the corresponding saved final-evaluation NPZ on both starts. These are repeated diagnostic starts, not new independent performance evidence.

**Finding:** Noisy-observation/physical-trajectory juggle counts are 2/2 and 3/2 respectively. Both use the same contact-distance/direction-reversal heuristic; physical-trajectory counts remove sensor corruption but are still heuristic, not a formal contact-event ground truth. No historical metric was overwritten. New reports expose both.

**Implementation:** Evaluator records actual Box2D body positions directly, without invoking sensor-producing `get_current_state` or advancing RNG. These values are stored in NPZ fields `true_paddle_xy` / `true_puck_xy` and used only for diagnostics, never policy inputs. Existing legacy observed-count summaries remain comparable. No sensing, physics, randomization or training changes.

**Reproduction:** `.venv/bin/python -u -m scripts.td3.juggle_survival --run-dir runs/td3/noisy_scratch_20261028_seed0/train --scenario-set training --episodes 2 --horizon 10000 --seed-base 200000 --gif-steps 0 --out-dir runs/td3/noisy_scratch_20261028_seed0/truth_diagnostic`. [Raw results](../runs/td3/noisy_scratch_20261028_seed0/truth_diagnostic/results.json). Compare all six old NPZ keys using `numpy.testing.assert_array_equal`. No new figure for this measurement check. [Task A008](agent_logbook.md#a008).

<a id="e006"></a>

## E006 — 2026-10-29 12:19:36 -0500 — Full learning-curve and failure-action analysis

Saved checkpoint evaluations show continued improvement with large fluctuations; terminal actions are more often saturated, without evidence that saturation causes failure.

**Analysis, completed now:** Reconciled all 19 saved checkpoint evaluations from E004 (100k–1.9M steps). Eight reused validation starts, 5,000-step cap: best observed mean survival 771.875 at 1.8M versus 663.375 at 1.9M; no genuine cap completions. The final actor's separate 32-start result is 416.125 steps and 3.78125 observed juggles. These different seed sets cannot measure a matched final-checkpoint regression.

![Historical checkpoint learning curves](juggling-noisy-progress.png)

Historical figure generated 2026-10-29 from saved E004 results; this task ran no policy evaluations. [Report and reproduction context](juggling-noisy-progress.md), [raw CSV](juggling-noisy-progress.csv), [report generator](../scripts/analysis/juggle_progress.py). Generate with `.venv/bin/python -m scripts.analysis.juggle_progress --roots runs/td3/noisy_scratch_20261028_seed0 runs/td3/noisy_iteration2_20261029_long runs/td3/noisy_iteration2_20261029_low_explore runs/td3/noisy_iteration2_20261029_large_low_explore --out docs/juggling-noisy-progress.md`.

**Failure diagnostics:** Computed from 32 saved final-evaluation NPZ trajectories, equally weighting episodes: normalized action components with absolute value >0.95 occupy 35.83% of all steps/components and 63.75% in the last ten steps. Latest puck-frame occlusion rates average 9.33% overall and 10.94% in the last ten steps. These are observational associations, not controlled causal tests; no sensing modifications are justified. The occlusion rate here measures observed flags, not the simulator's configured per-event probability.

**Live experiment:** E005 trials confirmed as live Slurm processes, completed initial collection and began CUDA-graph updates; approximately 25k–30k steps at inspection. No checkpoint evaluations yet. No additional experiments launched. [Task A007](agent_logbook.md#a007).

<a id="e005"></a>

## E005 — 2026-10-29 12:17:15 -0500 — Scratch run completed; longer-training comparison launched

The 2M-step scratch policy improved but failed every fresh 10,000-step rollout; three new scratch trials now test training duration, exploration and network capacity.

**Completed E004:** Finished 2026-10-28 13:57:31 -05:00 including final evaluation. On 32 independent starts (200000–200031), mean survival 416.1 steps, mean contact timesteps 20.4, genuine cap completions 0/32. Contact timesteps are not distinct juggle cycles. [Raw final results](../runs/td3/noisy_scratch_20261028_seed0/final_eval/results.json). These results supersede E004's running status; the goal remains unsolved.

![Scratch policy under canonical noisy sensing](assets/juggling/noisy-scratch-final.gif)

The existing evaluation clip was copied on 2026-10-29; it shows one rollout and does not establish aggregate reliability.

**New experiment, running:** Three 6M-step seed-0 trials, all fresh random actor/critics: [long baseline](../configs/td3/sustained/noisy_long.yaml), [reduced exploration](../configs/td3/sustained/noisy_low_explore.yaml), and [256-wide networks plus reduced exploration](../configs/td3/sustained/noisy_large_low_explore.yaml). Low exploration changes Gaussian noise 0.1→0.03 and primitive probability floor 0.05→0.001, annealed over 500k steps. Everything else matches E004. Baseline controls duration; second controls exploration; third compares capacity at matched exploration. One seed each limits generalization.

**Contract audit:** Compared simulator code against `94acd81932c59f3cd3b827d6f39a0283747a1beb`: optional history sampling remains false, original history path retained; base-env changes concern reward/terminal handling. Exact canonical sensing/control config plus established three-parameter DR checked for each trial. No simulator changes in this task.

**Artifacts/reproduction:** Roots `runs/td3/noisy_iteration2_20261029_{long,low_explore,large_low_explore}`; status.json and training.log record actual starts/progress. Existing allocations 112935/112936/112937. Command within allocation: `.venv/bin/python -u -m scripts.td3.juggle_noisy_scratch --root <root> --args-file <config>`. Final evaluation: 32 starts, seeds 200000–200031, cap 10k. Those starts are now a repeated benchmark, not a new untouched test; use new seeds for subsequent acceptance testing. No new trial performance available at recording time. [Related task](agent_logbook.md#a006).

<a id="e004"></a>

## E004 — 2026-10-28 12:36:09 -05:00 — Noisy-simulator training from scratch

A new TD3 policy is training from randomly initialized actor and critics in the canonical noisy simulator.

**Timing/status:** Started 2026-10-28 12:35:26 -05:00; running at this entry's recording time. No performance results yet.

**Setup:** Seed 0, 2M steps, gamma 0.995, canonical upper-half reward, random initial exploration for 20k steps, 25 critic and six actor updates every 32 environment steps. No checkpoint load or residual base. Simulator parameters exactly match the canonical noisy recipe; established three-parameter physics DR retained, episode cap 10,000. No dynamics or observation code changes.

**Evaluation:** Planned checkpoint validation every 100k steps, eight starts with 5,000-step cap, training-distribution scenario. Final actor evaluation uses 32 fresh starts (200000–200031), 10,000-step cap; the runner does not select checkpoints on these final starts.

**Artifacts/reproduction:** [TD3 config](../configs/td3/sustained/noisy_scratch.yaml), [guarded runner](../scripts/td3/juggle_noisy_scratch.py). Run command: `.venv/bin/python -u -m scripts.td3.juggle_noisy_scratch --root runs/td3/noisy_scratch_20261028_seed0` inside existing Slurm allocation 112935. Runner refuses an existing root. Local status, training log and saved launch args are under that run directory; training outputs under `train/`, final results under `final_eval/`.

**Figures and limits:** No rollout figure or learned-policy result available yet. One training seed cannot establish seed robustness or infinite survival.

**Related task:** [A005](agent_logbook.md#a005).

<a id="e003"></a>

## E003 — 2026-10-22 10:09:00 -05:00 — Original-sensing fine-tuning regression

All three original-sensing fine-tuned actors regressed relative to the matched original actor.

**Question/setup:** Preserve canonical sysid-v2 sensing/control while fine-tuning source or clean-trained actors with fresh critics and discounts 0.99/0.995. Three trials, 1M training steps each, one training seed each, established three-parameter ±25% physics randomization. Validation: 32 shared starts (100000–100031), 10,000-step cap.

**Timing and status:** Completed historical results; `84620a0` records the matched baseline correction, not experiment execution time.

| Actor | Mean steps | Median steps | Mean juggles | Genuine cap completion |
|---|---:|---:|---:|---:|
| Original, no new training | 660.1 | 264 | 8.625 | 0/32 |
| Source fine-tune, gamma 0.995 | 179.2 | 136.5 | 1.000 | 0/32 |
| Clean warm start, gamma 0.995 | 282.7 | 175 | 2.156 | 0/32 |
| Source fine-tune, gamma 0.99 | 342.8 | 246 | 2.844 | 0/32 |

**Interpretation and limits:** The campaign's selected trial was best among new actors, not better than the baseline. Retain the original actor as reference. These 32 starts were also used for selection; they are not an independent post-selection test. The earlier 16-start baseline is not a matched comparator.

**Figures:** No tracked GIF for this study is available in `docs/assets/juggling/`; local rollout paths are recorded in the [result note](../notes/scratch/experiments/2026-10-22_01-45_original-sensing-juggle.md).

**Artifacts and reproduction:** [Full setup, commands and baseline correction](juggling-original-sensing.md), [environment config](../configs/new_juggle/sustained/original_sensing.yaml), [launcher](../scripts/td3/juggle_original_sensing.py). Local raw results/checkpoints: `runs/td3/original_sensing/`, including `source_matched_validation/`. Baseline checkpoint: `latest_models/canonical/hist2_motion0_v2/` (local availability not checked).

**Related task:** [A003](agent_logbook.md#a003).

<a id="e002"></a>

## E002 — 2026-10-21 19:46:27 -05:00 — Clean-simulation iterations and failed transfer

Clean-simulation training achieved long bounded juggling but failed when simulated sensing corruption returned.

**Question/setup:** Compare upper-half reward, survival shaping and policy memory, then continuation, replication and a longer discount. Clean sensing and disabled physics randomization make these diagnostic trials outside the current training contract. Iteration 1 used 300k steps per trial; iteration 2 targeted 1M total steps. Each trial had one training seed; validation used 16 starts with a 5,000-step cap.

**Timing and status:** Completed historical studies; author-commit milestone `76d1d2c` records final clips/results, not execution timing.

**Results:** Iteration 1 upper/survival/memory mean lengths were 1011.9/1188.6/234.6 steps, with genuine cap completion 2/16, 1/16, 0/16. Iteration 2 continuation/replication/long-discount achieved 14/16, 15/16, 16/16. The selected policy completed the independent 20,000-step clean cap on 28/32 fresh starts; four failed within 334 steps. Sensor-corrupted transfer tests completed 0/32 for both nominal physics (80.4 mean steps) and 100 ms camera lag (78.4).

![Selected clean policy near the end of a long rollout](assets/juggling/final-clean-end.gif)

This individual episode illustrates clean bounded survival; it is not aggregate evidence.

![Clean-trained policy failing with simulated sensor corruption](assets/juggling/final-transfer-nominal.gif)

The corrupted-sensing rollout illustrates the transfer failure; it is simulation, not a hardware test.

**Interpretation and limits:** Long clean survival does not establish noisy-sim robustness or infinite juggling. Contact/pinning artifacts are excluded using a repeated-juggle heuristic, which still needs visual inspection.

**Artifacts and reproduction:** [Setup, commands, raw-data/checkpoint paths](juggling-iterations.md), [iteration 1 report](../notes/scratch/experiments/2026-10-21_21-14_clean-juggle-iteration-1.md), [iteration 2 report](../notes/scratch/experiments/2026-10-21_21-41_clean-juggle-iteration-2.md), [tracked GIF index](assets/juggling/README.md), [launcher](../scripts/td3/juggle_iterations.py). Local runs are under `runs/td3/juggle_iterations/`; availability is machine-dependent.

**Related task:** [A002](agent_logbook.md#a002).

<a id="e001"></a>

## E001 — 2026-10-21 15:50:04 -05:00 — Initial sustained-juggling studies

The initial campaign explored continuing-task TD3, scratch/warm-start training and residual CQL without establishing indefinite juggling.

**Question/setup:** Improve sustained juggling using corrected time-limit bootstrapping, longer discounts, warm starts and residual learning; the detailed recipes and budgets are preserved in the [study documentation](../notes/docs/training/sustained-juggling.md) and [experiment index](../notes/scratch/experiments/INDEX.md).

**Timing and status:** Historical implementation/recording milestone `4a64790`; individual pilot outcomes are recorded in the source notes, not inferred from the commit. Full campaign completion is not independently verified here.

**Results and limits:** Sensor-robust trials did not solve sustained juggling, as reported in the [subsequent analysis](juggling-iterations.md). No numerical aggregate is reconstructed without a matched saved report. Changes included an optional history-sampling mode, so this campaign must not be treated as proof under the preserved simulator contract.

**Figures:** No aggregate figure is available in the reviewed tracked study documentation.

**Artifacts and reproduction:** [Campaign launcher](../scripts/td3/juggle_campaign.py); commands/configurations and local checkpoint locations are documented in the study documentation. Do not rerun its historical recipes as the new noisy-simulator objective.

**Related task:** [A001](agent_logbook.md#a001).
