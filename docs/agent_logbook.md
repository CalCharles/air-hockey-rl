# Agent logbook

Scope: recent sustained-juggling work and this documentation setup. Backfilled 2026-10-28 12:23:43-05:00. Historical entries reconstruct tasks from commits and documents; original prompts and agent identities are unavailable. Commit author names do not establish agent identity. Historical times are author-commit recording milestones, not execution times. Timezone: America/Chicago. [Experiment logbook](experiment_logbook.md).

Entries and contents are ordered newest first; IDs remain permanent.

## Table of contents

- [A023 — Publish selected 20k clips with aggregate failure context](#a023)
- [A022 — Audit evaluation physics without touching live learners](#a022)
- [A021 — Measure newer history checkpoint and start paired longer test](#a021)
- [A020 — Publish history milestone and independent failure test](#a020)
- [A019 — Prepare independently testable actor and critic controls](#a019)
- [A018 — Evaluate and isolate an unsuccessful state-estimator candidate](#a018)
- [A017 — Reassign the plateauing pilot to survival-focused scratch training](#a017)
- [A016 — Publish the first multi-minute noisy scratch rollout and 2M comparison](#a016)
- [A015 — Prepare authorized reward changes without relaxing simulator fidelity](#a015)
- [A014 — Reassign an underperforming pilot GPU to fresh policy memory](#a014)
- [A013 — Publish the 1M milestone without promoting a validation-only winner](#a013)
- [A012 — Prepare guarded from-scratch history training](#a012)
- [A011 — Test whether deployable memory improves state estimation](#a011)
- [A010 — Publish the first matched exploration milestone](#a010)
- [A009 — Publish stronger scratch incumbent and attach checkpoint selection](#a009)
- [A008 — Separate physical-trajectory diagnostics from noisy juggle counts](#a008)
- [A007 — Add reproducible progress figures and inspect live learners](#a007)
- [A006 — Continue sustained-juggling research and publish progress](#a006)
- [A005 — Reverse logbooks and launch scratch training](#a005)
- [A004 — Preserve simulator contract and set up logbooks](#a004)
- [A003 — Restore original sensing and matched comparison](#a003)
- [A002 — Clean diagnostics and evaluation](#a002)
- [A001 — Continuing-task implementation](#a001)

<a id="a023"></a>

## A023 — 2026-10-29 15:30:37 -0500 — Publish selected 20k clips with aggregate failure context

Codex rendered and checked the first10seconds and final20seconds of a selected20k history-policy success.

**Deliverables:** [GIF index](assets/juggling/README.md), [E022](experiment_logbook.md#e022), [detailed note](../notes/scratch/experiments/2026-10-29_20-30_history-20k-selected-gifs.md). All eight20000-step arrays match the prior benchmark episode exactly; a decoded frame was visually inspected. Preserved durations and labeled selection, censoring and4/32 aggregate cap rate. Progress-report discovery excludes demo directories so a chosen1/1success is not presented as a benchmark. Saved-data report regeneration and documentation checks pass. Existing learners continue; no simulator or distribution change.

<a id="a022"></a>

## A022 — 2026-10-29 15:27:48 -0500 — Audit evaluation physics without touching live learners

Codex added a standalone reset-reconstruction audit and verified32stored observations before recovering randomized parameters.

**Evidence:** [E021](experiment_logbook.md#e021), [detailed record](../notes/scratch/experiments/2026-10-29_20-27_history-physics-audit.md). Actual reset replay and parameter-range assertions pass; mismatches fail before a CSV is written. The diagnostic process is separate from live training and performs no actions. Small exploratory subgroup differences do not authorize narrowing DR; no such change made. Existing baseline, history and survival learners continue. Link/anchor/TOC and whitespace checks performed for publication.

<a id="a021"></a>

## A021 — 2026-10-29 15:19:59 -0500 — Measure newer history checkpoint and start paired longer test

Codex completed the2.1M frozen history benchmark and started a matched20k comparison without changing live training.

**Evidence:** [E020](experiment_logbook.md#e020), [detailed note](../notes/scratch/experiments/2026-10-29_20-19_history-2m1-milestone.md), extended per-start comparison. Verified evaluation handle exited successfully and all three learner launcher processes remained live. Improvement is promising but9/32caps is insufficient. The new32-start20k test is running; its reused seeds must not be called fresh acceptance. Earlier GIF/logbook commit0869bce was confirmed pushed. Prepared [history seed1 replication](../configs/td3/sustained/noisy_history_scratch_seed1.yaml), changing only seed/run name; scratch and canonical guards pass. It is **not launched** and will use random initialization. No simulator changes or new implementation tests.

**Status update 2026-10-29 15:25:44 -0500:** Completed the paired20k test and published per-start results;4/32 genuine caps remains insufficient. Updated frozen candidate pointer/hash to2.1M and retained the earlier independent-test provenance with an explicit reused-start limitation. Seed1 replication remains planned.

<a id="a020"></a>

## A020 — 2026-10-29 15:15:24 -0500 — Publish history milestone and independent failure test

Codex evaluated frozen scratch policies, preserved a stronger research candidate and published both favorable and failed rollout evidence.

**Work:** Matched1.6M history/control benchmark, baseline through3M selection, extended selected rollouts and a checkpoint-locked independent32-start20k test. [E019](experiment_logbook.md#e019) and [detailed record](../notes/scratch/experiments/2026-10-29_20-15_history-benchmark-independent.md) document completed results and limits. Added portable per-start CSVs and hashed local checkpoint pointer; it is never a training initialization source. Extended saved-evaluation discovery in the progress report and regenerated curves.

**Status:** Not solved:0/32 independent20k caps. Baseline, history and survival scratch learners remain active. No sensing, physics, control or randomization modifications. Published two compact GIFs showing different episodes. Documentation/link/anchor/TOC and whitespace checks performed for publication; no expensive training launched for documentation.

<a id="a019"></a>

## A019 — 2026-10-29 14:40:03 -0500 — Prepare independently testable actor and critic controls

Codex implemented optional learned aim-point output and previous-action critic context, validated both from scratch on CPU and verified disabled-flag update compatibility.

**Deliverables:** [Aim-point actor](../scripts/td3/absolute_target_agent.py), [trainer](../scripts/td3/td3_training.py), [update helper](../scripts/td3/helper/td3_graphed_update.py), [checkpoint inference](../scripts/td3/eval_utils.py), scratch/argument guards, separate full configs and [Args reference](../notes/docs/training/td3-args-reference.md). Critic context consistently uses previous actions for current Q, taken actions for successor Q and matching context in actor/CQL losses. Replay/sensing are unchanged. Aim-point policy uses only observed pose and fixed public geometry.

**Evidence:** [E018](experiment_logbook.md#e018) / [detailed setup](../notes/scratch/experiments/2026-10-29_19-40_actor-critic-cpu-smokes.md) record actual two CPU smoke timestamps, saved optimizer/projection checks, 32 passed tests and additional guard-case recheck. [Compatibility driver](../scripts/analysis/verify_td3_update_compatibility.py) / [JSON](juggling-update-compatibility.json) demonstrate zero CPU eager output/gradient differences versus `c4c422b` with both options disabled. Enabled-mode GPU capture and performance still need actual trials. No simulator implementation, sensing/control recipe, randomization or active process configuration changed. No full new trial launched; all three existing learners remain live.

**Publication checks:** Reviewed staged implementation, relative links, image paths, unique anchors, TOC coverage and whitespace. Future policies must start with random weights; smoke/regression/incumbent models are not initialization sources. Goal remains active.

<a id="a018"></a>

## A018 — 2026-10-29 14:09:34 -0500 — Evaluate and isolate an unsuccessful state-estimator candidate

Codex tested a deterministic nominal-physics observer on saved noisy trajectories and retained it only as a reproducible negative analysis.

**Evidence/decision:** [E017](experiment_logbook.md#e017) / [detailed note](../notes/scratch/experiments/2026-10-29_19-09_fixed-observer-negative.md) record larger state-estimation errors than E010 regressors and the unequal-input/impact-model limitations. No policy integration, GPU training, simulator alteration, action filtering or checkpoint initialization resulted.

**Deliverables/checks:** [Experimental class](../scripts/analysis/puck_state_filter.py), [analysis driver](../scripts/analysis/juggle_filter_observability.py), [raw metrics](juggling-filter-observability.json), [figure](juggling-filter-observability.png), [tests](../scripts/td3/tests/test_puck_state_filter.py). All 26 targeted tests passed; three observer tests were rechecked after placing research code under analysis. Checked causal target alignment against E010, no observation/RNG mutation, latency/occlusion behavior and finite wall-prediction covariance. Relative links, image paths, anchors, TOC coverage and whitespace validated. Existing baseline/history/survival learners confirmed live and progressing; goal remains active.

<a id="a017"></a>

## A017 — 2026-10-29 13:49:22 -0500 — Reassign the plateauing pilot to survival-focused scratch training

Codex preserved and stopped the reduced-exploration pilot, completed its partial selected evaluation and launched the prepared reward/optimization candidate on the same existing allocation.

**Operations/evidence:** [E016](experiment_logbook.md#e016) / [launch note](../notes/scratch/experiments/2026-10-29_18-49_survival-candidate-start.md) record checkpoint/benchmark reasons, one-seed limits and intentional interruption instead of falsely claiming the 6M plan finished. Verified step/PID ownership, stopped only step 112936.3, confirmed it gone and GPU idle, preserved checkpoints and old waiting status directory. The replacement selector recognized interrupted state and completed. New training step 112936.5 and selected waiter launched after root/status existed.

**Contract audit:** Re-read full simulator/base-env diffs against the required merge. Effective config comparison shows only the three authorized reward fields. Scratch guard passes, model_path null, random initial exploration; no pretrained actor/critic, residual base or smoke/regression weights. No simulator implementation changed. The previously completed 23-test suite validates reward-only sensor neutrality; no historical test is presented as a fresh run. New GPU learner advanced beyond its 20k warmup; the saved 100k training state confirms 62,500 critic updates and populated actor optimizer state. Other baseline/history processes remain live; goal active and unsolved.

<a id="a016"></a>

## A016 — 2026-10-29 13:49:22 -0500 — Publish the first multi-minute noisy scratch rollout and 2M comparison

Codex verified the first 5k capped validation rollout, extended it to failure, benchmarked fixed 2M actors and prepared truthful GIF/data publication.

**Findings:** [E015](experiment_logbook.md#e015) / [detailed note](../notes/scratch/experiments/2026-10-29_18-49_first-noisy-cap-2m-results.md) distinguish one selected 6,859-step episode from zero 10k completions on the shared 32-start benchmark. Replayed all eight NPZ fields through the first 5k exactly; noisy sensing and dynamics unchanged. Baseline fixed-2M mean 1017.34375 versus reduced exploration 664.4375, paired interval crossing zero. No solved-policy claim.

**Artifacts/checks:** Two compact start/end GIFs with explicit time gap and selection caveat, per-start benchmark CSV, refreshed progress plot with genuine-cap fraction and runner statuses. Inspected a rendered GIF frame, checked source summary numbers, relative Markdown targets, image existence, anchors, TOC coverage and whitespace. No new implementation tests for these evaluation/artifact tasks. Prior 23-test result remains historical. Benchmark seeds are reused; untouched acceptance evidence is still missing.

<a id="a015"></a>

## A015 — 2026-10-29 13:31:51 -0500 — Prepare authorized reward changes without relaxing simulator fidelity

Codex prepared a survival-focused scratch candidate and added a narrowly scoped reward exception to the canonical simulator guard.

**Implementation:** [Runner](../scripts/td3/juggle_noisy_scratch.py) and [selector](../scripts/td3/juggle_select.py) accept only the three existing reward knobs while preserving exact equality for every other environment field. Boolean/numeric validation rejects malformed, negative or nonfinite shaping values. Scratch/privileged-actor guards remain in force. Added [reward environment](../configs/new_juggle/sustained/noisy_survival.yaml) and [future training config](../configs/td3/sustained/noisy_survival_scratch.yaml); no full new trial launched.

**Evidence:** All 23 targeted tests passed, including rejection of altered noise, task and DR, and a paired original/shaped trajectory through actual puck loss with exact sensor/terminal agreement and expected reward deltas. Training config matches Args and model_path is null. No simulator implementation file changed. Current running processes retain their settings and continue. [Planned experiment E014](experiment_logbook.md#e014) documents the combined candidate and its rationale; inspect 2M evidence before deciding to launch. Relative links, image existence, anchors, TOC coverage and whitespace checked.

<a id="a014"></a>

## A014 — 2026-10-29 13:22:22 -0500 — Reassign an underperforming pilot GPU to fresh policy memory

Codex preserved and intentionally stopped the weak width pilot, evaluated its selected checkpoint and launched the prepared from-scratch history policy on the freed existing GPU.

**Decision/evidence:** [E013](experiment_logbook.md#e013) / [detailed note](../notes/scratch/experiments/2026-10-29_18-22_width-stop-history-start.md) record ten matched negative pilot checkpoints, one-seed limitations and the adaptive change from the originally planned budget. No claim that the interrupted pilot completed. Both other learners and their waiters remain live.

**Operations:** Confirmed Slurm step 112937.3 and launcher PID matched the width run, signaled only that step, verified it gone and GPU idle, preserved all checkpoints. Marked durable status interrupted with recording timestamp and last logged/saved steps. Stopped the old width waiter, preserved its directory, and ran a fresh partial selector (completed). Launched history step 112937.5, PID 1241157, then its selected-evaluation waiter PID 1241813 after root/status existed. Exact sensing/DR guard and scratch-initialization guard passed; no simulator files changed.

**Implementation/checks:** [Selector](../scripts/td3/juggle_select.py) now exits its waiting loop on intentional interruption and records partial training status before evaluation. All 21 targeted tests pass, including the interruption regression test. [Report generator](../scripts/analysis/juggle_progress.py) displays runner status; refreshed report/CSV/figure show interrupted versus training states accurately. Relative-link, image, anchor, TOC and whitespace checks performed. No checkpoint from the pilot, smoke test or E010 regression initialized the new policy.

<a id="a013"></a>

## A013 — 2026-10-29 13:03:57 -0500 — Publish the 1M milestone without promoting a validation-only winner

Codex monitored all live trials through the first 1M checkpoints, benchmarked the fixed reduced-exploration checkpoint and retained the stronger prior incumbent.

**Evidence:** [E012](experiment_logbook.md#e012) and [detailed note](../notes/scratch/experiments/2026-10-29_18-03_exploration-1m-benchmark.md) distinguish the eight-start validation lead from the inconclusive shared 32-start comparison. New benchmark completed, current three production trainings and selected-checkpoint waiting workers remain live. No training, reward, architecture, simulator or randomization changes.

**Publication:** Updated learning-curve report/CSV/figure, per-start paired benchmark CSV and a new 1M noisy rollout GIF with original source preserved locally. Checked raw JSON summaries against report values, relative links, image existence, anchors, TOC coverage and whitespace. No new implementation tests for this artifact-only milestone; the previous 20-test result remains historical. Goal remains active, with no cap completions yet.

<a id="a012"></a>

## A012 — 2026-10-29 12:50:01 -0500 — Prepare guarded from-scratch history training

Codex generalized the scratch runner to the existing deployable long-history TD3 trainer and validated a future architecture candidate without interrupting current GPU trials.

**Implementation:** [Runner](../scripts/td3/juggle_noisy_scratch.py) selects the history trainer only for `rma_mode: history`, rejects privileged actor modes, all checkpoint/residual initialization, history warm starts and nonrandom initial exploration. Added [production config](../configs/td3/sustained/noisy_history_scratch.yaml); full run is planned, not launched. The existing simulator and its observations are unchanged; only policy memory is added.

**Evidence:** [E011](experiment_logbook.md#e011) / [detailed note](../notes/scratch/experiments/2026-10-29_17-50_scratch-history-smoke.md) record the completed CPU smoke, actual timestamps and optimizer evidence. All 20 targeted tests pass. New paired wrapper test proves bitwise-identical canonical observations, rewards and terminal flags; scratch guards reject prohibited initialization routes. Production config keys match `RMAArgs` and exact sensing/DR assertion passes. CPU smoke does not prove GPU capture or policy performance. No full new GPU training launched; existing three learners and selected-checkpoint workers remain active.

<a id="a011"></a>

## A011 — 2026-10-29 12:45:50 -0500 — Test whether deployable memory improves state estimation

Codex ran a supervised diagnostic on saved physical/noisy trajectory pairs to assess a longer-memory policy architecture while RL trials continued.

**Deliverables:** [Script](../scripts/analysis/juggle_observability.py), [raw held-out metrics](juggling-observability.json), [figure](juggling-observability.png), [detailed experiment](../notes/scratch/experiments/2026-10-29_17-45_noisy-state-observability.md). [E010](experiment_logbook.md#e010) records the modest memory benefit and its limits. Regressor weights were discarded; no policy warm start, simulator change or new rollout was used.

**Validation:** All 18 targeted continuing-task tests passed. Added a synthetic alignment/leakage test proving that inputs exclude current/future actions and states, targets refer to physical state before the evaluated action and seed groups remain separated. Control period is explicit (50 ms default, unchanged from the measured run). Checked generated plot/data paths, relative Markdown links, entry anchors, TOC coverage and whitespace. RL learners remained live and advanced to approximately 670k/665k/455k; no additional RL trial launched.

**Decision:** Keep the existing comparison running. New policy memory is evidence-supported but unproven for sustained control; any future policy must be initialized independently, without these regression weights.

<a id="a010"></a>

## A010 — 2026-10-29 12:39:35 -0500 — Publish the first matched exploration milestone

Codex checked live training through 500k, refreshed the learning-curve report and published a reduced-exploration checkpoint GIF.

**Evidence/decision:** [E009](experiment_logbook.md#e009) records the first post-annealing checkpoint: reduced exploration 233.125 mean survival steps versus baseline 149.75, both 0/8 capped completions. Retained all planned trials; no early winner declaration. Larger-network comparison is not yet matched in training steps.

**Reporting:** The [generator](../scripts/analysis/juggle_progress.py) now includes physical-trajectory juggle counts in CSV when available, and labels separate final, selected and paired benchmarks in its report. Executed it against all four run roots and checked source means, link existence, TOC coverage and unique anchors. GIF compacted from existing evaluation frames with duration preserved. No fresh rollouts or implementation tests required for this reporting-only update; the 17 prior implementation tests remain historical validation, not rerun here.

**Waiting state:** Launcher PIDs 1214379/1214383/1214388 confirmed live during this task; selection PIDs 1222845/1222846/1222847 also confirmed. Progress logs advanced on each learner. Next substantive comparison is later checkpoints after sustained low-exploration learning. Goal remains active.

<a id="a009"></a>

## A009 — 2026-10-29 12:31:39 -0500 — Publish stronger scratch incumbent and attach checkpoint selection

Codex evaluated validation-selected and final scratch policies on fresh paired starts, published a new noisy rollout GIF and attached selected-checkpoint evaluation to the live trials.

**Finding and evidence:** [E008](experiment_logbook.md#e008): selected policy averages 816.94 survival steps versus final 415.31, with 0/32 cap completion each. [Detailed note](../notes/scratch/experiments/2026-10-29_17-31_scratch-paired-checkpoints.md). This justifies evaluating selected checkpoints rather than trusting the final actor.

**Implementation:** [Selection worker](../scripts/td3/juggle_select.py) waits for durable run completion, ranks saved validation results, rejects evaluation-seed overlap and produces selected_eval outputs without training. Existing runner processes were left running; detached follow-up worker PIDs 1222845/1222846/1222847 were confirmed live. All 17 targeted unit tests passed, including genuine-juggling ranking and seed-overlap rejection. Real saved-run check selects the expected 1.8M checkpoint. Initial real-config check rejected trainer-saved top-level seed 51706749; inspection proved it was the sole difference, and the comparison now normalizes only that episode RNG seed, which survival evaluation overrides per episode. No physical/sensor key relaxed.

**Publication:** New selected GIF is 160×357, 200 frames at 10 fps (20 seconds), with original source preserved locally. Documented paired raw data, bootstrap method, limitations and reproduction commands. Link/anchor/TOC checks and git diff whitespace check performed. Live trainers around 325k/315k/220k at inspection. Goal remains unsolved.

<a id="a008"></a>

## A008 — 2026-10-29 12:24:45 -0500 — Separate physical-trajectory diagnostics from noisy juggle counts

Codex added read-only physical-trajectory logging to survival evaluation and verified that it leaves the policy rollout unchanged.

**Reason:** Existing noisy-position juggle classification can be affected by occluded direction-window endpoints. Preserve historical observed counts and add a separate physical-trajectory heuristic for subsequent acceptance checks. Actual simulated bodies are read after each action; no sensor calls, simulator settings or policy input changes.

**Deliverables/checks:** [Evaluator](../scripts/td3/juggle_survival.py), [continuing-task tests](../scripts/td3/tests/test_juggle_continuing.py). All 16 tests pass, including paired sensor traces with/without diagnostic reads and separation of observed/physical count summaries. Replayed two saved policy episodes and verified exact equality for all six historical trajectory fields; [E007](experiment_logbook.md#e007) documents results. Checkpoint learners remained live throughout; latest inspected training steps approximately 160k/155k/115k. No new training launched. Periodic GIF publication continues at substantive milestones, rather than publishing random-initialization clips.

<a id="a007"></a>

## A007 — 2026-10-29 12:19:36 -0500 — Add reproducible progress figures and inspect live learners

Codex added a read-only saved-evaluation report generator and analyzed completed-policy learning curves and terminal actions while three scratch trials train.

**Deliverables:** [Generator](../scripts/analysis/juggle_progress.py), [report](juggling-noisy-progress.md), [CSV](juggling-noisy-progress.csv), [figure](juggling-noisy-progress.png). The report separates reused checkpoint validation starts from final-actor results and labels limits of one training seed. [Findings E006](experiment_logbook.md#e006).

**Checks actually run:** Executed generator on all four run roots; inspected figure rendering, checked 19 source checkpoint means and the final summary against raw JSON, and checked relative Markdown links and unique anchors/TOC coverage. Existing 14 implementation tests passed in A006; not rerun for this reporting-only change. Live launcher PIDs 1214379/1214383/1214388 and Slurm allocations confirmed, logs show learning-phase progress. Commit da83c9f with the first noisy GIF was successfully pushed to origin/codex/sustained-juggling; unrelated puck-score config remains untracked.

**Next:** Wait for checkpoint evidence, publish milestone GIFs and evaluate whether reduced exploration improves sustained survival. Goal remains active and unsolved.

<a id="a006"></a>

## A006 — 2026-10-29 12:17:15 -0500 — Continue sustained-juggling research and publish progress

Codex analyzed the completed scratch run, launched a controlled next comparison and prepared a noisy-policy GIF for publication.

**Authorization:** User requests continued work until solved, documentation and periodic pushed GIFs. Stay on existing branch. Every new policy starts from scratch. No simulator or randomization changes.

**Work:** Generalized the guarded scratch runner with `--args-file`; generated three 6M-step configs and launched detached Slurm steps on existing idle GPUs. Compared effective configs and simulator diff against the required merge. Initial GPU probes failed due to Slurm CPU binding; retried with `--cpu-bind=none` successfully, before launch. No failed training run resulted from those probes.

**Validation:** All 14 continuing-task unit tests passed; trainer Args key validation and exact original-sensing assertions passed for all configs. Run status/logs checked separately after launch. Results and GIF linked in [E005](experiment_logbook.md#e005). Pending: new experiment results, independent acceptance testing and periodic progress publication. The problem is not solved.

<a id="a005"></a>

## A005 — 2026-10-28 12:36:09 -05:00 — Reverse logbooks and launch scratch training

Codex reordered both logbooks newest first and launched a fresh noisy-simulator TD3 run without pretrained weights.

**Scope/decisions:** User requested reverse chronological logbooks and training from scratch instead of fine-tuning. Persistent rules now require both; existing IDs and historical content are preserved. Canonical simulator dynamics/observations and established DR are unchanged.

**Deliverables:** [Scratch config](../configs/td3/sustained/noisy_scratch.yaml), [runner](../scripts/td3/juggle_noisy_scratch.py), [maintenance rules](../AGENTS.md). Runner checks canonical config equality, rejects checkpoint/residual initialization, writes durable status, trains then evaluates. Existing allocation 112935 had an idle A40; detached launcher PID 708960 started the run on this branch.

**Validation:** All 14 `scripts.td3.tests.test_juggle_continuing` tests passed. Scratch config keys match trainer Args; model path is null and the original-sensing config invariant passes. Slurm runner reports training started at 12:35:26 -05:00. Link, anchor, TOC ordering checks and `git diff --check` performed. Training has not completed and performance is not yet measured.

**Related experiment:** [E004](experiment_logbook.md#e004).

<a id="a004"></a>

## A004 — 2026-10-28 12:23:43-05:00 — Preserve simulator contract and set up logbooks

Codex reviewed the simulator changes and created persistent, evidence-backed logbooks on the existing branch.

**Scope:** User requires unchanged noisy-simulator dynamics/observations, permits algorithm/policy/reward changes and requests the [logbook setup](logbook_setup_guide.md).

**Work and decisions:** Compared simulator changes since Carl's merge `94acd81` of Rohan's `97c1ef4`, inspected canonical and experimental sensing configs, historical reports and tracked GIFs. Kept branch `codex/sustained-juggling`. Added persistent simulator-contract and maintenance rules without modifying simulator code or launching experiments. No restoration of existing optional code is claimed.

**Deliverables:** [Repository instructions](../AGENTS.md), [experiment logbook](experiment_logbook.md), this logbook, and discoverable links in both READMEs. Existing documentation and unrelated untracked config preserved.

**Validation:** Checked relative Markdown targets, image existence, unique explicit entry anchors, TOC coverage and `git diff --check`. No training, hardware operation or implementation tests required for these documentation-only changes.

**Unresolved work:** Future algorithm changes and fresh noisy-sim experiments remain to be designed; historical regression [E003](experiment_logbook.md#e003) is the starting evidence. The optional simulator history mode still exists and must remain unused for this objective.

<a id="a003"></a>

## A003 — 2026-10-22 10:09:00 -05:00 — Restore original sensing and matched comparison

Reconstructed task: train with canonical sensing and correct the baseline comparison (`9ad6bee`, `d06ff09`, `84620a0`).

**Work and decisions:** Added config equality checks and original-sensing trials; recorded that all fine-tuned actors regressed against the original actor on matched starts.

**Deliverables:** [Original-sensing analysis](juggling-original-sensing.md), [launcher](../scripts/td3/juggle_original_sensing.py).

**Validation:** Historical saved evaluations only; no newly executed policy tests during backfill.

**Unresolved work:** Address policy degradation while preserving dynamics and observations. [E003](experiment_logbook.md#e003).

<a id="a002"></a>

## A002 — 2026-10-21 19:46:27 -05:00 — Clean diagnostics and evaluation

Reconstructed task: compare clean-sim rewards/memory and record long-rollout evaluation (`af4eeec` through `76d1d2c`).

**Scope/work:** Adaptive trials, seeded initialization, artifact filtering and compact rollout clips; disabled sensing corruption in diagnostic configs.

**Deliverables:** [Setup and limitations](juggling-iterations.md), [GIF index](assets/juggling/README.md).

**Validation:** Saved historical results show strong clean survival and failed sensor-corrupted transfer. No fresh execution performed during backfill.

**Unresolved work:** Clean results do not satisfy the deployment objective. [E002](experiment_logbook.md#e002).

<a id="a001"></a>

## A001 — 2026-10-21 15:50:04 -05:00 — Continuing-task implementation

Reconstructed task: add continuing-task TD3 support and sustained-juggling experiments (`4a64790`).

**Scope/work:** Time-limit handling, replay collection, reward shaping, smoothness, optional simulator history sampling and campaign tools. That optional sampling change is outside the newly reaffirmed observation contract.

**Deliverables:** [Implementation and verification account](../notes/docs/training/sustained-juggling.md), [campaign](../scripts/td3/juggle_campaign.py).

**Validation:** Historical notes report targeted tests and a GPU smoke test; they also report an unrelated full-suite failure. None was rerun for this backfill.

**Unresolved work:** Sustained noisy-sim juggling remained unsolved. [E001](experiment_logbook.md#e001).
