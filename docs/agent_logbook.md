# Agent logbook

Scope: recent sustained-juggling work and this documentation setup. Backfilled 2026-10-28 12:23:43-05:00. Historical entries reconstruct tasks from commits and documents; original prompts and agent identities are unavailable. Commit author names do not establish agent identity. Historical times are author-commit recording milestones, not execution times. Timezone: America/Chicago. [Experiment logbook](experiment_logbook.md).

Entries and contents are ordered newest first; IDs remain permanent.

## Table of contents

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
