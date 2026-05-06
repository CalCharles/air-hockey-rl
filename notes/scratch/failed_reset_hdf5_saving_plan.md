# Failed-reset HDF5 saving — current state and plan

**Goal:** make sure every reset trajectory the robot performs (success *and* failure) lands on disk, clearly partitioned, in a way that's modular and unintrusive to add. Audience: future implementer.

This is a planning doc, not a fix — no code changes are proposed beyond the design sketch in §4.

---

## TL;DR

Failure saving is **mostly already wired up** — `partition="failure"` exists end-to-end, and other runs on this machine (`real_runs/online_run/`, `real_runs/async_td3_explore/`, several `runs/td3_training/.../`) do have `reset_hdf5/failure/` directories populated. The eval directory the user pointed at simply had no failed resets in it (all 22 resets succeeded).

That said, there are **three real gaps** worth fixing while we're here, in roughly decreasing order of impact:

1. **Tail reset dropped on loop exit.** The reset that runs *after* the final policy episode is buffered in `pending_reset_artifact` and never flushed — there's no next episode to trigger the flush. Both eval and modular have this. Costs us one trajectory per run, which matters most when the loop exits *because* something went wrong.
2. **Stage-2 retry attempts within a single FSM call are concatenated into one HDF5.** A "success on the 4th try" and "success on the 1st try" produce indistinguishable artifacts at the file level. The row-level `reset_stage_id` field marks stage 0 / stage 1 / unknown, but it does *not* mark per-attempt boundaries within stage-2 retries.
3. **`HARD_SKIP_FSM` produces no artifact at all.** This is the operator-pause-only path; the robot doesn't move so there's nothing to save, but it also leaves no trace in `reset_summaries.jsonl`. Whether this matters depends on the use case.

---

## 1. How the pipeline saves resets today

**Files involved:**

- FSM that produces the trajectory: `scripts/real/rollout_reset_policy_real.py` (`ResetPolicyFSM`)
- Per-run wrapper that builds the artifact: `scripts/smooth_policy/amp_history/amp_training/td3/helper/real_reset_runner.py`
  - `_run_fsm_once` — drives the FSM, accumulates rows, builds `PendingResetArtifact` (lines 200–254)
  - `_reset_artifact_partition` — `"success" if done_reason=="success" else "failure"` (lines 68–69)
  - `ResetRunner.run` — orchestrates startup / soft / hard-with-fsm / hard-skip-fsm (lines 381–515)
- Buffer merge: `scripts/smooth_policy/amp_history/amp_training/td3/helper/real_collector_reset.py` — `merge_reset_fsm_artifact_into_pending` (lines 17–55)
- Disk write & summary append: `scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_modular.py` — `_save_episode_artifacts_and_pending_reset`, reset-flush block at lines 319–382 (called from both modular at line 1085 and eval at line 500)
- Bucketing: `scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real.py` — `_episode_length_bucket_name` (1134–1142), `_reset_output_dir` (1155–1158)
- Output layout: `<reset_artifact_dir>/{success|failure}/{<50|50-100|100-200|>200}/trajectory_data{N}.hdf5`

**Loop shape (eval; modular is the same):**

```
startup_reset → buffers pending
loop:
  policy_runner.run_episode()                        # episode N
  _save_episode_artifacts_and_pending_reset(...)     # saves ep N + flushes pending reset (one preceding ep N)
  pending_reset_artifact = None
  reset_runner.run(...)                              # buffers reset that will precede ep N+1
  pending_reset_artifact = reset_result.pending_reset_artifact
```

Failure handling is "passive": the FSM either terminates with `done_reason="success"` or with `done_reason="hard_reset_required"` (after `max_stage2_cycles` exhausted). Either way, `_run_fsm_once` builds an artifact with the right partition and returns it. The flush block at modular:319 does **no** partition filtering — both partitions write.

**Empirical sanity check.** I greplfried `reset_summaries.jsonl` across `runs/` and `real_runs/`:

- Training runs that ran for hours: contain both `partition: success` and `partition: failure` entries; `failure/` directories exist on disk.
- Recent eval runs (3 found, all under `runs/evaluate/base/latest_model/...`): only `partition: success` appears, because all resets succeeded. Not a bug.

So the user's specific observation ("no `failure/` dir under that eval data dir") is not because the code drops failures — it's because no failure happened. But the gaps below are real and worth addressing.

---

## 2. Gap A — tail reset dropped on loop exit

**Issue.** The `pending_reset_artifact` is only flushed inside `_save_episode_artifacts_and_pending_reset`, which is called *after* a policy episode. The last reset of a run (which runs after the final episode but before the loop body re-checks the exit condition) is buffered and then thrown away when the function returns.

**Why it matters.** In the common case this just costs us one extra reset trajectory per run. The case it actually hurts:

- Eval ends because `eval_max_attempts` was hit and the run was failing repeatedly. The very last reset (a likely failure) is the most diagnostic one.
- Modular training ends because the operator killed the run after a problem; same reasoning.

**Where to fix.** A single tail-flush call after the loop exits in:

- `async_td3_real_eval.py` after the `while` at line 461 (after line 638, before aggregate).
- `async_td3_real_modular.py` wherever its main loop ends (mirror the eval addition).

The flush logic is already a function — `_save_episode_artifacts_and_pending_reset`. It just needs to be re-entered with `result=None` (or a thin "no episode this time" path). Two viable shapes:

- **Shape 1 (lighter):** factor the reset-flush block (modular:319–382) into a small helper `_flush_pending_reset_artifact(args, pending_reset_artifact, …) -> kept_summary_row` and call it (a) inside `_save_episode_artifacts_and_pending_reset` like today, and (b) once on loop exit.
- **Shape 2 (heavier):** make `_save_episode_artifacts_and_pending_reset` accept `result=None` to mean "skip the episode side, only flush pending reset". Same outcome but mixes responsibilities.

Shape 1 wins on "modular & unintrusive". The new helper has no dependencies on the policy episode, and the existing call site shrinks by ~60 lines.

---

## 3. Gap B — stage-2 retry attempts collapsed into one HDF5

**Issue.** Inside `_run_fsm_once`, the loop runs until `fsm.done`. Stage-2 retries (`done_reason="restart_round"`) keep `fsm.done=False`, so all retries' steps append to the **same** `reset_rows` list. The single artifact produced has `partition` set from the *final* outcome only.

**Why it matters.** If we want to evaluate the FSM's per-attempt success rate (a metric the new reset-policy redesign explicitly wants — see [`reset_policy_redesign.md`](reset_policy_redesign.md) §3a "Be measurable"), we need to know how many strikes happened and how many of them succeeded. Right now that's only inferable by re-parsing `reset_stage_id` transitions in the rows, and even then we lose the per-attempt success label.

**Two options, in increasing intrusiveness:**

**Option B1 (minimal — recommended first):** add an *attempt boundary marker* to the row schema. Either bump `reset_stage_id` semantics, or add a sibling integer column `reset_attempt_idx` that increments on every `restart_round` event. This requires:

- Editing `_run_fsm_once` to track the current attempt index (incremented when the FSM emits `restart_round`).
- Threading it through `build_split_episode_row` (in helpers).

That single field unlocks per-attempt analysis without changing the file/partition layout. It also gives us a clean way to answer "did the puck enter the upper half from any *non-final* attempt?" if we want to extract more granular success metrics later.

**Option B2 (heavier — defer):** save one HDF5 per attempt, with each failed attempt going under `reset_hdf5/failure/<bucket>/...` and the final attempt under whichever partition matches its outcome. Pros: GIFs of failed-attempt-only trajectories are immediately viewable; the file system *is* the index. Cons: substantial change to `_run_fsm_once` (it would need to flush at each `restart_round` boundary), more files to manage, and a non-trivial rework of `merge_reset_fsm_artifact_into_pending` since one FSM run would now produce N artifacts.

Recommend B1 for now — it's a one-line schema addition and a counter — and revisit B2 only if the granular per-attempt GIFs prove worth the complexity.

---

## 4. Gap C — `HARD_SKIP_FSM` is silent

**Issue.** When `_should_run_reset_policy_at_episode_start` returns False after a hard pause, we take the `HARD_SKIP_FSM` branch (real_reset_runner.py:496–500). The operator hand-places the puck; no robot motion → no rows → no artifact → no `reset_summaries.jsonl` row.

**Why it might matter.** Today there is no way to know post-hoc that a hard-skip-fsm reset happened — only the *transition_reason* in the next episode's row hints at it. If we want a reset timeline that matches the policy timeline 1:1 (e.g., for plotting), this is missing.

**Suggested fix (cheap):** synthesize a zero-step reset-summary row in the `HARD_SKIP_FSM` branch — no HDF5, just a JSONL entry with `partition: "skip_fsm"`, `done_reason: "hard_reset_skip_fsm"`, `step_count: 0`, plus the wall time. The flush helper from Gap A already needs to handle `pending_reset_artifact is not None` separately from "I want to log a reset event" — easiest is a second small helper `append_skip_fsm_summary(...)` called inline.

Whether to do this depends on how the user uses `reset_summaries.jsonl`. If they only care about HDF5 presence, skip this.

---

## 5. Pre-existing concerns to verify (not gaps yet)

- **Overwrite-warning path in `merge_reset_fsm_artifact_into_pending` (real_collector_reset.py:31–36).** The branch silently drops a previously-buffered artifact. I traced the orchestrators in modular and eval and could not find a path that triggers this in normal operation (each `ResetRunner.run` call merges exactly once, and `_save_episode_artifacts_and_pending_reset` flushes between runs). If this warning has *ever* fired in a log, it indicates a real lost-trajectory bug worth investigating before shipping the above changes. Quick `grep` over saved logs is worthwhile before/after the fix.
- **Failures during the `_save_episode_artifacts_and_pending_reset` call itself.** If the HDF5 write throws (disk full, OS error), the pending artifact is lost and nothing is logged about why. Wrapping the flush in a try/except that emits a clear `[collector_reset_artifact] FLUSH FAILED ...` line would be a one-line resilience improvement, naturally bundled with the Gap A refactor.

---

## 6. Suggested implementation order

The smallest patches first:

1. **Gap A first.** Extract the reset-flush block at modular:319–382 into `_flush_pending_reset_artifact(args, pending_reset_artifact, *, flush_after_policy_episode_id) -> bool`. Call it (a) inside `_save_episode_artifacts_and_pending_reset` exactly where the block lives today, (b) once after the eval/modular main loops exit. This is a strict refactor + one new call site — no behavior change for existing data, only adds the tail flush. Verify by running an eval with `--eval_episodes 2` and confirming there's now one extra reset HDF5 written for the post-final-episode reset.
2. **Gap B1** (attempt index in row schema) as a separate commit. Verify with an existing failure HDF5 from `real_runs/online_run/reset_hdf5/failure/` that the new column shows monotone-non-decreasing per-attempt indices and that GIFs still render via the trim+gif path (`trim_reset_hdf5_post_first_upward.py`).
3. **Gap C** only if `reset_summaries.jsonl` is a primary analysis artifact — it's a cosmetic gap otherwise.

All three changes are scoped to helpers + a couple of orchestrator call sites; none of them require touching the FSM itself or the on-disk format used by downstream tooling (the bucket dir layout, the column schema for existing fields, the `reset_summaries.jsonl` shape — only an *additional* column / row type).

---

## 7. Out of scope (defer)

- A learned reset policy that produces per-step success probabilities. Lives in [`reset_policy_redesign.md`](reset_policy_redesign.md). Pre-requisite: Gap B1's per-attempt label, which gives the redesign a comparable success metric out of the box.
- Restructuring `reset_hdf5/` away from the step-count buckets. The buckets are useful for browsing GIFs by trajectory length. Leave them alone for now.
- Saving raw camera frames during the operator pause itself. Out of scope of the FSM, and probably never worth the disk cost.
