# A CPU continuation preserves the fixed long-reference cohort if its first allocation ends early

- **Date**: 2026-11-01T23:32:07.537535+00:00 queue recording, UTC.
- **Status**: queued; original job remains running.
- **Run dirs**: original `runs/td3/juggle_long_reference_20261101`; continuation `runs/td3/juggle_long_reference_20261101_completion`.
- **Related**: [original reference launch](2026-11-01_22-52_long-juggle-reference.md), [physical motion analysis](2026-11-01_23-18_smoothness-motion-curves.md).

## Runtime issue and response

The original CPU-only 32-start, 100k-step reference test (job 115864) remained authoritatively live but was slow. Its first completed trajectory recorded 100,000 steps and 1,403 physical juggles; the remaining cohort was not yet complete. Requested a time-limit extension from four to twelve hours. Slurm returned **Access/permission denied**; inspection confirmed the original four-hour limit and running state unchanged. This was a cluster permission denial, not an observation timeout or evidence that the evaluator stopped. No process was cancelled or restarted.

Source inspection shows growing full-history snapshot copies in the canonical simulator, which may contribute to long-episode runtime; that is a performance hypothesis, not a profiled causal measurement. No history truncation, changed sampling, sensing shortcut or simulator optimization was introduced. Preserving the simulator contract takes precedence over making long tests cheaper.

Queued CPU-only job **115869** with Slurm dependency **afterany:115864**. Confirmed `PENDING (Dependency)` while 115864 remains live. It requests six CPUs, 16G memory, no GPU and a twelve-hour limit, and runs at most four missing cases concurrently. It will only run after the parent is terminal, then checks authoritative `sacct` state (retrying transient accounting delay) before any rollout. If the original test finishes all 32 cases, no cases need rerunning.

## Fixed policy, cohort and data integrity

The actor remains frozen SHA256 `c8a1df78ade41cfbe6845a84f4b21480f512973be7a89f0baf66daf5d452b31b`. The cohort remains exactly seeds 9000000–9000031 with 100,000-step recordings. Original valid completed trajectories are reused. Missing or partial/corrupt cases are evaluated from their original reset seeds only after the parent is terminal. An interrupted unsaved case may therefore start again; this is evaluation of a fixed actor, not training warm start. Existing partial artifacts are preserved, and retries use fresh output paths.

The [worker](../../../scripts/analysis/juggle_long_reference_continue.py) validates all required NPZ arrays, lengths, canonical observation shape, finite values and either a recorded terminal flag or full recording boundary. It recomputes physical/sensed juggles, steps, reward, command changes and sampled position-motion distributions, persisting each case for recovery. Original NPZ files do not store every termination reason; unknown causes remain explicitly unknown unless a completed source results JSON supplies them. A complete raw aggregate is written only when the full planned cohort is present. No binary acceptance gate is used.

Model, bundle metadata and reviewed source/config hashes are checked before each new evaluation. Sources include simulator/base-env/observation code, policy/history loading, counter, evaluator and motion analysis. Original eight GPU learners remain unchanged. Continuation inference is CPU-only with empty `CUDA_VISIBLE_DEVICES`; total user GPU allocation remains 11 including the three preexisting jobs. No robot operation, sensing/control/DR changes or new policy initialization is involved.

## Evidence and reproduction

- [Dependency, exact batch source, actor hash and reviewed manifest](../../../docs/juggling-long-reference-completion-queue.json).
- [Worker source](../../../scripts/analysis/juggle_long_reference_continue.py).
- [Reuse/partial-artifact tests](../../../scripts/td3/tests/test_juggle_long_reference_continue.py).
- Local future outputs: continuation `status.json`, `case_SEED.json`, individual `seed_SEED/` result/NPZ files and `raw_results.json` for the complete cohort. Source paths/hashes distinguish reused from newly evaluated cases.

Reproduce only after parent termination: `.venv/bin/python -m scripts.analysis.juggle_long_reference_continue --manifest MANIFEST_JSON`, using the preserved manifest in the queue record and a suitable fresh output root. The Slurm dependency and worker-side terminal-state guard prevent concurrent replacement of a live evaluation. Cases can be distributed differently across CPU processes, but actor, reset seeds, code and simulator recipe remain fixed.

Three tests passed: full-length or truly terminated episodes are reusable; unterminated partial episodes are rejected; corrupt artifacts are rejected. The actual first saved 100k trajectory also validates at 1,403 physical juggles. Checked Python source compilation, live parent handle, pending dependency, source hashes, documentation links/anchors/TOCs and whitespace. No continuation rollout has run yet, and no completed aggregate or figure is claimed. The full research goal remains active.
