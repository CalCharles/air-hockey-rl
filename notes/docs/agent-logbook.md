# Agent logbook

Record substantive agent work and handoffs here, oldest first. Append dated
entries with UTC times; preserve prior entries and add corrections explicitly.
Include the request, actions, evidence/validation, and outstanding work. For
training checks, record job IDs, nodes, run paths, and observation times. Verify
processes, GPU activity, and log freshness before calling training active.
Do not record secrets.

Experiment results belong in the [experiment logbook](../scratch/experiments/INDEX.md),
following its [dated-file convention](../scratch/experiments/README.md).
Link experiment writeups rather than duplicating them here.

## 2026-11-22 22:54 UTC — Codex: FB status and logbook setup

- **Request:** Inspect FB training, specifically allocation `1159444`, and
  ensure documentation, an experiment logbook, and an agent logbook exist.
- **Actions:** Read Slurm accounting and FB status/checkpoint/metrics files;
  inspected live processes and GPU on `c610-072`. Found existing `docs/`,
  detailed `notes/docs/`, and dated experiment writeups with an index.
  Created this logbook and linked it from documentation and agent context.
- **Evidence:** At 22:52 UTC (16:52 CST), allocation `1159444` was running but
  had no Python/FB trainers. GPU utilization was 0%, memory usage 0 MiB, and
  no GPU processes were present. Its batch process was `sleep 172800`.
  Allocation ends 2026-11-24 06:11 CST. Process records for
  `runs/fb/realplant_full_50M` and `runs/fb/realplant_current_50M` point to
  old allocation `1154576` on `c620-022`. Both report `paused_for_allocation`,
  at 7,944,864 and 8,000,000 steps respectively; logs last changed around
  06:46 CST. Prototype job `1159911` completed at 16:29 CST; all four seeds
  reached 1M steps and exited with code 0.
- **Correction:** The first response checked scheduler state and artifacts
  without inspecting processes inside `1159444`. The follow-up checked the
  node directly and confirmed no training was running there.
- **Validation:** Documentation links and whitespace checked; no code changed.
- **Outstanding:** No submission evidence for the prepared
  `runs/fb/slurm/50m_continuation/submit.sh` appeared in inspected scheduler
  history or logs. Training resume remains outstanding. No jobs were started,
  stopped, or modified during these checks.

## 2026-11-22 22:59 UTC — Codex: resume FB trainers on 1159444

- **Request:** Launch the 50M jobs on existing allocation `1159444`.
- **Actions:** Verified configs, checkpoints, node, allocation deadline, and
  absence of existing trainers. Launched full/current trainers detached with
  latest-checkpoint resume; updated run process metadata and experiment index.
- **Validation:** Both restored 1M replay entries without refill and advanced
  beyond saved counters to 7.96M/8.02M with finite metrics. GPU at 100%.
- **Outstanding:** Training remains in flight; checkpoint stop is configured
  for 2026-11-24 06:06 CST, five minutes before allocation expiry.
- **Details:** [Resume writeup](../scratch/experiments/2026-11-22_22-57_fb-50m-resume-1159444.md).

## 2026-11-22 23:41 UTC — Codex: latent inference convergence

- **Request:** Compare reward-inferred z stability across state dataset sizes.
- **Actions:** Froze matched 8.75M full/current policies and 1M replay buffers;
  added CPU analysis script and ran 30 repeats at nine dataset sizes for three tasks.
- **Validation:** Cached computation matched actual reward inference on 1k states;
  all analysis completed, plot inspected, whitespace checked. Live training continued.
- **Finding:** 100k samples gives <5° p95 error against full-buffer z across all
  task/model combinations; 300k gives <2°. No inference settings changed.
- **Outstanding:** Policy-return sensitivity and later-checkpoint convergence.
- **Details:** [Experiment](../scratch/experiments/2026-11-22_23-41_fb-z-convergence.md).

## 2026-11-22 23:57 UTC — Codex: adopt 300k reward-z samples

- **Request:** Document the convergence study and use 300k samples in future.
- **Changes:** Updated shared FB Args default, four canonical configs, and both
  prepared 50M continuation YAMLs to 300000. Documented decision, evidence,
  replay-size cap, and existing-process/saved-dataset behavior in canonical docs
  and agent context. Prior experiment writeup remains an immutable record.
- **Validation:** Parsed configurations and checked default and diff whitespace.
- **Scope:** Running trainers were not restarted; they retain their loaded 10k
  setting. Future launches/resumes with updated configs use 300k. Other historical
  prepared bundles must be regenerated or have explicit overrides updated.

## 2026-11-23 00:03 UTC — Codex: prepare 50M sbatch continuation

- **Request:** Set up sbatch training through 50M steps.
- **Actions:** Generated `runs/fb/slurm/50m_after_1159444/` using the repository
  handoff generator. Both runs resume latest checkpoints with 300k inference,
  one GH node for 48h, dependency `afterany:1159444`.
- **Validation:** Both shell scripts pass `bash -n`; parsed YAMLs have correct
  targets, inference counts, resume paths, and safety margins. Both latest
  checkpoints were complete at 9.25M. Dependency verified.
- **Blocker:** Read-only SSH probe to login1 with BatchMode failed authentication
  (interactive password/token required). Bundle remains NOT SUBMITTED; no job ID.
- **Handoff:** User runs bundle `submit.sh` from an authenticated login-node shell.
  Running trainers and scheduler jobs were not modified.

## 2026-11-23 16:20 UTC — Codex: live FB training status

- **Request:** Check current training progress.
- **Evidence:** Both original trainer PIDs remain live on allocation 1159444,
  c610-072; GPU utilization 100%. Fresh metrics: full 29.57M at 346 steps/s,
  current 29.92M at 351 steps/s. Latest checkpoints 29.50M/29.75M. No nonfinite
  scalar training metrics found. Latest juggle evaluation return 83.0/27.9
  (full/current); current-only touch/reach results also weaker. Evaluations use
  10 episodes and still use the loaded 10k inference setting.
- **Submission update:** scontrol confirms continuation job 1162255 was submitted
  2026-11-22 18:06 CST and is pending afterany:1159444. This supersedes earlier
  not-submitted status. Its command points to 50m_after_1159444/job.sbatch.
- **Forecast:** About 16 hours remaining at observed average throughput, likely
  finishing before allocation expiry Nov 24 06:11 CST. Estimate, not guarantee.
- **Actions:** Read-only scheduler/process/metrics inspection; no jobs changed.

## 2026-11-23 16:24 UTC — Codex: stop trainers and execute 300k inference

- **Request:** Stop current jobs and perform 300k reward-z inference now.
- **Actions:** SIGTERM to both trainers; verified complete checkpoint saves at
  29,621,664 / 29,966,976 and process exits. Performed 300k inference for all
  three tasks on both saved models; saved new datasets, latents, and evaluations.
- **Validation:** Exactly 300k rows per model; finite 50D latents with norm sqrt(50);
  10 evaluation episodes/task completed; script exited successfully.
- **State:** Trainers remain stopped. Allocation 1159444 and pending continuation
  1162255 were not cancelled; continuation configs use 300k.
- **Correction:** Initial commentary proposed restart; this was corrected before
  execution to honor the explicit stop request. No trainers were restarted.
- **Details:** [Results](../scratch/experiments/2026-11-23_16-22_fb-300k-inference.md).

## 2026-11-23 16-28 UTC — Codex: compare 10k and 300k rewards

- **Request:** Controlled reward comparison between inference sample sizes.
- **Actions:** Used the stopped checkpoints and identical 10 evaluation seeds;
  10k inference uses a prefix of the previously saved 300k dataset.
- **Validation:** Evaluation script completed successfully; metadata records
  checkpoint, sampling seed, episode seeds, and raw metrics. Training remains stopped.
- **Finding:** Full returns improve on all three tasks; current juggle improves,
  touch worsens, reach unchanged. One inference draw and 10 episodes limit certainty.
- **Details:** [Experiment](../scratch/experiments/2026-11-23_16-28_fb-10k-vs-300k-rewards.md).

## 2026-11-23 16-31 UTC — Codex: GIFs and 300k-enabled training resume

- **Request:** Render 300k-inference performance GIFs, then continue training;
  explain exploration.
- **Actions:** Rendered 12 GIFs from saved task latents; validated animated frames.
  Restarted full/current on 1159444 from 29,621,664/29,966,976, PIDs 503624/503625.
- **Validation:** Both status files running; resolved args.yaml confirms 300000
  inference samples for both processes. Original stop checkpoints retained at launch.
- **Details:** [Writeup](../scratch/experiments/2026-11-23_16-31_fb-300k-gifs-resume.md).

## 2026-11-23 17:47 UTC — Codex: document and publish FB work

- **Request:** Document GIF generation, 300k-enabled training, and exploration;
  commit and push the work.
- **Changes:** Canonical docs explain random spherical z plus action noise versus
  deterministic reward-inferred evaluation. Added a versioned 12-GIF gallery,
  300k defaults, convergence analysis and results, controlled reward comparison,
  and agent/experiment handoffs from this session.
- **Validation:** CPU FB tests: 7 passed, 2 CUDA cases skipped; one upstream
  pytest-plugin deprecation warning. Gallery and training-document links resolve;
  git diff whitespace check passed. Remote fb branch had no new commits.
- **Publication:** Prepared for commit and push to origin/fb; training continues.
