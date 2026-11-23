# Continuing both FB models to 50 million steps

Use `scripts.fb.fast_training` with `configs/fb/fb_50M_{full,current}.yaml`.
Both targets are **50M cumulative simulator steps**, continuing the corresponding
1M checkpoints (49M additional steps). Architectures, FB losses, batch size 256,
and one learner update per eligible transition are retained.

## Throughput changes

- Capture GPU replay sampling, observation normalization, latent sampling,
  FB/actor updates, fused Adam and target updates in one CUDA graph.
- Capture warmup restores model/target parameters, normalization buffers, Adam
  moments/step counters and CUDA RNG in place; it does not count as training.
- Collect 64 transitions using a CPU actor and transfer them together. Refresh
  actor parameters and normalization once per block; then apply one update per
  newly eligible transition. This changes rollout/update interleaving and makes
  the actor up to 64 transitions stale, rather than acting on every update.
- Use PyTorch `high` float32 matmul precision (TF32 on supported CUDA hardware).
  This and fused Adam preserve objectives, not bitwise numerical trajectories.
- Check finite metrics once per block, aggregate episode logs, log every 10k,
  and evaluate/save GIFs every 250k rather than every 10k steps.
- Evaluate a CPU snapshot so graph-owned model state is never repurposed for
  evaluation. Ten episodes/task, two GIFs/task, W&B and local GIF galleries remain.

On the GH200, update-only benchmarks improved from ~87 updates/s to ~750;
short single-process simulator training measured ~420–500 steps/s. With both
50M jobs running together, early measurements are ~360 steps/s **per model**
(~720 combined). These are throughput observations, not convergence guarantees.
At sustained 360 steps/s, the remaining 49M steps take about 38 hours per model.

## Continuation semantics

Original 1M checkpoints lack replay. The first continuation restores all model,
target and optimizer state, collects 5,000 transitions with the restored policy
to refill replay, then resumes learning. This is an explicit warm continuation,
not an exact replay of the original run's RNG/data trajectory.

New checkpoints include the complete active replay ring, indices, optimizer
state, model/targets, counters and Python/NumPy/Torch/CUDA RNG states. The
simulator and active episode are **reset at every restart**: Box2D state, its
action-latency queue and the unfinished rollout are not serialized. Therefore
restarts preserve learning state and data but are not bit-exact uninterrupted
rollouts. Resetting an episode does not connect unrelated transitions in replay.

- `latest_checkpoint.json` points to an atomically completed checkpoint.
- `checkpoints/step_*/continuation.pt` holds replay/RNG state.
- Only the latest two complete continuation checkpoints are retained.
- `policies/step_*` retains smaller policy exports and inference samples for
  analysis. The normal saved-policy evaluation CLI and grid evaluator can load them.
- `status.json` distinguishes running, completed, failed and allocation/signal
  stops. A file lock prevents two trainers writing to the same run.
- `--resume-latest` restores the saved pointer; otherwise the initial 1M source
  in `--resume-from` is used. Counter targets remain cumulative across allocations.
- A resumed allocation currently opens a new W&B segment; local metrics append
  to the run, and `wandb_run.json` links to the active segment.

The trainer handles SIGTERM/SIGINT at block boundaries and saves. With
`SLURM_JOB_END_TIME`, it stops five minutes before allocation expiry. The handoff
also sets a 47h50m runtime cap inside a 48h job. Hard kills can still lose progress
since the last checkpoint; checkpoints are every 250k steps.

## Local runs and allocation handoff

Local runs:

```bash
.venv/bin/python -m scripts.fb.fast_training --args-file configs/fb/fb_50M_full.yaml
.venv/bin/python -m scripts.fb.fast_training --args-file configs/fb/fb_50M_current.yaml
```

Both were started on the existing dev allocation. No Slurm jobs were submitted
from the compute node. A one-node/two-process handoff is prepared at
`runs/fb/slurm/50m_continuation/`. It waits for allocation 1154576 to end, then
resumes these same run folders. From a **Vista login node**, submit explicitly:

```bash
bash /work/09312/rudolph/documents/air-hockey-rl/runs/fb/slurm/50m_continuation/submit.sh
```

Inspect `job.sbatch`, `full.yaml` and `current.yaml` first. The job requests
one GH node, 72 CPU cores, account ASC26008 and 48h. The wrapper refuses compute
nodes or active Slurm allocations. Queue delay and permission to request 48h
are scheduler-dependent. Keep this shared checkout unchanged while queued/running.
If a job ends before 50M, resubmit a continuation from a login node; do not mistake
an allocation stop for target completion. Do not reuse the original four-seed
prototype to launch these two continuations.

Generate a new handoff for a different allocation:

```bash
.venv/bin/python -m scripts.fb.prepare_50m_continuation \
  --folder runs/fb/slurm/new_handoff --after-job YOUR_DEV_ALLOCATION_ID
```

## Validation

Fast-update tests cover upstream math on CPU/CUDA, both backward architectures,
graph capture restoration and deterministic parity, fresh graph RNG/replay-size
changes, replay/optimizer reload, checkpoint retention, single-writer locking,
and allocation-stop saves. Both original checkpoint types were also exercised
through CUDA continuation; current-only CPU evaluation/GIF generation succeeded.
The original trainer tests still pass. The long-run configurations preserve a
separate baseline entrypoint for comparisons.

## Reward-inference sample size

A [fixed-checkpoint convergence study](../../scratch/experiments/2026-11-22_23-41_fb-z-convergence.md)
at 8.75M steps compared 100–1M replay draws against each full-buffer latent.
Across juggle/touch/reach in both models, 100k draws gave <5° and 300k gave <2°
95th-percentile angular error (30 repeats). The former 10k setting was notably
noisier on touch. These measure sampling stability, not task performance.
Reproduction: `scripts.fb.analyze_z_convergence`.

**Decision (2026-11-22): use `inference_samples: 300000` for future reward-z
inference.** The shared Args default, all four canonical FB YAMLs, and the
prepared `runs/fb/slurm/50m_continuation/{full,current}.yaml` configs now use
300k. New handoffs inherit the canonical values. Evaluation and saved inference
samples use `min(replay_size, 300000)`, sampled uniformly with replacement.
Existing saved inference datasets are not enlarged retroactively.

The initial processes on allocation 1159444 retained 10k until the explicit
stop/inference/restart on 2026-11-23. Both replacement processes (503624 and
503625) now have 300k in their resolved args.yaml; new training metrics confirmed
progress beyond the stop checkpoints.
Other older prepared bundles may retain explicit 10k overrides; update or
regenerate those before using them for future runs.

## Prepared batch continuation after allocation 1159444

Prepared 2026-11-23 UTC in `runs/fb/slurm/50m_after_1159444/`, using the new
300k inference setting and a dependency on `1159444`. Both canonical run
folders resume to 50M cumulative steps. One GH node, 72 CPUs, ASC26008, 48h.
As verified 2026-11-23 16:20 UTC, this bundle **has been submitted as job
1162255**, pending dependency `afterany:1159444`. The original session could
not authenticate to login1; submission subsequently occurred at 2026-11-22
18:06 CST. The submission command below is retained for reference; do not
submit a duplicate while 1162255 is queued:

```bash
bash /work/09312/rudolph/documents/air-hockey-rl/runs/fb/slurm/50m_after_1159444/submit.sh
```

Shell syntax, YAML values, dependency, and latest checkpoint completeness were
verified. Logs go under the bundle's `logs/`; status/checkpoints remain in the
original run folders. Keep the shared trainer checkout stable. To regenerate
in a new directory, use `python -m scripts.fb.prepare_50m_continuation --folder
<new-folder> --after-job 1159444`. A 48h budget is requested; completion still
depends on throughput. If already at 50M, the trainer exits without more updates.

## 300k performance GIFs and resumed training

[GIF gallery](../../../docs/assets/fb-300k/README.md): two fixed-seed episodes
per task (juggle, touch, reach) for both models, using saved 300k reward-inferred
latents and deterministic actions. All 12 GIFs are versioned with the docs.
Full/current checkpoints were 29,621,664/29,966,976 steps. After rendering,
both trainers resumed on allocation 1159444 with 300k loaded. See the
[resume record](../../scratch/experiments/2026-11-23_16-31_fb-300k-gifs-resume.md)
and [controlled reward comparison](../../scratch/experiments/2026-11-23_16-28_fb-10k-vs-300k-rewards.md).
