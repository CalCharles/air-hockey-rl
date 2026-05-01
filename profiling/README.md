# Profiling `train_lsgan_eipo`

This directory contains everything needed to profile the AMP-LSGAN + EIPO PPO
training script ([`scripts/smooth_policy/amp_history/amp_training/ppo/train_lsgan_eipo.py`](../scripts/smooth_policy/amp_history/amp_training/ppo/train_lsgan_eipo.py))
and reason about its bottlenecks. Two complementary tools are used:

| Tool             | Sees           | Best at                                           |
| ---------------- | -------------- | ------------------------------------------------- |
| **py-spy**       | Python / CPU   | Fast first-pass: which Python frames are hot?     |
| **torch.profiler** | CPU + CUDA   | Per-operator and per-CUDA-kernel timings, GPU util |

`py-spy` is a sampling profiler that requires no code changes — it tells you
*which Python functions accumulate the most time*. It cannot see GPU work
(CUDA is asynchronous; the CPU launches kernels and returns immediately).

`torch.profiler` is built into PyTorch and integrates with TensorBoard. It
records every CPU op and CUDA kernel with timestamps, so you can see GPU
utilization, kernel breakdown, and CPU↔GPU sync points. It requires lightly
instrumenting the training script — that's why `train_lsgan_eipo_profiled.py`
exists as a copy of the original.

---

## Directory layout

```
profiling/
├── README.md                          ← this file
├── train_lsgan_eipo_profiled.py       ← copy of the training script with
│                                        torch.profiler instrumentation around
│                                        the main training loop. The original
│                                        script in scripts/.../ppo/ is untouched.
│
├── configs/
│   └── eipo_profiling.yaml            ← Profiling config. Mirrors
│                                        eipo_target.yaml (full normal-training
│                                        params: num_steps=512, num_envs=8, etc.)
│                                        so the trace is representative of real
│                                        training. Only run_name / log_parent_dir
│                                        / num_iterations differ.
│
├── pyspy/                             ← py-spy outputs (already collected)
│   ├── eipo_profile.svg               ← Run 1: full sampling (--idle --subprocesses)
│   │                                    Includes idle time of vec-env workers
│   │                                    and TensorBoard writer thread.
│   ├── eipo_active.svg                ← Run 2: active-only sampling (no --idle)
│   │                                    Filters out queue-wait time, shows
│   │                                    only main-thread compute.
│   ├── run1_pyspy_full/               ← training-side artifacts for run 1
│   │                                    (args.yaml, config.yaml, TB events)
│   └── run2_pyspy_active/             ← training-side artifacts for run 2
│
└── torch_profiler/
    └── traces/                        ← One subdirectory per profile run.
        │                                Each subdir name == args.run_name.
        │                                TensorBoard --logdir at this level
        │                                shows each subdir as a separate run
        │                                in the dropdown.
        └── baseline_smallsteps/       ← num_steps=64 baseline (391 MB trace)
            └── *.pt.trace.json
```

---

## How to run

### 1. py-spy (already done; documented for reference)

py-spy is installed at `.venv/bin/py-spy` (v0.4.1). Two preset commands:

```bash
# Full sampling: includes idle workers (large flamegraph, useful first time)
.venv/bin/py-spy record \
    -o profiling/pyspy/eipo_profile.svg \
    --rate 100 --subprocesses --idle \
    -- .venv/bin/python scripts/smooth_policy/amp_history/amp_training/ppo/train_lsgan_eipo.py \
        --args-file scripts/smooth_policy/amp_history/configs/ppo/eipo/eipo_target.yaml \
        --device cuda:1 --log-parent-dir profiling/pyspy --run-name run1_pyspy_full \
        --disc-stationarity-mode target --disc-ema-tau 0.005 \
        --eipo-alpha-lr 0.01 --eipo-alpha-init 1.0
# Run for ~3 minutes past warmup, then Ctrl+C once.
# Open the SVG in a browser or VS Code.

# Active-only sampling: drop --idle and --subprocesses for cleaner main-thread view
.venv/bin/py-spy record \
    -o profiling/pyspy/eipo_active.svg \
    --rate 100 \
    -- .venv/bin/python scripts/smooth_policy/amp_history/amp_training/ppo/train_lsgan_eipo.py \
        --args-file scripts/smooth_policy/amp_history/configs/ppo/eipo/eipo_target.yaml \
        --device cuda:1 --log-parent-dir profiling/pyspy --run-name run2_pyspy_active \
        --disc-stationarity-mode target --disc-ema-tau 0.005 \
        --eipo-alpha-lr 0.01 --eipo-alpha-init 1.0
```

**Common gotchas:**
- ptrace is restricted (`/proc/sys/kernel/yama/ptrace_scope = 1`). The above
  commands work because py-spy launches the child itself. Attaching to an
  already-running process via `--pid` requires `sudo` or
  `sudo sysctl -w kernel.yama.ptrace_scope=0`.
- The "lagging behind in sampling" warning is harmless — drop `--rate` to 50
  or omit `--subprocesses` to silence it.

### 2. torch.profiler

The instrumented script at `train_lsgan_eipo_profiled.py` wraps the main
training loop with a `torch.profiler.profile(...)` context. Current schedule:

```python
schedule=schedule(wait=2, warmup=2, active=1, repeat=1)
```

| Iteration | Phase  | Notes                                            |
| --------- | ------ | ------------------------------------------------ |
| 1, 2      | wait   | Replay buffer, normalizer warm into steady state |
| 3, 4      | warmup | CUDA cache / autotune stabilizes                 |
| **5**     | **active** | Recorded into trace                          |
| 6         | flush  | `prof.step()` triggers `on_trace_ready`, then `sys.exit(0)` |

The trace handler writes to `profiling/torch_profiler/traces/{args.run_name}/`,
so each run lands in its own subdirectory automatically.

**Run command:**

```bash
cd /home/air-hockey/air-hockey-rl
.venv/bin/python profiling/train_lsgan_eipo_profiled.py \
    --args-file profiling/configs/eipo_profiling.yaml
```

That's it — `eipo_profiling.yaml` already specifies `device`, `run_name`,
`log_parent_dir`, and all EIPO/AMP knobs. Override anything on the CLI
(`--num-steps 64`) for cheaper/smaller runs.

**Expected output:**
- Wall time: ~1.5 minutes (6 iterations at full `num_steps=512`)
- Trace size: ~3 GB (single active iteration; `record_shapes=False`)
- Final line: `[torch.profiler] traces written to ... — exiting`
- Trace path: `profiling/torch_profiler/traces/profiling_baseline/*.pt.trace.json`

**To do a comparative profile** (e.g. before/after an optimization), bump the
`run_name` in `eipo_profiling.yaml` to something distinctive
(`run_name: optimized_v1`) and re-run. TB will show both as side-by-side runs.

---

## Viewing traces in TensorBoard

Install the profiler plugin once per environment (uses `uv`, not `pip`):

```bash
uv pip install torch-tb-profiler
```

Launch TB pointed at the parent `traces/` dir so all run subdirectories show up:

```bash
.venv/bin/tensorboard --logdir profiling/torch_profiler/traces --bind_all --port 6009
```

**On a remote server**, port-forward 6009 to your laptop. Three options:

1. **VS Code Remote-SSH (easiest):** open the **PORTS** panel in VS Code,
   click *Forward a Port*, enter `6009`, then click the `localhost:6009` link.
2. **SSH tunnel from laptop:** `ssh -L 6009:localhost:6009 user@host`
3. **VPN + direct hostname:** only works if firewall permits (`http://host:6009`).

Once the page loads:
- Top tab: **PYTORCH_PROFILER**
- Left **Runs** dropdown: select the run subdirectory
- Left **Views** dropdown: switch between `Overview`, `Operator`, `Kernel`, `Trace`, `Memory`
- For comparative analysis: top **DIFF** tab compares two runs side by side

**TB load times** scale with trace size and are slow:
- ~400 MB trace → 1–2 min
- ~3 GB trace → 5–15 min
- ~6 GB trace → 30+ min, may stall

If TB seems frozen, check it's still working: `ps -p <PID> -o pcpu,rss,etime`.
RAM should be growing while CPU stays > 5%. If RAM stops growing for several
minutes, kill it and re-profile with `--num-steps` smaller or `active=1`.

---

## Findings so far (baseline, num_steps=64)

The first torch.profiler run (`baseline_smallsteps`) gave a clear diagnosis:

| Metric                         | Value      | Verdict        |
| ------------------------------ | ---------- | -------------- |
| GPU Utilization                | **2.2%**   | Catastrophically low |
| Est. SM Efficiency             | **0.25%**  | GPU barely doing anything |
| Est. Achieved Occupancy        | **2.85%**  | Same |
| Average step time (one iter)   | ~14.2 s    |                |
| Kernel time (GPU compute)      | 0.31 s     | ~2.2% of step  |
| **CPU Exec time**              | **11.5 s** | **81.4% of step — main bottleneck** |
| Other (sync, launch overhead)  | 2.32 s     | ~16.4%         |

**Conclusion: severely CPU-bound.** The GPU is idle ~98% of the time while
the main thread is busy doing CPU-side work. This contradicts an initial
reading of py-spy ("CPU is waiting") — what looked like waiting was actually
the vec-env workers blocked in `queue.get()` while the **main thread** was
doing CPU work elsewhere.

**Caveat:** the baseline used `--num-steps 64` (vs default 512) to keep the
trace small. This artificially inflates the relative weight of per-iteration
fixed costs (TB logging, EIPO state updates, kernel-launch overhead). Under
default `num_steps=512`, GPU utilization should rise modestly (estimated
~10–20%), but the qualitative conclusion is unchanged: the script is
CPU-bound, not GPU-bound.

**Likely culprits (to confirm in the Operator / Trace views):**
1. **AsyncVectorEnv IPC overhead** — Python `multiprocessing.Pipe` send/recv
   per env step is expensive at 4096 steps/iter.
2. **AMP feature processing on CPU** — normalizers and feature builders
   may be running in numpy / torch-CPU before being moved to GPU.
3. **`.item()` / `.cpu()` / `aten::to` sync points** — these stall the GPU
   while the CPU reads a single scalar (e.g. for logging, KL early-stop).

The next profile (`profiling_baseline` run, full `num_steps=512`) should
verify these and pinpoint exact functions via the **Operator view**
(sort by `Self Host Duration`).

---

## Quick reference: what each view in TB tells you

| TB view    | Use it to find                                              |
| ---------- | ----------------------------------------------------------- |
| Overview   | Headline GPU utilization, Step Time Breakdown pie chart, perf recommendation |
| Operator   | Top CPU/GPU ops by time. Sort by `Self Host Duration` for CPU bottlenecks |
| Kernel     | Per-CUDA-kernel ranking. Useful when GPU-bound (not your case yet) |
| Trace      | Chrome-style timeline. Look for **gaps in the GPU lane** = stalls |
| Memory     | Peak GPU memory over time. Useful for OOM debugging        |

---

## Tips & gotchas

- **Always profile through a tmux session** if running > a few minutes,
  to survive SSH disconnects:
  ```bash
  tmux new -s prof
  # run your command
  # detach: Ctrl+B then D
  # reattach: tmux attach -t prof
  ```
- **Trace files are huge.** With `record_shapes=True`, expect 5–10× larger.
  Always start with `record_shapes=False` and only enable it if needed.
- **`active=1` is usually enough** for finding bottlenecks. More iterations
  = larger trace + slower TB load with diminishing returns.
- **Kill stale TBs before re-launching.** Multiple TBs on the same machine
  will collide on ports. Check with `pgrep -af tensorboard`.
- **Don't kill processes from other users.** `pgrep` shows everyone's
  TBs — only `kill` your own PIDs (you'll see `/home/<your-username>/...`
  in the path).
- **Comparing two profiles:** use distinct `run_name` values in the yaml
  config; both traces auto-organize into separate subdirs and TB's `DIFF`
  tab compares them.

---

## File reference

| File / dir                                          | What it is                                  |
| --------------------------------------------------- | ------------------------------------------- |
| `train_lsgan_eipo_profiled.py`                      | Instrumented training script               |
| `configs/eipo_profiling.yaml`                       | Profile-friendly config, normal training params |
| `pyspy/eipo_profile.svg`                            | py-spy flamegraph, full (with idle workers) |
| `pyspy/eipo_active.svg`                             | py-spy flamegraph, active-only             |
| `pyspy/run{1,2}_*/`                                 | Training-side artifacts for each py-spy run |
| `torch_profiler/traces/<run_name>/*.pt.trace.json`  | Chrome-trace JSON, viewable in TB profiler |
