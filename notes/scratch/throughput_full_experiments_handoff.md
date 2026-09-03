# Handoff: full-budget throughput / parity experiments (new trainer)

*Written 2026-09-03 for the agent that runs the full experiments. Background:
[`notes/docs/training/training-throughput.md`](../docs/training/training-throughput.md)
and the 100k-step sweep in
[`experiments/2026-09-03_04-20_training-throughput-optimization.md`](experiments/2026-09-03_04-20_training-throughput-optimization.md).*

## What to run

Ten full-budget training runs with the **new** trainer, one per (task, setting):

| Task (config key) | Env task name | DR setting (2M steps, `td3_paramrand_pm25` recipe) | No-DR setting (1M steps, `td3_recommended_top50_hist2` recipe) |
|---|---|---|---|
| `reach` | `paddle_reach_position` | `configs/td3/throughput_bench/full/reach_dr.yaml` | `configs/td3/throughput_bench/full/reach_nodr.yaml` |
| `reach_vel` | `paddle_reach_position_velocity` | `.../full/reach_vel_dr.yaml` | `.../full/reach_vel_nodr.yaml` |
| `juggle` | `puck_juggle_upper_half_reward` (canonical) | `.../full/juggle_dr.yaml` | `.../full/juggle_nodr.yaml` |
| `puck_vel` | `puck_velocity` | `.../full/puck_vel_dr.yaml` | `.../full/puck_vel_nodr.yaml` |
| `touch` | `puck_touch` | `.../full/touch_dr.yaml` | `.../full/touch_nodr.yaml` |

Each args YAML is the canonical recipe with only `config:` (→
`configs/new_juggle/throughput_bench/sim_{dr,nodr}_<task>.yaml`), `run_name`
and `checkpoint_interval: 25000` changed. The sim configs are the canonical
`sim_paramrand_pm25.yaml` (DR) / `sysid_best_params_hist2.yaml` (no-DR) with
the task swapped (reach tasks add the goal keys and `num_pucks: 0`).

Entry points (the runner picks the right one from `eval_param_seed`):
- DR runs → `python -m scripts.td3.td3_training_dr --args-file <yaml>`
- No-DR runs → `python -m scripts.td3.td3_training --args-file <yaml>`

## The one command

`scripts/td3/run_experiments.py` is the canonical batch entrypoint: a set of
args YAMLs (files / dirs / globs), `--mode dr|nodr|auto`, a GPU list; jobs
run in order, at most one per GPU.

```bash
cd /home/air-hockey/daliu/air-hockey-rl
nohup .venv/bin/python -u -m scripts.td3.run_experiments \
    --mode dr --configs 'configs/td3/throughput_bench/full/*_dr.yaml' \
    --gpus 0 2 3 --out-root runs/td3/full_dr \
    > runs/td3/full_dr_runner.log 2>&1 &

# then (or on other free GPUs, in parallel)
nohup .venv/bin/python -u -m scripts.td3.run_experiments \
    --mode nodr --configs 'configs/td3/throughput_bench/full/*_nodr.yaml' \
    --gpus 0 2 3 --out-root runs/td3/full_nodr \
    > runs/td3/full_nodr_runner.log 2>&1 &
```

(`--mode auto` with the whole directory does both in one batch.) Anything
after `--` is forwarded to every trainer call, e.g. `-- --total-timesteps 500000`.

- Runs **at most one job per GPU**. Check `nvidia-smi` first and pass only
  GPUs that are idle (on 2026-09-03 GPU 1 was occupied by another user's
  process; do not use a GPU someone else is on).
- Each job's stdout: `<out-root>/<job>.stdout.log`; run dir:
  `<out-root>/<job>/` (checkpoints every 25k with `multi_env_eval.json` for
  DR runs, `eval.log`, GIFs, `run_meta.json` with wall-clock).
- Progress: `tail <runner log>` (`[start]` / `[done ] ... rc=0` lines) and
  `grep SPS <job>.stdout.log`.
- When all jobs finish the runner writes `<out-root>/summary.md` (wall,
  pre-learning SPS, training SPS, mean episode length, final eval return).
  Re-generate anytime with the same command plus `--summarise-only`.

Expected wall-clock with the new trainer: DR 2M runs ≈ 1–1.5 h each, no-DR
1M runs ≈ 0.5–0.8 h each (rate rises as episodes get longer); ten runs on
three GPUs ≈ 4 h.

## Optional: old-trainer reference at full budget

Only if a full-length learning-curve parity check is wanted (the 100k sweep
already showed parity at short budget). The old code lives in the git
worktree `runs/td3/throughput_opt/_work/baseline_wt` (commit `bf9936e`; if it
is gone, `git worktree add runs/td3/throughput_opt/_work/baseline_wt bf9936e`).

```bash
nohup .venv/bin/python -u -m scripts.td3.extras.throughput_bench \
    --args-dir configs/td3/throughput_bench/full \
    --out-root runs/td3/throughput_bench_full \
    --versions old \
    --gpus 0 2 3 \
    > runs/td3/throughput_bench_full_old_runner.log 2>&1 &
```

`reach` now works for the old code too (the array/scalar reward mix in
`paddle_reach_position_reward.py` was fixed on 2026-09-03; the old worktree
shares the editable `airhockey` package). Budget:
~10 h per DR run and ~5 h per no-DR run at 50–60 SPS, i.e. ~20 h wall on
three GPUs. Passing `--versions old new` runs both into the same
`--out-root` and the summary prints per-task speedups.

## What to report

1. The `summary.md` table (wall, phase SPS, final eval) for all ten cells.
2. For DR runs, the multi-env eval trajectory: `aggregate.mean_return_across_envs`
   from every `checkpoint_*/multi_env_eval.json`, as peak, rolling-5 mean at
   peak, and back-half (last 25 % of checkpoints) mean — the same summary the
   ablation READMEs use. Do not reduce a noisy trajectory to a single number
   without the shape (see the `feedback_no_clean_summary_overclaim` note).
   The historical reference for `juggle_dr` is the `paramrand_pm25` 2M run:
   rolling-5 peak 132.7 at 1M, back-half plateau ~118
   (`latest_models/ablations/README.md`).
3. For no-DR runs, `charts/avg_episodic_return` from TensorBoard over the
   last 10 % of steps (there is no multi-env eval without `eval_param_seed`);
   `training_summary.png` in each run dir has the curve.
4. Wall-clock per run and the projected old-trainer time (old rates from the
   100k sweep: ~56 SPS training phase for the ~45-step-episode tasks,
   ~125 SPS for `reach_vel`).

Write the results as a **new** dated file in `notes/scratch/experiments/`
(never edit the 2026-09-03 file), add a line to
`notes/scratch/experiments/INDEX.md`, and if the juggle_dr run reproduces the
historical curve, say so in `notes/docs/training/training-throughput.md`
under "Learning is unchanged".

## Things to know before running

- First training cycle compiles (`torch.compile`) for ~15 s; the cumulative
  `SPS` line dips there and recovers. `TD3_PROFILE_SECTIONS=1` prints a
  per-section breakdown if throughput looks wrong.
- Per-checkpoint eval runs in a background CPU subprocess; its summary line
  `[eval step N] ok ...` appears at the *next* checkpoint. If a run ends with
  `[eval step N] exit=1`, read `<run>/checkpoint_N/eval.log`.
- If the runner is killed, kill its children too (`pkill -f td3_training`),
  delete the half-written run dirs under `--out-root`, and relaunch — the
  trainer appends `r1` to an existing run dir instead of overwriting it.
- Do not pre-create run directories for the same reason.
- Everything under `runs/` is gitignored; the configs under
  `configs/td3/throughput_bench/` and `configs/new_juggle/throughput_bench/`
  are the only inputs.
