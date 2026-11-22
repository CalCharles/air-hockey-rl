# FB Slurm prototype for Vista

This prototype prepares a single-node `gh` batch job with a bounded queue of FB
runs. It does not submit automatically. Four processes share the node's GPU;
each uses one PyTorch/BLAS thread. This concurrency is a starting point, not a
benchmarked optimum. Training settings, W&B, and GIFs come from the supplied
FB configs. Seeds and output paths are assigned per run.

Prepare from the repository root (safe on a compute node):

```bash
.venv/bin/python -m scripts.fb.slurm_jobs prepare \
  --folder runs/fb/slurm/prototype_4seeds \
  --configs configs/fb/fb_online_realplant_current_backward.yaml \
  --seeds 0 1 2 3 --parallel 4 \
  --account ASC26008 --partition gh --walltime 24:00:00
```

Review `manifest.json`, `job.sbatch`, and the snapshotted config under that folder.
Then, **from a Vista login node**, submit explicitly:

```bash
bash runs/fb/slurm/prototype_4seeds/submit.sh
```

The wrapper refuses submission inside a Slurm allocation or from a host whose
short name does not start with `login` followed by a digit. No remote SSH or
submission is performed by preparation. The `.venv` interpreter is an absolute
path; the job assumes its dependencies and CUDA libraries are available on the
allocated node. It inherits the submission environment and never sources a
personal interactive shell configuration. Verify that environment on the login
node before submitting.

Defaults: one node, 72 allocated CPU cores, account `ASC26008`, partition `gh`,
24 hours. These reflect the current Vista allocation/template, not a guarantee
of queue eligibility or allowed walltime. All are configurable. There is no
`--gres=gpu` directive because the existing Vista GH template allocates the GPU
through the node partition. This is not portable to arbitrary Slurm clusters.

Pass multiple configs to queue their Cartesian product with the seeds, keeping
at most `--parallel` training processes active. Each process has a distinct run
folder, log, W&B name, checkpoints, and GIF gallery. Slurm output is under `logs/`;
completion codes are written to `results.jsonl`. Any failed child makes the job
exit nonzero after the queue completes. Running the same bundle again is refused.

The manifest records the prepared Git HEAD and refuses execution if HEAD changes.
It snapshots trainer argument YAMLs but **does not snapshot code or simulator
configs or detect uncommitted changes**. Use a clean, stable checkout when moving
beyond this prototype, and prepare a new bundle after committing/revising code.
Keep the shared checkout unchanged while jobs run. Exact training resume is not
implemented: Slurm timeouts or preemption may lose work after the latest checkpoint.

Validation is limited to local manifest generation, command quoting, shell syntax,
and guard tests. No scheduler submission or batch-node execution has been tested.
