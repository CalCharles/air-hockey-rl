# Online FB real-plant training with W&B and periodic GIFs

- **Date**: 2026-11-21 22:28 UTC
- **Status**: in-flight
- **Run dirs**: `runs/fb/realplant_seed0_20261121`
- **Configs**: `configs/fb/fb_online_realplant.yaml`
- **Training code commit**: `77ee92e` on branch `fb`
- **Node / allocation**: `c620-022.vista.tacc.utexas.edu`, SLURM 1154576
- **PID at launch**: 1543329
- **W&B**: https://forge.coreweave.com/wandb/maxrudolph/air-hockey-fb/runs/lpdyb9si

## Question

Does reward-free online FB develop useful juggling, contact and position-reaching
skills on the real-plant proxy? This run begins testing the full-budget recipe.

## Setup

One million simulator steps, seed 0, GH200 GPU. Uses the requested
`sysid_v2_hist4_realplant.yaml` plus two-step action latency. See the
[online FB guide](../../docs/training/fb-online.md) for algorithm and task details.
Default batch 256, 5,000 uniform-action warmup steps, one FB+actor update per step.

```bash
.venv/bin/python -u -m scripts.fb.fb_training \
  --args-file configs/fb/fb_online_realplant.yaml \
  --output-dir runs/fb/realplant_seed0_20261121 \
  --wandb-name fb-realplant-seed0-20261121
```

Launched in a detached session with stdout/stderr in `train.log`; `process.json`
records the PID, invocation, node and code revision. This is a process in the
existing allocation, not a new batch job. Allocation ends 2026-11-22 06:51 CST.

## Initial validation

- Original six tests passed on this node.
- Six tests passed again with GIF generation enabled, including CPU/CUDA updates,
  GIF decoding/size/timing, checkpoints, and task relabeling (29.39 seconds).
- An online W&B smoke run (`u0l5ev8b`, `runs/fb/wandb_gif_smoke_20261121`)
  finished and uploaded 12 GIFs. The saved reach clip was visually inspected.
- Full run reached 5,000 steps and began updating with finite losses.

## Monitoring

W&B logs scalar metrics and `videos/*` GIF panels. Evaluation is after warmup,
every 10,000 steps, and at completion. Each task uses ten fixed-seed episodes;
the first two produce 160 px, 20 fps GIFs. Local gallery:
`runs/fb/realplant_seed0_20261121/gifs/index.html`, with stable clips in `gifs/latest/`.
Checkpoints are saved every 25,000 steps and at completion.

## Conclusion

Training is running; convergence and final performance are not yet known.
Contact/reach still use juggling starts and termination rules. Exact resume is
not implemented; periodic checkpoints preserve weights/optimizers and inference
samples, but not the full replay or environment state.

## Next

Inspect task curves and videos after more updates, then record final results in
a new experiment note. Avoid interpreting the warmup evaluation as learned skill.
