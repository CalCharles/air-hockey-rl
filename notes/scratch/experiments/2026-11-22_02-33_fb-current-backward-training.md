# Train online FB with only current observations in the backward model

- **Date**: 2026-11-22 02:33 UTC
- **Status**: in-flight
- **Run dirs**: `runs/fb/realplant_current_backward_seed0_20261122`
- **Configs**: `configs/fb/fb_online_realplant_current_backward.yaml`
- **Code commit**: `bbbdac9` on `fb`
- **Node / allocation / PID**: `c620-022.vista.tacc.utexas.edu` / 1154576 / 1552490
- **W&B**: https://forge.coreweave.com/wandb/maxrudolph/air-hockey-fb/runs/i4hou14q

## Question

Does restricting B to the current observed paddle and puck improve task inference,
compared with B seeing the full history and previous action?

## Setup

New model trained from scratch, seed 0, one million steps. The backward network
receives six features: newest paddle `(x, y, occlusion)` and newest puck
`(x, y, occlusion)`, indices `[12,13,14,27,28,29]` of the full policy observation.
These are noisy/delayed observations, not ground-truth simulator positions.
The mask convention is 1 for occluded/missing. Older history and previous action
are excluded from online/target B, goal-derived training latents and inference.

F and the actor retain their original architectures and 32-feature history plus
previous-action inputs. All other hyperparameters, reward definitions, simulator
config and two-step latency match the [baseline run](2026-11-21_22-28_fb-online-realplant-training.md).
Weights are newly initialized; this is not fine-tuning the baseline. The original
run remains running and its files are unchanged. Both initially share the GH200.

```bash
.venv/bin/python -u -m scripts.fb.fb_training \
  --args-file configs/fb/fb_online_realplant_current_backward.yaml \
  --output-dir runs/fb/realplant_current_backward_seed0_20261122 \
  --wandb-name fb-realplant-current-backward-seed0-20261122
```

## Initial validation

Nine tests passed in 80.10 seconds, including full/current B variants on CPU and
CUDA, finite updates, checkpoint and policy reloads, and GIF generation. Explicit
invariance and gradient checks confirm that old observations and the previous
action cannot affect either online or target B. Reward-inferred and goal-derived
latents also obey the restriction. F and actor parameter shapes match baseline.

A separate default-width CUDA smoke run completed 100 transitions / 69 updates
with finite losses. CPU CLI checkpoint reload reproduced its evaluation results
(to floating-point precision). Artifacts: `runs/fb/current_backward_smoke_20261122`.

## Monitoring

Same W&B project as baseline. GIFs after 5k warmup, every 10k steps, and at finish;
ten episodes/task, first two recorded. Local gallery is `gifs/index.html` under
the run directory; newest clips are in `gifs/latest/`. Checkpoints every 25k.
The detached process is in the existing allocation, ending 06:51 CST Nov 22.

## Conclusion

Implementation and execution checks pass. Full-budget learning results remain
pending. Compare paired evaluation seeds against the baseline after training.

## Next

Record full-budget results and task curves in a new experiment note. Compare
juggling/contact/position metrics and videos; do not infer quality from the
short smoke test or warmup evaluation.
