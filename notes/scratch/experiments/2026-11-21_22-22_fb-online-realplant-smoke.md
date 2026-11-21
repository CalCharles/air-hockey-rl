# Online FB trains and reloads on the real-plant simulator

- **Date**: 2026-11-21 22:22 UTC writeup
- **Status**: done
- **Run dirs**: `runs/fb/smoke_realplant`
- **Configs**: `configs/fb/fb_online_realplant.yaml`

## Question

Can the local Metamotivo FB agent train from online simulator interaction with no
dataset, then infer juggling, contact and paddle-position tasks?

## Setup

See [online FB instructions](../../docs/training/fb-online.md). Simulator config:
`sysid_v2_hist4_realplant.yaml`, external two-step latency, 32-dimensional policy
observation (history plus previous issued action). Metamotivo revision
`ff8dcc55cf58f766d365ab0be23a021a7e34d53d`. PyTorch 2.5.1, Python 3.12.2,
Box2D 2.3.10 built from Git, NVIDIA GH200. Local `.venv` inherits CUDA PyTorch;
only dependencies needed by this path were installed, not all repo extras.

```bash
.venv/bin/python -m scripts.fb.fb_training \
  --args-file configs/fb/fb_online_realplant.yaml \
  --output-dir runs/fb/smoke_realplant --total-timesteps 1000 \
  --learning-starts 128 --batch-size 64 --buffer-size 2000 \
  --eval-every-steps 500 --eval-episodes 2 --checkpoint-every-steps 500
```

## Results

- Six tests passed in 30.88 s, including CPU and CUDA training on the requested
  simulator, ring-buffer wraparound, changed network parameters, finite metrics,
  agent/policy checkpoint reload, deterministic evaluation, terminal semantics,
  latency queue reset, task reward relabeling and RNG isolation.
- 1,000 transitions, 873 FB/actor updates, finite final FB loss -26.7547 and actor
  loss -8.2129. Negative FB losses are expected from the diagonal objective.
- Measured 51.76 interaction steps/s, including periodic evaluation through the
  last training log (excludes final checkpoint/evaluation).
- Final two-episode juggling return: 17 ± 1 (episode standard deviation).
- Contact return: 1.5 ± 1.5, contact success 1/2 episodes; only 0.7% of inference
  samples contain a contact event.
- Position reward return: 0.02551 ± 0.02548, target-entry success 0/2 episodes at
  table position (0.70, 0.0), 5 cm radius.
- Reloaded the exported CUDA-trained policy on CPU via `--eval-checkpoint`;
  juggling/contact returns match exactly, reaching return agrees to floating-point
  precision. Checkpoints at 500, 1000 and `final` are available locally.

## Conclusion

Collection, learning, task inference, logging and checkpoint evaluation work.
This is an execution smoke test with one seed, not a convergence experiment.
Contact and reaching use the original juggling environment's resets/endings;
these numbers are not the separate canonical touch or no-puck reach benchmarks.
No full-budget training run has been launched.

## Next

Run the default million-step preset across seeds and inspect exploration coverage
and task curves. Exact training resume and no-puck benchmark adapters remain
separate extensions; the current checkpoint stores agent weights/optimizers and
an inference sample, not the full replay or simulator state.
