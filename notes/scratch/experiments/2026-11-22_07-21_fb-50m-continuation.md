# Both FB models continue toward 50M steps with faster updates

- **Date**: 2026-11-22 07:21 UTC
- **Status**: in-flight
- **Run dirs**: `runs/fb/realplant_full_50M`, `runs/fb/realplant_current_50M`
- **Configs**: `configs/fb/fb_50M_full.yaml`, `configs/fb/fb_50M_current.yaml`
- **Node / allocation**: c620-022, Slurm 1154576
- **PIDs at launch**: full 1566382, current 1566383
- **W&B full**: https://forge.coreweave.com/wandb/maxrudolph/air-hockey-fb/runs/vmurui9i
- **W&B current**: https://forge.coreweave.com/wandb/maxrudolph/air-hockey-fb/runs/fahl79s8

## Question

Does substantially more reward-free FB experience improve the weak puck-goal
results? Continue both 1M models to 50M cumulative interactions, retaining their
architectures and objectives while making the budget feasible.

## Setup

See [50M trainer details](../../docs/training/fb-50m.md). GPU graphs, fused Adam,
CPU actor collection in 64-transition blocks, TF32 matmul precision, one update
per eligible transition. Original replay was not saved: restore model/targets
and optimizers, collect 5k policy transitions, then resume learning. New saves
include replay/RNG/counters. Simulators reset at allocation restarts.

## Initial validation / performance

- Update-only benchmark: eager ~87 updates/s; graphs ~640; fused graph ~750.
- End-to-end short continuation: ~420–500 steps/s for one process.
- Both real jobs initially sustain ~360 steps/s each, finite metrics after 30k
  additional transitions (~720 steps/s aggregate).
- CPU/CUDA upstream-parity tests, graph capture restoration, deterministic graph
  parity, checkpoint replay/optimizer continuation and baseline FB tests passed.
- Actual 1M full checkpoint continued to 1,008,000, then reloaded its replay and
  continued to 1,009,024 without refill; counters and weights/optimizers preserved.
- Current-only 1M checkpoint continued on CUDA, saved/reloaded, and generated
  task evaluation metrics and GIFs through the fast trainer.

## Allocation plan

The dev allocation ends 2026-11-22 06:51 CST. Both trainers checkpoint and stop
five minutes before that. A 48h one-node continuation job is prepared at
`runs/fb/slurm/50m_continuation/`, dependent on the dev allocation ending.
**It has not been submitted**; the user must run its submit wrapper on a login
node. The separate four-seed 1M prototype job remains unchanged.

## Conclusion

The two 50M-target runs have started successfully; 50M completion and policy
quality are not yet established. Walltime estimates depend on sustained
throughput and queue availability. Actor/update interleaving and numeric kernels
changed, so this is not a bit-identical extension of the original implementation.

## Next

Submit the prepared handoff from a login node. Inspect long-run curves/GIFs and
repeat the paired puck-goal grid at later milestones. Record final outcomes in
a new note; do not treat this launch record as a completed experiment.
