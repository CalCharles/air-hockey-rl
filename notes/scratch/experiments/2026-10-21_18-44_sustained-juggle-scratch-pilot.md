# Scratch pilot for sustained juggling exposed failure-label and initialization issues

- **Date**: 2026-10-21 18:44 UTC
- **Status**: abandoned (checkpoints retained)
- **Run dirs**: `runs/td3/sustained_20261021/{g0975,g099,g0995}/seed0/`
- **Configs**: immutable run-local `args.yaml` / `config.yaml`; current `configs/td3/sustained/` files describe the subsequent warm-start campaign, not this pilot

## Question

Can longer-horizon, fixed-step TD3 learn sustained juggling while preserving identified control gains and introducing more realistic observation timing?

## Setup

Three parallel GPU runs, one training seed each, discounts 0.975 / 0.99 / 0.995. Sysid-v2 hist2 controller and measured workspace; 2,000-step cap, flat replay, eight critic and two actor updates per 32 collected steps, actor action-change penalty 0.05, new actor and critics from scratch. Physics and sensor timing were randomized. New observation history sampled once per control step instead of twice at delay integration splits.

Implementation and verification ran in the main session only, without subagents. Independent GPU lanes ran in parallel after targeted tests and a GPU smoke run.

## Results

At 100k training steps, nominal evaluation mean observed survival was 40.6 / 38.2 / 40.1 steps for the three discounts, with 0.0 / 0.4 / 0.0 recorded contacts per episode. Each evaluation had eight paired episodes and a 2,000-step cap; none reached the cap. These are single training-seed pilot values, not across-seed estimates.

Inspection of evaluation end reasons exposed a second terminal-label issue: after removing the public step wrapper's promotion of all truncations to terminal flags, puck-pass-paddle failures were still labeled truncations internally, allowing inappropriate value bootstrapping. Low-motion failures had the same semantic issue. Both were corrected before the next campaign. Early evaluator `failures` counters also missed failure truncations; mean lengths and reached-cap fractions remain interpretable, but do not use those old failure counts as authoritative.

The runs were stopped around 210k steps. Their checkpoint artifacts remain. The initial launch also failed because Slurm resolved `env` to a non-executable user wrapper; using `/usr/bin/env` fixed launch. No training ran in those failed steps, and allocations were retained.

## Conclusion

This pilot did not learn a reasonable juggling policy. It justified correcting true failure labels and using the existing source actor for useful initial contact data, rather than claiming that longer episodes alone solve the task.

## Next

Follow-up: [warm-start campaign](2026-10-21_18-50_sustained-juggle-warm-start.md). No controller gain or control-authority increase was used.
