# Aim-point actors and previous-action critics pass independent scratch CPU smoke checks

- **Date**: 2026-10-29 19:40 UTC
- **Status**: done (implementation/CPU validation); both full GPU trials planned, not launched
- **Run dirs**: `runs/td3/noisy_{absolute_target,critic_context}_smoke_20261029/`
- **Configs**: [aim-point full candidate](../../../configs/td3/sustained/noisy_absolute_target_scratch.yaml), [critic-context full candidate](../../../configs/td3/sustained/noisy_critic_context_scratch.yaml)

## Questions
Can a learned absolute aim-point representation provide a useful policy prior while retaining canonical normalized commands? Separately, does giving the critic the previous action already available to the actor help account for target smoothing? These are untested control hypotheses; the checks below validate implementation rather than performance.

## Independent candidates
Both full configs derive from the existing high-exploration baseline, seed 0, 6M planned, gamma .995, 64-wide residual networks, unchanged upper-half reward and established three-parameter DR. Only one flag changes in each candidate.

1. `actor_absolute_target: true`: the same 32 policy inputs (canonical history plus previous action) produce a learned workspace aim point. The actor computes `tanh((aim - observed_paddle_xy) / [.26, .12])` and sends that normalized action through the original env/PID/smoother. Aim center [.6959, −.0076], half-span [.19, .36] derive from fixed public workspace geometry. No actual simulator state, teacher or residual base. This bounded nonlinear output prior changes policy expressivity as well as coordinates; it is not a one-to-one equivalent parameterization of all original mean commands. Observed paddle latency remains present. Geometry buffers identify the checkpoint automatically; old checkpoints retain their original class.
2. `critic_last_action: true`: actor input/output remains the baseline. Critics condition on `[observation, previous_action]`. At the Bellman successor, action context is the transition's taken action, with existing true-terminal masking retained. Q input width is 34 (30 observation + 2 previous + 2 candidate action), versus baseline 32. Replay and simulator observations remain 30-dimensional. The actor-loss and optional CQL branches use the same consistent critic context.

Flags default false. Both variants require standard non-HER/non-RMA TD3 and scratch initialization. Actor/critic weights are freshly randomized; target-network copies are standard in-run initializations. Neither smoke, incumbent, regression nor observer weights may initialize a full trial. The fixed observer from E017 is not used.

## CPU smoke runs completed

| Variant | Actual start (America/Chicago) | Completed including runner final evaluation |
|---|---|---|
| absolute_target | 2026-10-29T14:22:49.798099-05:00 | 2026-10-29T14:23:07.333904-05:00 |
| critic_context | 2026-10-29T14:22:49.797780-05:00 | 2026-10-29T14:23:07.328421-05:00 |


Each smoke: CPU/eager, compilation and CUDA graphs disabled, 96 environment steps, random collection starts learning after 8, minibatch 8, one critic and actor update per 32 steps, saved checkpoint at 64. Saved training states confirm **3 critic updates** and **36 actor optimizer parameter entries** each. Critic first projection shapes are `(64,32)` for aim-point and `(64,34)` for action context. Aim-point model class reloads as `absolute_target_agent`; context actor reloads as unchanged `deterministic_agent`. Checkpoint/final evaluation completed. Missing sparse logging plots are expected after only three learner cycles.

No learned-control conclusion or meaningful policy figure follows from these 96-step runs. Random smoke GIFs remain local under their evaluation directories; no new tracked qualitative clip. Full GPU training and CUDA capture for enabled flags remain untested.

## Tests and default compatibility
All **32 targeted tests** passed. They cover canonical sensor/DR contracts, scratch guards, reward neutrality, observer checks, aim-point monotonic observed-pose feedback and gradients, exact saved-policy action round trips for both new and legacy classes, critic successor context, terminal masking, CQL widths and warm-start rejection. Additional history-trainer flag-rejection cases were rechecked separately.

A reproducible [CPU compatibility driver](../../../scripts/analysis/verify_td3_update_compatibility.py) loads the pre-change update implementation from commit `c4c422bf359989c5bf5e3ecdcc2c89a3837ab0ef` and compares disabled-flag outputs and gradients on fixed real 64-wide residual actor/twin critics. Batch 16, torch seed 20261029, fixture gamma .9, epsilon .001, actor-delta penalty .005, both no-CQL and CQL alpha .1 (two random actions/sample). **All maximum absolute differences are exactly zero**, including critic/actor outputs and parameter gradients. [Raw verification artifact](../../../docs/juggling-update-compatibility.json) records source hash and parameters. This is CPU eager evidence, not CUDA graph equivalence or a full training-distribution proof.

## Reproduce

```sh
.venv/bin/python -m unittest scripts.td3.tests.test_juggle_continuing scripts.td3.tests.test_puck_state_filter scripts.td3.tests.test_absolute_target_agent scripts.td3.tests.test_critic_action_context -q
.venv/bin/python -m scripts.analysis.verify_td3_update_compatibility
# Existing local smoke args (run only with new roots):
.venv/bin/python -u -m scripts.td3.juggle_noisy_scratch --root <new-aim-smoke-root> --args-file runs/td3/noisy_absolute_target_smoke_20261029_args.yaml
.venv/bin/python -u -m scripts.td3.juggle_noisy_scratch --root <new-context-smoke-root> --args-file runs/td3/noisy_critic_context_smoke_20261029_args.yaml
```

Full candidate launch commands use the two tracked configs linked above instead of local smoke args. Do not launch until current evidence/resources justify it; neither full run has started in this record.

## Next
Continue existing baseline, history and survival trials. History has improved to 612.875 mean validation steps at 1.4M, with 0/8 caps, so immediate interruption is not justified by its weak early checkpoints alone. The new critic context is a compact additional-control-state hypothesis; aim-point output is a separate structural prior. Evaluate candidates under unchanged simulator/DR and require untouched acceptance starts if sustained survival emerges. Goal remains unsolved.
