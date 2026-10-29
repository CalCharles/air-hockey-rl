# From-scratch long-history training passes a canonical-sensing CPU smoke test

- **Date**: 2026-10-29 17:50 UTC
- **Status**: done (smoke test); full trial planned, not launched
- **Run dirs**: `runs/td3/noisy_history_smoke_20261029/`
- **Configs**: `runs/td3/noisy_history_smoke_20261029_args.yaml` (actual smoke); [planned full config](../../../configs/td3/sustained/noisy_history_scratch.yaml)

## Question
Can the existing deployable history TD3 architecture train from scratch through the guarded noisy runner while retaining the exact canonical sensor stream? E010 supports exploring memory, but gives no control-performance guarantee.

## Setup
Reused `scripts.rma.train_base_policy` in `rma_mode: history` (no privileged actor input, no adaptation teacher/phase 2). Actor uses 50 prior latest-frame/action pairs, a 32-wide temporal encoder, eight-dimensional learned latent and 64-wide policy. Both critics see deployment observations and history. No actor, critic, encoder checkpoint or residual base loaded. The normalizer metadata describes DR ranges; no actual episode parameters are policy input in history mode.

Canonical `original_sensing.yaml` and established three-parameter DR checked exactly before launch. Policy memory is appended after the environment returns its original observation; simulator history sampling is untouched. Production config derives from the reduced-exploration recipe (seed 0, 6M steps); no full production run started.

CPU smoke: 96 steps, initial random exploration 8 steps, batch 8, one critic and one actor update per 32 steps, target frequency 1, checkpoint at 64, eager CPU updates and rollout, compile/CUDA graphs disabled, replay success capacity 1,000. Checkpoint validation 2 starts with cap 80; runner final evaluation 32 starts with 10k cap. These budgets validate plumbing, not control capability.

## Results
Actual smoke start/end: 2026-10-29 12:48:15–12:48:34 -05:00, from durable status. Completed successfully. Saved training state reports global_step 96 and total_critic_updates 3; optimizer state contains 72 critic and 48 actor parameter entries. History metadata: raw obs 30, replay obs 430, H=50, per-step features 8. Saved bundle reloaded through the existing `HistoryAgent` survival-evaluation path.

All 20 targeted continuing-task tests pass, including guards against actor/critic checkpoints, history warm starts and privileged actor mode. Paired plain/history wrappers produce bitwise-identical first 30 observation components, rewards and termination/truncation traces on the same canonical seeded simulation and actions. Existing tests cover sensor-neutral body telemetry and pre-action diagnostic alignment.

No rollout GIF published for this nearly random smoke actor; it is not a meaningful qualitative performance artifact. Missing sparse logging plots in the short run are expected because only three learner cycles occurred and metrics log every 20 cycles. This is not a GPU compilation/performance test.

## Reproduce

```sh
.venv/bin/python -u -m scripts.td3.juggle_noisy_scratch --root <new-smoke-root> --args-file runs/td3/noisy_history_smoke_20261029_args.yaml
.venv/bin/python -m unittest scripts.td3.tests.test_juggle_continuing -q
```

Planned full run, when the current comparison has been assessed and resources are free:

```sh
.venv/bin/python -u -m scripts.td3.juggle_noisy_scratch --root <new-full-history-root> --args-file configs/td3/sustained/noisy_history_scratch.yaml
```

## Conclusion
The deployable history architecture is ready for a from-scratch noisy-simulator trial. No performance conclusion follows from 96 steps. The three current short-history trials remain uninterrupted.

## Next
Use current trial results to decide whether/when to launch the full history comparison. Preserve scratch initialization, canonical simulator and DR; do not reuse E010 regression weights or this smoke checkpoint. Evaluate saved validation-selected checkpoints and then new untouched acceptance starts.
