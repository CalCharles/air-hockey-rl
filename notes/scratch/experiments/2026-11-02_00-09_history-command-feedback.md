# Deployment history now records final commands after holds and overrides

- **Date**: 2026-11-02 00:09 UTC analysis/writeup; America/Chicago date 2026-11-01, offset -0600.
- **Status**: done implementation and offline checks; juggling research/training remains running.
- **Related**: [normal-path parity and identified limitation](2026-11-01_23-54_first-1m-smoothness-comparison.md).

## Problem and resulting behavior

The history/RMA deployment adapter called the agent, which immediately stored its intended action. The real policy runner could subsequently substitute a zero hold, clip/noise the action, apply an exploration primitive or force standstill. The private window then described an action different from the final normalized command sent to the environment. This differed from training-side history, which records the command passed to the environment, before its internal workspace clipping. The mismatch could persist for the 50-step history window.

The adapter now uses a deployment-only history-buffer subclass with a method that replaces the newest action. After successful `env.step`, the runner calls the actor's optional `on_env_action` hook with a copy of the final normalized command captured before environment-internal clipping. The adapter replaces that newest action in place; it does not append a second time step or alter state features. Episode resets clear the feedback state. A callback without a preceding prediction, duplicate callback or invalid action fails explicitly. If `env.step` raises, feedback is not committed.

Normal predicted actions remain the same. Following a hold or override, future policy predictions now use a window containing the final commands. PID, action filters, workspace limits, hold commands, stop checks and existing `last_action_mode` semantics are unchanged. The callback records the command sent to the environment, not proof of hardware execution. Stateful-buffer changes are confined to the deployment adapter; shared simulation/training history and all protected live source manifests remain unchanged.

## Implementation and verification

- [Deployment adapter](../../../scripts/td3/helper/real_eval_agents.py): deployment-only command history and `on_env_action`.
- [Policy runner](../../../scripts/td3/helper/real_policy_runner.py): step-before-feedback ordering and protection against env-internal mutation of the action argument.
- [Tests](../../../scripts/td3/tests/test_history_command_feedback.py): normal paths/resets; zero holds and nonzero overrides; history-network predictions versus the training-side flat history layout over two 120-step episodes; adapted-RMA network versus explicit command-history unroll over 90 steps; no duplicate time step; finite/action-shape checks; callback ordering; failed env steps; actors without a hook.

**Nine tests passed** in `test_history_command_feedback` and the existing `test_real_eval_agents` suite. The neural tests use randomly initialized CPU models to verify bookkeeping, without training new policies. They cover repeated 50-step rollovers and hold/override histories with the exact network interfaces used by the policies.

The updated offline verifier invokes the new final-command hook on every normal action. The retained 3.6M target still produces **9,146 exactly matching commands across 32 saved episodes**, up to 300 steps each. Inputs are the original noisy observations and known previous commands; episode histories reset correctly. [Post-change parity results/source hashes](../../../docs/assets/juggling/history-deployment-parity-command-feedback-20261102.json); [pre-change normal parity](../../../docs/assets/juggling/history-deployment-parity-20261101.json). These are dated records for the fixed model SHA256 `c8a1df78ade41cfbe6845a84f4b21480f512973be7a89f0baf66daf5d452b31b`, not a blanket guarantee for future checkpoints.

No real environment was constructed or robot connected. Hardware timing, command delivery, arm dynamics and protective-stop risk remain unverified. Hold/override correctness here is a policy-memory test, not a hardware motion experiment or a claim of infinite juggling. The [eval pipeline documentation](../../docs/training/real-world-eval-pipeline.md#stateful-policies-and-final-commands) records the hook contract and remaining limits.

## Reproduction and live-work preservation

Tests: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest scripts.td3.tests.test_history_command_feedback scripts.td3.tests.test_real_eval_agents`.

Normal parity: `.venv/bin/python -m scripts.analysis.juggle_deployment_parity --bundle runs/td3/noisy_history_20261029_seed0/target_actor_3m6_bundle --results runs/td3/noisy_history_20261029_seed0/independent_selected_target_3m6_20k_eval/results.json --out NEW_OUTPUT_JSON --steps 300`. This requires the original local ignored model/trajectory artifacts and performs CPU inference only.

All eight live trainer manifests and the queued long-reference completion manifest still match their reviewed hashes. No checkpoint warm start, live learner/evaluator change, observation-history sampling change, noise/occlusion/delay simplification, controller change, geometry/contact/workspace change or altered DR distribution. GPU allocation remains 11 including three preexisting jobs. Live learners and CPU long-reference test were verified through Slurm; the CPU completion remains pending on its parent. No additional training was launched for this fix.

Validation includes the nine implementation tests, retained-policy action parity, source/hash preservation, reviewed production diff, links/images, unique ordered logbook anchors/TOCs and whitespace. No new performance figures are warranted for this bookkeeping change. Continue full-budget policy comparisons and longer independent tests; the research goal remains active and unsolved.
