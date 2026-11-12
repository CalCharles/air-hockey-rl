# Online fine-tuning of the new-sim cap-7000 history_privileged policies: code + sim dry run

- **Date**: 2026-11-03 20:57 CST
- **Status**: code done and dry-run in sim; real-robot runs not started
- **Policies**: `data/newsim_scratch_cap7000_swa/seed<S>/training_state.pth` (SWA actor + critics fitted to it)
- **Dry-run output**: `runs/td3/online_finetune_dryrun/newsim_cap7000_recycle/hist4/seed0/trial{0..3}/`

## What was added

- `td3_online_real_finetune.py` now supports RMA `history_privileged` checkpoints (detected from `rma_mode`). Header item 8 describes it.
  - **Actor:** `PrivilegedHistoryActor`.
  - **Critics:** they see [x_t 30 | window 400 | state 8], so the replay observation is 438-d.
  - **γ:** must equal the training γ (0.995).
  - **Checkpoints:** `checkpoint_ep<i>/` gets `rma_meta.json`, so a fine-tuned policy evaluates with `async_td3_real_eval.py --agent history`.
- `helper/history_privileged_obs.py` builds the critic observation and the reward from observed positions, every step.
  - **State:** the latest puck and paddle position plus finite-difference velocities.
  - **Window:** pushed with the executed action.
  - **Reward:** the sim training reward, i.e. task reward + strike shaping via `motion_shaping.shaping_step`. The sim trainer now uses the same function, and it is verified identical to the old inline code.
  - **Edge termination:** a contact less than 3 cm from the top edge ends the robot episode.
- `async_td3_real.collector_process_modular` has two new hooks: `policy_env_wrapper` (policy runner only) and `collector_actor_factory` (strict weight sync).
- New args file: `configs/td3/td3_online_real_finetune/newsim_cap7000.yaml`.
- Sim dry-run driver: `scripts/td3/sim_dryrun_history_online_finetune.py`. It uses the real script's learner, recycling and logging, the training sim with 2-step action latency, and the observed-position state/reward.
- Brought in commit e1bc2fc (the real-plant sim, `scripts/rma/action_latency.py`) as uncommitted working-tree changes.
- Tests: `scripts/td3/tests/test_history_privileged_obs.py` (4 tests, all pass).
  - The window is identical to the sim trainer's.
  - The actor matches the eval agent.
  - `main()` runs end to end through the real `PolicyRunner` on Box2D.
  - Observed-position shaping agrees with the true-state shaping on 99.4 % of steps and 77.5 % of contact steps (the gap is finite-difference speed noise against the 0.3 m/s weak / 1.5 m/s pout thresholds). Over a 1500-step episode the shaping totals −105.7 (observed) vs −104.3 (true).

## Dry run (seed 0, 4 trials × 5 episodes, 500-step cap instead of 10,000 for time)

| Trial | D0 transitions | Returns per episode (shaped) | q_loss round 1 → 5 |
|---|---|---|---|
| 0 | 0 | 316, 331, 315, 302, 330 | 0.049 → 0.004 |
| 1 | 2510 | 320, 350, 307, −3 (37 steps, puck_hit_bottom), 318 | 0.109 → 0.005 |
| 2 | 4555 | 337, 346, 345, 331, 325 | 0.133 → 0.003 |
| 3 | 7065 | 330, 342, 326, 285, 324 | 0.145 → 0.002 |

- The pipeline runs end to end: D0 / α annealing, K = 5·T, ⌊K/20⌋ actor updates, per-round checkpoints, trial buffers.
- The online returns are flat. This is not a learning result: one seed, 500-step episodes.
- **Learner cost: about 8.6 ms per critic update** on the RTX 2070 SUPER. It is CPU-bound by kernel launches (GPU time is about 1.6 ms). So with K = 5·T the robot idles for about 0.9× the episode's own duration after each episode, e.g. about 7 min after a 10,000-step episode. Capturing the update in a CUDA graph would cut this; not done.

## Not covered by the dry run

The robot orchestrator itself is not covered: resets, transition holds, e-stops, the camera, and the real contact detector (geometric, 2 cm slop). The real path also keeps `terminate_on_puck_pass_paddle` off and `terminate_on_puck_stop` on, as in the earlier real fine-tuning runs; the sim training had them the other way round.
