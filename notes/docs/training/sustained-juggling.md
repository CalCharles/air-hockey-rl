# Sustained juggling campaign

The objective is repeated paddle contacts with very low drop probability over long rollouts, while retaining the real robot's measured controller and workspace. Passing a finite rollout cap is censored survival, not proof of indefinite juggling or hardware safety.

## Changes made on 2026-10-21

- The Gymnasium environment now distinguishes actual failure from a time limit. A time limit does not set `terminated`; losing the puck past the paddle and the enabled low-motion stop condition do. The time-limit transition retains its task reward instead of replacing it with `truncate_rew`.
- `train_every_steps: 32` enables fixed-step replay insertion and learning. This mode requires flat replay. It drains staged transitions during an episode without resetting the cumulative episode return; completed episodes still update statistics normally. Default `0` preserves episode-end updates for existing recipes.
- `actor_learning_starts` lets new critics warm up before a pretrained actor changes. The default retains the original schedule.
- `actor_action_delta_penalty` adds a soft squared difference from the previous action to the normalized-Q actor objective. It is default-off and is not a hardware acceleration limiter. The campaign uses 0.005.
- Box2D's opt-in `history_sample_once_per_step` records one observation-history entry at the observation snapshot per control step, instead of one per integration substep. This matches the robot's control-step sampling more closely. Existing configs retain legacy history sampling. It changes the temporal meaning of the same 30-value observation and must be treated as a training/deployment contract.

These changes are wired into `scripts.td3.td3_training`. The subsequent clean-sim iteration work also wires fixed-step collection, actor warmup, and action-change regularization into the non-HER RMA/history trainer. HER retains episode-based collection; fixed-step RMA/HER is explicitly rejected.

## Training setup

Generated files:

- Environment: `configs/new_juggle/sustained/juggle.yaml`.
- Algorithm settings: `configs/td3/sustained/{g0975,g099,g0995}.yaml`.
- Generator, launcher, status and selection: `scripts/td3/juggle_campaign.py`.
- Evaluation and GIFs: `scripts/td3/juggle_survival.py`.

The environment derives from `sysid_v2_hist2.yaml`, including measured table offsets and robot workspace limits. PID gains, controller smoothing, action scaling, force limit, and speed limit are not increased. Per-reset randomization covers paddle density, gravity, damping, shared observation timing, puck camera latency, and puck noise. A 30 fps puck frame grid and plain occlusions remain enabled. The training horizon is 2,000 steps at the configured 20 Hz.

Actors initialize from `latest_models/canonical/hist2_motion0_v2/model.pth`; critics and optimizers start fresh. This is simulation-only policy training from a historical source actor, not continuation of its replay or value estimates. Critics start at 10k steps, actor updates at 50k. The recipe uses 16 critic and 2 actor updates every 32 steps, a 300k flat replay, lower actor learning rate, and reduced exploration. The three lanes vary only discount (0.975, 0.99, 0.995), with three sequential seeds each and 1M steps per run.

Implementation and integration were performed in the main session only, with no subagents. GPU training lanes run in parallel; seeds within a lane run sequentially. Targeted tests and a CUDA graph smoke run preceded launch. Final long evaluations follow each completed training run.

## Run and monitor

Current campaign: `runs/td3/sustained_20261021_warm/`. Three existing A40 allocations were used (106217, 106218, 106219); no new allocations were requested. The earlier `runs/td3/sustained_20261021/` is a stopped scratch pilot, retained with its checkpoint configs and baseline artifacts.

At the 200k checkpoint, all three full-policy seed-0 trials remained below the unchanged source actor. A separate bounded residual/CQL comparison was therefore launched in `runs/td3/sustained_20261021_residual/`: `res_g0975`, `res_g099`, and `res_g0995`, 500k steps each, seed 0 only. Each shares its allocation with a full-policy run; observed GPU utilization before adding these jobs was about 48%. These comparison configs use five conservative critics (`cql_alpha: 20`), batches of 256, two critic updates per 32 steps, and a frozen source actor plus a correction bounded by 0.15 per action component. They retain the same controller gains and environment. They are experimental comparisons, not established improvements.

```bash
# Monitor without loading torch or requiring a GPU.
.venv/bin/python -m scripts.td3.juggle_campaign --status

# Monitor the separate bounded residual comparison.
.venv/bin/python -m scripts.td3.juggle_campaign --status \
  --root runs/td3/sustained_20261021_residual \
  --recipes res_g0975 res_g099 res_g0995

# Recreate recipe files before a future campaign, after reviewing changes.
.venv/bin/python -m scripts.td3.juggle_campaign --prepare

# Start a NEW campaign in existing idle allocations; refuses an existing manifest.
.venv/bin/python -m scripts.td3.juggle_campaign --launch \
  --root runs/td3/sustained_NEW --job-ids JOB1 JOB2 JOB3 \
  --steps 1000000 --seeds 0 1 2
```

Launchers run detached Slurm steps with durable output, not dependent on an interactive tool cell. Each lane has a `status.json`, `seedN.log`, and per-seed run directory. `launch.json` records exact commands, allocations, and launcher PIDs. `--stop --root <campaign>` terminates only verified launcher process groups from that manifest and retains allocations and artifacts. It does not create a new checkpoint; stop after a checkpoint when preservation of recent progress matters.

Checkpoint directories contain saved model/config/args and `survival_eval/{results.json,summary.md,*.gif}`. Every 100k steps, eight paired validation episodes run to a 2,000-step cap on nominal physics, randomized conditions, and fixed 50/100 ms camera-latency probes. The final actor gets 32 episodes per scenario with a 10,000-step cap and separate seeds starting at 100000, written to `seedN/long_eval/`. A run is not complete until training and that evaluation have finished.

Select intermediate checkpoints using validation episodes, then evaluate separately on unused seeds:

```bash
.venv/bin/python -m scripts.td3.juggle_campaign --select-run <seed-run-directory>
.venv/bin/python -m scripts.td3.juggle_survival \
  --run-dir <selected-checkpoint-directory> --horizon 10000 \
  --episodes 100 --seed-base 200000 --out-dir <new-test-output-directory>
```

Selection maximizes worst-scenario survival fraction, then worst-scenario mean observed length. The automatic final evaluation tests the final actor, not a retrospectively selected best checkpoint. Once final-test results guide another selection or recipe change, use fresh seeds for the next final test.

Reports include survival at multiple durations, contacts, reasons for episode ends, simulator paddle speed/acceleration, and command changes. At least three recorded contacts are required for the additional `sustained_juggling_fraction` metric; this remains a coarse check, not a complete detector of valid repeated juggling. No policy has been promoted for robot deployment by this campaign setup.

New evaluator output saves the first 200 frames to `<scenario>_start.gif` and a rolling last-400-frame clip to `<scenario>.gif`, so a late failure can be inspected. Older checkpoint GIFs may show only the beginning of a rollout. Rendering uses vertical orientation, width 160, and 20 fps with reward/contact overlays.

Evaluation also snapshots its effective config/args and writes `<scenario>_seed<seed>.npz` containing observations, next observations, actions, rewards, and termination/truncation flags. These are policy-visible observations, not a privileged physics-state recording. The per-episode JSON links each trajectory file. Earlier evaluations predate this recording addition.

## Verification and limitations

The short GPU smoke test exercised CUDA graph optimization, replay chunking, checkpoints, and final evaluation. The new targeted tests cover time-limit bootstrapping/reward, actual failures, history sampling, replay chunking and restored returns, smoothness gradients, configuration invariants, and censored statistics.

The initial full TD3 unittest run found one unrelated existing failure: `test_direct_args_keys_are_applied` expects an `updates_per_second` field absent from the real runtime mapping. Both its test and implementation were byte-identical to HEAD. Do not interpret the whole suite as passing.

Preserving measured gains and penalizing command changes does not establish safe deployment. Camera latency is not fully identified, physics randomization covers only selected parameters, observation timing still depends on the robot implementation, and simulation motion metrics do not predict protective stops exactly. Require long paired tests, inspect GIFs and worst trajectories, and compare command/speed/acceleration metrics before considering real deployment.
