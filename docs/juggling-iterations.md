# Clean juggling iterations

Branch: `codex/sustained-juggling`. The prior sensor-robust trials did not solve sustained juggling. Their “nominal” evaluation referred to nominal physics while retaining simulated sensor corruption; it was not a clean sensor benchmark.

The new configurations explicitly disable puck noise, occlusions, observation/action delay, camera-frame lag, delay interpolation, and physics randomization. They retain the sysid-v2 measured PID gains, hist2 command smoothing, real workspace, action scaling, and speed/force limits. Source actor clean baseline: 137.2 mean steps on eight starts, cap 10,000; no capped successes. Artifacts: `runs/td3/juggle_iterations/iteration_00/source_clean/`.

## Hypotheses and schedule

All implementation and experiment decisions use the main session, without subagents. Three independent Slurm jobs run in parallel per iteration. Iteration 2 waits for completed validation in iteration 1.

1. **Iteration 1, 300k steps per trial, seed 0:** compare the existing upper-half reward, that reward plus a 0.25 survival bonus and a 5-point actual-puck-loss cost, and the same shaped reward with a 50-step observation/action memory. All use discount 0.99. Memory initializes from the same plain actor with zero weights on the new latent input columns, so initial actions match rather than replacing a working policy with a random one. Critics are fresh and warm up before actor updates.
2. **Iteration 2, chosen from validation:** continue the winning final checkpoint to 1M total steps, independently train its recipe with seed 1, and test discount 0.995. This is an adaptive choice of architecture/reward, not a claim that the first hypothesis will work.
3. **Independent final tests:** evaluate the selected model on 32 fresh starts for up to 20,000 steps in clean simulation, then separately against the earlier sensor/physics robustness configuration. These tests do not authorize hardware operation or imply infinite survival.

Model initialization, global randomization, and action-space exploration are now seeded. Earlier trainers only seeded resets and selected helper RNGs, so previous `seed` labels did not make whole runs reproducible. Resume preserves existing checkpoint RNG semantics; it is not an exact restoration of the simulator state.

## Measuring actual juggling

The evaluator uses the existing direction-flip juggle counter. Selection ranks genuine capped juggling first, then duration with repeated juggles, then observed survival. Long stationary contact or pinning the puck cannot count as successful juggling. This remains a heuristic quality check; inspect the GIFs and saved trajectories.

Each evaluation saves start and end/failure GIFs, raw policy-observation/action trajectories, sensor/control configuration, motion diagnostics, contacts, and distinct juggles. The clips are width 160, vertical, 20 fps. Final evaluation uses bounded GIF buffers even for long successful episodes.

## Running and monitoring

```bash
.venv/bin/python -m scripts.td3.juggle_iterations --prepare
.venv/bin/python -m scripts.td3.juggle_iterations --launch
cat runs/td3/juggle_iterations/status.json
tail -n 20 runs/td3/juggle_iterations/iteration_01/memory.log
```

The supervisor uses the three existing allocations 106217/106218/106219; it requests no new allocation. Root `status.json` and `supervisor.log` track decisions and errors. Each trial has a status JSON, training log, checkpoint GIFs, and final validation. `iteration_01.md` and `iteration_02.md` under the run root contain comparison tables with GIF links. These artifact reports update as trials finish.

After a completed iteration, the supervisor writes a new immutable experiment note and commits that note on this branch. It commits only its own new note and an otherwise-clean experiment index; it skips report commits if the active branch has changed. Training code/config changes are committed separately after testing. No weights or media are added to Git.

The existing unrelated untracked `configs/new_juggle/sysid_best_params_hist2_puck_score.yaml` is outside this work and must remain untouched.
