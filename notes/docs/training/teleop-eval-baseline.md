# Human teleop baseline (paper user study)

A "user study / human baseline" entrypoint that runs the **same evaluation
protocol** as `async_td3_real_eval.py` but with a human moving the paddle
via the mouse. The point is to make policy and human numbers directly
comparable — same task config, same termination rules, same juggle counter,
same JSONL/JSON output layout — so a paper table that shows policy vs.
human rests on a single shared protocol.

Entrypoint:
[`scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_teleop_eval.py`](../../../scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_teleop_eval.py)

---

## What it shares with the policy eval

| Surface | Source | Mirrored in teleop eval |
|---------|--------|-------------------------|
| Task / reward | `args.config` (e.g. `configs/real_configs/rollout_td3_config_hist4.yaml`) | yes — same YAML |
| Termination conditions | `terminate_on_puck_*`, `max_timesteps` from the config | yes — env decides episode end |
| Per-step HDF5 row format | `_build_split_episode_row` in `helper/real_td3_runtime.py` | yes — same helper, same fields |
| Juggle counter | `count_juggles_from_rows` (`helper/juggle_counter.py`) | yes |
| Eval aggregate stats | `compute_eval_aggregate` (`helper/real_eval_stats.py`) | yes |
| `eval_per_episode.jsonl` / `eval_summary.json` | `helper/real_eval_stats.py` | yes — identical schemas |
| `episode_summaries.jsonl` / `run_events.jsonl` | `helper/run_event_log.py` | yes |
| Run-data dir layout | `_setup_run_data_dir` | yes |

A `kept_index`-style row in the human eval set is directly comparable to
the policy's `kept_index` row produced by the same target_episodes count.

## What it changes

* **Control mode**: forced to `mouse` (same UX as
  [`scripts/real/teleoperate.py`](../../../scripts/real/teleoperate.py)).
  The user's cursor over the existing `image` window drives the paddle.
* **No actor**: there is nothing to load. `--model-path` is ignored;
  exploration / checkpointing knobs are forced off via `_force_teleop_mode`.
* **Reset path**: the autonomous reset FSM only honors `env.step(action)` in
  non-mouse control modes, so it is *not* used here. Instead, the script
  watches the puck position and auto-advances once the user has pushed the
  puck back into the upper half (`reset_puck_upper_half_frames` consecutive
  non-occluded frames past the table midpoint).
* **No motion-reward / no GIF / no camera-video / no replay**: irrelevant
  for a baseline. Episode HDF5s are still written so trajectories are
  re-enactable and can be re-scored offline.

## Phase banner

A second cv2 window (`teleop_status`, default 720x420) is opened by the
script and shows the current phase with a thick colored border so the user
can read the mode from across the room:

| Phase | Border | Meaning |
|-------|--------|---------|
| `RESET PHASE` | blue | Push the puck back into the upper half. Auto-advances once the puck is observed in the upper half for N consecutive non-occluded frames. |
| `GET READY` | yellow | 3-2-1 countdown. Place the cursor on the paddle's current position so user control begins without a jerk. |
| `USER CONTROL` | green | Episode is running. Live `step / juggles / contacts / return` are shown. Episode ends on env terminations / truncations. |
| `EPISODE OVER` | red | Brief pause showing `end_reason` before the next reset begins. |

The user controls the paddle in the existing `image` window (mouse follows
cursor, same as `scripts/real/teleoperate.py`); the status window is
peripheral.

Keyboard shortcuts (active during reset and user control phases via
`NonBlockingConsole`):

* `space` / `s` — skip the remainder of the reset wait (advance to handoff now)
* `q` — abort the current user-control attempt (artifact still saved as a
  discarded episode, then the next reset begins). Mirrors `q` in the
  existing teleop script.
* `x` — exit the run. Partial summary is preserved on disk.

## Outputs

Under the standard run-data dir (created by `_setup_run_data_dir`):

```
<data_root>/no_model/data_<TIMESTAMP>/
    eval_summary.json          ← run_meta + aggregate + per_episode (the eval set)
    eval_per_episode.jsonl     ← one row per kept episode, written incrementally
    episode_summaries.jsonl    ← one row per attempt (kept and discarded)
    run_events.jsonl           ← run_start / eval_done
    episode_hdf5/<bucket>/trajectory_data*.hdf5   ← per-step trajectories
```

`run_meta` carries `control_mode: mouse` and `operator: human`, and
`per_episode` rows carry `reset_phase_reason` so a downstream reader can
distinguish "puck was ready" from "max-wait fired" resets.

## Running

The two YAML files needed are the same ones the policy eval consumes —
the `--config` (env config) and the `--args-file` (online-behavior args).
Architecture is irrelevant here, so `--train-args` is optional and
`--model-path` is ignored.

Minimum-arg launch (20 episodes, `puck_juggle_upper_half_reward`):

```bash
python -m scripts.smooth_policy.amp_history.amp_training.td3.extras.async_td3_real_teleop_eval \
    --config     configs/real_configs/rollout_td3_config_hist4.yaml \
    --args-file  scripts/smooth_policy/amp_history/configs/td3_real_world/td3_online.yaml \
    --eval-episodes 20
```

Long-form launch with explicit reset / handoff timing (matches what we
used in the paper user study):

```bash
python -m scripts.smooth_policy.amp_history.amp_training.td3.extras.async_td3_real_teleop_eval \
    --config     configs/real_configs/rollout_td3_config_hist4.yaml \
    --args-file  scripts/smooth_policy/amp_history/configs/td3_real_world/td3_online.yaml \
    --eval-episodes              20 \
    --eval-max-attempts          40 \
    --reset-min-wait-s           2.5 \
    --reset-max-wait-s           30.0 \
    --reset-puck-upper-half-frames 20 \
    --reset-upper-half-margin-m  0.05 \
    --handoff-countdown-s        3.0 \
    --post-episode-pause-s       1.5 \
    --data-root-dir              data/teleop_eval/<participant_id>
```

Notes:
* Per-participant `--data-root-dir` keeps each user-study session in its
  own folder (otherwise `no_model/data_<TIMESTAMP>/` collides on a busy
  rig).
* Set `--eval-max-attempts` slightly higher than `--eval-episodes` so a
  few short / discarded episodes do not exhaust the cap.
* For a lower-stakes warm-up before the timed run, override
  `--eval-episodes 5` and `--reset-max-wait-s 60`.

## Comparing human vs. policy

Both scripts produce `eval_summary.json` with the same `aggregate.series`
keys (`episode_return`, `episode_juggles`, `episode_contacts`,
`episode_task_reward`, `episode_length`) and the same `aggregate.rates`
(`episode_juggle_success`, `episode_success`). A side-by-side table for
the paper is just a matter of loading both JSONs and reading
`aggregate.series.<field>`.

## Why we cannot reuse the policy eval's reset FSM

`ResetPolicyFSM` drives the robot via `env.step(action)`. In
`control_mode='mouse'`, the simulator (`airhockey/sims/air_hockey_real.py`,
the `if self.control_mode in ["mouse", "mimic"]:` branch) ignores the
action and reads the user's cursor instead — so an FSM action would never
reach the controller. The teleop window's `camera_callback` subprocess
also continuously overwrites the shared mouse-position field at camera FPS,
so even if we wrote a synthetic cursor target each step from the main
process, the next camera tick would clobber it. Switching control modes
mid-run is also not supported (camera capture is owned by different
processes for `mouse` vs. `RL`). The visual reset phase + auto-detect
("puck back in upper half") is the simplest mirror of the FSM intent that
keeps the rest of the protocol intact.
