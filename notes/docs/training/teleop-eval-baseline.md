# Human teleop baseline (paper user study)

Same evaluation protocol as `async_td3_real_eval.py` — same task, same
termination rules, same juggle counter, same JSONL/JSON output schema —
but with a human moving the paddle via the mouse. So policy and human
numbers go side-by-side in the paper from a single shared protocol.

**Entrypoint:**
[`scripts/td3/extras/async_td3_real_teleop_eval.py`](../../../scripts/td3/extras/async_td3_real_teleop_eval.py)

**Run command (always identical across sessions):**

```bash
/home/pearl/miniconda3/envs/air/bin/python -m scripts.td3.extras.async_td3_real_teleop_eval --args-file configs/td3_real_world/td3_online.yaml --data-root-dir runs/teleop_user_study --eval-episodes 20
```

The script prompts for the participant id at runtime, so the launch
command stays the same for every subject — only the typed id changes.
Each session lands at `runs/teleop_user_study/<participant_id>/data_<TIMESTAMP>/`.

The sim config is pinned to `configs/real_configs/rollout_td3_config.yaml`
inside the script (matches the policy eval); whatever the args-file's
`config:` field says is overridden — you'll see a `[teleop_eval]
overriding args.config …` line on launch confirming it.

---

## What it shares with the policy eval

| Surface | Source |
|---|---|
| Task / termination | `configs/real_configs/rollout_td3_config.yaml` |
| Autonomous reset between episodes | `ResetPolicyFSM` (same as `async_td3_real_eval.py`) |
| Per-step HDF5 row format | `_build_split_episode_row` |
| Juggle counter, eval aggregate, JSON output | `helper/juggle_counter.py`, `helper/real_eval_stats.py` |

A `kept_index` row in the human eval set is directly comparable to the
policy's `kept_index` row at the same `target_episodes`.

## What it changes

* **Control mode**: `mouse` — cursor over the live `image` window drives
  the paddle (same UX as `scripts/real/teleoperate.py`).
* **No actor / replay / checkpointing** — irrelevant for a baseline.
  Episode HDF5s are still written, so trajectories are re-scorable
  offline.
* **Single cv2 window**: the phase banner (border color + header) is
  drawn directly on the `image` window so the participant has exactly
  one window to look at and one window to drag the cursor on.
* **Clean camera feed**: green/red/yellow puck / paddle / target circles
  and edge / region overlays are stripped from the camera feed (only
  the phase banner is drawn on top).

## Phase banner (border colors on the `image` window)

| Phase | Border | What's happening |
|---|---|---|
| `ROBOT RESETTING` | blue | Autonomous `ResetPolicyFSM` is driving the robot. Stand back. |
| `USER CONTROL` | green | Episode is live. Drag the cursor over the `image` window to control the paddle. Live `step / juggles / contacts` shown. |
| `EPISODE OVER` | red | Brief pause before the next reset begins. |

Handoff is **immediate** after each reset — no countdown.

## Operator keyboard shortcuts

Active throughout the run via `HumanInterruptListener` (also via
`/tmp/airhockey_human_interrupt`):

* `s` — STOP. Truncates the current episode and routes through the same
  hard-reset path as `async_td3_real` (HARD_WITH_FSM). Use this for
  e-stop recovery or to abort a bad attempt.
* `r` — RESET. Clears the STOP state when the robot is safe to resume.

## Outputs

```
runs/teleop_user_study/<participant_id>/data_<TIMESTAMP>/
    eval_summary.json          ← run_meta + aggregate + per_episode
    eval_per_episode.jsonl     ← one row per kept episode (incremental)
    episode_summaries.jsonl    ← every attempt (kept and discarded)
    reset_summaries.jsonl      ← every reset event
    run_events.jsonl           ← run_start / eval_done
    episode_hdf5/<bucket>/trajectory_data*.hdf5   ← per-step trajectories
```

`run_meta` carries `control_mode: mouse`, `operator: human`, and
`participant_id` for downstream analysis. `per_episode` rows carry
`reset_phase_reason` so a downstream reader can distinguish "puck was
ready" from "max-wait fired" resets.

## Running

The two YAML files needed are the same ones the policy eval consumes —
the `--config` (env config) and the `--args-file` (online-behavior args).
Architecture is irrelevant here, so `--train-args` is optional and
`--model-path` is ignored.

Minimum-arg launch (20 episodes, `puck_juggle_upper_half_reward`):

```bash
python -m scripts.td3.extras.async_td3_real_teleop_eval \
    --config     configs/real_configs/rollout_td3_config.yaml \
    --args-file  configs/td3_real_world/td3_online.yaml \
    --eval-episodes 20
```

Long-form launch with explicit reset / handoff timing (matches what we
used in the paper user study):

```bash
python -m scripts.td3.extras.async_td3_real_teleop_eval \
    --config     configs/real_configs/rollout_td3_config.yaml \
    --args-file  configs/td3_real_world/td3_online.yaml \
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

Both scripts produce `eval_summary.json` with the same
`aggregate.series` keys (`episode_return`, `episode_juggles`,
`episode_contacts`, `episode_task_reward`, `episode_length`) and the
same `aggregate.rates`. A side-by-side paper table is just loading both
JSONs and reading `aggregate.series.<field>`.

## Useful CLI knobs

* `--eval-episodes N` — target kept episodes (default 20).
* `--eval-max-attempts M` — cap total attempts so a few short / e-stopped
  episodes do not exhaust the run. Default 0 (unlimited).
* `--post-episode-pause-s 1.5` — red banner pause before the next reset.
* `--verbose` — restore noisy per-step debug prints from the
  training-side helpers (default `--quiet`).
