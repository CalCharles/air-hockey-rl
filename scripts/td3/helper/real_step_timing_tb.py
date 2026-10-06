"""Per-step control-loop timing -> TensorBoard, for the real-robot eval.

Called once per episode *after* it ends (``async_td3_real_eval.run_eval``), from
the rows the runner already recorded, so nothing is added to the 20 Hz control
loop. Every attempt is logged, including the short / e-stopped episodes whose
HDF5 the eval later discards.

Each episode is its own TensorBoard run (``<run_data_dir>/step_timing_tb/ep<NNN>``)
writing the same tags, so TensorBoard overlays one curve per episode on a single
chart per tag, with the episode step on the x axis:

  step_timing/step_period_ms   wall-clock time between the starts of the previous
                               and this control step (target: block_time ~ 49-50 ms).
                               Not logged for an episode's first step, whose
                               predecessor is the end of the reset.
  step_timing/compute_ms       time the loop spent working before it slept: policy
                               forward pass, previous step's bookkeeping, telemetry,
                               camera read + puck detection. Above 50 ms the loop
                               overruns and the step period stretches.
  step_timing/sleep_ms         time slept to pad the step out to block_time
  step_timing/policy_inference_ms
                               actor forward pass only; logged when the eval runs
                               with --enable-latency-profiling

The ``timing`` row layout comes from ``real_td3_runtime._build_split_episode_row``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Columns of the per-step ``timing`` row (see _build_split_episode_row).
_STEP_START_S = 1
_SLEEP_BEFORE_STEP_S = 6
_LOOP_RUNTIME_BEFORE_SLEEP_S = 7

TB_SUBDIR = "step_timing_tb"


def log_episode_step_timing(run_data_dir, episode_index: int, rows, policy_inference_ms=None) -> None:
    if not rows:
        return
    timing = np.stack([np.asarray(r["timing"], dtype=np.float64).reshape(-1) for r in rows])
    sleep_ms = timing[:, _SLEEP_BEFORE_STEP_S] * 1000.0
    compute_ms = timing[:, _LOOP_RUNTIME_BEFORE_SLEEP_S] * 1000.0
    # Missing env timing is recorded as -1 s.
    valid = (timing[:, _SLEEP_BEFORE_STEP_S] >= 0) & (timing[:, _LOOP_RUNTIME_BEFORE_SLEEP_S] >= 0)
    step_start = timing[:, _STEP_START_S]
    inference = list(policy_inference_ms or [])
    if len(inference) != len(rows):
        inference = []

    writer = SummaryWriter(log_dir=str(Path(run_data_dir) / TB_SUBDIR / f"ep{int(episode_index):03d}"))
    try:
        for step in range(len(rows)):
            if step > 0 and step_start[step] > 0 and step_start[step - 1] > 0:
                writer.add_scalar(
                    "step_timing/step_period_ms", (step_start[step] - step_start[step - 1]) * 1000.0, step
                )
            if valid[step]:
                writer.add_scalar("step_timing/compute_ms", compute_ms[step], step)
                writer.add_scalar("step_timing/sleep_ms", sleep_ms[step], step)
            if inference:
                writer.add_scalar("step_timing/policy_inference_ms", float(inference[step]), step)
    finally:
        writer.close()
