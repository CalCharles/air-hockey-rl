"""Action latency for a live env: every action reaches the simulator ``k`` control steps late.

Models the real robot's slower paddle response (servoL lookahead + the robot's TCP-target reference),
measured by replaying real-robot actions in the sim (``scripts/analysis/replay_real_actions.py``):
the real paddle trails the no-observation-delay sim by 2 control steps. The first ``k`` steps of an
episode apply zero actions. The simulator itself is not modified.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def install_action_latency(env, steps: int):
    """Delay ``env.step`` actions by ``steps`` control steps, in place. Returns ``env``."""
    if steps <= 0:
        return env
    env_reset, env_step = env.reset, env.step
    queue = deque()

    def reset(*args, **kwargs):
        queue.clear()
        queue.extend(np.zeros(env.action_space.shape, dtype=np.float32) for _ in range(steps))
        return env_reset(*args, **kwargs)

    def step(action):
        queue.append(np.asarray(action, dtype=np.float32).copy())
        return env_step(queue.popleft())

    env.reset, env.step = reset, step
    return env
