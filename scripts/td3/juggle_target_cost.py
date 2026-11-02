"""Reward-only cost on changes of the cached, filtered PID position target.

This is an experimental training wrapper, not a controller or safety limit.
It reads no sensors and takes no additional simulator steps. The first target
after reset has no previous commanded target and incurs zero cost. Evaluation
must continue to use the ordinary environment and report its original reward.
"""
from __future__ import annotations

import math
import numpy as np


class TargetStepCost:
    """Forward the environment unchanged, subtracting c*||target delta||^4.

    Coefficient units are reward/m^4. A constant coefficient keeps replay-buffer
    rewards stationary. This is neither motor energy nor UR5 stop prediction.
    Training targets are privileged reward telemetry only, never observations.
    """

    def __init__(self, env, coefficient):
        coefficient = float(coefficient)
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError('Target cost coefficient must be finite and nonnegative')
        self.env = env
        self.coefficient = coefficient
        self.previous_target = None
        self.cost_return = 0.0
        self.shaped_return = 0.0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        self.previous_target = None
        self.cost_return = self.shaped_return = 0.0
        return result

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        target = np.asarray(self.env.simulator.last_target_position, dtype=np.float64)
        if target.shape != (2,) or not np.all(np.isfinite(target)):
            raise ValueError('Expected a finite cached 2D PID target')
        delta = target - self.previous_target if self.previous_target is not None else np.zeros(2)
        distance = float(np.linalg.norm(delta))
        fourth_power = distance ** 4
        cost = self.coefficient * fourth_power
        self.previous_target = target.copy()
        shaped_reward = float(reward) - cost
        self.cost_return += cost
        self.shaped_return += shaped_reward
        info = dict(info)
        info.update(target_step_m=distance, target_step_fourth_power_m4=fourth_power,
                    target_step_cost=cost, original_reward=float(reward))
        if terminated or truncated:
            # Preserve the underlying task's bookkeeping and expose both returns.
            info['original_episode_return'] = info['episode_return']
            info['episode_return'] = self.shaped_return
            info['target_step_cost_return'] = self.cost_return
        return observation, shaped_reward, terminated, truncated, info

    def close(self):
        return self.env.close()
