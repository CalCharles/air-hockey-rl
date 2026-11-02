"""Reward arithmetic, episode bookkeeping and reset isolation."""
from types import SimpleNamespace
import numpy as np
import unittest
from scripts.td3.juggle_target_cost import TargetStepCost


class ScriptedEnv:
    def __init__(self):
        self.simulator = SimpleNamespace(last_target_position=np.zeros(2))
        self.actions = []

    def reset(self, **kwargs):
        self.index = 0
        return np.array([7.,8.]), {'seed': kwargs.get('seed')}

    def step(self, action):
        self.actions.append(action.copy())
        # Mutate the cached array in place to detect accidental target aliasing.
        self.simulator.last_target_position[:] = [(0,0),(.3,.4),(.3,.4)][self.index]
        self.index += 1
        done = self.index == 3
        return np.array([self.index]), 2., done, False, ({'episode_return':6.} if done else {})


class TargetStepCostTests(unittest.TestCase):
    def test_reward_and_episode_bookkeeping(self):
        base = ScriptedEnv(); env = TargetStepCost(base, 4.)
        env.reset(seed=3)
        action = np.array([.2,-.1])
        first = env.step(action)
        second = env.step(action)
        third = env.step(action)
        assert first[1] == 2. and first[-1]['target_step_cost'] == 0.
        assert second[1] == 1.75 and second[-1]['target_step_cost'] == .25
        assert third[-1]['original_episode_return'] == 6.
        assert third[-1]['episode_return'] == 5.75
        assert third[-1]['target_step_cost_return'] == .25
        assert all(np.array_equal(a, action) for a in base.actions)
        obs, info = env.reset(seed=9)
        assert np.array_equal(obs, [7.,8.]) and info == {'seed':9}
        assert env.step(action)[-1]['target_step_cost'] == 0.
        assert env.cost_return == 0. and env.shaped_return == 2.


    def test_reject_invalid_coefficient(self):
        for coefficient in [-1.,float('nan'),float('inf')]:
            with unittest.TestCase().assertRaises(ValueError):
                TargetStepCost(ScriptedEnv(),coefficient)


if __name__ == '__main__':
    unittest.main()
