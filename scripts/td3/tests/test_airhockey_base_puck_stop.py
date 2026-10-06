import unittest
from types import SimpleNamespace

from airhockey.airhockey_base import AirHockeyBaseEnv


class _EnvShell(AirHockeyBaseEnv):
    """Concrete stand-in: object.__new__ refuses the abstract base class."""


_EnvShell.__abstractmethods__ = frozenset()


def _stationary_history(n: int, x: float = 0.2, y: float = 0.1, occluded: float = 0.0):
    return [(x, y, occluded) for _ in range(n)]


class PuckStopLowMotionFallbackTests(unittest.TestCase):
    def _make_env_shell(self, simulator_name: str, simulator_puck_history):
        # Build a minimal shell object without running full env initialization.
        env = object.__new__(_EnvShell)
        env.puck_low_motion_window_clean = 10
        env.puck_low_motion_window_occluded = 20
        env.puck_low_motion_radius_m = 0.03
        env.simulator_name = simulator_name
        env.simulator = SimpleNamespace(puck_history=simulator_puck_history)
        return env

    def test_real_mode_uses_simulator_history_fallback(self):
        env = self._make_env_shell(
            simulator_name="real",
            simulator_puck_history=_stationary_history(10),
        )
        state_info = {
            "pucks": [
                {
                    "history": _stationary_history(5),
                }
            ]
        }

        low_motion_cluster, active_window = env._puck_low_motion_cluster_window(state_info)

        self.assertTrue(low_motion_cluster)
        self.assertEqual(active_window, 10)

    def test_non_real_mode_does_not_use_simulator_history_fallback(self):
        env = self._make_env_shell(
            simulator_name="box2d",
            simulator_puck_history=_stationary_history(10),
        )
        state_info = {
            "pucks": [
                {
                    "history": _stationary_history(5),
                }
            ]
        }

        low_motion_cluster, active_window = env._puck_low_motion_cluster_window(state_info)

        self.assertFalse(low_motion_cluster)
        self.assertEqual(active_window, 0)


class PuckStopTopBlindSpotTests(unittest.TestCase):
    """Real robot: a puck hidden in the camera's top blind spot is held at its
    last seen position (occluded=1) and must not read as a stopped puck."""

    def _make_env_shell(self, simulator_puck_history):
        env = object.__new__(_EnvShell)
        env.puck_low_motion_window_clean = 20
        env.puck_low_motion_window_occluded = 60  # real-robot value
        env.puck_low_motion_radius_m = 0.03
        env.simulator_name = "real"
        env.simulator = SimpleNamespace(puck_history=simulator_puck_history)
        return env

    @staticmethod
    def _rising_then_hidden(hidden_steps: int):
        # Puck travels up the table (x decreasing) and disappears at x = -0.85.
        rising = [(0.5 - 0.05 * i, 0.1, 0.0) for i in range(28)]
        last_x = rising[-1][0]
        return rising + _stationary_history(hidden_steps, x=last_x, occluded=1.0)

    def _check(self, history):
        state_info = {"pucks": [{"history": history[-5:]}]}
        return self._make_env_shell(history)._puck_low_motion_cluster_window(state_info)

    def test_puck_hidden_in_blind_spot_does_not_end_episode(self):
        low_motion_cluster, active_window = self._check(self._rising_then_hidden(25))
        self.assertFalse(low_motion_cluster)
        self.assertEqual(active_window, 60)

    def test_puck_hidden_for_whole_long_window_still_ends_episode(self):
        low_motion_cluster, active_window = self._check(self._rising_then_hidden(60))
        self.assertTrue(low_motion_cluster)
        self.assertEqual(active_window, 60)

    def test_visible_stationary_puck_still_ends_after_clean_window(self):
        history = [(0.5 - 0.05 * i, 0.1, 0.0) for i in range(28)] + _stationary_history(20)
        low_motion_cluster, active_window = self._check(history)
        self.assertTrue(low_motion_cluster)
        self.assertEqual(active_window, 20)


if __name__ == "__main__":
    unittest.main()
