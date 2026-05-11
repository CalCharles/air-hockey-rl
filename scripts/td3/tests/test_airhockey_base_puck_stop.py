import unittest
from types import SimpleNamespace

from airhockey.airhockey_base import AirHockeyBaseEnv


def _stationary_history(n: int, x: float = 0.2, y: float = 0.1, occluded: float = 0.0):
    return [(x, y, occluded) for _ in range(n)]


class PuckStopLowMotionFallbackTests(unittest.TestCase):
    def _make_env_shell(self, simulator_name: str, simulator_puck_history):
        # Build a minimal shell object without running full env initialization.
        env = object.__new__(AirHockeyBaseEnv)
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


if __name__ == "__main__":
    unittest.main()
