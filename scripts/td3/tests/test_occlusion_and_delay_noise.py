"""Perception-noise knobs: geometric occlusion run lengths, delay interpolation.

Occlusion run lengths follow a geometric scheme (each extra occluded frame
``random_occlusion_decay`` times as likely as the previous one) instead of a
hardcoded weight list, and the puck delay interpolation jitters within
[0.9, 1.1] rather than [0.75, 1.25].
"""

import unittest
from pathlib import Path

import numpy as np
import yaml

from airhockey import AirHockeyEnv


REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL = REPO_ROOT / "configs" / "new_juggle" / "sysid_best_params_hist2.yaml"


def _make_env(**sim_overrides):
    with CANONICAL.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["air_hockey"]
    cfg["seed"] = 9
    cfg["simulator_params"].update(sim_overrides)
    return AirHockeyEnv(cfg)


class OcclusionRunLengthTests(unittest.TestCase):
    def test_weights_halve_with_each_extra_frame(self):
        env = _make_env()
        try:
            weights = np.asarray(env.simulator.random_occlusion_length_weights, dtype=float)
            self.assertEqual(weights.size, 7)
            np.testing.assert_allclose(weights, 0.5 ** np.arange(7))
            # Each length is half as likely as the one before it.
            ratios = weights[1:] / weights[:-1]
            np.testing.assert_allclose(ratios, np.full(6, 0.5))
        finally:
            env.close()

    def test_decay_and_max_run_are_configurable(self):
        env = _make_env(random_occlusion_decay=0.25, random_occlusion_max_run=4)
        try:
            weights = np.asarray(env.simulator.random_occlusion_length_weights, dtype=float)
            np.testing.assert_allclose(weights, 0.25 ** np.arange(4))
            self.assertEqual(env.simulator._occlusion_max_run, 4)
        finally:
            env.close()

    def test_explicit_weight_list_still_overrides(self):
        env = _make_env(random_occlusion_length_weights=[3, 2, 1])
        try:
            np.testing.assert_allclose(
                env.simulator.random_occlusion_length_weights, [3.0, 2.0, 1.0]
            )
            self.assertEqual(env.simulator._occlusion_max_run, 3)
        finally:
            env.close()

    def test_sampled_runs_respect_the_cap_and_favor_short_runs(self):
        env = _make_env(random_occlusion_rate=1.0)
        try:
            sim = env.simulator
            env.reset()
            runs = [sim._sample_occlusion_run_length() for _ in range(4000)]
            self.assertGreaterEqual(min(runs), 1)
            self.assertLessEqual(max(runs), sim._occlusion_max_run)
            share_of_ones = np.mean(np.asarray(runs) == 1)
            self.assertGreater(share_of_ones, 0.4)
            self.assertLess(share_of_ones, 0.6)
        finally:
            env.close()


class PuckDelayInterpolationTests(unittest.TestCase):
    def test_default_range_is_narrowed(self):
        env = _make_env()
        try:
            self.assertAlmostEqual(env.simulator.puck_delay_interpolation_min, 0.9)
            self.assertAlmostEqual(env.simulator.puck_delay_interpolation_max, 1.1)
        finally:
            env.close()

    def test_reset_clears_the_previous_episode_position(self):
        """Otherwise the first frame of an episode extrapolates from the last one."""
        env = _make_env(puck_noise=False, enable_random_occlusions=False)
        try:
            spawn_x = []
            for _ in range(200):
                env.reset()
                # Run a few steps so the previous-position cache is far from the
                # next episode's spawn.
                for _ in range(5):
                    env.step(np.zeros(2, dtype=np.float32))
                env.reset()
                spawn_x.append(env.current_state["pucks"][0]["position"][0])
            x_low = env.table_x_top + env.puck_radius
            x_high = (
                env.table_x_top
                + env.puck_spawn_top_fraction * env.length
                - env.puck_radius
            )
            self.assertGreaterEqual(min(spawn_x), x_low - 1e-9)
            self.assertLessEqual(max(spawn_x), x_high + 1e-9)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
