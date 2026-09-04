"""Per-task truncation budgets for the benchmark sim configs.

Each task gets the horizon its reward needs: reach is a short point-to-point
move, the velocity / touch tasks need room to line a stroke up, and juggling
needs a long enough episode to chain several hits.
"""

import unittest
from pathlib import Path

import yaml

from airhockey import AirHockeyEnv


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH_DIR = REPO_ROOT / "configs" / "new_juggle" / "tasks"

EXPECTED_MAX_TIMESTEPS = {
    "reach": 50,
    "reach_vel": 100,
    "puck_vel": 100,
    "touch": 100,
    "juggle": 250,
}


class TaskTimestepBudgetTests(unittest.TestCase):
    def test_configs_declare_the_expected_budget(self):
        for task, expected in EXPECTED_MAX_TIMESTEPS.items():
            for variant in ("sysid", "dr"):
                config_path = BENCH_DIR / f"sim_{variant}_{task}.yaml"
                with self.subTest(config=config_path.name):
                    with config_path.open("r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)["air_hockey"]
                    self.assertEqual(cfg["max_timesteps"], expected)

    def test_episodes_truncate_at_the_budget(self):
        # An idle paddle never finishes reach / reach_vel, so those episodes run
        # the full budget and truncate.
        for task in ("reach", "reach_vel"):
            with self.subTest(task=task):
                config_path = BENCH_DIR / f"sim_sysid_{task}.yaml"
                with config_path.open("r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)["air_hockey"]
                cfg["seed"] = 3
                env = AirHockeyEnv(cfg)
                try:
                    import numpy as np

                    env.reset()
                    # Park the goal out of reach so the episode cannot end early.
                    steps = 0
                    while True:
                        env.goal_pos = np.array([10.0, 10.0])
                        _, _, done, truncated, info = env.step(
                            np.zeros(2, dtype=np.float32)
                        )
                        steps += 1
                        if done or truncated:
                            break
                    self.assertTrue(truncated)
                    self.assertIn("max_timesteps_exceeded", info["truncation_reasons"])
                    # has_finished truncates once current_timestep exceeds the
                    # budget, and current_timestep is incremented after that
                    # check, so an episode runs max_timesteps + 2 steps.
                    self.assertEqual(steps, cfg["max_timesteps"] + 2)
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
