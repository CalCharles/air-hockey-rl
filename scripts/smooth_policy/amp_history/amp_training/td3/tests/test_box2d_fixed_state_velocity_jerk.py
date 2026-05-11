import unittest
from pathlib import Path

import numpy as np
import yaml

from airhockey.sims.airhockey_box2d import AirHockeyBox2D


class Box2DFixedStateVelocityJerkTests(unittest.TestCase):
    @staticmethod
    def _default_simulator_params() -> dict:
        repo_root = Path(__file__).resolve().parents[6]
        config_path = (
            repo_root
            / "scripts"
            / "smooth_policy"
            / "amp_history"
            / "configs"
            / "new_juggle"
            / "sysid_best_params.yaml"
        )
        with config_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        simulator_params = dict(loaded["air_hockey"]["simulator_params"])
        simulator_params["seed"] = int(loaded["air_hockey"].get("seed", 0))
        return simulator_params

    def _build_sim(self, *, enable_mask: bool) -> AirHockeyBox2D:
        sim_params = self._default_simulator_params()
        sim_params.update(
            {
                "enable_fixed_state_velocity_jerk": enable_mask,
                "fixed_state_paddle_velocity": (0.0, 0.0),
                "fixed_state_paddle_jerk": (0.0, 0.0),
                "fixed_state_puck_velocity": (0.0, 0.0),
                "mask_puck_velocity": True,
                "enable_random_occlusions": False,
                "puck_noise": False,
            }
        )
        return AirHockeyBox2D(
            **sim_params,
        )

    def _populate_with_motion(self, sim: AirHockeyBox2D) -> None:
        sim.spawn_paddle(pos=(-0.6, 0.1), vel=(0.45, -0.2), name="paddle_ego")
        sim.spawn_puck(pos=(-0.2, 0.0), vel=(0.3, 0.25), name="puck0")
        # Stored internally in Box2D frame; state export converts it back to base frame.
        sim.paddles["paddle_ego_jerk"] = sim.base_coord_to_box2d((0.35, -0.4))

    def test_state_retrieval_can_mask_velocity_and_jerk(self):
        unmasked = self._build_sim(enable_mask=False)
        self._populate_with_motion(unmasked)
        unmasked_state = unmasked.get_current_state()
        unmasked_internal_velocity = np.array(
            [
                unmasked.paddles["paddle_ego"].linearVelocity[0],
                unmasked.paddles["paddle_ego"].linearVelocity[1],
            ],
            dtype=float,
        )

        self.assertGreater(float(np.linalg.norm(unmasked_internal_velocity)), 1e-6)
        self.assertGreater(
            float(np.linalg.norm(np.asarray(unmasked_state["paddles"]["paddle_ego"]["velocity"], dtype=float))),
            1e-6,
        )
        self.assertGreater(
            float(np.linalg.norm(np.asarray(unmasked_state["paddles"]["paddle_ego"]["jerk"], dtype=float))),
            1e-6,
        )
        self.assertGreater(
            float(np.linalg.norm(np.asarray(unmasked_state["pucks"][0]["velocity"], dtype=float))),
            1e-6,
        )

        masked = self._build_sim(enable_mask=True)
        self._populate_with_motion(masked)
        masked_state = masked.get_current_state()
        masked_internal_velocity = np.array(
            [
                masked.paddles["paddle_ego"].linearVelocity[0],
                masked.paddles["paddle_ego"].linearVelocity[1],
            ],
            dtype=float,
        )

        np.testing.assert_allclose(
            np.asarray(masked_state["paddles"]["paddle_ego"]["velocity"], dtype=float),
            np.zeros(2, dtype=float),
            atol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(masked_state["paddles"]["paddle_ego"]["jerk"], dtype=float),
            np.zeros(2, dtype=float),
            atol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(masked_state["pucks"][0]["velocity"], dtype=float),
            np.zeros(2, dtype=float),
            atol=1e-8,
        )
        self.assertGreater(float(np.linalg.norm(masked_internal_velocity)), 1e-6)


if __name__ == "__main__":
    unittest.main()
