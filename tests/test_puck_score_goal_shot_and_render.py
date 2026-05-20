"""
Integration checks for ``puck_score``: fire the puck at the top-edge goal,
expect a positive velocity-scaled success reward, and optionally build a
GIF render with reward text in a right-hand sidebar.

Requires Box2D (``pip install Box2D`` or ``uv sync --extra box2d``).

The render test writes ``puck_score_render_test_output.gif`` next to this file
and prints its absolute path; use ``pytest -s`` (or ``--capture=no``) so the
print is visible.
"""
from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np
import yaml

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    imageio = None

try:
    from airhockey import AirHockeyEnv
    from airhockey.renderers.render import AirHockeyRenderer
except ImportError as e:  # pragma: no cover
    AirHockeyEnv = None  # type: ignore[misc, assignment]
    AirHockeyRenderer = None  # type: ignore[misc, assignment]
    _AIRHOCKEY_IMPORT_ERROR = e
else:
    _AIRHOCKEY_IMPORT_ERROR = None

try:
    import Box2D  # noqa: F401
except ImportError:  # pragma: no cover
    Box2D = None


def _score_yaml_path() -> Path:
    return Path(__file__).resolve().parents[2] / "air_hockey_configs" / "score.yaml"


def _load_score_cfg(**overrides):
    with _score_yaml_path().open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["air_hockey"]
    cfg = copy.deepcopy(cfg)
    cfg.update(overrides)
    return cfg


def _set_puck_base_position_velocity(env, pos_xy, vel_xy):
    """Set puck pose in base frame (matches ``spawn_puck`` coordinate maps)."""
    sim = env.simulator
    body = sim.pucks["puck_0"]
    body.position = sim.base_coord_to_box2d(pos_xy)
    body.linearVelocity = sim.base_coord_to_box2d(vel_xy)
    body.angularVelocity = 0.0
    body.awake = True


def composite_table_with_reward_sidebar(
    table_bgr: np.ndarray,
    *,
    step_reward: float,
    cum_reward: float,
    step_idx: int,
    extra_lines: list[str] | None = None,
    side_width: int = 168,
) -> np.ndarray:
    """Append a white panel on the right with reward readouts (BGR uint8)."""
    if cv2 is None:
        raise unittest.SkipTest("opencv-python not installed")
    h, w = table_bgr.shape[:2]
    panel = np.full((h, side_width, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 20
    lines = [
        "puck_score",
        f"step {step_idx}",
        f"r_step {step_reward:.4f}",
        f"r_cum {cum_reward:.4f}",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    for line in lines:
        cv2.putText(panel, line, (6, y), font, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        y += 20
    return np.concatenate([table_bgr, panel], axis=1)


class PuckScoreGoalShotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if Box2D is None or _AIRHOCKEY_IMPORT_ERROR is not None:
            raise unittest.SkipTest(
                f"needs Box2D and airhockey importable ({_AIRHOCKEY_IMPORT_ERROR})"
            )

    def _make_env(self):
        cfg = _load_score_cfg(
            seed=0,
            terminate_on_puck_pass_paddle=False,
            terminate_on_out_of_bounds=False,
            terminate_on_enemy_goal=False,
            max_timesteps=800,
            obs_type="pos",
        )
        sp = cfg.setdefault("simulator_params", {})
        if isinstance(sp, dict):
            sp["gravity"] = 0.0
        return AirHockeyEnv(cfg)

    def test_shot_toward_goal_yields_positive_score_reward(self):
        env = self._make_env()
        try:
            env.reset(seed=42)
            length = float(env.length)
            x0 = env.table_x_top + 0.42 * length
            y0 = 0.0
            _set_puck_base_position_velocity(env, (x0, y0), (-12.0, 0.0))

            cum_r = 0.0
            scored = False
            last_r = 0.0
            zero_act = np.zeros(2, dtype=np.float32)
            for t in range(500):
                _, last_r, done, trunc, info = env.step(zero_act)
                cum_r += float(last_r)
                reasons = info.get("termination_reasons") or []
                if done and "top_edge_goal" in reasons:
                    scored = True
                    break
                if trunc:
                    self.fail(f"unexpected truncation: {info.get('truncation_reasons')}")
            self.assertTrue(scored, "expected top_edge_goal before step budget")
            self.assertGreater(last_r, 0.0, "success step should have positive base reward")
            scale = float(env.top_edge_goal_velocity_reward_scale)
            self.assertLess(last_r, scale * 20.0)
        finally:
            if hasattr(env, "close"):
                env.close()

    def test_render_gif_shows_goal_and_reward_sidebar(self):
        if cv2 is None or imageio is None or AirHockeyRenderer is None:
            self.skipTest("cv2 / imageio / renderer not available")
        env = self._make_env()
        try:
            env.reset(seed=0)
            _set_puck_base_position_velocity(
                env,
                (env.table_x_top + 0.42 * float(env.length), 0.0),
                (-10.0, 0.0),
            )
            renderer = AirHockeyRenderer(env, orientation="vertical")
            zero_act = np.zeros(2, dtype=np.float32)
            cum_r = 0.0
            scored = False
            frames_rgb = []
            for t in range(500):
                _, step_r, done, trunc, info = env.step(zero_act)
                cum_r += float(step_r)
                table = renderer.get_frame()
                self.assertEqual(len(table.shape), 3)
                reasons = info.get("termination_reasons") or []
                composed = composite_table_with_reward_sidebar(
                    table,
                    step_reward=float(step_r),
                    cum_reward=cum_r,
                    step_idx=t + 1,
                    extra_lines=[f"goal={'yes' if 'top_edge_goal' in reasons else 'no'}"],
                )
                # Keep GIFs lightweight and consistent for qualitative checks.
                h, w = composed.shape[:2]
                out_w = 160
                out_h = max(1, int(round(h * out_w / w)))
                composed = cv2.resize(
                    composed, (out_w, out_h), interpolation=cv2.INTER_AREA
                )
                frames_rgb.append(cv2.cvtColor(composed, cv2.COLOR_BGR2RGB))
                if done and "top_edge_goal" in reasons:
                    scored = True
                    break
                if trunc:
                    self.fail(f"unexpected truncation: {info.get('truncation_reasons')}")

            self.assertTrue(scored, "expected top_edge_goal before step budget")
            self.assertGreater(len(frames_rgb), 1, "expected a multi-frame trajectory")
            out_path = (
                Path(__file__).resolve().parent / "puck_score_render_test_output.gif"
            )
            imageio.mimsave(str(out_path), frames_rgb, duration=50)
            print(f"\nRendered composite GIF saved to: {out_path.resolve()}\n")
        finally:
            if hasattr(env, "close"):
                env.close()


if __name__ == "__main__":
    unittest.main()
