"""Training-only reward shaping for natural juggling strikes (the simulator is not modified).

The sustained-juggling policies of ``configs/ceiling/final_privcritic_pretanh.yaml`` learned to
park the paddle on the top edge of its workspace and dribble the puck (85-97 % of contacts within
2 cm of the edge, ~0 m/s forward paddle speed at contact), where the original hist4 policies strike
the puck 9-12 cm below the edge at ~0.9 m/s. These terms, read from the true simulator state
(training only, like the true-state critic), push toward the latter:

    edge   every step:      -edge_coef * clip((edge_margin - d) / edge_margin, 0, 1)
    hit    contact steps:   +hit_coef  * clip(v_fwd / hit_vref, 0, 1)
    weak   contact steps:   -weak_coef if v_fwd < weak_vmin
    edgehit contact steps:  -edge_hit_coef * clip((edge_hit_margin - d) / edge_hit_margin, 0, 1)
    pout   contact steps:   +pout_coef * clip((|v_puck| - pout_min) / (pout_ref - pout_min), 0, 1)
    term   contact steps:   end the episode (terminated) if d < edge_term_margin
    cock   contact steps:   +cock_coef * clip(cock / cock_ref, 0, 1)
    fast   contact steps:   -fast_coef * clip((|v_puck| - fast_min) / fast_ref, 0, 1)

d is the paddle's distance to the top edge of its workspace (``x_min_lim``, robot frame) and
v_fwd its forward speed (toward that edge) over the control step, from the true paddle position.
|v_puck| is the true puck speed at the end of the contact step. cock is the backswing before the hit, as in
``scripts/ceiling/hit_motion.py``: the largest backward travel in the last ``cock_window`` steps (max x - first x). A contact step is one with
``info["paddle_puck_collision_count"] > 0``. Evaluation always uses the unshaped task reward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MotionShaping:
    edge_coef: float = 0.0
    edge_margin: float = 0.05
    hit_coef: float = 0.0
    hit_vref: float = 1.0
    weak_coef: float = 0.0
    weak_vmin: float = 0.3
    edge_hit_coef: float = 0.0
    edge_hit_margin: float = 0.08
    pout_coef: float = 0.0
    pout_min: float = 1.5
    pout_ref: float = 3.5
    edge_term_margin: float = 0.0
    cock_coef: float = 0.0
    cock_ref: float = 0.10
    cock_window: int = 12
    fast_coef: float = 0.0
    fast_min: float = 3.5
    fast_ref: float = 1.5

    @property
    def enabled(self) -> bool:
        return any(c > 0 for c in (self.edge_coef, self.hit_coef, self.weak_coef, self.edge_hit_coef, self.pout_coef,
                                       self.edge_term_margin, self.cock_coef,
                                       self.fast_coef))

    @classmethod
    def from_args(cls, args) -> "MotionShaping":
        return cls(edge_coef=args.shape_edge_coef, edge_margin=args.shape_edge_margin, hit_coef=args.shape_hit_coef,
                   hit_vref=args.shape_hit_vref, weak_coef=args.shape_weak_coef, weak_vmin=args.shape_weak_vmin,
                   edge_hit_coef=args.shape_edge_hit_coef, edge_hit_margin=args.shape_edge_hit_margin,
                   pout_coef=args.shape_pout_coef, pout_min=args.shape_pout_min, pout_ref=args.shape_pout_ref,
                   edge_term_margin=args.shape_edge_term_margin, cock_coef=args.shape_cock_coef,
                   cock_ref=args.shape_cock_ref, fast_coef=args.shape_fast_coef, fast_min=args.shape_fast_min,
                   fast_ref=args.shape_fast_ref)


def _paddle_x(sim) -> float:
    """True paddle x in the robot frame (top edge of the workspace at ``sim.x_min_lim``)."""
    return float(-sim.paddles["paddle_ego"].position[1] - sim.center_offset_constant)


def install_motion_shaping(env, shaping: MotionShaping):
    """Add the shaping terms to the reward of ``env.step``, in place. Returns ``env``."""
    env_reset, env_step = env.reset, env.step
    state = {"x": None, "xs": []}

    def reset(*args, **kwargs):
        out = env_reset(*args, **kwargs)
        state["x"] = _paddle_x(env.simulator)
        state["xs"] = [state["x"]]
        return out

    def step(action):
        obs, reward, terminated, truncated, info = env_step(action)
        sim = env.simulator
        x = _paddle_x(sim)
        prev = x if state["x"] is None else state["x"]
        state["x"] = x
        state["xs"] = (state["xs"] + [x])[-(shaping.cock_window + 1):]
        d = x - float(sim.x_min_lim)
        v_fwd = -(x - prev) / float(sim.time_per_step)
        bonus = -shaping.edge_coef * float(np.clip((shaping.edge_margin - d) / shaping.edge_margin, 0.0, 1.0))
        if int(info.get("paddle_puck_collision_count", 0)) > 0:
            bonus += shaping.hit_coef * float(np.clip(v_fwd / shaping.hit_vref, 0.0, 1.0))
            if v_fwd < shaping.weak_vmin:
                bonus -= shaping.weak_coef
            bonus -= shaping.edge_hit_coef * float(np.clip((shaping.edge_hit_margin - d) / shaping.edge_hit_margin, 0.0, 1.0))
            puck_speed = float(np.linalg.norm(next(iter(sim.pucks.values())).linearVelocity))
            bonus += shaping.pout_coef * float(np.clip((puck_speed - shaping.pout_min) / (shaping.pout_ref - shaping.pout_min), 0.0, 1.0))
            bonus -= shaping.fast_coef * float(np.clip((puck_speed - shaping.fast_min) / shaping.fast_ref, 0.0, 1.0))
            win = state["xs"]
            bonus += shaping.cock_coef * float(np.clip((max(win) - win[0]) / shaping.cock_ref, 0.0, 1.0))
            if d < shaping.edge_term_margin:
                terminated = True
        info["motion_shaping"] = bonus
        return obs, reward + bonus, terminated, truncated, info

    env.reset, env.step = reset, step
    return env
