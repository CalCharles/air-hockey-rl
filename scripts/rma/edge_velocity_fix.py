"""Counterfactual fix of the paddle's top-edge velocity, applied to a live env (no simulator edit).

After every Box2D sub-step the simulator clips the paddle position back into its workspace
(``airhockey/sims/airhockey_box2d.py``, the ``_clip_pid_target_to_workspace(pos)`` call after
``world.Step``) but keeps the paddle body's velocity, so a paddle pinned against the boundary still
carries its outward velocity into the paddle-puck restitution model. This patch removes the outward
component of that velocity whenever the clip moves the paddle, as a clamped real paddle would.

It patches instance methods of ``env.simulator`` after every ``reset`` (the simulator is rebuilt
there under domain randomization). Undo by not calling it: ``airhockey/`` is unchanged.
"""

from __future__ import annotations

import numpy as np


def _patch_simulator(sim) -> None:
    if getattr(sim, "_edge_velocity_fix", False):
        return
    orig_pid, orig_clip = sim._apply_pid_force, sim._clip_pid_target_to_workspace
    state = {"in_pid": False}

    def pid(pos, act):
        # _apply_pid_force clips the PID target with the same method; leave that call alone.
        state["in_pid"] = True
        try:
            return orig_pid(pos, act)
        finally:
            state["in_pid"] = False

    def clip(q):
        out = orig_clip(q)
        if not state["in_pid"]:
            d = np.asarray(q, float) - np.asarray(out, float)
            dn = float(np.linalg.norm(d))
            if dn > 1e-9:
                u = d / dn
                body = sim.paddles["paddle_ego"]
                v = np.array(body.linearVelocity, float)
                vo = float(v @ u)
                if vo > 0:
                    v -= vo * u
                    body.linearVelocity = (float(v[0]), float(v[1]))
        return out

    sim._apply_pid_force, sim._clip_pid_target_to_workspace = pid, clip
    sim._edge_velocity_fix = True


def install_edge_velocity_fix(env):
    """Patch ``env`` in place so every simulator it builds has the fix. Returns ``env``."""
    env_reset = env.reset

    def reset(*args, **kwargs):
        out = env_reset(*args, **kwargs)
        _patch_simulator(env.simulator)
        return out

    env.reset = reset
    _patch_simulator(env.simulator)
    return env


def patch_env_class(cls) -> None:
    """Give every ``cls`` env created in this process the fix (used by the eval scripts' ``--edge-velocity-fix``)."""
    if getattr(cls, "_edge_velocity_fix", False):
        return
    cls_reset = cls.reset

    def reset(self, *args, **kwargs):
        out = cls_reset(self, *args, **kwargs)
        _patch_simulator(self.simulator)
        return out

    cls.reset = reset
    cls._edge_velocity_fix = True
