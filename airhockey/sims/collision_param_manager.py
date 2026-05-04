"""
CollisionParamManager
---------------------
Manages per-speed-tier restitution scales for the Box2D simulator and persists
training-time collision statistics so an external optimizer can improve them.

Usage (inside a training loop)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    from airhockey.sims.collision_param_manager import CollisionParamManager

    manager = CollisionParamManager(
        status_path="runs/collision_status.json",
        params_path="runs/collision_params.json",
    )
    # Give the manager a handle to the sim so it can push params directly.
    manager.attach_sim(sim)

    # --- after each episode ---
    stats = sim.get_episode_collision_stats()
    manager.on_episode_end(
        episode=episode_idx,
        collision_stats=stats,
        episode_outcome={"total_reward": total_rew, "juggle_count": juggles, ...},
    )

External optimizer (same process or separate process)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read current state
    import json
    status = json.load(open("runs/collision_status.json"))

    # Write new params
    new_params = {
        "wall_scales":   [0.95, 1.0, 1.05],
        "paddle_scales": [0.90, 1.0, 1.10],
        "speed_breakpoints": [0.25, 0.75],
    }
    json.dump(new_params, open("runs/collision_params.json", "w"))

    # The manager will pick these up on the next on_episode_end call.
"""

import json
import os
import tempfile
from typing import Optional, Sequence


_DEFAULT_SPEED_BREAKPOINTS = (0.25, 0.75)
_DEFAULT_WALL_SCALES = [1.0, 1.0, 1.0]
_DEFAULT_PADDLE_SCALES = [1.0, 1.0, 1.0]


class CollisionParamManager:
    """Persists collision params and stats; bridges the training loop and an
    external optimizer.

    Parameters
    ----------
    status_path : str
        Path for the JSON status file written after each episode (or every
        ``write_every_n_episodes`` episodes).  The external optimizer reads
        this to understand what happened.
    params_path : str
        Path the external optimizer writes new params to.  The manager polls
        this file on each ``on_episode_end`` call.  Once consumed, the file is
        deleted so it is not re-applied.
    write_every_n_episodes : int
        How often to write the status file.  Default 1 (every episode).
    """

    def __init__(
        self,
        status_path: str,
        params_path: str,
        write_every_n_episodes: int = 1,
    ):
        self.status_path = status_path
        self.params_path = params_path
        self.write_every_n_episodes = max(1, int(write_every_n_episodes))

        self._wall_scales: list = list(_DEFAULT_WALL_SCALES)
        self._paddle_scales: list = list(_DEFAULT_PADDLE_SCALES)
        self._speed_breakpoints: tuple = _DEFAULT_SPEED_BREAKPOINTS

        self._sim = None  # set via attach_sim()
        self._episode_count: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def attach_sim(self, sim) -> None:
        """Attach the Box2D simulator so params can be pushed automatically."""
        self._sim = sim
        self._push_to_sim()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_params(self) -> dict:
        return {
            "wall_scales": list(self._wall_scales),
            "paddle_scales": list(self._paddle_scales),
            "speed_breakpoints": list(self._speed_breakpoints),
        }

    # ------------------------------------------------------------------
    # Training-loop API
    # ------------------------------------------------------------------

    def set_collision_scales(
        self,
        wall_scales: Sequence[float],
        paddle_scales: Sequence[float],
        speed_breakpoints: Optional[Sequence[float]] = None,
    ) -> None:
        """Manually set scales (e.g. from an in-process optimizer)."""
        self._wall_scales = [float(s) for s in wall_scales]
        self._paddle_scales = [float(s) for s in paddle_scales]
        if speed_breakpoints is not None:
            self._speed_breakpoints = (float(speed_breakpoints[0]), float(speed_breakpoints[1]))
        self._push_to_sim()

    def on_episode_end(
        self,
        episode: int,
        collision_stats: dict,
        episode_outcome: Optional[dict] = None,
    ) -> bool:
        """Call at the end of every episode.

        Writes the status file periodically and checks for an external params
        file.  Returns True if new params were loaded from disk.

        Parameters
        ----------
        episode : int
            Current episode index (used in the status file).
        collision_stats : dict
            Output of ``sim.get_episode_collision_stats()``.
        episode_outcome : dict, optional
            Arbitrary episode-level metrics (reward, juggle count, etc.).
        """
        self._episode_count += 1
        loaded = self._check_for_external_params()

        if self._episode_count % self.write_every_n_episodes == 0:
            self._write_status(episode, collision_stats, episode_outcome or {})

        return loaded

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _push_to_sim(self) -> None:
        if self._sim is None:
            return
        self._sim.set_collision_scales(
            wall_scales=self._wall_scales,
            paddle_scales=self._paddle_scales,
            speed_breakpoints=self._speed_breakpoints,
        )

    def _write_status(self, episode: int, collision_stats: dict, episode_outcome: dict) -> None:
        payload = {
            "episode": episode,
            "current_params": self.current_params,
            "collision_stats": collision_stats,
            "episode_outcome": episode_outcome,
        }
        _atomic_json_write(self.status_path, payload)

    def _check_for_external_params(self) -> bool:
        """Read and consume the external params file if it exists.

        Returns True if new params were loaded.
        """
        if not os.path.exists(self.params_path):
            return False
        try:
            with open(self.params_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False

        # Consume file immediately to avoid re-applying on the next episode.
        try:
            os.remove(self.params_path)
        except OSError:
            pass

        wall = data.get("wall_scales", self._wall_scales)
        paddle = data.get("paddle_scales", self._paddle_scales)
        bp = data.get("speed_breakpoints", None)
        self.set_collision_scales(wall, paddle, bp)
        return True


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _atomic_json_write(path: str, payload: dict) -> None:
    """Write JSON atomically using a temp file + rename."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    dir_ = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, default=_json_default)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _json_default(obj):
    """Fallback serializer for numpy scalars etc."""
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)
