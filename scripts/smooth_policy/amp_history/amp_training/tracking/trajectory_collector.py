"""Per-episode paddle trajectory collector."""

from collections import deque
import numpy as np


class TrajectoryCollector:
    """Collect per-episode paddle trajectories from vectorized rollouts.

    Args:
        max_episodes: Maximum number of completed episodes to retain in memory.
    """

    def __init__(self, max_episodes: int = 200):
        self.max_episodes = max_episodes
        self._episodes: deque = deque(maxlen=max_episodes)
        # Per-env in-progress episode buffer (list of positions)
        self._in_progress: list[list[np.ndarray]] = []

    def _ensure_in_progress(self, num_envs: int):
        """Lazily initialise per-env in-progress buffers."""
        if len(self._in_progress) != num_envs:
            self._in_progress = [[] for _ in range(num_envs)]

    def push_rollout(
        self,
        paddle_positions: np.ndarray,
        dones: np.ndarray,
    ):
        """Ingest one rollout from the vectorised environment.

        Args:
            paddle_positions: float array of shape [T, num_envs, 2].
            dones: bool array of shape [T, num_envs].
                   dones[step, env_i] == True means env i was reset *before*
                   step t (i.e. the new episode already started at step t).
        """
        T, num_envs = dones.shape
        self._ensure_in_progress(num_envs)

        for step in range(T):
            for env_i in range(num_envs):
                if dones[step, env_i]:
                    # The previous episode ended; store it if non-empty.
                    if self._in_progress[env_i]:
                        episode = np.array(
                            self._in_progress[env_i], dtype=np.float32
                        )  # (Ti, 2)
                        self._episodes.append(episode)
                        self._in_progress[env_i] = []

                # Append current position to the running episode.
                pos = paddle_positions[step, env_i]  # (2,)
                self._in_progress[env_i].append(pos.copy())

    def get_recent(self, n: int) -> list[np.ndarray]:
        """Return the last *n* completed episodes (oldest-first)."""
        episodes = list(self._episodes)
        return episodes[-n:] if n < len(episodes) else episodes

    def get_all_positions(self) -> np.ndarray:
        """Return a flat (N, 2) array of all positions across all stored episodes."""
        if not self._episodes:
            return np.zeros((0, 2), dtype=np.float32)
        return np.concatenate(list(self._episodes), axis=0)
