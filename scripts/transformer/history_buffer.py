
import torch
import numpy as np


from collections import deque
from typing import Dict, List, Optional, Tuple


PADDLE_POS_SLICE = slice(12, 14)   # current paddle (x, y) in raw 30-dim obs
PADDLE_VALID_INDEX = 14              # current paddle valid flag in raw 30-dim obs
PUCK_POS_SLICE   = slice(27, 29)   # current puck   (x, y) in raw 30-dim obs
PUCK_VALID_INDEX   = 29              # current puck   valid flag in raw 30-dim obs
HISTORY_ENTRY_DIM  = 6               # [paddle_x, paddle_y, paddle_valid, puck_x, puck_y, puck_valid]


class HistoryBuffer:
    """
    Circular buffer of observations

    Maintains the last `context_len` steps so that at any point the
    training loop can call `sample()` to get the tensors for the encoder.

    Parameters
    ----------
    obs_dim     : observation dimension
    context_len : history length T fed to the encoder
    device      : torch device for snapshot tensors
    include_action : if True, append action_dim to each history entry (RMA)
    action_dim  : action dimension when include_action is True (default 2)
    """

    def __init__(
        self,
        context_len: int,
        device: torch.device | str = "cpu",
        include_action: bool = False,
        action_dim: int = 2,
    ):
        self.context_len = context_len
        self.device = torch.device(device)
        self.include_action = bool(include_action)
        self.action_dim = int(action_dim)

        # deques act as efficient circular buffers
        self._buf = deque(maxlen=context_len)
        self.entry_dim = HISTORY_ENTRY_DIM + (self.action_dim if self.include_action else 0)

        self.reset_env()

    # @staticmethod
    # def extract_entry(obs: np.ndarray) -> np.ndarray:
    #     """Extract the 4-dim current-timestep position from a raw 30-dim obs."""
    #     return np.concatenate([
    #         obs[PADDLE_POS_SLICE],   # (2,)
    #         obs[PUCK_POS_SLICE],     # (2,)
    #     ]).astype(np.float32)        # (4,)

    @staticmethod
    def extract_entry(obs: np.ndarray) -> np.ndarray:
        """Extract the 6-dim current-timestep position + validity from a raw 30-dim obs."""
        return np.concatenate([
            obs[PADDLE_POS_SLICE],                             # (2,) paddle x, y
            obs[PADDLE_VALID_INDEX : PADDLE_VALID_INDEX + 1],   # (1,) paddle valid flag
            obs[PUCK_POS_SLICE],                                # (2,) puck x, y
            obs[PUCK_VALID_INDEX : PUCK_VALID_INDEX + 1],       # (1,) puck valid flag
        ]).astype(np.float32)                                   # (6,)

    def add(
        self,
        obs: np.ndarray,
        action=None,
        done: bool = False,
    ):
        """
        Push the current timestep's puck and paddle position into the history buffer.

        If `include_action` is True, concatenates `action` (or zeros if None)
        onto the 6-dim entry.

        If `done` is True the buffer is reset to zeros so
        the new episode starts with a clean history.
        """
        entry = self.extract_entry(obs)
        if self.include_action:
            if action is None:
                action_arr = np.zeros(self.action_dim, dtype=np.float32)
            else:
                action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
                if action_arr.shape[0] != self.action_dim:
                    raise ValueError(
                        f"action dim {action_arr.shape[0]} != expected {self.action_dim}"
                    )
            entry = np.concatenate([entry, action_arr], axis=0)
        self._buf.append(entry)

    def sample(self) -> torch.Tensor:
        """
        Returns (1, context_len, entry_dim) — ready to feed into the transformer
        or flatten for direct policy concatenation.
        """
        seq = np.stack(list(self._buf), axis=0)          # (T, entry_dim)
        return torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, T, entry_dim)



    def reset_env(self):
        """Fill the env's buffer with zeros (called on episode reset)."""
        self._buf.clear()
        for _ in range(self.context_len):
            self._buf.append(np.zeros(self.entry_dim, dtype=np.float32))
