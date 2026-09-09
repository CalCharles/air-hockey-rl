
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
    """

    def __init__(
        self,
        context_len: int,
        device: torch.device | str = "cpu",
    ):
        self.context_len = context_len
        self.device = torch.device(device)

        # deques act as efficient circular buffers
        self._buf = deque(maxlen=context_len)
        self.entry_dim = HISTORY_ENTRY_DIM
        
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
        done: bool = False,       
    ):
        """
        Push the current timestep's puck and paddle position into the history buffer.

        If `done` is True the buffer is reset to zeros so
        the new episode starts with a clean history.
        """

        self._buf.append(self.extract_entry(obs))
        

    def sample(self) -> torch.Tensor:
        """
        Returns (1, context_len, 4) — ready to feed into the transformer
        or flatten for direct policy concatenation.
        """
        seq = np.stack(list(self._buf), axis=0)          # (T, 4)
        return torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, T, 4)



    def reset_env(self):
        """Fill the env's buffer with zeros (called on episode reset)."""
        self._buf.clear()
        for _ in range(self.context_len):
            self._buf.append(np.zeros(self.entry_dim, dtype=np.float32))


