
import torch
import numpy as np


from collections import deque
from typing import Dict, List, Optional, Tuple


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
        obs_dim: int,
        context_len: int,
        device: torch.device | str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.context_len = context_len
        self.device = torch.device(device)

        # deques act as efficient circular buffers
        self._obs_buf = deque(maxlen=context_len)
        
        self._reset_env()



    def add(
        self,
        obs: np.ndarray,            # (obs_dim)
        done: bool = False,       # bool
    ):
        """
        Push the current (obs) into the history buffer.

        If `done` is True the buffer is reset to zeros so
        the new episode starts with a clean history.
        """

        self._obs_buf.append(obs.astype(np.float32))

        if done:
            self._reset_env()
        

    def sample(self):
        obs_seq = torch.tensor( np.stack(list(self._obs_buf), axis=0),
            dtype=torch.float32,
            device=self.device
        )                                          # (context_len, obs_dim)
        return obs_seq.unsqueeze(0)               # (1, context_len, obs_dim)


    def _reset_env(self):
        """Fill the env's buffer with zeros (called on episode reset)."""
        self._obs_buf.clear()
        for _ in range(self.context_len):
            self._obs_buf.append(np.zeros(self.obs_dim, dtype=np.float32))


