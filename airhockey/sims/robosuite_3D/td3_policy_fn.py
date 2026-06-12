# scripts/td3/helper/td3_policy_fn.py

from __future__ import annotations
import numpy as np
import torch
from scripts.transformer.history_buffer import HistoryBuffer


def make_policy_fn(
    actor,
    device: str,
    use_last_action: bool = False,
    use_history: bool = False,
    use_transformer: bool = False,
    transformer=None,
    context_len: int = 30,
    act_dim: int = 2,
):
    """
    Wrap a TD3 actor into the (obs, deterministic) -> action signature
    that rollout.py expects.

    Maintains its own HistoryBuffer so it can be used as a stateless
    callable from rollout.py's episode loop.
    """
    history_buf = HistoryBuffer(context_len=context_len, device=device)
    last_action = torch.zeros((1, act_dim), dtype=torch.float32, device=device)

    def policy_fn(obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        if use_history:
            history_buf.add(obs)
            state_history = history_buf.sample()  # (1, T, 4)
            if use_transformer and transformer is not None:
                with torch.no_grad():
                    context = transformer(state_history)
            else:
                context = state_history.view(1, -1)
            obs_tensor = torch.cat([obs_tensor, context], dim=-1)

        if use_last_action:
            obs_tensor = torch.cat([obs_tensor, last_action], dim=-1)

        with torch.no_grad():
            action = actor.get_action(obs_tensor)

        last_action.copy_(
            torch.as_tensor(action, dtype=torch.float32, device=device).reshape(1, act_dim)
        )
        return action.cpu().numpy().flatten()

    def reset_fn():
        """Call at the start of each episode to clear history state."""
        history_buf.reset_env()
        last_action.zero_()

    policy_fn.reset = reset_fn
    return policy_fn