"""
Episode rollout for collision adaptation.

rollout_episodes() runs n_episodes of env interaction using a deterministic actor
and returns aggregated per-tier paddle collision statistics.  Wall collisions are
ignored; only paddle stats are returned (per the adaptation algorithm design).
"""

from __future__ import annotations

import numpy as np
import torch


def rollout_episodes(
    env,
    actor,
    n_episodes: int,
    device: str,
    use_last_action: bool = True,
) -> dict:
    """
    Run n_episodes and return aggregate paddle-only collision stats.

    Parameters
    ----------
    env          : AirHockeyEnv instance (already configured with correct collision scales)
    actor        : DeterministicAgent with .get_action(obs_tensor) method
    n_episodes   : number of full episodes to roll out
    device       : torch device string
    use_last_action : whether to append last action to observation before actor forward pass

    Returns
    -------
    dict of the form:
        {
          "paddle": {
            "low":  {"count": int, "mean_speed_in": float, "mean_speed_out": float},
            "mid":  {...},
            "high": {...},
          }
        }
    Counts and speed sums are averaged across all episodes.
    """
    act_dim = int(np.prod(env.action_space.shape))
    action_low = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)
    action_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)

    # Accumulated stats across all episodes
    total: dict[str, dict[str, dict]] = {
        "paddle": {
            tier: {"count": 0, "speed_in_sum": 0.0, "speed_out_sum": 0.0}
            for tier in ("low", "mid", "high")
        }
    }

    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        last_action = torch.zeros((1, act_dim), dtype=torch.float32, device=device)
        done = False

        while not done:
            with torch.no_grad():
                if use_last_action:
                    policy_obs = torch.cat([obs_tensor, last_action], dim=-1)
                else:
                    policy_obs = obs_tensor
                action_tensor = actor.get_action(policy_obs)
                action_tensor = torch.clamp(action_tensor, action_low, action_high)

            action_np = action_tensor.squeeze(0).detach().cpu().numpy()
            next_obs, _, terminated, truncated, _ = env.step(action_np)
            done = bool(terminated or truncated)
            obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_action = action_tensor.detach().clone()
            if done:
                last_action.zero_()

        # Collect per-episode stats (resets sim counters)
        ep_stats = env.simulator.get_episode_collision_stats()
        for tier in ("low", "mid", "high"):
            bucket = ep_stats["paddle"][tier]
            count = int(bucket.get("count", 0))
            if count > 0:
                total["paddle"][tier]["count"] += count
                total["paddle"][tier]["speed_in_sum"] += (
                    float(bucket.get("mean_speed_in", 0.0)) * count
                )
                total["paddle"][tier]["speed_out_sum"] += (
                    float(bucket.get("mean_speed_out", 0.0)) * count
                )

    # Convert to mean stats
    result: dict[str, dict[str, dict]] = {"paddle": {}}
    for tier in ("low", "mid", "high"):
        count = total["paddle"][tier]["count"]
        result["paddle"][tier] = {
            "count": count,
            "mean_speed_in": (
                total["paddle"][tier]["speed_in_sum"] / count if count > 0 else 0.0
            ),
            "mean_speed_out": (
                total["paddle"][tier]["speed_out_sum"] / count if count > 0 else 0.0
            ),
        }

    return result
