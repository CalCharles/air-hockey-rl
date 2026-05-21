"""CQL (conservative Q-learning, Kumar et al. 2020) helpers for TD3.

Adds `cql_alpha * (logsumexp_a Q(s,a) - Q(s, pi(s)))` to each critic's task
loss. The logsumexp is approximated by `n_random` uniform actions in
[-1, 1]^act_dim per state. Push Q down for OOD actions while keeping Q up for
the policy's action — the canonical fix for the Q-overestimation drift
mechanism §8.13 documents.
"""

import math
from typing import NamedTuple

import torch


class CQLTerms(NamedTuple):
    """Pre-computed per-minibatch tensors reused across critics."""
    random_actions: torch.Tensor  # (n_random * bsz, act_dim)
    obs_repeat: torch.Tensor      # (n_random * bsz, obs_dim)
    policy_action: torch.Tensor   # (bsz, act_dim)
    n_random: int
    bsz: int


def precompute_cql_terms(
    *,
    sampled_observations: torch.Tensor,
    policy_action: torch.Tensor,
    act_dim: int,
    n_random: int,
) -> CQLTerms:
    """Build the random-action / obs-repeat tensors used by every critic's CQL
    term. `policy_action` is the (no-grad) deterministic action at the
    sampled states — the caller computes it once outside.
    """
    bsz = sampled_observations.shape[0]
    random_actions = torch.empty(
        n_random * bsz, act_dim, device=sampled_observations.device
    ).uniform_(-1.0, 1.0)
    obs_repeat = (
        sampled_observations.unsqueeze(0)
        .expand(n_random, -1, -1)
        .reshape(n_random * bsz, -1)
    )
    return CQLTerms(
        random_actions=random_actions,
        obs_repeat=obs_repeat,
        policy_action=policy_action,
        n_random=n_random,
        bsz=bsz,
    )


def cql_penalty(
    q,
    sampled_observations: torch.Tensor,
    terms: CQLTerms,
) -> torch.Tensor:
    """Single-critic CQL penalty: logsumexp_a Q(s,a) - Q(s, pi(s)), averaged."""
    q_rand_h = q(terms.obs_repeat, terms.random_actions).view(terms.n_random, terms.bsz)
    q_pi_h = q(sampled_observations, terms.policy_action).view(-1)
    logsumexp = torch.logsumexp(q_rand_h, dim=0) - math.log(float(terms.n_random))
    return (logsumexp - q_pi_h).mean()
