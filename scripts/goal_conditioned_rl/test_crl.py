"""Unit tests for the PyTorch contrastive-RL port.  pytest scripts/goal_conditioned_rl/test_crl.py -q"""

import numpy as np
import torch

torch.set_num_threads(1)  # tiny nets: multithreaded CPU kernels are ~300x slower on a loaded box

from scripts.goal_conditioned_rl.crl_agent import CRLAgent, CRLConfig
from scripts.goal_conditioned_rl.crl_replay import EpisodeReplay


def _replay_with_episodes(lengths, capacity=10_000):
    rp = EpisodeReplay(capacity, 1, 1, 2, "cpu")
    for e, n in enumerate(lengths):
        t = np.arange(n, dtype=np.float32)
        # goal view encodes (episode id, time index of the NEXT state)
        rp.add_episode(np.c_[t], np.zeros((n, 1)), np.c_[np.full(n, e), t + 1])
    return rp


def test_positives_stay_in_episode_and_are_future():
    rp = _replay_with_episodes([5, 300, 1, 40, 1000])
    s, _, g, rows, offsets = rp.sample(50_000, 0.99)
    ep_of_row = torch.repeat_interleave(torch.arange(5), torch.tensor([5, 300, 1, 40, 1000]))[rows]
    assert torch.equal(g[:, 0].long(), ep_of_row), "goal crossed an episode boundary"
    assert torch.all(g[:, 1] >= s[:, 0] + 1), "goal is not strictly in the future"


def test_offset_distribution_matches_truncated_geometric():
    rp = EpisodeReplay(10, 1, 1, 1, "cpu")
    gamma, n = 0.9, 20
    k = rp.sample_offsets(torch.full((400_000,), n), gamma).numpy()
    want = gamma ** np.arange(n)
    want /= want.sum()
    got = np.bincount(k, minlength=n) / len(k)
    assert np.abs(got - want).max() < 3e-3
    assert k.max() <= n - 1


def test_ring_wrap_keeps_futures_in_episode():
    rp = _replay_with_episodes([700, 700, 700], capacity=1500)  # third episode wraps over the first
    s, _, g, rows, _ = rp.sample(20_000, 0.99)
    assert torch.all(g[:, 1] >= s[:, 0] + 1)
    assert torch.all(g[:, 1] - s[:, 0] <= 700)


def test_actor_goal_modes():
    for rg, rows in ((0.0, 8), (0.5, 16), (1.0, 8)):
        agent = CRLAgent(CRLConfig.paper(random_goals=rg, hidden_sizes=(16, 16), repr_dim=4), 3, 2, 2)
        s, g = torch.randn(8, 3), torch.arange(16.0).reshape(8, 2)
        s2, g2 = agent.actor_goals(s, g)
        assert s2.shape[0] == rows and g2.shape[0] == rows
        if rg == 1.0:
            assert torch.equal(g2, torch.roll(g, 1, 0))
        if rg == 0.5:
            assert torch.equal(g2[:8], g) and torch.equal(g2[8:], torch.roll(g, 1, 0))


def test_update_runs_and_critic_learns_both_presets():
    torch.manual_seed(0)
    for cfg in (CRLConfig.paper(hidden_sizes=(64, 64), repr_dim=16, batch_size=64),
                CRLConfig.c0(width=64, depth=4, repr_dim=16, batch_size=64)):
        agent = CRLAgent(cfg, 2, 2, 2)
        accs = []
        # goal = s + a: learnable, so contrastive accuracy should climb. Binary NCE (1 positive vs 63 negatives
        # per row) sits at chance for ~300 steps before separating (0.39 at 1000), InfoNCE is faster.
        for _ in range(1000 if cfg.critic_loss == "nce" else 300):
            s, a = torch.rand(64, 2) * 2 - 1, torch.rand(64, 2) * 2 - 1
            info = agent.update(s, a, s + a)
            accs.append(float(info["accuracy"]))
        assert np.mean(accs[-30:]) > 0.2, (cfg.critic_loss, np.mean(accs[-30:]))


def test_tanh_log_prob_matches_torch_distributions():
    from torch.distributions import Normal, TanhTransform, TransformedDistribution

    loc, std = torch.randn(1000, 2), torch.rand(1000, 2) + 0.1
    u = loc + std * torch.randn_like(loc)
    ref = TransformedDistribution(Normal(loc, std), [TanhTransform()]).log_prob(torch.tanh(u)).sum(-1)
    from scripts.goal_conditioned_rl.crl_agent import TanhGaussianActor

    ours = TanhGaussianActor.tanh_log_prob(u, loc, std)
    mask = torch.tanh(u).abs() < 0.999
    assert torch.allclose(ours[mask.all(-1)], ref[mask.all(-1)], atol=1e-3)
