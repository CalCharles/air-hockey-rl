"""Unit tests for the PyTorch Y3H4 port.  pytest scripts/goal_conditioned_rl/test_y3h4.py -q
(Numerical parity with the JAX y3h4.py is parity_y3h4.py, run in a JAX venv.)"""

import numpy as np
import torch

torch.set_num_threads(1)

from scripts.goal_conditioned_rl.train_y3h4 import CommandedReplay, rest_view  # noqa: E402
from scripts.goal_conditioned_rl.y3h4_agent import Y3H4Agent, Y3H4Config  # noqa: E402

SMALL = dict(width=16, depth=4, repr_dim=8, batch_size=32)


def _agent(heads=3, **kw):
    torch.manual_seed(0)
    return Y3H4Agent(Y3H4Config(heads=heads, **SMALL, **kw), 5, 2, 4)


def test_replay_returns_each_rows_commanded_goal():
    rp = CommandedReplay(1000, 1, 1, 4, "cpu")
    for e, n in enumerate([7, 50, 3, 200]):
        t = np.arange(n, dtype=np.float32)
        cmd = np.tile([e, 0, 0, 0], (n, 1)).astype(np.float32)
        rp.add_episode(np.c_[t + 1000 * e], np.zeros((n, 1)), np.c_[np.full(n, e), t + 1, t, t], cmd)
    s, a, vg, ag, c = rp.sample_y3h4(5000, 0.99)
    ep = torch.div(s[:, 0], 1000, rounding_mode="floor")
    assert torch.equal(c[:, 0], ep) and torch.equal(vg[:, 0], ep), "commanded / positive goal from another episode"
    assert torch.equal(vg, ag), "actor goal must be the same hindsight goal"
    assert torch.all(vg[:, 1] >= s[:, 0] - 1000 * ep + 1), "positive not in the future"


def test_rest_view_zeroes_velocity_only():
    g = rest_view(np.array([[0.1, -0.2, 0.5, -0.7]]), (2, 3))
    assert np.allclose(g, [[0.1, -0.2, 0.0, 0.0]])


def test_heads_are_independent():
    ag = _agent()
    g = torch.randn(10, 4)
    before = ag.psi(g).detach().clone()
    with torch.no_grad():
        ag.g_encoder.mlp.dense[0].weight[1] += torch.randn_like(ag.g_encoder.mlp.dense[0].weight[1])  # not a
        # constant: the LayerNorm after the layer would cancel a uniform shift
    after = ag.psi(g)
    assert torch.allclose(before[0], after[0]) and torch.allclose(before[2], after[2])
    assert not torch.allclose(before[1], after[1])


def test_projection_head_first_coordinate_is_zero():
    ag = _agent()
    assert torch.all(ag.psi(torch.randn(6, 4))[..., 0] == 0)


def test_identical_heads_vote_returns_their_action_on_every_lane():
    ag = _agent()
    with torch.no_grad():
        for p in ag.parameters():
            p.copy_(p[:1].expand_as(p))
    s, g = torch.randn(8, 5), torch.randn(8, 4)
    loc, _ = ag.policy(s, ag.psi(g))
    want = torch.tanh(loc[0])
    assert torch.allclose(ag.act(s, g), want, atol=1e-6)
    assert torch.allclose(ag.act(s, g, lane_ids=torch.arange(8), n_lanes=8), want, atol=1e-6)


def test_pinned_lanes_use_their_head():
    ag = _agent(heads=3, commit_fraction=0.25)
    s, g = torch.randn(8, 5), torch.randn(8, 4)
    loc, _ = ag.policy(s, ag.psi(g))
    means = torch.tanh(loc)
    out = ag.act(s, g, lane_ids=torch.arange(8), n_lanes=8)  # commit = 2: lanes 2..7 -> heads 0,1,2,0,1,2
    for lane in range(2, 8):
        assert torch.allclose(out[lane], means[(lane - 2) % 3, lane], atol=1e-6)


def test_rolled_accuracy_equals_accuracy_when_phi_ignores_z():
    ag = _agent(goal_jitter=0.0)
    with torch.no_grad():
        ag.sa_encoder.mlp.dense[0].weight[:, 7:, :] = 0.0  # inputs [s(5), a(2), z(8)]: zero the z columns
    s, a, vg, c = torch.randn(32, 5), torch.rand(32, 2) * 2 - 1, torch.randn(32, 4), torch.randn(32, 4)
    _, info, _, _ = ag.critic_loss(s, a, vg, c, jitter_noise=torch.zeros(3, 32, 4))
    assert abs(float(info["accuracy"]) - float(ag.rolled_code_accuracy(s, a, vg, c))) < 1e-6


def test_actor_loss_leaves_critic_untouched_and_update_changes_all_heads():
    ag = _agent()
    s, a, vg, c = torch.randn(32, 5), torch.rand(32, 2) * 2 - 1, torch.randn(32, 4), torch.randn(32, 4)
    a_loss, _ = ag.actor_loss(s, vg)
    grads = torch.autograd.grad(a_loss, ag.actor_params + [ag.log_alpha], allow_unused=False)
    assert all(torch.isfinite(gr).all() for gr in grads)
    w0 = ag.sa_encoder.mlp.dense[0].weight.detach().clone()
    info = ag.update(s, a, vg, vg, c)
    assert float(info["skipped"]) == 0.0
    changed = (ag.sa_encoder.mlp.dense[0].weight - w0).abs().flatten(1).max(1).values
    assert torch.all(changed > 0), "every head's critic should move"


def test_critic_learns_toy_future():
    """Goal = deterministic function of (s, a): the critic should pick the right goal out of the batch."""
    torch.manual_seed(1)
    ag = Y3H4Agent(Y3H4Config(heads=2, width=32, depth=4, repr_dim=16, batch_size=64, goal_jitter=0.0), 3, 2, 4)
    W = torch.randn(5, 4)
    for _ in range(400):
        s, a = torch.randn(64, 3), torch.rand(64, 2) * 2 - 1
        g = torch.tanh(torch.cat([s, a], 1) @ W)
        info = ag.update(s, a, g, g, torch.zeros(64, 4))
    assert float(info["accuracy"]) > 0.3, info["accuracy"]
