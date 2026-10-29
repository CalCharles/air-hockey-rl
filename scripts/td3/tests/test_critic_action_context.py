"""Previous-action conditioning on both sides of the Bellman update."""
import unittest

import torch
from torch import nn

from scripts.td3.helper.td3_args_validation import validate_args
from scripts.td3.helper.td3_graphed_update import GraphedTD3Update, h_transform
from scripts.td3.td3_training import Args


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(2))

    def get_action(self, state):
        return self.offset.expand(len(state), -1)


class RecordingCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.ones(1))
        self.inputs = []

    def forward(self, state, action):
        self.inputs.append(state.detach().clone())
        return h_transform(self.bias + state[:, -1:] + action[:, :1], eps=.001)


def updater(context=True):
    obj = object.__new__(GraphedTD3Update)
    obj.actor, obj.actor_target = Actor(), Actor()
    obj.qfs = [RecordingCritic(), RecordingCritic()]
    obj.qfs_target = [RecordingCritic(), RecordingCritic()]
    obj.use_last_action, obj.critic_last_action = True, context
    obj.gamma, obj.h_eps, obj.target_subset = .9, .001, None
    obj.num_critics, obj.cql_n_random, obj.cql_alpha = 2, 2, .1
    obj.action_low, obj.action_high = -torch.ones(2), torch.ones(2)
    obj.actor_action_delta_penalty = 0.
    return obj


class CriticContextTests(unittest.TestCase):
    def test_bellman_uses_previous_action_and_taken_action_at_next_state(self):
        obj = updater()
        obs, next_obs = torch.zeros(2, 30), torch.ones(2, 30)
        actions = torch.tensor([[.2, -.4], [.8, .7]])
        previous = torch.tensor([[.9, -.6], [-.2, .3]])
        output = obj._critic_core(obs, next_obs, actions, previous, torch.tensor([1., 2.]),
                                  torch.tensor([0., 1.]), torch.ones(2), torch.zeros(2, 2), None)
        for q in obj.qfs:
            torch.testing.assert_close(q.inputs[0], torch.cat([obs, previous], dim=-1))
        expected_next = torch.cat([next_obs, torch.tensor([[.2, -.4], [0., 0.]])], dim=-1)
        for q in obj.qfs_target:
            torch.testing.assert_close(q.inputs[0], expected_next)
        self.assertAlmostEqual(output[3].item(), (1. + .9 * .6 + 2.) / 2, places=4)
        loss, _ = obj._actor_core(obs, previous)
        loss.backward()
        torch.testing.assert_close(obj.qfs[0].inputs[-1], torch.cat([obs, previous], dim=-1))
        self.assertLess(obj.actor.offset.grad[0].item(), 0.)

    def test_cql_and_disabled_context_keep_correct_observation_widths(self):
        obs = torch.zeros(2, 30)
        for enabled, width in ((False, 30), (True, 32)):
            obj = updater(enabled)
            obj._critic_core(obs, obs, torch.zeros(2, 2), torch.ones(2, 2), torch.ones(2),
                             torch.zeros(2), torch.ones(2), torch.zeros(2, 2), torch.zeros(4, 2))
            for q in obj.qfs:
                self.assertEqual([value.shape[-1] for value in q.inputs], [width, width, width])

    def test_context_requires_matching_actor_state_and_scratch_initialization(self):
        with self.assertRaisesRegex(ValueError, "matching actor"):
            validate_args(Args(critic_last_action=True, use_last_action_in_policy_state=False))
        with self.assertRaisesRegex(ValueError, "scratch"):
            validate_args(Args(critic_last_action=True, use_last_action_in_policy_state=True, model_path="old.pth"))


if __name__ == "__main__":
    unittest.main()
