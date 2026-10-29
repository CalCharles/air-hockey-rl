"""Output-coordinate, gradient and checkpoint tests for scratch aim-point actors."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import gymnasium as gym
import numpy as np
import torch

from scripts.td3.absolute_target_agent import AbsoluteTargetAgent
from scripts.td3.deterministic_agent import DeterministicAgent
from scripts.td3.eval_utils import infer_policy_class_from_state_dict, load_policy_for_evaluation
from scripts.td3.helper.td3_args_validation import validate_args
from scripts.td3.helper.td3_graphed_update import deterministic_actor_action
from scripts.td3.td3_training import Args


def view():
    return SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(32,)),
                           single_action_space=gym.spaces.Box(-1., 1., shape=(2,)))


class AbsoluteTargetTests(unittest.TestCase):
    def test_same_random_weight_initialization_and_observed_pose_feedback(self):
        torch.manual_seed(123)
        plain = DeterministicAgent(view())
        torch.manual_seed(123)
        actor = AbsoluteTargetAgent(view())
        for (name, p), (other_name, q) in zip(plain.named_parameters(), actor.named_parameters()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        with torch.no_grad():
            for p in actor.parameters():
                p.zero_()
        state = torch.zeros(3, 32)
        state[:, 12:14] = actor.target_center
        state[1, 12] -= .05
        state[2, 12] += .05
        action = deterministic_actor_action(actor, state)
        torch.testing.assert_close(action[0], torch.zeros(2))
        self.assertGreater(action[1, 0].item(), 0.)
        self.assertLess(action[2, 0].item(), 0.)
        self.assertTrue((action.abs() <= 1).all())
        action.sum().backward()
        self.assertTrue((actor.actor_mean_head.bias.grad > 0).all())

    def test_checkpoint_inference_preserves_absolute_and_legacy_actions(self):
        state = torch.randn(8, 32)
        state[:, 12:14] = torch.tensor([.7, .1])
        with tempfile.TemporaryDirectory() as directory:
            for actor_class, label in ((AbsoluteTargetAgent, "absolute_target_agent"),
                                       (DeterministicAgent, "deterministic_agent")):
                actor = actor_class(view())
                self.assertEqual(infer_policy_class_from_state_dict(actor.state_dict()), label)
                path = Path(directory) / (label + ".pth")
                torch.save(actor.state_dict(), path)
                loaded = load_policy_for_evaluation(path, view(), 1., 64, 2)
                torch.testing.assert_close(loaded(state), actor(state), rtol=0, atol=0)

    def test_absolute_target_training_rejects_checkpoint_initialization(self):
        args = Args(actor_absolute_target=True, model_path="old.pth")
        with self.assertRaisesRegex(ValueError, "scratch"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
