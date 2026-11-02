"""Deployment history follows final commands, including holds and overrides."""
import unittest
import copy
from types import SimpleNamespace
import numpy as np
import torch
from scripts.rma.history import StepHistoryBuffer
from scripts.td3.helper.real_eval_agents import _RMAActorAdapter
from scripts.td3.helper.real_policy_runner import PolicyRunner


class Agent:
    def __init__(self):
        self.history=StepHistoryBuffer(4,4,2)

    def reset(self, obs, unused=None):
        self.history.reset(obs)

    def predict(self):
        return np.clip(self.history.window()[-1,-2:]+np.array([.1,-.1],dtype=np.float32),-1,1)

    def act(self, obs, last_action):
        action=self.predict();self.history.push(obs,action);return action


def adapter():
    return _RMAActorAdapter(Agent(),obs_dim=2,act_dim=2,use_last_action=True,device=torch.device('cpu'))


class HistoryCommandTests(unittest.TestCase):
    def test_adapted_agent_also_uses_committed_commands(self):
        from scripts.rma.evaluate import AdaptedRMAAgent
        from scripts.rma.networks import RMAActor, AdaptationModule, step_state_features
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.set_num_threads(1);torch.manual_seed(62)
        model=RMAActor(obs_dim=30,act_dim=2,env_param_dim=3).eval()
        phi=AdaptationModule(feature_dim=8,history_len=50).eval()
        actor=_RMAActorAdapter(AdaptedRMAAgent(copy.deepcopy(model),copy.deepcopy(phi),'latest_frame'),
            obs_dim=30,act_dim=2,use_last_action=True,device=torch.device('cpu'))
        reference=StepHistoryBuffer(50,8,2);rng=np.random.default_rng(62)
        first=rng.normal(size=30).astype(np.float32);actor.on_episode_start(first)
        reference.reset(step_state_features(torch.from_numpy(first),'latest_frame').numpy())
        previous=np.zeros(2,dtype=np.float32)
        for step in range(90):
            obs=rng.normal(size=30).astype(np.float32)
            with torch.no_grad():
                expected=model.get_action_from_latent(torch.from_numpy(obs).reshape(1,-1),
                    torch.from_numpy(previous).reshape(1,-1),phi(torch.from_numpy(reference.window()).unsqueeze(0)))
                actual=actor.get_action(torch.from_numpy(np.concatenate((obs,previous))).reshape(1,-1))
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            command=actual.numpy().reshape(-1).copy()
            if step%9==0:command[:]=0
            actor.on_env_action(command)
            reference.push(step_state_features(torch.from_numpy(obs),'latest_frame').numpy(),command)
            np.testing.assert_array_equal(actor._agent.history.window(),reference.window())
            previous=command.copy()

    def test_history_network_matches_training_layout_after_overrides(self):
        from scripts.rma.evaluate import HistoryAgent
        from scripts.rma.networks import HistoryActor, step_state_features
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.set_num_threads(1)
        torch.manual_seed(61)
        model=HistoryActor(obs_dim=30,act_dim=2,history_len=50,feature_dim=8).eval()
        actor=_RMAActorAdapter(HistoryAgent(copy.deepcopy(model)),obs_dim=30,act_dim=2,
                               use_last_action=True,device=torch.device('cpu'))
        reference=StepHistoryBuffer(50,8,2)
        rng=np.random.default_rng(61)
        for episode in range(2):
            first=rng.normal(size=30).astype(np.float32)
            actor.on_episode_start(first)
            reference.reset(step_state_features(torch.from_numpy(first),'latest_frame').numpy())
            previous=np.zeros(2,dtype=np.float32)
            for step in range(120):
                obs=rng.normal(size=30).astype(np.float32)
                policy=torch.from_numpy(np.concatenate((obs,previous))).reshape(1,-1)
                training_layout=torch.from_numpy(np.concatenate((obs,reference.window().reshape(-1),previous))).reshape(1,-1)
                with torch.no_grad():
                    expected=model.get_action(training_layout)
                    actual=actor.get_action(policy)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                command=actual.numpy().reshape(-1).copy()
                if step%7==0:
                    command[:]=0
                elif step%11==0:
                    command=np.clip(command+np.array([.2,-.3],dtype=np.float32),-1,1)
                actor.on_env_action(command)
                reference.push(step_state_features(torch.from_numpy(obs),'latest_frame').numpy(),command)
                np.testing.assert_array_equal(actor._agent.history.window(),reference.window())
                previous=command.copy()

    def test_normal_commands_and_episode_resets_match_reference(self):
        actor=adapter();reference=Agent()
        for episode in range(2):
            first=np.array([episode,.2],dtype=np.float32)
            actor.on_episode_start(first);reference.reset(first);previous=np.zeros(2,dtype=np.float32)
            for step in range(20):
                obs=np.array([step*.01,.2],dtype=np.float32)
                actual=actor.get_action(torch.from_numpy(np.concatenate((obs,previous))).reshape(1,-1)).numpy().reshape(-1)
                expected=reference.act(obs,previous)
                np.testing.assert_array_equal(actual,expected)
                actor.on_env_action(actual)
                np.testing.assert_array_equal(actor._agent.history.window(),reference.history.window())
                previous=actual.copy()

    def test_holds_and_nonzero_overrides_affect_the_next_prediction(self):
        actor=adapter();reference=Agent();first=np.zeros(2,dtype=np.float32)
        actor.on_episode_start(first);reference.reset(first);previous=np.zeros(2,dtype=np.float32)
        for step in range(24):
            obs=np.array([step,.5],dtype=np.float32)
            actual=actor.get_action(torch.from_numpy(np.concatenate((obs,previous))).reshape(1,-1)).numpy().reshape(-1)
            np.testing.assert_array_equal(actual,reference.predict())
            command=np.zeros(2,dtype=np.float32) if step%3==0 else np.array([.4,-.6],dtype=np.float32)
            actor.on_env_action(command);reference.history.push(obs,command)
            np.testing.assert_array_equal(actor._agent.history.window(),reference.history.window())
            previous=command.copy()

    def test_command_feedback_requires_one_query_and_finite_action(self):
        actor=adapter()
        with self.assertRaises(RuntimeError):actor.on_env_action(np.zeros(2))
        actor.on_episode_start(np.zeros(2));actor.get_action(torch.zeros((1,4)))
        with self.assertRaises(ValueError):actor.on_env_action(np.array([np.nan,0]))
        with self.assertRaises(ValueError):actor.on_env_action(np.zeros(3))
        actor.on_env_action(np.zeros(2))
        with self.assertRaises(RuntimeError):actor.on_env_action(np.zeros(2))
        actor.on_episode_start(np.ones(2))
        with self.assertRaises(RuntimeError):actor.on_env_action(np.zeros(2))

    def test_runner_reports_after_step_using_command_before_internal_mutation(self):
        events=[];received=[]
        class Env:
            def step(self, action):
                events.append('step');action[:]=0;return ('outcome',)
        actor=SimpleNamespace(on_env_action=lambda command:(events.append('history'),received.append(command.copy())))
        runner=object.__new__(PolicyRunner);runner._env=Env();runner._actor=actor
        result=runner._step_env_action(np.array([.4,-.6],dtype=np.float32))
        self.assertEqual(result,('outcome',));self.assertEqual(events,['step','history'])
        np.testing.assert_array_equal(received[0],np.array([.4,-.6],dtype=np.float32))
        runner._actor=object()
        self.assertEqual(runner._step_env_action(np.zeros(2)),('outcome',))

    def test_failed_env_step_does_not_commit_command(self):
        class Env:
            def step(self, action):raise OSError('step failed')
        calls=[];runner=object.__new__(PolicyRunner);runner._env=Env();runner._actor=SimpleNamespace(on_env_action=calls.append)
        with self.assertRaises(OSError):runner._step_env_action(np.zeros(2))
        self.assertEqual(calls,[])


if __name__=='__main__':
    unittest.main()
