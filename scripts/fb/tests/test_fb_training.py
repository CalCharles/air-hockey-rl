"""Integration checks use real Metamotivo networks and the requested Box2D config."""
from dataclasses import replace
import json
import random

import numpy as np
import pytest
import torch
from metamotivo.fb import FBAgent, FBModel

from scripts.fb.fb_training import (Args, create_agent, evaluate, isolated_rng,
    policy_obs, train, transition)


def test_terminal_semantics_and_action_history():
    obs = np.arange(30, dtype=np.float32)
    action = np.array([.2, -.3], dtype=np.float32)
    next_state = policy_obs(obs, action)
    assert next_state.shape == (32,)
    np.testing.assert_array_equal(next_state[-2:], action)
    for terminated, truncated, reasons, expected in [
        (True, False, [], True), (False, False, [], False),
        (False, True, ["max_timesteps_exceeded"], False),
        (False, True, ["puck_hit_bottom"], True),
        (True, True, ["max_timesteps_exceeded"], True),
    ]:
        data = transition(next_state, action, next_state, 1., terminated, truncated,
                          {"truncation_reasons": reasons})
        assert data["next"]["terminated"].shape == (1, 1)
        assert data["next"]["terminated"].item() == expected


def test_latency_queue_resets():
    from scripts.rma.action_latency import install_action_latency
    class Env:
        action_space = type("Space", (), {"shape": (2,)})()
        def reset(self):
            pass
        def step(self, action):
            return action
    env = install_action_latency(Env(), 2)
    env.reset()
    np.testing.assert_array_equal(env.step([1, 1]), [0, 0])
    np.testing.assert_array_equal(env.step([2, 2]), [0, 0])
    np.testing.assert_array_equal(env.step([3, 3]), [1, 1])
    env.reset()
    np.testing.assert_array_equal(env.step([4, 4]), [0, 0])


def test_rng_isolation():
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)
    with isolated_rng(99):
        random.random(), np.random.rand(), torch.rand(1)
    actual = random.random(), np.random.rand(), torch.rand(1)
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)
    expected = random.random(), np.random.rand(), torch.rand(1)
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No CUDA"))])
@pytest.mark.parametrize("backward_input", ["full", "current"])
def test_online_training_and_checkpoint(tmp_path, device, backward_input):
    args = Args(device=device, backward_input=backward_input, output_dir=str(tmp_path / device), total_timesteps=40,
                learning_starts=8, batch_size=8, buffer_size=24, hidden_dim=32,
                backward_hidden_dim=32, z_dim=8, eval_every_steps=20,
                checkpoint_every_steps=20, eval_episodes=1, inference_samples=16,
                log_every_steps=20, tensorboard=False, gif_episodes=1)
    checkpoint = train(args)
    metadata = json.loads((checkpoint / "run.json").read_text())
    assert metadata["updates"] == 33
    from PIL import Image
    gifs = list((checkpoint.parent / "gifs" / "step_000000040").glob("*.gif"))
    assert len(gifs) == 3
    for path in gifs:
        with Image.open(path) as clip:
            assert clip.width == 160
            assert clip.n_frames > 1
            assert clip.info["duration"] == 50
    assert (checkpoint.parent / "gifs" / "index.html").is_file()
    assert metadata["sim_config"]["air_hockey"]["simulator_params"]["hist_len"] == 4
    from scripts.fb.models import AirHockeyFBAgent, load_policy
    cls = AirHockeyFBAgent if backward_input == "current" else FBAgent
    agent = cls.load(str(checkpoint), device=device)
    model = load_policy(checkpoint / "policy", device=device)
    z = torch.load(checkpoint / "reward_z.pt", map_location=device, weights_only=True)
    obs = torch.zeros((1, 32), device=device)
    torch.testing.assert_close(agent.act(obs, z), model.act(obs, z))
    assert torch.isfinite(model.act(obs, z)).all()
    # Ensure optimization actually changed weights, not just collection/logging.
    torch.manual_seed(args.seed)
    initial = create_agent(args, 32, 2)
    assert any(not torch.equal(a, b) for a, b in zip(initial._model._forward_map.parameters(),
                                                     model._forward_map.parameters()))
    rows = [json.loads(line) for line in (checkpoint.parent / "metrics.jsonl").read_text().splitlines()]
    assert any("train/fb_loss" in row for row in rows)
    assert any("eval/return_mean" in row for row in rows)
    assert all(np.isfinite(value) for row in rows for value in row.values())
    result = evaluate(model, z, metadata["sim_config"], args)
    assert result["eval/return_mean"] == rows[-1]["eval/return_mean"]
    with pytest.raises(FileExistsError):
        train(args)


def test_task_reward_relabeling():
    from scripts.fb.fb_training import task_rewards
    args = Args()
    batch = {"next": {"reward": torch.tensor([[3.], [4.]]),
                      "contact": torch.tensor([[0.], [1.]]),
                      "paddle_position": torch.tensor([[.7, 0.], [.9, 0.]])}}
    torch.testing.assert_close(task_rewards(batch, "juggle", args), batch["next"]["reward"])
    torch.testing.assert_close(task_rewards(batch, "touch", args), batch["next"]["contact"])
    reach = task_rewards(batch, "reach", args)
    assert reach[0].item() == 1.
    assert reach[1].item() < .001
    assert task_rewards(batch, "reach", replace(args, paddle_goal_x=.9))[1].item() == 1.


def test_backward_uses_only_current_observation():
    from scripts.fb.models import CURRENT_INDICES
    torch.set_num_threads(1)
    args = Args(device="cpu", backward_input="current", hidden_dim=32,
                backward_hidden_dim=32, z_dim=8, batch_size=8)
    agent = create_agent(args, 32, 2)
    full = create_agent(replace(args, backward_input="full"), 32, 2)
    model = agent._model
    for name in ("_forward_map", "_actor"):
        assert {k: v.shape for k, v in getattr(model, name).state_dict().items()} == {
            k: v.shape for k, v in getattr(full._model, name).state_dict().items()}
    x = torch.randn(8, 32)
    other = torch.randn_like(x) * 100
    other[:, list(CURRENT_INDICES)] = x[:, list(CURRENT_INDICES)]
    torch.testing.assert_close(model.backward_map(x), model.backward_map(other), rtol=0, atol=0)
    reward = torch.rand(8, 1)
    torch.testing.assert_close(model.reward_inference(x, reward),
                               model.reward_inference(other, reward), rtol=0, atol=0)
    for backward in (model._backward_map, model._target_backward_map):
        x_grad = x.clone().requires_grad_()
        y = backward(x_grad)
        (y * torch.randn_like(y)).sum().backward()
        excluded = [i for i in range(32) if i not in CURRENT_INDICES]
        assert torch.count_nonzero(x_grad.grad[:, excluded]) == 0
        assert torch.count_nonzero(x_grad.grad[:, list(CURRENT_INDICES)]) > 0
    # Goal-derived training latents must also ignore all older history and action.
    torch.manual_seed(44)
    first = agent.sample_mixed_z(x)
    torch.manual_seed(44)
    second = agent.sample_mixed_z(other)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
