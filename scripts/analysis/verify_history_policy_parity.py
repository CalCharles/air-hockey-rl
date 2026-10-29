"""Compare training-wrapper and evaluation-agent history on noisy rollouts.

The short episode budget exercises automatic resets, not policy performance.
No learning, simulator modification or clean sensing is used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from airhockey import AirHockeyEnv
from scripts.rma.bundle import load_phase1
from scripts.rma.env_wrapper import HistoryEnvVector
from scripts.rma.evaluate import HistoryAgent
from scripts.td3.juggle_noisy_scratch import assert_canonical_scratch_env


@torch.inference_mode()
def verify(checkpoint: Path, output: Path):
    torch.set_num_threads(1)
    seed = 600013  # Previously consumed research seed; no acceptance claim.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    bundle = load_phase1(str(checkpoint))
    if bundle["mode"] != "history":
        raise ValueError("Requires a deployable history policy")
    actor = bundle["actor"].eval()
    config = yaml.safe_load((checkpoint / "config.yaml").read_text())["air_hockey"]
    assert_canonical_scratch_env(dict(config, seed=0))
    # Only an episode budget changes, to exercise time-limit auto-reset.
    config.update(seed=seed, max_timesteps=73)
    env = HistoryEnvVector(lambda: AirHockeyEnv(copy.deepcopy(config)),
                           actor.history_len, actor.step_features)
    agent = HistoryAgent(actor)
    resets = terminal_windows = 0
    max_error = 0.0
    try:
        obs, _ = env.reset(seed=seed)
        agent.reset(obs[0, :actor.obs_dim])
        previous = np.zeros(actor.act_dim, dtype=np.float32)
        for _ in range(256):
            window = obs[0, actor.obs_dim:].astype(np.float32).reshape(
                actor.history_len, actor.feature_dim)
            np.testing.assert_array_equal(window, agent.history.window())
            policy_obs = torch.as_tensor(np.concatenate([obs[0], previous]), dtype=torch.float32)
            expected = actor.get_action(policy_obs.unsqueeze(0)).numpy().reshape(-1)
            actual = agent.act(obs[0, :actor.obs_dim], previous)
            max_error = max(max_error, float(np.max(np.abs(expected - actual))))
            np.testing.assert_array_equal(expected, actual)
            obs, _, _, _, info = env.step(actual[None])
            if "final_observation" in info:
                terminal = info["final_observation"][0]
                np.testing.assert_array_equal(
                    terminal[actor.obs_dim:].astype(np.float32).reshape(
                        actor.history_len, actor.feature_dim), agent.history.window())
                terminal_windows += 1
                agent.reset(obs[0, :actor.obs_dim])
                previous = np.zeros(actor.act_dim, dtype=np.float32)
                resets += 1
            else:
                previous = actual.copy()
        if resets < 2:
            raise AssertionError("Insufficient automatic reset coverage")
    finally:
        env.close()
    result = dict(checkpoint=str(checkpoint),
                  model_sha256=hashlib.sha256((checkpoint / "model.pth").read_bytes()).hexdigest(),
                  seed=seed, steps=256, diagnostic_episode_budget=73,
                  automatic_resets=resets, terminal_history_windows=terminal_windows,
                  history_len=actor.history_len, max_action_difference=max_error,
                  all_windows_and_actions_exact=True,
                  limitation="CPU eager comparison; short budgets test reset semantics, not survival quality")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.checkpoint, args.out)
