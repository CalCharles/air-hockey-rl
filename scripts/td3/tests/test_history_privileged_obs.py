"""Online fine-tuning of the RMA history_privileged policies (helper/history_privileged_obs.py).

* the refactored ``motion_shaping.shaping_step`` reproduces the original inline shaping;
* the builder's [x_t | window] equals the sim trainer's ``PrivilegedHistoryEnvVector`` and its
  shaping (from observed positions) matches the sim's (true state) on the same rollout;
* a ``history_privileged`` checkpoint loads into the online learner (438-d critic obs) and the
  learner actor gives the same actions as the deployed eval agent.
"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
CHECKPOINT = REPO / "data/newsim_scratch_cap7000_swa/seed0/training_state.pth"
SIM_CONFIG = REPO / "data/newsim_scratch_cap7000_swa/seed0/config.yaml"

pytestmark = pytest.mark.skipif(not CHECKPOINT.exists(), reason="newsim cap7000 checkpoint not on this machine")


def _air_hockey_cfg(max_timesteps=None):
    cfg = yaml.safe_load(open(SIM_CONFIG))["air_hockey"]
    if max_timesteps is not None:
        cfg["max_timesteps"] = int(max_timesteps)
    return cfg


def _make_env(max_timesteps):
    """The training sim exactly as scripts/rma/train_base_policy.py builds it (2-step action latency)."""
    from airhockey import AirHockeyEnv
    from scripts.rma.action_latency import install_action_latency

    return install_action_latency(AirHockeyEnv(copy.deepcopy(_air_hockey_cfg(max_timesteps))), 2)


def _ckpt():
    return torch.load(CHECKPOINT, map_location="cpu", weights_only=False)


def _spec(checkpoint=None):
    from scripts.td3.helper.history_privileged_obs import history_privileged_spec

    return history_privileged_spec(checkpoint or _ckpt(), str(CHECKPOINT), 30, 2)


def _eval_agent(spec):
    """The deployed policy: scripts/rma/evaluate.HistoryAgent on model.pth (a plain HistoryActor)."""
    from scripts.rma.evaluate import HistoryAgent
    from scripts.rma.networks import HistoryActor

    actor = HistoryActor(obs_dim=30, act_dim=2, history_len=spec.history_len, feature_dim=spec.feature_dim,
                         latent_dim=spec.latent_dim, use_last_action=True, hidden_layer_size=spec.hidden_layer_size,
                         num_hidden_layers=spec.num_hidden_layers, embed_dim=spec.embed_dim,
                         conv_channels=spec.conv_channels)
    actor.load_state_dict(torch.load(CHECKPOINT.parent / "model.pth", map_location="cpu"))
    return HistoryAgent(actor, spec.step_features)


def _original_install_motion_shaping():
    """``install_motion_shaping`` as committed before the shaping_step refactor."""
    source = subprocess.run(
        ["git", "show", "c986c73:scripts/rma/motion_shaping.py"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    namespace: dict = {}
    exec(compile(source, "motion_shaping_original", "exec"), namespace)
    return namespace["install_motion_shaping"], namespace["MotionShaping"]


def _rollout(env, steps, seed, actor_fn):
    obs, _ = env.reset(seed=seed)
    out = []
    for t in range(steps):
        action = actor_fn(obs, t)
        obs, reward, term, trunc, info = env.step(action)
        out.append((obs.copy(), float(reward), bool(term), float(info.get("motion_shaping", 0.0))))
        if term or trunc:
            break
    return out


def test_shaping_step_matches_original_inline_shaping():
    from airhockey import AirHockeyEnv
    from scripts.rma.motion_shaping import install_motion_shaping

    original_install, OriginalShaping = _original_install_motion_shaping()
    spec = _spec()
    shaping_old = OriginalShaping.from_args(SimpleNamespace(**_ckpt()["args"]))
    contacts = nonzero = 0
    for seed in range(3):
        env_new = install_motion_shaping(_make_env(1500), spec.shaping)
        env_old = original_install(_make_env(1500), shaping_old)
        obs, _ = env_new.reset(seed=100 + seed)
        obs_old, _ = env_old.reset(seed=100 + seed)
        agent = _eval_agent(spec)
        agent.reset(obs)
        last = np.zeros(2, np.float32)
        for _ in range(1500):
            a = agent.act(obs.astype(np.float32), last)
            obs, r1, d1, t1, i1 = env_new.step(a)
            obs_old, r2, d2, t2, i2 = env_old.step(a)
            np.testing.assert_allclose(obs, obs_old)
            assert r1 == pytest.approx(r2) and d1 == d2 and i1["motion_shaping"] == pytest.approx(i2["motion_shaping"])
            contacts += int(i1.get("paddle_puck_collision_count", 0) > 0)
            nonzero += int(i1["motion_shaping"] != 0)
            last = a
            if d1 or t1:
                break
    assert contacts > 20 and nonzero > 20  # the contact branches were exercised


def test_builder_matches_sim_trainer_window_and_tracks_true_state():
    from airhockey import AirHockeyEnv
    from scripts.rma.env_wrapper import PrivilegedHistoryEnvVector, true_state_features
    from scripts.rma.motion_shaping import install_motion_shaping
    from scripts.td3.helper.history_privileged_obs import HistoryPrivilegedObs, HistoryPrivilegedPolicyEnv

    spec = _spec()
    n_x = 30 + spec.window_dim
    seed, steps = 7, 1500

    # sim trainer: PrivilegedHistoryEnvVector over the shaped env (true-state shaping)
    vec = PrivilegedHistoryEnvVector(
        lambda: install_motion_shaping(_make_env(steps), spec.shaping),
        spec.history_len, spec.step_features,
    )
    # sim trainer reward, unwrapped (keeps info["motion_shaping"])
    shaped_env = install_motion_shaping(_make_env(steps), spec.shaping)
    # robot view: plain env + builder (observed-position state and shaping)
    raw_env = _make_env(steps)
    policy_env = HistoryPrivilegedPolicyEnv(raw_env, HistoryPrivilegedObs.for_env(spec, raw_env))

    o_vec, _ = vec.reset(seed=seed)
    raw_true, _ = shaped_env.reset(seed=seed)
    o_pol = policy_env.begin_episode(raw_env.reset(seed=seed)[0])
    np.testing.assert_allclose(o_vec[0][:n_x], o_pol[:n_x], atol=1e-6)
    agent = _eval_agent(spec)
    agent.reset(raw_true)
    last = np.zeros(2, np.float32)
    pos_err, vel_err, bonus_true, bonus_obs, contact = [], [], [], [], []
    for _ in range(steps):
        a = agent.act(raw_true.astype(np.float32), last)
        last = a
        o_vec, _, _, _, info_vec = vec.step(a[None])
        raw_true, r_true, term_true, _, info_true = shaped_env.step(a)
        o_pol, r_pol, term_pol, trunc_pol, info_pol = policy_env.step(a)
        if "final_observation" in info_vec:
            break
        # identical x_t + window (same seed -> same physics and observation noise)
        np.testing.assert_allclose(o_vec[0][:n_x], o_pol[:n_x], atol=1e-5)
        truth = true_state_features(shaped_env)
        pos_err.append(np.abs(truth[4:6] - o_pol[n_x + 4:n_x + 6]).max())   # paddle position
        vel_err.append(np.abs(truth[6:8] - o_pol[n_x + 6:n_x + 8]).max())   # paddle velocity
        bonus_true.append(float(info_true["motion_shaping"]))
        bonus_obs.append(float(info_pol["motion_shaping"]))
        contact.append(int(info_true.get("paddle_puck_collision_count", 0)) > 0)
        # same task reward underneath both
        assert info_pol["task_reward"] == pytest.approx(r_true - info_true["motion_shaping"], abs=1e-5)
        if term_true or term_pol or trunc_pol:
            break
    assert len(pos_err) > 50
    # observed paddle position = true position (no paddle obs noise in the sim)
    assert float(np.max(pos_err)) < 1e-4
    # finite-difference paddle velocity tracks the true velocity (one-step lag)
    assert float(np.median(vel_err)) < 0.5
    bonus_true, bonus_obs, contact = map(np.asarray, (bonus_true, bonus_obs, contact))
    assert contact.sum() > 10
    # shaping from observed positions agrees with the true-state shaping on most steps,
    # contact steps included (weak strike uses the paddle speed, pout the puck speed)
    assert np.mean(np.isclose(bonus_true, bonus_obs, atol=0.05)) > 0.9
    print(f"[builder] {len(bonus_true)} steps, {int(contact.sum())} contacts; shaping agrees on "
          f"{np.mean(np.isclose(bonus_true, bonus_obs, atol=0.05)):.3f} of steps, "
          f"{np.mean(np.isclose(bonus_true[contact], bonus_obs[contact], atol=0.05)):.3f} of contact steps; "
          f"episode return true {bonus_true.sum():.1f} vs observed {bonus_obs.sum():.1f} (shaping part)")


def test_checkpoint_loads_into_learner_and_matches_eval_agent():
    from scripts.td3 import td3_online_real_finetune as ft
    from scripts.td3.helper.history_privileged_obs import HistoryPrivilegedObs

    checkpoint = _ckpt()
    spec = _spec(checkpoint)
    assert spec.obs_dim == 438 and spec.gamma == pytest.approx(0.995)
    args = ft.OnlineFinetuneArgs(checkpoint=str(CHECKPOINT), learner_device="cpu", gamma=0.995, policy_lr=1e-5, q_lr=1e-3)
    train_args, _ = ft._resolve_train_args(args, checkpoint)
    state = ft._build_learner_from_checkpoint(
        args=args, train_args=train_args, checkpoint=checkpoint, obs_dim=spec.obs_dim, act_dim=2,
        action_low_np=-np.ones(2, np.float32), action_high_np=np.ones(2, np.float32),
        tb_log_dir="/tmp/claude-1000/-home-pearl-rohan-air-hockey-rl/bb344814-a83c-4c52-abf5-dbe2cb859c17/scratchpad/test_tb",
        spec=spec,
    )
    assert state.actor_optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)
    assert state.q_optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)

    agent = _eval_agent(spec)

    from airhockey import AirHockeyEnv

    env = _make_env(300)
    builder = HistoryPrivilegedObs.for_env(spec, env)
    raw, _ = env.reset(seed=3)
    aug = builder.start(raw)
    agent.reset(raw)
    last = np.zeros(2, np.float32)
    collector = spec.build_actor("cpu")
    collector.load_state_dict(state.actor.state_dict(), strict=True)
    for t in range(150):
        a_eval = agent.act(raw.astype(np.float32), last)
        with torch.no_grad():
            a_learn = collector.get_action(torch.as_tensor(np.concatenate([aug, last]), dtype=torch.float32)[None])[0].numpy()
        np.testing.assert_allclose(a_learn, a_eval, atol=1e-5)
        raw, r, term, trunc, info = env.step(a_eval)
        aug, _, term, _ = builder.step(raw, a_eval, r, term, info)
        last = a_eval
        if term or trunc:
            break

    # one critic + one actor update on a fake batch run and change the networks
    n = 64
    batch = {
        "observations": torch.randn(n, spec.obs_dim), "next_observations": torch.randn(n, spec.obs_dim),
        "actions": torch.rand(n, 2) * 2 - 1, "prev_actions": torch.rand(n, 2) * 2 - 1,
        "rewards": torch.randn(n), "dones": torch.zeros(n),
    }
    buffer = SimpleNamespace(sample=lambda bs: {k: v[torch.randint(0, n, (bs,))] for k, v in batch.items()})
    args.batch_size = 32
    before = [p.clone() for p in state.actor.parameters()]
    q_loss, _ = ft._critic_update(args, train_args, buffer, state)
    actor_loss, _ = ft._actor_update(args, train_args, buffer, state)
    assert torch.isfinite(q_loss) and torch.isfinite(actor_loss)
    assert any(not torch.equal(b, p) for b, p in zip(before, state.actor.parameters()))


def test_real_main_wiring_with_stub_orchestrator(tmp_path=None):
    """td3_online_real_finetune.main() end to end with the robot orchestrator replaced by a stub
    that drives the real PolicyRunner on a Box2D env through the hooks main() passes
    (policy_env_wrapper, collector_actor_factory, replay push, learner step)."""
    import tempfile
    from collections import deque

    from scripts.td3 import td3_online_real_finetune as ft
    from scripts.td3.helper import real_td3_runtime as rt
    from scripts.td3.helper.real_collector_factories import build_primitive_exploration_selector_for_real_collector
    from scripts.td3.helper.real_policy_runner import PolicyRunner
    from scripts.td3.helper.real_transition_hold import (
        RolloutContext,
        TransitionHoldState,
        normalize_transition_last_action_mode,
    )

    tmp = Path(tmp_path or tempfile.mkdtemp(dir="/tmp/claude-1000/-home-pearl-rohan-air-hockey-rl/bb344814-a83c-4c52-abf5-dbe2cb859c17/scratchpad"))
    defaults, _, _ = ft._build_finetune_args_file_defaults(str(REPO / "configs/td3/td3_online_real_finetune/newsim_cap7000.yaml"))
    args = ft.OnlineFinetuneArgs(**defaults)
    args.checkpoint = str(CHECKPOINT)
    args.model_path = args.checkpoint
    args.config = str(REPO / args.config)
    args.data_root_dir = str(tmp)
    args.num_online_episodes = 2
    args.learner_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.utd_ratio = 0.5  # keep the test short; the recipe value is 5
    checkpoint = ft._load_checkpoint(args.checkpoint)
    train_args, _ = ft._resolve_train_args(args, checkpoint)
    curve = ft._prepare_curve(args, checkpoint)
    rt._setup_run_data_dir(args, "stub orchestrator test")
    seen = {}

    def fake_collector(args_, train_args_, replay, stats, learner_state, obs_dim, act_dim, low, high, tb_dir, *,
                       add_episode_to_replay_fn, learner_step_fn, should_stop_fn, policy_env_wrapper,
                       collector_actor_factory, **_):
        assert obs_dim == 438 and replay.state_dict()["success"]["observations"].shape[1] == 438
        device = torch.device("cpu")
        env = _make_env(150)  # Box2D stand-in for the robot env (no robot I/O)
        policy_env = policy_env_wrapper(env)
        actor = collector_actor_factory(device)
        actor.load_state_dict({k: v.detach().cpu() for k, v in learner_state.actor.state_dict().items()}, strict=True)
        runner = PolicyRunner(
            policy_env, actor, device=device, args=args_, train_args=train_args_,
            action_low=torch.as_tensor(low)[None], action_high=torch.as_tensor(high)[None],
            primitive_selector=build_primitive_exploration_selector_for_real_collector(args_, device, initial_total_steps=0),
            transition_hold=TransitionHoldState(last_action_mode=normalize_transition_last_action_mode(args_.transition_last_action_mode)),
            ctx=RolloutContext(last_action_for_policy=torch.zeros((1, act_dim)), last_executed_action=torch.zeros((1, act_dim))),
            reset_primitive_rollout_state=rt._reset_primitive_rollout_state,
            deterministic_actor_action=rt.deterministic_actor_action,
            augment_policy_observation=rt.augment_policy_observation,
            primitive_exploration_chance_for_step=rt.primitive_exploration_chance_for_step,
            latest_camera_frame=lambda e: None, env_timing_info=lambda e: {}, safe_nonnegative_ms=rt._safe_nonnegative_ms,
            build_split_episode_row=lambda **kw: {}, readiness_fn=lambda e: (True, "ok"),
        )
        episodes = 0
        while should_stop_fn(stats) is None:
            raw, _ = env.reset(seed=11 + episodes)
            runner.seed_after_reset(raw)
            runner.set_artifact_episode_id(episodes)
            result = runner.run_episode()
            traj = result.trajectory
            assert traj.observations[0].shape[0] == 438
            stats.update(last_episode_id=float(episodes), last_episode_juggles=0.0, last_episode_contacts=0.0,
                         last_episode_estop_flag=0.0)
            # keyword calls, exactly as async_td3_real.collector_process_modular makes them
            add_episode_to_replay_fn(replay=replay, episode_trajectory=traj,
                                     recent_episode_returns=deque(maxlen=10), success_top_fraction=0.0)
            changed = learner_step_fn(args=args_, train_args=train_args_, replay=replay, stats=stats,
                                      state=learner_state)
            assert changed
            actor.load_state_dict({k: v.detach().cpu() for k, v in learner_state.actor.state_dict().items()}, strict=True)
            episodes += 1
        seen["episodes"] = episodes

    real_collector = ft.collector_process_modular
    ft.collector_process_modular = fake_collector
    try:
        end = ft.main(args, train_args, checkpoint, seed_dir=curve["seed_dir"], hist_len=curve["hist_len"],
                      sim_seed=curve["sim_seed"], quiet=False)
    finally:
        ft.collector_process_modular = real_collector
    assert end == "completed" and seen["episodes"] == 2
    rows = [json.loads(line) for line in open(curve["seed_dir"] / ft._PROGRESS_FILE)]
    assert len(rows) == 2 and all(r["trained"] for r in rows)
    assert all("episode_task_return" in r and "episode_shaping_return" in r for r in rows)
    ckpts = sorted(Path(args.checkpoint_root_dir).glob("checkpoint_ep*"))
    assert ckpts and all((c / "rma_meta.json").exists() and (c / "training_state.pth").exists() for c in ckpts)
