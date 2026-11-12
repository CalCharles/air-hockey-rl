"""Residual online fine-tuning of the RMA history_privileged policy (td3_online_real_finetune.py
``residual_finetune: true`` + ``update_schedule: sim``; helper/residual_finetune.py).

* the residual is a second PrivilegedHistoryActor: same architecture, random init, zeroed output
  layer, output scale residual_scale; the wrapped policy equals the sim policy at the start; only
  the residual is optimised; the critics are new;
* the "sim" schedule runs floor((carry + T) / 32) iterations of 8 critic + 2 actor updates;
* main() end to end (stub orchestrator driving the real PolicyRunner on Box2D): the base stays
  frozen, the residual learns, checkpoints carry the residual marker and evaluate through
  scripts.rma.bundle.load_phase1 / HistoryAgent with the learner's actions, and resume restores
  the residual learner exactly;
* CQL adds its penalty to the critic loss.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.td3.tests.test_history_privileged_obs import CHECKPOINT, REPO, _ckpt, _make_env, _spec

pytestmark = pytest.mark.skipif(not CHECKPOINT.exists(), reason="newsim cap7000 checkpoint not on this machine")

ARGS_FILE = REPO / "configs/td3/td3_online_real_finetune/newsim_cap7000_residual.yaml"
SCRATCH = Path(tempfile.gettempdir()) / "residual_online_finetune_tests"


def _args(**overrides):
    from scripts.td3 import td3_online_real_finetune as ft

    defaults, _, ignored = ft._build_finetune_args_file_defaults(str(ARGS_FILE))
    assert not ignored, f"args file keys ignored: {ignored}"
    args = ft.OnlineFinetuneArgs(**defaults)
    args.checkpoint = str(CHECKPOINT)
    args.model_path = args.checkpoint
    args.learner_device = "cpu"
    for key, value in overrides.items():
        setattr(args, key, value)
    ft._validate_args(args)
    return args


def _learner(args, checkpoint=None):
    """As main(): architecture (spec, train args) from the sim checkpoint, weights from ``checkpoint``."""
    from scripts.td3 import td3_online_real_finetune as ft

    sim_checkpoint = _ckpt()
    checkpoint = checkpoint or sim_checkpoint
    spec = _spec(sim_checkpoint)
    train_args, _ = ft._resolve_train_args(args, sim_checkpoint)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    state = ft._build_learner_from_checkpoint(
        args=args, train_args=train_args, checkpoint=checkpoint, obs_dim=spec.obs_dim, act_dim=2,
        action_low_np=-np.ones(2, np.float32), action_high_np=np.ones(2, np.float32),
        tb_log_dir=tempfile.mkdtemp(dir=SCRATCH), spec=spec,
    )
    return state, spec, train_args


def _fake_buffer(obs_dim, n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    batch = {
        "observations": torch.randn(n, obs_dim, generator=g), "next_observations": torch.randn(n, obs_dim, generator=g),
        "actions": torch.rand(n, 2, generator=g) * 2 - 1, "prev_actions": torch.rand(n, 2, generator=g) * 2 - 1,
        "rewards": torch.randn(n, generator=g), "dones": torch.zeros(n),
    }
    return SimpleNamespace(sample=lambda bs: {k: v[torch.randint(0, n, (bs,))] for k, v in batch.items()})


def test_residual_actor_structure_and_start_equals_base():
    from scripts.rma.networks import PrivilegedHistoryActor
    from scripts.td3.residual_agent import ResidualActor

    checkpoint = _ckpt()
    state, spec, _ = _learner(_args(), checkpoint)
    actor = state.actor
    assert isinstance(actor, ResidualActor)
    assert type(actor.base) is PrivilegedHistoryActor and type(actor.residual) is PrivilegedHistoryActor
    base_sd, res_sd = actor.base.state_dict(), actor.residual.state_dict()
    assert list(base_sd) == list(res_sd) and all(base_sd[k].shape == res_sd[k].shape for k in base_sd)
    # base = the sim actor, frozen
    for key, value in checkpoint["actor"].items():
        assert torch.equal(base_sd[key], value), key
    assert not any(p.requires_grad for p in actor.base.parameters())
    # residual: random init (not a copy of the base), output layer zero, output scale 0.15
    assert float(actor.residual.action_scale) == pytest.approx(0.15) and float(actor.base.action_scale) == 1.0
    assert torch.count_nonzero(actor.residual.actor_mean_head.weight) == 0
    assert torch.count_nonzero(actor.residual.actor_mean_head.bias) == 0
    others = [k for k in res_sd if res_sd[k].dim() > 1 and not k.startswith("actor_mean_head")]
    assert others and all(not torch.equal(res_sd[k], base_sd[k]) for k in others)
    # starts equal to the base policy
    pobs = torch.randn(128, actor.base.policy_obs_dim)
    with torch.no_grad():
        assert torch.equal(actor.get_action(pobs), torch.clamp(actor.base.get_action(pobs), -1, 1))
    # optimizer = residual only; target shares the frozen base and copies the residual
    opt_params = {id(p) for g in state.actor_optimizer.param_groups for p in g["params"]}
    assert opt_params == {id(p) for p in actor.residual.parameters()}
    assert state.actor_target.base is actor.base and state.actor_target.residual is not actor.residual
    # new critics: 2, same architecture as the sim critics, not the sim weights, fresh Adam
    assert len(state.qfs) == 2
    for i, q in enumerate(state.qfs, start=1):
        sim_q = checkpoint[f"qf{i}"]
        assert all(q.state_dict()[k].shape == sim_q[k].shape for k in sim_q)
        assert not all(torch.equal(q.state_dict()[k], sim_q[k]) for k in sim_q)
    assert not state.q_optimizer.state and not state.actor_optimizer.state


def test_sim_schedule_counts_and_frozen_base():
    """One kept episode of T=100: 3 iterations x (8 critic + 2 actor), carry 4; base unchanged."""
    from scripts.td3 import td3_online_real_finetune as ft

    args = _args(batch_size=64, actor_learning_starts=0)
    state, spec, train_args = _learner(args)
    base_before = {k: v.clone() for k, v in state.actor.base.state_dict().items()}
    res_before = {k: v.clone() for k, v in state.actor.residual.state_dict().items()}
    seed_dir = Path(tempfile.mkdtemp(dir=SCRATCH))
    progress = ft._OnlineProgress(seed_dir=seed_dir, experiment="t", hist_len=4, sim_seed=0)
    buffer = _fake_buffer(spec.obs_dim)
    step = ft._make_online_learner_step(progress, {}, make_buffer=lambda rb, dev: buffer, spec=None)
    replay = SimpleNamespace(len=lambda part: 256)
    args.checkpoint_every_online_episodes = 0
    for T, (iters, carry) in ((100, (3, 4)), (60, (2, 0))):
        stats = {"online_pending_episode_steps": T, "online_pending_episode_return": 1.0}
        q0, a0 = state.total_updates, state.total_actor_updates
        assert step(args, train_args, replay, stats, state)
        assert state.total_updates - q0 == 8 * iters and state.total_actor_updates - a0 == 2 * iters
        assert progress.schedule_carry_steps == carry
    assert all(torch.equal(base_before[k], v) for k, v in state.actor.base.state_dict().items())
    assert any(not torch.equal(res_before[k], v) for k, v in state.actor.residual.state_dict().items())
    rows = [json.loads(line) for line in open(seed_dir / ft._PROGRESS_FILE)]
    assert [r["sim_iterations"] for r in rows] == [3, 2]
    assert [r["critic_updates_K"] for r in rows] == [24, 16] and [r["actor_updates"] for r in rows] == [6, 4]
    # below the learning threshold (batch 64 > buffer): no update, nothing banked
    stats = {"online_pending_episode_steps": 50, "online_pending_episode_return": 1.0}
    q0 = state.total_updates
    assert not step(args, train_args, SimpleNamespace(len=lambda part: 10), stats, state)
    assert state.total_updates == q0 and progress.schedule_carry_steps == 0


def test_cql_adds_penalty():
    from scripts.td3 import td3_online_real_finetune as ft

    losses = {}
    for alpha in (0.0, 20.0):
        args = _args(cql_alpha=alpha, batch_size=32)
        torch.manual_seed(0)
        state, spec, train_args = _learner(args)
        torch.manual_seed(1)
        q_loss, _ = ft._critic_update(args, train_args, _fake_buffer(spec.obs_dim), state)
        assert torch.isfinite(q_loss)
        losses[alpha] = float(q_loss)
    assert losses[20.0] != losses[0.0]
    with pytest.raises(ValueError):
        _args(residual_finetune=False, cql_alpha=1.0)


def test_main_end_to_end_eval_bundle_and_resume():
    """main() with the stub orchestrator of test_history_privileged_obs (real PolicyRunner on Box2D)."""
    from collections import deque

    from scripts.rma.bundle import load_phase1
    from scripts.rma.evaluate import HistoryAgent
    from scripts.td3 import td3_online_real_finetune as ft
    from scripts.td3.helper import real_td3_runtime as rt
    from scripts.td3.helper.history_privileged_obs import HistoryPrivilegedObs
    from scripts.td3.helper.real_collector_factories import build_primitive_exploration_selector_for_real_collector
    from scripts.td3.helper.real_policy_runner import PolicyRunner
    from scripts.td3.helper.real_transition_hold import (
        RolloutContext,
        TransitionHoldState,
        normalize_transition_last_action_mode,
    )
    from scripts.td3.helper.residual_finetune import ResidualHistoryActor

    tmp = Path(tempfile.mkdtemp(dir=SCRATCH))
    args = _args(batch_size=64, actor_learning_starts=200)  # actor starts during episode 2 (150-step episodes)
    args.config = str(REPO / args.config)
    args.data_root_dir = str(tmp)
    args.num_online_episodes = 3
    args.learner_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint = ft._load_checkpoint(args.checkpoint)
    train_args, _ = ft._resolve_train_args(args, checkpoint)
    curve = ft._prepare_curve(args, checkpoint)
    rt._setup_run_data_dir(args, "residual stub orchestrator test")
    seen = {}

    def fake_collector(args_, train_args_, replay, stats, learner_state, obs_dim, act_dim, low, high, tb_dir, *,
                       add_episode_to_replay_fn, learner_step_fn, should_stop_fn, policy_env_wrapper,
                       collector_actor_factory, **_):
        device = torch.device("cpu")
        env = _make_env(150)
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
            traj = runner.run_episode().trajectory
            stats.update(last_episode_id=float(episodes), last_episode_juggles=0.0, last_episode_contacts=0.0,
                         last_episode_estop_flag=0.0)
            add_episode_to_replay_fn(replay=replay, episode_trajectory=traj,
                                     recent_episode_returns=deque(maxlen=10), success_top_fraction=0.0)
            if learner_step_fn(args=args_, train_args=train_args_, replay=replay, stats=stats, state=learner_state):
                actor.load_state_dict({k: v.detach().cpu() for k, v in learner_state.actor.state_dict().items()}, strict=True)
            episodes += 1
        seen["episodes"] = episodes
        seen["state"] = learner_state

    real_collector = ft.collector_process_modular
    ft.collector_process_modular = fake_collector
    try:
        end = ft.main(args, train_args, checkpoint, seed_dir=curve["seed_dir"], hist_len=curve["hist_len"],
                      sim_seed=curve["sim_seed"], quiet=False)
    finally:
        ft.collector_process_modular = real_collector
    assert end == "completed" and seen["episodes"] == 3
    state = seen["state"]
    rows = [json.loads(line) for line in open(curve["seed_dir"] / ft._PROGRESS_FILE)]
    trained = [r for r in rows if r["trained"]]
    assert trained and all(r["critic_updates_K"] == 8 * r["sim_iterations"] for r in trained)
    # critic-only until 200 transitions: none in round 1, part of round 2, all of round 3
    assert trained[0]["actor_updates"] == 0 and 0 < trained[1]["actor_updates"] < 2 * trained[1]["sim_iterations"]
    assert trained[-1]["actor_updates"] == 2 * trained[-1]["sim_iterations"]
    # base frozen at the sim weights; residual moved off zero
    for key, value in checkpoint["actor"].items():
        assert torch.equal(state.actor.base.state_dict()[key].cpu(), value), key
    assert torch.count_nonzero(state.actor.residual.actor_mean_head.weight) > 0

    # latest per-round checkpoint: residual marker + eval loader reproduces the learner's actions
    ckpt_dir = sorted(Path(args.checkpoint_root_dir).glob("checkpoint_ep*"))[-1]
    meta = json.loads((ckpt_dir / "rma_meta.json").read_text())
    assert meta["residual"]["scale"] == pytest.approx(0.15)
    bundle = load_phase1(str(ckpt_dir))
    assert isinstance(bundle["actor"], ResidualHistoryActor)
    saved = torch.load(ckpt_dir / "training_state.pth", map_location="cpu", weights_only=False)
    learner = ft.make_actor_factory(args=args, train_args=train_args, spec=_spec(checkpoint), obs_dim=438, act_dim=2,
                                    action_low_np=-np.ones(2, np.float32), action_high_np=np.ones(2, np.float32))("cpu")
    learner.load_state_dict(saved["actor"], strict=True)
    agent = HistoryAgent(bundle["actor"])
    spec = _spec(checkpoint)
    env = _make_env(200)
    builder = HistoryPrivilegedObs.for_env(spec, env)
    raw, _ = env.reset(seed=5)
    aug = builder.start(raw)
    agent.reset(raw)
    last = np.zeros(2, np.float32)
    max_diff = 0.0
    for _ in range(120):
        a_eval = agent.act(raw.astype(np.float32), last)
        with torch.no_grad():
            a_learn = learner.get_action(torch.as_tensor(np.concatenate([aug, last]), dtype=torch.float32)[None])[0].numpy()
        max_diff = max(max_diff, float(np.abs(a_eval - a_learn).max()))
        raw, r, term, trunc, info = env.step(a_eval)
        aug, _, term, _ = builder.step(raw, a_eval, r, term, info)
        last = a_eval
        if term or trunc:
            break
    assert max_diff < 1e-5

    # resume: the residual learner is restored exactly from the online checkpoint
    resumed, _, _ = _learner(_args(batch_size=64), saved)
    for a, b in ((resumed.actor, saved["actor"]), (resumed.actor_target, saved["actor_target"])):
        assert all(torch.equal(a.state_dict()[k].cpu(), v.cpu()) for k, v in b.items())
    for i, q in enumerate(resumed.qfs, start=1):
        assert all(torch.equal(q.state_dict()[k].cpu(), v.cpu()) for k, v in saved[f"qf{i}"].items())
    assert resumed.actor_optimizer.state and resumed.q_optimizer.state
    online_state = json.loads((ckpt_dir / ft._ONLINE_STATE_FILE).read_text())
    assert "schedule_carry_steps" in online_state


def test_critic_only_warmup_actor_learning_starts():
    """No residual-actor updates before actor_learning_starts collected transitions; the critic
    trains throughout; the actor starts at the first iteration past the count (sim semantics)."""
    from scripts.td3 import td3_online_real_finetune as ft

    args = _args(batch_size=64, checkpoint_every_online_episodes=0)
    assert args.actor_learning_starts == 2000 and args.cql_alpha == 0.0 and args.actor_loss_normalized
    state, spec, train_args = _learner(args)
    progress = ft._OnlineProgress(seed_dir=Path(tempfile.mkdtemp(dir=SCRATCH)), experiment="t", hist_len=4, sim_seed=0)
    step = ft._make_online_learner_step(progress, {}, make_buffer=lambda rb, dev: _fake_buffer(spec.obs_dim), spec=None)
    replay = SimpleNamespace(len=lambda part: 4096)
    res_before = {k: v.clone() for k, v in state.actor.residual.state_dict().items()}

    def episode(T):
        q0, a0 = state.total_updates, state.total_actor_updates
        step(args, train_args, replay, {"online_pending_episode_steps": T, "online_pending_episode_return": 0.0}, state)
        return state.total_updates - q0, state.total_actor_updates - a0

    # episode 1: 1000 steps -> 31 iterations, carry 8, collected 1000: critic only
    assert episode(1000) == (248, 0)
    assert all(torch.equal(res_before[k], v) for k, v in state.actor.residual.state_dict().items())
    # episode 2: (8 + 1000) // 32 = 31 iterations ending at 992 + 32 (i + 1) <= 1984 < 2000: critic only
    assert episode(1000) == (248, 0) and progress.actor_enabled_at is None
    # episode 3: (16 + 1000) // 32 = 31 iterations ending at 1984 + 32 (i + 1) >= 2016: actor on from iteration 0
    assert episode(1000) == (248, 62)
    assert progress.actor_enabled_at["transitions_collected"] == 2016
    assert progress.actor_enabled_at["critic_updates_before"] == 2 * 248 + 8
    assert any(not torch.equal(res_before[k], v) for k, v in state.actor.residual.state_dict().items())
    assert progress.online_state(None)["transitions_collected"] == 3000

    # mid-round start: threshold inside an episode
    args2 = _args(batch_size=64, checkpoint_every_online_episodes=0, actor_learning_starts=500)
    state2, _, _ = _learner(args2)
    progress2 = ft._OnlineProgress(seed_dir=Path(tempfile.mkdtemp(dir=SCRATCH)), experiment="t", hist_len=4, sim_seed=0)
    step2 = ft._make_online_learner_step(progress2, {}, make_buffer=lambda rb, dev: _fake_buffer(spec.obs_dim), spec=None)
    step2(args2, train_args, replay, {"online_pending_episode_steps": 1000, "online_pending_episode_return": 0.0}, state2)
    # iterations end at 32, 64, ..., 992; those >= 500 are 512 .. 992 -> 16 iterations x 2 actor updates
    assert state2.total_updates == 248 and state2.total_actor_updates == 32


def test_actor_loss_normalized_is_sim_form():
    from scripts.td3 import td3_online_real_finetune as ft

    losses = {}
    for normalized in (False, True):
        args = _args(batch_size=32, actor_loss_normalized=normalized)
        torch.manual_seed(0)
        state, spec, train_args = _learner(args)
        torch.manual_seed(1)
        loss, norm_q = ft._actor_update(args, train_args, _fake_buffer(spec.obs_dim), state)
        losses[normalized] = (float(loss), float(norm_q))
    assert losses[True][0] == pytest.approx((1 - 0.995) * losses[False][0], rel=1e-5)
    assert losses[True][0] == pytest.approx(-losses[True][1], rel=1e-5)
