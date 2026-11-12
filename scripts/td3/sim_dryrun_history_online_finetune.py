"""Sim dry run of the real-robot online fine-tuning of a history_privileged policy (no robot).

Runs the data-recycling recipe of ``scripts/td3/td3_online_real_finetune.py`` (``--num-trials``)
on the Box2D training sim of the policy, with the real script's own learner pieces:

* ``_build_learner_from_checkpoint`` (in-place load of actor / critics / targets / Adam state,
  LRs re-applied), ``_make_online_replay_push``, ``_make_online_learner_step`` (K = round(η·T)
  critic updates, one actor update per M, per-round checkpoint + ``rma_meta.json``),
  ``_MixedReplaySnapshot`` / ``_recycle_alpha`` / ``_save_trial_buffer`` / ``_load_recycle_d0``
  (trial t ≥ 1 samples (1 − α)·D0 + α·D_online, α 0.5 → 1 over the trial);
* ``HistoryPrivilegedPolicyEnv`` (helper/history_privileged_obs.py): the same critic observation
  [x_t | window | s_t], shaped reward and edge termination as on the robot, built from the
  observed positions — not from the simulator's true state;
* the real args file (``configs/td3/td3_online_real_finetune/newsim_cap7000.yaml``), with the
  robot-only fields replaced: the env is the policy's training sim (``config.yaml`` next to the
  checkpoint) with the training action latency (2 control steps) and the robot episode cap
  (``--max-timesteps`` 10,000), and the robot's termination overrides
  (``terminate_on_puck_pass_paddle`` off, as in the real script).

Only the collector differs from the robot: a plain Box2D episode loop (exploration noise, clamp,
last action, done = termination) instead of the robot orchestrator (no resets / holds / e-stops).
The output layout and TensorBoard tags are the real script's:
``<out>/<experiment>/hist<H>/seed<S>/trial<t>/online_tb`` etc.

    .venv/bin/python -m scripts.td3.sim_dryrun_history_online_finetune \\
        --checkpoint data/newsim_scratch_cap7000_swa/seed0/training_state.pth
    tensorboard --logdir runs/td3/online_finetune_dryrun/newsim_cap7000_recycle
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from airhockey import AirHockeyEnv  # noqa: E402
from scripts.rma.action_latency import install_action_latency  # noqa: E402
from scripts.td3 import td3_online_real_finetune as ft  # noqa: E402
from scripts.td3.helper.history_privileged_obs import (  # noqa: E402
    HistoryPrivilegedObs,
    HistoryPrivilegedPolicyEnv,
    history_privileged_spec,
)
from scripts.td3.helper.juggle_counter import count_juggles_from_arrays  # noqa: E402
from scripts.td3.helper.shared_replay import SharedTD3Replay  # noqa: E402
from scripts.td3.helper.td3_episode_collection import EpisodeTrajectory  # noqa: E402


def _make_env(sim_config: Path, max_timesteps: int, latency: int, seed: int, overrides: dict):
    cfg = copy.deepcopy(yaml.safe_load(open(sim_config))["air_hockey"])
    cfg["max_timesteps"] = int(max_timesteps)
    cfg["seed"] = int(seed)
    cfg.update(overrides)
    env = AirHockeyEnv(cfg)
    if env.simulator_name != "box2d":
        raise SystemExit(f"the dry run only runs on the Box2D simulator, got {env.simulator_name!r}")
    return install_action_latency(env, int(latency))


def _juggles(raw_obs_seq) -> int:
    obs = np.asarray(raw_obs_seq)
    return int(count_juggles_from_arrays(obs[:, 12:14], obs[:, 27:29], obs[:, 29]).n_juggles)


def run_trial(cli, trial: int, finished_dirs: list[Path], checkpoint: dict, base_args: ft.OnlineFinetuneArgs) -> Path:
    args = copy.deepcopy(base_args)
    train_args, _ = ft._resolve_train_args(args, checkpoint)
    seed_dir = Path(cli.out).resolve() / cli.experiment_name / f"hist{cli.hist_len}" / f"seed{cli.sim_seed}" / f"trial{trial}"
    if (seed_dir / ft._PROGRESS_FILE).exists():
        moved = seed_dir.with_name(f"{seed_dir.name}_old_{time.strftime('%Y%m%d-%H%M%S')}")
        seed_dir.rename(moved)
        print(f"[dryrun] moved the previous trial{trial} run to {moved}", flush=True)
    seed_dir.mkdir(parents=True, exist_ok=True)
    args.data_root_dir = str(seed_dir)
    args.experiment_name = cli.experiment_name
    # the real script's per-launch folder (data_<timestamp>/: checkpoints, learner_tb, run_events.jsonl)
    run_dir = ft._setup_run_data_dir(args, "sim dry run (no robot)")

    env = _make_env(cli.sim_config, cli.max_timesteps, cli.action_latency, 1000 * trial + cli.sim_seed,
                    {"terminate_on_puck_pass_paddle": bool(args.terminate_on_puck_pass_paddle)})
    raw_obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    low = np.asarray(env.action_space.low, np.float32)
    high = np.asarray(env.action_space.high, np.float32)
    spec = history_privileged_spec(checkpoint, args.checkpoint, raw_obs_dim, act_dim)
    if spec is None:
        raise SystemExit("the dry run is for history_privileged checkpoints")
    ft._check_history_privileged_args(args, spec)
    builder = HistoryPrivilegedObs.for_env(spec, env)
    policy_env = HistoryPrivilegedPolicyEnv(env, builder)

    replay = SharedTD3Replay(success_capacity=int(args.online_buffer_size), failure_capacity=1,
                             obs_shape=(spec.obs_dim,), action_shape=(act_dim,))
    state = ft._build_learner_from_checkpoint(
        args=args, train_args=train_args, checkpoint=checkpoint, obs_dim=spec.obs_dim, act_dim=act_dim,
        action_low_np=low, action_high_np=high, tb_log_dir=str(run_dir / "learner_tb"), spec=spec,
    )
    progress = ft._OnlineProgress(seed_dir=seed_dir, experiment=cli.experiment_name, hist_len=cli.hist_len,
                                  sim_seed=cli.sim_seed)
    target = int(args.num_online_episodes)
    anneal_rounds = int(args.recycle_anneal_episodes) or max(1, target - int(args.warm_start_episodes))
    d0, d0_sources = ft._load_recycle_d0(finished_dirs, args.learner_device)
    d0_size = 0 if d0 is None else int(d0["rewards"].shape[0])
    with open(seed_dir / ft._TRIAL_INFO_FILE, "w") as f:
        json.dump({"trial": trial, "num_trials": cli.num_trials, "episodes_per_trial": target,
                   "sim_checkpoint": str(args.checkpoint), "d0_transitions": d0_size, "d0_sources": d0_sources,
                   "alpha_start": float(args.recycle_alpha_start) if d0 is not None else 1.0,
                   "anneal_rounds": anneal_rounds if d0 is not None else 0, "dry_run": True,
                   "sim_config": str(cli.sim_config), "action_latency": cli.action_latency,
                   "max_timesteps": cli.max_timesteps}, f, indent=2)
    make_buffer = None
    if d0 is not None:
        def make_buffer(rb, device):
            rounds = progress.episodes_trained
            alpha = ft._recycle_alpha(rounds, float(args.recycle_alpha_start), anneal_rounds)
            progress.writer.add_scalar("online_finetune_recycle/alpha", alpha, rounds + 1)
            progress.writer.add_scalar("online_finetune_recycle/d0_size", float(d0_size), rounds + 1)
            return ft._MixedReplaySnapshot(ft._DeviceReplaySnapshot(rb, device), d0, alpha)
    print(f"\n[dryrun] ===== trial {trial}/{cli.num_trials - 1}: "
          + (f"D0 = {d0_size} transitions from {len(d0_sources)} earlier trial(s), α {args.recycle_alpha_start:g} -> 1 "
             f"over {anneal_rounds} rounds" if d0 is not None else "online data only") + f" -> {seed_dir} =====", flush=True)

    stats: dict = {}
    pending: dict = {}
    push = ft._make_online_replay_push(stats, args, pending)
    learner_step = ft._make_online_learner_step(progress, pending, make_buffer=make_buffer, spec=spec)
    # Same structure as the learner actor (a base + residual wrapper with residual_finetune).
    collector = ft.make_actor_factory(args=args, train_args=train_args, spec=spec, obs_dim=spec.obs_dim, act_dim=act_dim,
                                      action_low_np=low, action_high_np=high)("cpu").eval()
    collector.load_state_dict({k: v.detach().cpu() for k, v in state.actor.state_dict().items()}, strict=True)
    rng = torch.Generator().manual_seed(1_000_003 * trial + cli.sim_seed)
    recent = deque(maxlen=500)
    attempts = 0
    while progress.launch_kept_episodes < target:
        attempts += 1
        raw, _ = env.reset(seed=500_000 + 10_000 * trial + 100 * cli.sim_seed + attempts)
        obs = policy_env.begin_episode(raw)
        last = torch.zeros(act_dim)
        traj = EpisodeTrajectory.empty()
        raw_seq, contacts, info = [raw], 0, {}
        t_collect = time.time()
        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action = collector.get_action(torch.cat([obs_t, last]).unsqueeze(0))[0]
                if args.exploration_noise > 0:
                    action = action + torch.randn(action.shape, generator=rng) * float(args.exploration_noise)
                action = torch.clamp(action, torch.as_tensor(low), torch.as_tensor(high))
            next_obs, reward, term, trunc, info = policy_env.step(action.numpy())
            contacts += int(int(info.get("paddle_puck_collision_count", 0) or 0) > 0)
            traj.append_step(obs=obs_t, next_obs=torch.as_tensor(next_obs, dtype=torch.float32), action=action,
                             reward=torch.tensor(float(reward)), done=torch.tensor(float(term)), prev_action=last)
            raw_seq.append(np.asarray(next_obs[:raw_obs_dim]))
            obs, last = next_obs, action.clone()
            if term or trunc:
                break
        collect_s = time.time() - t_collect
        if len(traj.observations) < int(args.min_episode_steps):
            print(f"[online] discarded attempt ({len(traj.observations)} < {args.min_episode_steps} steps)", flush=True)
            continue
        stats.update(last_episode_id=float(attempts), last_episode_juggles=float(_juggles(raw_seq)),
                     last_episode_contacts=float(contacts), last_episode_estop_flag=0.0,
                     collector_total_steps=float(stats.get("collector_total_steps", 0.0)) + len(traj.observations))
        stats.update(builder.episode_stats())
        task_return, shaping_return = builder.episode_task_return, builder.episode_shaping_return
        push(replay, traj, recent, 0.0)
        changed = learner_step(args, train_args, replay, stats, state)
        report = stats.pop("online_episode_report", {}) or {}
        if changed:
            collector.load_state_dict({k: v.detach().cpu() for k, v in state.actor.state_dict().items()}, strict=True)
        end = info.get("episode_end_reason") or ("truncated" if trunc else "terminated")
        detail = (f"round {report['round']}: K={report['K']}, actor updates {report['actor_updates']}, "
                  f"q_loss {report['q_loss']:.4f}, actor_loss {report['actor_loss']:.4f} ({report['learner_s']:.0f}s)"
                  if report.get("kind") == "trained" else str(report.get("kind")))
        print(f"[online] trial {trial} ep {progress.launch_kept_episodes}/{target} | return {traj.episode_return:.1f} "
              f"(task {task_return:.1f}, shaping {shaping_return:.1f}) | len {len(traj.observations)} | "
              f"contacts {contacts} | end {end} | collect {collect_s:.0f}s | {detail} | buffer {replay.len(ft._ONLINE_PARTITION)}",
              flush=True)
    saved = ft._save_trial_buffer(seed_dir, replay)
    print(f"[dryrun] trial {trial}: online buffer ({saved} transitions) -> {seed_dir / ft._RECYCLE_BUFFER_FILE}", flush=True)
    progress.close()
    state.writer.close()
    env.close()
    del d0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return seed_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--args-file", default="configs/td3/td3_online_real_finetune/newsim_cap7000.yaml")
    ap.add_argument("--sim-config", default=None, help="default: config.yaml next to the checkpoint (the training sim)")
    ap.add_argument("--action-latency", type=int, default=2, help="control steps (the training value)")
    ap.add_argument("--max-timesteps", type=int, default=10_000, help="episode cap (the robot's)")
    ap.add_argument("--num-trials", type=int, default=4)
    ap.add_argument("--num-online-episodes", type=int, default=5)
    ap.add_argument("--experiment-name", default="newsim_cap7000_recycle")
    ap.add_argument("--out", default="runs/td3/online_finetune_dryrun")
    ap.add_argument("--learner-device", default=None)
    cli = ap.parse_args()

    defaults, applied, ignored = ft._build_finetune_args_file_defaults(str(REPO_ROOT / cli.args_file))
    args = ft.OnlineFinetuneArgs(**defaults)
    args.checkpoint = str(Path(cli.checkpoint).resolve())
    args.args_file = cli.args_file
    args.num_online_episodes = int(cli.num_online_episodes)
    args.num_trials = int(cli.num_trials)
    args.warm_start_episodes = 0
    args.no_warmup_no_sim_data = True
    args.enable_periodic_checkpointing = False   # the per-round checkpoints (checkpoint_ep<i>/) stay on
    if cli.learner_device:
        args.learner_device = cli.learner_device
    if args.learner_device.startswith("cuda") and not torch.cuda.is_available():
        args.learner_device = "cpu"
    cli.sim_config = Path(cli.sim_config or Path(args.checkpoint).parent / "config.yaml")
    args.config = str(cli.sim_config)  # copied into each checkpoint's config.yaml
    ft._validate_args(args)
    checkpoint = ft._load_checkpoint(args.checkpoint)
    cli.sim_seed = ft._sim_seed_of(checkpoint, args.checkpoint)
    cli.hist_len = int(ft._hist_len_of(str(cli.sim_config)))
    print(f"[dryrun] args file {cli.args_file}: applied {', '.join(applied)}; ignored {', '.join(ignored) or 'none'}")
    print(f"[dryrun] sim {cli.sim_config} (hist{cli.hist_len}), action latency {cli.action_latency}, cap {cli.max_timesteps}, "
          f"learner {args.learner_device}; {cli.num_trials} trials x {cli.num_online_episodes} episodes", flush=True)
    finished: list[Path] = []
    t0 = time.time()
    for trial in range(cli.num_trials):
        finished.append(run_trial(cli, trial, finished, checkpoint, args))
    print(f"[dryrun] all {cli.num_trials} trials done in {(time.time() - t0) / 60:.1f} min under {finished[0].parent}")


if __name__ == "__main__":
    main()
