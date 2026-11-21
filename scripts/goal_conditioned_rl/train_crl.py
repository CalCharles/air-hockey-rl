"""Online contrastive RL on the new sim (reward-free goal-conditioned juggling).

Loop (official contrastive_rl recipe for state-based envs, see crl_agent.py for the learner):
  * ``num_envs`` new-sim envs (crl_env.py: unchanged sim config + 2-step action latency), stepped in turn;
    the actor is the CRL agent's own tanh-Gaussian policy (stochastic while collecting).
  * One commanded goal per episode: the goal view of a uniformly random replay state (the juggle env has no
    goal distribution of its own). Before ``min_replay_size`` transitions the actions are uniform random
    (the official InitiallyRandomActor).
  * Finished episodes enter the replay whole; goals are relabelled at sample time (crl_replay.py).
  * ``updates_per_env_step`` gradient steps per collected transition (1 = the official samples_per_insert
    256 at batch 256), done in groups every ``train_every`` env steps.
  * The env's reward is never used; it is only logged (``juggle_return``) as a survival indicator.

    python -m scripts.goal_conditioned_rl.train_crl --config configs/crl/juggle_crl_paper.yaml --seed 0 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from scripts.goal_conditioned_rl.crl_agent import CRLAgent, CRLConfig
from scripts.goal_conditioned_rl.crl_env import (ACTION_DIM, GOAL_DIM, StateBuilder, load_sim_config, make_env,
                                                 puck_goal, state_dim)
from scripts.goal_conditioned_rl.crl_replay import EpisodeReplay
from scripts.goal_conditioned_rl.eval_crl import EVAL_SEED_BASE, RADIUS, crl_policy, run_goal_episode


def in_training_eval(agent, sim_cfg, goals, horizon, device, latency):
    reset, act = crl_policy(agent, device)
    n = len(goals)
    eps = [run_goal_episode(reset, act, sim_cfg, goals[i], goals[(i + n // 2) % n],
                            EVAL_SEED_BASE + 10_000 + i, horizon, latency) for i in range(n)]
    return {k: float(np.mean([e[k] for e in eps])) for k in ("min_dist", "min_dist_control", "hit", "hit_still",
                                                             "speed_at_min", "steps")}


def converged(hist, window, rel_tol):
    """True when every tracked series' mean over the last ``window`` steps is within ``rel_tol`` (relative)
    of its mean over the ``window`` before that. ``hist`` = list of (step, {key: value})."""
    if not hist:
        return False, {}
    end = hist[-1][0]
    last = [v for t, v in hist if t > end - window]
    prev = [v for t, v in hist if end - 2 * window < t <= end - window]
    if len(prev) < 3 or len(last) < 3:
        return False, {}
    rel = {}
    for k in last[0]:
        a, b = np.mean([v[k] for v in last]), np.mean([v[k] for v in prev])
        rel[k] = float(abs(a - b) / max(abs(b), 1e-6))
    return all(r < rel_tol for r in rel.values()), rel


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--resume", default=None,
                   help="checkpoint to continue from (weights + optimizer state + step). The replay is not saved, "
                        "so it is refilled with the loaded policy (no random-action phase); logs continue in the same "
                        "run dir at the checkpoint's step, so TensorBoard curves extend the existing ones")
    p.add_argument("--no-convergence", action="store_true", help="ignore the config's convergence stop")
    p.add_argument("--final-name", default="final.pt")
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.total_steps is not None:
        cfg["total_steps"] = a.total_steps
    seed, device = a.seed, torch.device(a.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)  # graph capture streams are created on the current device
    run_dir = Path(a.run_dir or f"runs/crl/{cfg['name']}_s{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)

    agent_cfg = CRLConfig.from_dict(cfg["agent"])
    sim_cfg = load_sim_config(cfg.get("sim_config", "configs/new_juggle/sysid_v2_hist4_realplant.yaml"),
                              cfg["max_timesteps"])
    latency = int(cfg.get("action_latency", 2))
    gamma = float(cfg.get("gamma", 0.99))
    total = int(cfg["total_steps"])
    n_envs = int(cfg.get("num_envs", 4))
    min_replay = int(cfg.get("min_replay_size", 10_000))
    train_every = int(cfg.get("train_every", 16))
    upd_per_step = float(cfg.get("updates_per_env_step", 1.0))
    # Commanded goals (collection + in-training eval) with the velocity set to 0: "stop the puck here".
    zero_vel = bool(cfg.get("command_zero_velocity", False))
    conv = None if a.no_convergence else cfg.get("convergence")  # {min_steps, window, check_every, rel_tol}: stop early once converged
    conv_hist, next_conv = [], int(conv["min_steps"]) if conv else None
    with open(run_dir / ("config_resume.yaml" if a.resume else "config.yaml"), "w") as f:
        yaml.safe_dump(dict(cfg, seed=seed, resume=a.resume, device=str(device), agent_resolved=agent_cfg.to_dict(),
                            sim_config_resolved=sim_cfg), f)

    use_graph = device.type == "cuda" and bool(cfg.get("cuda_graph", True))
    # Policy-state size: obs + a_{t-1}, plus the optional TD3 inputs (agent.history_len / *_privileged; off = 32).
    s_dim = state_dim(agent_cfg.history_len, agent_cfg.uses_privileged)
    agent = CRLAgent(agent_cfg, s_dim, ACTION_DIM, GOAL_DIM, device=device, capturable=use_graph)
    update_fn = None  # built (CUDA graph, 3 real warm-up batches) once learning starts
    replay = EpisodeReplay(int(cfg.get("replay_size", 1_000_000)), s_dim, ACTION_DIM, GOAL_DIM, device)
    eval_goals = np.load(cfg["eval_goals"])["goals"][: int(cfg.get("train_eval_goals", 10))].astype(np.float32)

    def command(g):
        g = np.array(g, dtype=np.float32)
        if zero_vel:
            g[..., 2:] = 0.0
        return g

    eval_goals = command(eval_goals)
    writer = SummaryWriter(str(run_dir))

    envs = [make_env(sim_cfg, seed * 1000 + i, latency) for i in range(n_envs)]
    rng = np.random.default_rng(seed)
    obs = [env.reset(seed=seed * 1000 + i)[0] for i, env in enumerate(envs)]
    builders = [StateBuilder.for_agent_config(agent_cfg) for _ in range(n_envs)]
    for bld, env, o in zip(builders, envs, obs):
        bld.reset(env, o)
    last = np.zeros((n_envs, ACTION_DIM), dtype=np.float32)
    goals = np.zeros((n_envs, GOAL_DIM), dtype=np.float32)
    buf = [dict(s=[], a=[], g=[], ret=0.0) for _ in range(n_envs)]
    ep_lens, ep_rets = deque(maxlen=100), deque(maxlen=100)
    info_acc, n_updates, update_debt = {}, 0, 0.0
    step, t0, t_log = 0, time.time(), time.time()
    log_every, ckpt_every, eval_every = (int(cfg.get(k, d)) for k, d in
                                         (("log_interval", 10_000), ("checkpoint_interval", 250_000),
                                          ("eval_interval", 500_000)))
    next_log, next_ckpt, next_eval = log_every, ckpt_every, eval_every
    resumed = False
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        agent.load_state_dict(ck["model"])
        for opt, key in ((agent.critic_opt, "critic_opt"), (agent.actor_opt, "actor_opt"), (agent.alpha_opt, "alpha_opt")):
            opt.load_state_dict(ck[key])
            for grp in opt.param_groups:  # keep the agent's own lr tensor (the graphed NaN skip writes into it)
                grp["lr"] = agent._lr
                grp["capturable"] = agent.capturable  # saved from a CUDA-graph run; must match this device
        step, n_updates, resumed = int(ck["step"]), int(ck["updates"]), True
        nxt = lambda every: (step // every + 1) * every
        next_log, next_ckpt, next_eval = nxt(log_every), nxt(ckpt_every), nxt(eval_every)
        if conv:
            next_conv = max(next_conv, nxt(int(conv.get("check_every", 250_000))))
        print(json.dumps(dict(resumed_from=a.resume, step=step, updates=n_updates)), flush=True)

    def save(name):
        torch.save(dict(agent.checkpoint(), step=step, updates=n_updates), run_dir / name)

    started = False
    while step < total:
        learning = replay.size >= min_replay
        if learning and not started:  # episodes begun under the random actor get a real goal from now on
            started = True
            goals[:] = command(replay.random_goals(n_envs))
        states = [bld.state(env, o, l) for bld, env, o, l in zip(builders, envs, obs, last)]
        if learning or resumed:
            s_t = torch.as_tensor(np.stack(states), device=device)
            g_t = torch.as_tensor(goals, device=device)
            actions = agent.act(s_t, g_t, deterministic=False).cpu().numpy()
        else:
            actions = rng.uniform(-1.0, 1.0, (n_envs, ACTION_DIM)).astype(np.float32)
        for i, env in enumerate(envs):
            b = buf[i]
            b["s"].append(states[i])
            b["a"].append(actions[i])
            builders[i].push(obs[i], actions[i])
            o, r, term, trunc, _ = env.step(actions[i])
            b["g"].append(puck_goal(env))
            b["ret"] += float(np.asarray(r).reshape(-1)[0])
            last[i] = actions[i]
            obs[i] = o
            if term or trunc:
                replay.add_episode(np.stack(b["s"]), np.stack(b["a"]), np.stack(b["g"]))
                ep_lens.append(len(b["s"]))
                ep_rets.append(b["ret"])
                buf[i] = dict(s=[], a=[], g=[], ret=0.0)
                obs[i] = env.reset()[0]
                builders[i].reset(env, obs[i])
                last[i] = 0.0
                if replay.size > 0:
                    goals[i] = command(replay.random_goals(1))[0]
        step += n_envs

        if learning:
            update_debt += n_envs * upd_per_step
        if learning and update_debt >= train_every:
            agent.train()
            if update_fn is None:
                if use_graph:
                    update_fn = agent.build_graphed_update(
                        [replay.sample(agent_cfg.batch_size, gamma)[:3] for _ in range(3)])
                    n_updates += 3
                else:
                    update_fn = agent.update
            for _ in range(int(update_debt)):
                s, act_b, g, _, _ = replay.sample(agent_cfg.batch_size, gamma)
                info = update_fn(s, act_b, g)  # graphed: tensors are overwritten by the next call
                for k, v in info.items():
                    v = torch.as_tensor(v, device=device).float()
                    info_acc[k] = v.clone() if k not in info_acc else info_acc[k] + v
                info_acc["_n"] = info_acc.get("_n", 0) + 1
                n_updates += 1
            update_debt -= int(update_debt)

        if step >= next_log:
            next_log += log_every
            now = time.time()
            sps = log_every / max(now - t_log, 1e-6)
            t_log = now
            n_acc = info_acc.pop("_n", 0)
            means = {k: float(total_v) / n_acc for k, total_v in info_acc.items()} if n_acc else {}
            for k, v in means.items():
                writer.add_scalar(f"train/{k}", v, step)
            if conv and means and ep_lens:
                conv_hist.append((step, dict(critic_loss=means["critic_loss"], actor_loss=means["actor_loss"],
                                             episode_length=float(np.mean(ep_lens)))))
            info_acc = {}
            if ep_lens:
                writer.add_scalar("rollout/episode_length", float(np.mean(ep_lens)), step)
                writer.add_scalar("rollout/juggle_return", float(np.mean(ep_rets)), step)
            writer.add_scalar("perf/env_steps_per_s", sps, step)
            writer.add_scalar("perf/updates", n_updates, step)
            writer.add_scalar("perf/replay_size", replay.size, step)
            print(json.dumps(dict(step=step, updates=n_updates, sps=round(sps, 1), replay=replay.size,
                                  ep_len=float(np.mean(ep_lens)) if ep_lens else None,
                                  elapsed_h=round((now - t0) / 3600, 2))), flush=True)
        if step >= next_ckpt:
            next_ckpt += ckpt_every
            save(f"ckpt_{step}.pt")
        if step >= next_eval and learning:
            next_eval += eval_every
            agent.eval()
            ev = in_training_eval(agent, sim_cfg, eval_goals, int(cfg.get("train_eval_horizon", 2000)), device, latency)
            for k, v in ev.items():
                writer.add_scalar(f"eval/{k}", v, step)
            print(json.dumps(dict(step=step, eval=ev)), flush=True)

        if conv and step >= next_conv:
            next_conv += int(conv.get("check_every", 250_000))
            done, rel = converged(conv_hist, int(conv.get("window", 500_000)), float(conv.get("rel_tol", 0.10)))
            print(json.dumps(dict(step=step, convergence={k: round(v, 4) for k, v in rel.items()}, converged=done)),
                  flush=True)
            if done:
                break

    save(a.final_name)
    writer.close()
    print(json.dumps(dict(done=True, step=step, updates=n_updates, hours=round((time.time() - t0) / 3600, 2))), flush=True)


if __name__ == "__main__":
    main()
