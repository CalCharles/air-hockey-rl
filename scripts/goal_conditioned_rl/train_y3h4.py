"""Online Y3H4 (z-conditioned CRL, ensemble of heads) on the new sim: reward-free "stop the puck here" goals.

Learner = ``y3h4_agent.py`` (port of the colleague's y3h4.py, parity-checked by parity_y3h4.py). The loop is the
air-hockey CRL loop of ``train_crl.py`` (same env, policy state, true-puck goal view, replay relabelling, logging
and in-training eval) with y3h4's collection rules:

  * Commanded goal per episode = the goal view of a uniformly random replay state with the rest coordinates
    (vx, vy) set to 0: "stop the puck at (x, y)". It is stored with every row (the critic's z).
  * Before ``min_replay_size`` transitions the actions are uniform random; those steps get y3h4's seed-data
    commanded goal: the rest view of where their episode ended.
  * Collection acts with temperature 1; lane i = env i: the first ``commit_fraction`` of the envs follow the
    heads' vote, env i >= commit is pinned to head (i - commit) % heads. Eval acts on the vote at temperature 0.
  * Batches: uniform rows, positive = hindsight future goal (P(k) ~ gamma^k, within the episode), the actor goal
    = the same hindsight goal (velocity kept), commanded goal = the row's stored one.

What differs from y3h4's four-rooms run (env-specific): episodes end when the puck is lost (cap
``max_timesteps``) instead of fixed 1000-step lanes; ``num_envs`` lanes instead of 500; updates are interleaved
with collection at ``updates_per_env_step`` instead of 16,000 updates per 500k-transition round (0.032).

    python -m scripts.goal_conditioned_rl.train_y3h4 --config configs/crl/juggle_y3h4.yaml --seed 0 --device cuda:0
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

from scripts.goal_conditioned_rl.crl_env import (ACTION_DIM, GOAL_DIM, STATE_DIM, load_sim_config, make_env,
                                                 policy_state, puck_goal)
from scripts.goal_conditioned_rl.crl_replay import EpisodeReplay
from scripts.goal_conditioned_rl.eval_crl import EVAL_SEED_BASE, run_goal_episode
from scripts.goal_conditioned_rl.train_crl import converged
from scripts.goal_conditioned_rl.y3h4_agent import Y3H4Agent, Y3H4Config


class CommandedReplay(EpisodeReplay):
    """EpisodeReplay plus each row's commanded goal (the z the critic is given)."""

    def __init__(self, capacity, state_dim, action_dim, goal_dim, device):
        super().__init__(capacity, state_dim, action_dim, goal_dim, device)
        self.commanded = torch.zeros((capacity, goal_dim), dtype=torch.float32, device=self.device)

    def add_episode(self, states, actions, next_goals, commanded=None):
        n = len(states)
        if n and commanded is not None:
            slots = (self.written + torch.arange(n, device=self.device)) % self.capacity
            self.commanded[slots] = torch.as_tensor(commanded, dtype=torch.float32, device=self.device)
        super().add_episode(states, actions, next_goals)

    def sample_y3h4(self, batch_size, gamma):
        s, a, g, rows, _ = self.sample(batch_size, gamma)
        return s, a, g, g, self.commanded[rows]  # value goals, actor goals (same hindsight goal), commanded


def rest_view(g, rest):
    g = np.array(g, dtype=np.float32)
    g[..., list(rest)] = 0.0
    return g


def in_training_eval(agent, sim_cfg, goals, horizon, device, latency):
    def act(obs, last, goal):
        s = torch.as_tensor(policy_state(obs, last), device=device).unsqueeze(0)
        g = torch.as_tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
        return agent.act(s, g, temperature=0.0).cpu().numpy().reshape(2)

    n = len(goals)
    eps = [run_goal_episode(lambda o: None, act, sim_cfg, goals[i], goals[(i + n // 2) % n],
                            EVAL_SEED_BASE + 10_000 + i, horizon, latency) for i in range(n)]
    return {k: float(np.mean([e[k] for e in eps])) for k in ("min_dist", "min_dist_control", "hit", "hit_still",
                                                             "speed_at_min", "steps")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--resume", default=None, help="checkpoint to continue from (weights, optimizer, step); the "
                   "replay is refilled by the loaded policy, logs continue in the same run dir")
    p.add_argument("--no-convergence", action="store_true")
    p.add_argument("--final-name", default="final.pt")
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.total_steps is not None:
        cfg["total_steps"] = a.total_steps
    seed, device = a.seed, torch.device(a.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    run_dir = Path(a.run_dir or f"runs/crl/{cfg['name']}_s{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)

    agent_cfg = Y3H4Config(**cfg.get("agent", {}))
    sim_cfg = load_sim_config(cfg.get("sim_config", "configs/new_juggle/sysid_v2_hist4_realplant.yaml"),
                              cfg["max_timesteps"])
    latency = int(cfg.get("action_latency", 2))
    gamma = float(cfg.get("gamma", 0.99))
    total = int(cfg["total_steps"])
    n_envs = int(cfg.get("num_envs", 8))
    min_replay = int(cfg.get("min_replay_size", 10_000))
    train_every = int(cfg.get("train_every", 16))
    upd_per_step = float(cfg.get("updates_per_env_step", 1.0))
    rest = tuple(cfg.get("rest_coordinates", [2, 3]))
    conv = None if a.no_convergence else cfg.get("convergence")
    conv_hist, next_conv = [], int(conv["min_steps"]) if conv else None
    with open(run_dir / ("config_resume.yaml" if a.resume else "config.yaml"), "w") as f:
        yaml.safe_dump(dict(cfg, seed=seed, device=str(device), resume=a.resume, agent_resolved=agent_cfg.to_dict(),
                            sim_config_resolved=sim_cfg), f)

    use_graph = device.type == "cuda" and bool(cfg.get("cuda_graph", True))
    agent = Y3H4Agent(agent_cfg, STATE_DIM, ACTION_DIM, GOAL_DIM, device=device, capturable=use_graph)
    update_fn = None
    replay = CommandedReplay(int(cfg.get("replay_size", 1_000_000)), STATE_DIM, ACTION_DIM, GOAL_DIM, device)
    eval_goals = rest_view(np.load(cfg["eval_goals"])["goals"][: int(cfg.get("train_eval_goals", 10))], rest)
    writer = SummaryWriter(str(run_dir))

    envs = [make_env(sim_cfg, seed * 1000 + i, latency) for i in range(n_envs)]
    rng = np.random.default_rng(seed)
    obs = [env.reset(seed=seed * 1000 + i)[0] for i, env in enumerate(envs)]
    last = np.zeros((n_envs, ACTION_DIM), dtype=np.float32)
    goals = np.zeros((n_envs, GOAL_DIM), dtype=np.float32)
    lane_ids = torch.arange(n_envs, device=device)
    buf = [dict(s=[], a=[], g=[], c=[], ret=0.0) for _ in range(n_envs)]
    ep_lens, ep_rets = deque(maxlen=100), deque(maxlen=100)
    info_acc, rolled_acc, n_updates, update_debt = {}, [], 0, 0.0
    step, t0, t_log = 0, time.time(), time.time()
    log_every, ckpt_every, eval_every = (int(cfg.get(k, d)) for k, d in
                                         (("log_interval", 10_000), ("checkpoint_interval", 250_000),
                                          ("eval_interval", 500_000)))
    next_log, next_ckpt, next_eval = log_every, ckpt_every, eval_every
    resumed = False
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        agent.load_state_dict(ck["model"])
        agent.opt.load_state_dict(ck["opt"])
        for grp in agent.opt.param_groups:
            grp["lr"] = agent._lr
            grp["capturable"] = agent.capturable
        step, n_updates, resumed = int(ck["step"]), int(ck["updates"]), True
        nxt = lambda every: (step // every + 1) * every  # noqa: E731
        next_log, next_ckpt, next_eval = nxt(log_every), nxt(ckpt_every), nxt(eval_every)
        if conv:
            next_conv = max(next_conv, nxt(int(conv.get("check_every", 250_000))))
        print(json.dumps(dict(resumed_from=a.resume, step=step, updates=n_updates)), flush=True)

    def save(name):
        torch.save(dict(agent.checkpoint(), step=step, updates=n_updates), run_dir / name)

    def new_goal():
        return rest_view(replay.random_goals(1)[0], rest) if replay.size > 0 else np.zeros(GOAL_DIM, np.float32)

    started = False
    while step < total:
        learning = replay.size >= min_replay
        if learning and not started:
            started = True
            goals[:] = np.stack([new_goal() for _ in range(n_envs)])
        acting = learning or resumed
        if acting:
            s_t = torch.as_tensor(np.stack([policy_state(o, l) for o, l in zip(obs, last)]), device=device)
            g_t = torch.as_tensor(goals, device=device)
            actions = agent.act(s_t, g_t, temperature=1.0, lane_ids=lane_ids, n_lanes=n_envs).cpu().numpy()
        else:
            actions = rng.uniform(-1.0, 1.0, (n_envs, ACTION_DIM)).astype(np.float32)
        for i, env in enumerate(envs):
            b = buf[i]
            b["s"].append(policy_state(obs[i], last[i]))
            b["a"].append(actions[i])
            b["c"].append(goals[i].copy() if acting else None)  # None: random step, set at episode end
            o, r, term, trunc, _ = env.step(actions[i])
            b["g"].append(puck_goal(env))
            b["ret"] += float(np.asarray(r).reshape(-1)[0])
            last[i] = actions[i]
            obs[i] = o
            if term or trunc:
                end = rest_view(b["g"][-1], rest)  # y3h4 seed data: rest view of where the episode ended
                cmd = np.stack([end if c is None else c for c in b["c"]])
                replay.add_episode(np.stack(b["s"]), np.stack(b["a"]), np.stack(b["g"]), cmd)
                ep_lens.append(len(b["s"]))
                ep_rets.append(b["ret"])
                buf[i] = dict(s=[], a=[], g=[], c=[], ret=0.0)
                obs[i] = env.reset()[0]
                last[i] = 0.0
                goals[i] = new_goal()
        step += n_envs

        if learning:
            update_debt += n_envs * upd_per_step
        if learning and update_debt >= train_every:
            agent.train()
            if update_fn is None:
                if use_graph:
                    update_fn = agent.build_graphed_update(
                        [replay.sample_y3h4(agent_cfg.batch_size, gamma) for _ in range(3)])
                    n_updates += 3
                else:
                    update_fn = agent.update
            for _ in range(int(update_debt)):
                batch = replay.sample_y3h4(agent_cfg.batch_size, gamma)
                if n_updates % agent_cfg.diagnostic_interval == 0:  # before the update: the pre-update critic
                    rolled_acc.append(agent.rolled_code_accuracy(batch[0], batch[1], batch[2], batch[4]))
                info = update_fn(*batch)
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
            means = {k: float(v) / n_acc for k, v in info_acc.items()} if n_acc else {}
            if rolled_acc:
                means["rolled_code_accuracy"] = float(torch.stack(rolled_acc).mean())
            for k, v in means.items():
                writer.add_scalar(f"train/{k}", v, step)
            if conv and "critic_loss" in means and ep_lens:
                conv_hist.append((step, dict(critic_loss=means["critic_loss"], actor_loss=means["actor_loss"],
                                             episode_length=float(np.mean(ep_lens)))))
            info_acc, rolled_acc = {}, []
            if ep_lens:
                writer.add_scalar("rollout/episode_length", float(np.mean(ep_lens)), step)
                writer.add_scalar("rollout/juggle_return", float(np.mean(ep_rets)), step)
            writer.add_scalar("perf/env_steps_per_s", sps, step)
            writer.add_scalar("perf/updates", n_updates, step)
            writer.add_scalar("perf/replay_size", replay.size, step)
            print(json.dumps(dict(step=step, updates=n_updates, sps=round(sps, 1), replay=replay.size,
                                  ep_len=float(np.mean(ep_lens)) if ep_lens else None,
                                  acc=round(means.get("accuracy", float("nan")), 3),
                                  rolled=round(means.get("rolled_code_accuracy", float("nan")), 3),
                                  elapsed_h=round((now - t0) / 3600, 2))), flush=True)
        if step >= next_ckpt:
            next_ckpt += ckpt_every
            save(f"ckpt_{step}.pt")
        if step >= next_eval and learning:
            next_eval += eval_every
            agent.eval()
            ev = in_training_eval(agent, sim_cfg, eval_goals, int(cfg.get("train_eval_horizon", 2000)), device,
                                  latency)
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
    print(json.dumps(dict(done=True, step=step, updates=n_updates, hours=round((time.time() - t0) / 3600, 2))),
          flush=True)


if __name__ == "__main__":
    main()
