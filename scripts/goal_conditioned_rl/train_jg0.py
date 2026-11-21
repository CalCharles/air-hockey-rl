"""Train the colleague's JAX learner (scripts/goal_conditioned_rl/jg0.py, unchanged) on the air-hockey Box2D sim.

jg0.py is imported as a library; this file is only the harness around it, following his y3h4.py harness:

  * Rounds: ``lanes`` (500) parallel lanes x ``round_steps`` (1000) env steps = 500k transitions per round. Every
    lane starts a fresh episode at the start of a round; an episode ends when the puck is lost (env terminated)
    and the lane then starts a new one, or at the end of the round. Round 0 uses uniform random actions.
  * After each round: ``updates_per_round`` (16,000) jg0 updates on batches from a 1M FIFO replay
    (jg0.sample_replay: episode-truncated geometric future goals, gamma 0.99, never crossing an episode).
  * Commanded goals: one per episode, (x, y, 0, 0) with (x, y) uniform over the upper half of the table (the half
    away from the paddle, one puck radius from the walls). Critic and actor train on jg0's hindsight goals (the
    true future puck (x, y, vx, vy)), exactly as jg0.sample_replay returns them.
  * Acting: Learner.sample_actions with jg0's lane rule (the first 25 % of lanes act on the heads' vote, the others
    are pinned to one head each); evaluation uses the deterministic vote.
  * Two GPUs: the 4 heads are independent learners trained on the same batches, so heads 0-1 live in a learner
    process on the first GPU and heads 2-3 in one on the second. Both processes hold the same replay and use the
    same sampling key, so they draw identical batches: the math is that of one 4-head jg0 Learner. The acting copy
    (all 4 heads) is refreshed from them after every training phase.
  * State = the 30-dim history obs + the last action (scripts/goal_conditioned_rl/crl_env.policy_state); goal view
    = the true Box2D puck (x, y, vx, vy) (crl_env.puck_goal). Sim = crl_env defaults (new sim + 2-step action
    latency; puck noise / occlusion / lag on).
  * Lattice eval (as in the colleague's video): a 5 x 7 grid of goals (x, y, 0, 0) over the upper half, each
    launched from the same env seed, ``eval_horizon`` steps (600 = 30 s); several launches.

Run (the JAX venv):  .venv-jax/bin/python -m scripts.goal_conditioned_rl.train_jg0 \
    --config configs/crl/juggle_jg0_best.yaml --seed 0 --gpus 0,1
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
OBS_STATE_DIM, ACTION_DIM, GOAL_DIM = 32, 2, 4
RADIUS, STILL_SPEED = 0.10, 0.10  # = eval_crl
TABLE_X_TOP, TABLE_Y = -0.9652, 0.4318
PUCK_R = 0.03175


# ---------------------------------------------------------------------------------------------------------- goals
def sample_goals(rng, n):
    """(x, y, 0, 0), (x, y) uniform over the upper half (x in [top, 0]) one puck radius from the walls."""
    g = np.zeros((n, GOAL_DIM), np.float32)
    g[:, 0] = rng.uniform(TABLE_X_TOP + PUCK_R, 0.0, n)
    g[:, 1] = rng.uniform(-TABLE_Y + PUCK_R, TABLE_Y - PUCK_R, n)
    return g


def lattice_goals(rows=5, cols=7):
    """Cell centres of a rows x cols grid over the upper half; rows along x (top -> midline), cols along y."""
    xs = TABLE_X_TOP + (np.arange(rows) + 0.5) * (-TABLE_X_TOP) / rows
    ys = -TABLE_Y + (np.arange(cols) + 0.5) * (2 * TABLE_Y) / cols
    g = np.zeros((rows * cols, GOAL_DIM), np.float32)
    g[:, 0] = np.repeat(xs, cols)
    g[:, 1] = np.tile(ys, rows)
    return g


# ---------------------------------------------------------------------------------------------------- env worker
def env_worker(conn, worker_id, lane_ids, seed, sim_overrides, latency, max_timesteps):
    """Runs a group of lanes. No JAX here. Commands: start_round / step / finish_round / eval_start / eval_step /
    eval_finish / close."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import sys

    sys.path.insert(0, str(REPO))
    from scripts.goal_conditioned_rl.crl_env import load_sim_config, make_env, policy_state, puck_goal

    cfg = load_sim_config(max_timesteps=max_timesteps)
    for k, v in (sim_overrides or {}).items():
        cfg["simulator_params"][k] = v
    rng = np.random.default_rng([seed, 7, worker_id])
    n = len(lane_ids)
    envs = [make_env(cfg, int(rng.integers(2**31)), latency) for _ in range(n)]
    obs = [None] * n
    last = np.zeros((n, ACTION_DIM), np.float32)
    goals = np.zeros((n, GOAL_DIM), np.float32)
    buf = [None] * n  # per lane: lists of s, a, next goal for the open episode
    done_eps = []  # finished episodes of this round: (s, a, g_next)
    stats = []  # per finished episode: (length, min_dist, stopped, terminated)
    track = np.zeros((n, 2))  # min dist to commanded goal, stopped flag

    def new_episode(i):
        o, _ = envs[i].reset(seed=int(rng.integers(2**31)))
        obs[i] = o
        last[i] = 0.0
        goals[i] = sample_goals(rng, 1)[0]
        buf[i] = ([], [], [])
        track[i] = (np.inf, 0.0)

    def close_episode(i, terminated):
        s, a, g = buf[i]
        if s:
            done_eps.append((np.stack(s), np.stack(a), np.stack(g)))
            stats.append((len(s), float(track[i, 0]), float(track[i, 1]), bool(terminated)))
        buf[i] = None

    def states():
        return np.stack([policy_state(obs[i], last[i]) for i in range(n)]).astype(np.float32)

    ev = None
    while True:
        cmd, payload = conn.recv()
        if cmd == "start_round":
            done_eps.clear()
            stats.clear()
            for i in range(n):
                new_episode(i)
            conn.send((states(), goals.copy()))
        elif cmd == "step":
            actions = payload
            for i in range(n):
                s = policy_state(obs[i], last[i])
                a = np.clip(actions[i], -1.0, 1.0).astype(np.float32)
                o, _, term, trunc, _ = envs[i].step(a)
                pg = puck_goal(envs[i])
                buf[i][0].append(s)
                buf[i][1].append(a)
                buf[i][2].append(pg)
                d = float(np.linalg.norm(pg[:2] - goals[i, :2]))
                track[i, 0] = min(track[i, 0], d)
                if d < RADIUS and float(np.linalg.norm(pg[2:])) < STILL_SPEED:
                    track[i, 1] = 1.0
                obs[i] = o
                last[i] = a
                if term or trunc:
                    close_episode(i, term)
                    new_episode(i)
            conn.send((states(), goals.copy()))
        elif cmd == "finish_round":
            for i in range(n):
                close_episode(i, False)
            if done_eps:
                s = np.concatenate([e[0] for e in done_eps])
                a = np.concatenate([e[1] for e in done_eps])
                g = np.concatenate([e[2] for e in done_eps])
                rem = np.concatenate([np.arange(len(e[0]), 0, -1) for e in done_eps]).astype(np.int32)
            else:
                s = np.zeros((0, OBS_STATE_DIM), np.float32)
                a = np.zeros((0, ACTION_DIM), np.float32)
                g = np.zeros((0, GOAL_DIM), np.float32)
                rem = np.zeros((0,), np.int32)
            conn.send(((s, a, g, rem), list(stats)))
            done_eps.clear()
            stats.clear()
        elif cmd == "eval_start":
            items, horizon = payload  # list of (goal, env_seed)
            ev = dict(envs=[], obs=[], last=np.zeros((len(items), ACTION_DIM), np.float32),
                      goals=np.stack([g for g, _ in items]).astype(np.float32), alive=np.ones(len(items), bool),
                      min_d=np.full(len(items), np.inf), stop=np.zeros(len(items)), steps=np.zeros(len(items), int),
                      within=np.zeros(len(items)), dropped=np.zeros(len(items), bool))
            for g, sd in items:
                e = make_env(dict(cfg, max_timesteps=horizon + 1), int(sd), latency)
                o, _ = e.reset(seed=int(sd))
                ev["envs"].append(e)
                ev["obs"].append(o)
            conn.send(np.stack([policy_state(ev["obs"][i], ev["last"][i]) for i in range(len(items))]).astype(np.float32))
        elif cmd == "eval_step":
            actions = payload
            for i, e in enumerate(ev["envs"]):
                if not ev["alive"][i]:
                    continue
                a = np.clip(actions[i], -1.0, 1.0).astype(np.float32)
                o, _, term, trunc, _ = e.step(a)
                pg = puck_goal(e)
                d = float(np.linalg.norm(pg[:2] - ev["goals"][i, :2]))
                ev["min_d"][i] = min(ev["min_d"][i], d)
                ev["within"][i] += d < RADIUS
                if d < RADIUS and float(np.linalg.norm(pg[2:])) < STILL_SPEED:
                    ev["stop"][i] = 1.0
                ev["steps"][i] += 1
                ev["obs"][i] = o
                ev["last"][i] = a
                if term or trunc:
                    ev["alive"][i] = False
                    ev["dropped"][i] = bool(term)
            conn.send(np.stack([policy_state(ev["obs"][i], ev["last"][i])
                                for i in range(len(ev["envs"]))]).astype(np.float32))
        elif cmd == "eval_finish":
            for e in ev["envs"]:
                e.close()
            conn.send(dict(min_d=ev["min_d"], stop=ev["stop"], steps=ev["steps"], within=ev["within"],
                           dropped=ev["dropped"]))
            ev = None
        elif cmd == "close":
            for e in envs:
                e.close()
            conn.close()
            return


class EnvPool:
    def __init__(self, n_workers, lanes, seed, sim_overrides, latency, max_timesteps):
        ctx = mp.get_context("spawn")
        self.groups = np.array_split(np.arange(lanes), n_workers)
        self.conns, self.procs = [], []
        for w, ids in enumerate(self.groups):
            a, b = ctx.Pipe()
            p = ctx.Process(target=env_worker, args=(b, w, ids, seed, sim_overrides, latency, max_timesteps),
                            daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def _all(self, cmd, payloads=None):
        for i, c in enumerate(self.conns):
            c.send((cmd, None if payloads is None else payloads[i]))
        return [c.recv() for c in self.conns]

    def start_round(self):
        out = self._all("start_round")
        return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])

    def step(self, actions):
        out = self._all("step", [actions[ids] for ids in self.groups])
        return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])

    def finish_round(self):
        out = self._all("finish_round")
        data = tuple(np.concatenate([o[0][k] for o in out]) for k in range(4))
        stats = [s for o in out for s in o[1]]
        return data, stats

    def evaluate(self, act_fn, items, horizon):
        """items: list of (goal, seed). act_fn(states [N, S], goals [N, G]) -> actions [N, A]."""
        split = np.array_split(np.arange(len(items)), len(self.conns))
        used = [(c, ids) for c, ids in zip(self.conns, split) if len(ids)]
        for c, ids in used:
            c.send(("eval_start", ([items[i] for i in ids], horizon)))
        states = np.concatenate([c.recv() for c, _ in used])
        goals = np.stack([g for g, _ in items]).astype(np.float32)
        for _ in range(horizon):
            actions = act_fn(states, goals)
            for c, ids in used:
                c.send(("eval_step", actions[ids]))
            states = np.concatenate([c.recv() for c, _ in used])
        for c, _ in used:
            c.send(("eval_finish", None))
        res = [c.recv() for c, _ in used]
        return {k: np.concatenate([r[k] for r in res]) for k in res[0]}

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=10)


# ------------------------------------------------------------------------------------------------ learner process
def learner_proc(conn, gpu, head_slice, seed, cfg_overrides, capacity, batch_size, future, chunk):
    """Holds heads[head_slice] of the jg0 Learner and a full replay copy on one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import sys

    sys.path.insert(0, str(REPO))
    import flax
    import jax
    import jax.numpy as jnp

    from scripts.goal_conditioned_rl import jg0

    learner = build_learner(jax, jnp, jg0, seed, cfg_overrides)
    sl = slice(*head_slice)
    learner = learner.replace(heads=jax.tree_util.tree_map(lambda x: x[sl], learner.heads))
    learner = jax.device_put(learner, jax.devices()[0])
    replay = dict(observations=jnp.zeros((capacity, OBS_STATE_DIM), jnp.float32),
                  actions=jnp.zeros((capacity, ACTION_DIM), jnp.float32),
                  next_goal_observations=jnp.zeros((capacity, GOAL_DIM), jnp.float32),
                  remaining=jnp.ones((capacity,), jnp.int32))
    written = 0

    @jax.jit
    def insert(rep, slots, s, a, g, r):
        return dict(observations=rep["observations"].at[slots].set(s), actions=rep["actions"].at[slots].set(a),
                    next_goal_observations=rep["next_goal_observations"].at[slots].set(g),
                    remaining=rep["remaining"].at[slots].set(r))

    def train_chunk(learner, rep, key, size):
        def body(c, _):
            learner, key = c
            key, batch = jg0.sample_replay(rep, key, size, batch_size=batch_size, future=future)
            learner, info = learner.update(batch)
            return (learner, key), info
        (learner, key), info = jax.lax.scan(body, (learner, key), None, length=chunk)
        return learner, key, jax.tree_util.tree_map(jnp.mean, info)

    train_chunk = jax.jit(train_chunk)
    conn.send("ready")
    while True:
        cmd, payload = conn.recv()
        if cmd == "insert":
            s, a, g, r = payload
            n = len(s)
            if n > capacity:
                s, a, g, r = s[-capacity:], a[-capacity:], g[-capacity:], r[-capacity:]
                r = np.minimum(r, np.arange(capacity, 0, -1))  # cannot happen with 500k rounds; kept for safety
                n = capacity
            slots = jnp.asarray((written + np.arange(n)) % capacity)
            replay = insert(replay, slots, jnp.asarray(s), jnp.asarray(a), jnp.asarray(g), jnp.asarray(r))
            written += n
            conn.send(min(written, capacity))
        elif cmd == "train":
            n_updates, key_seed = payload
            key = jax.random.PRNGKey(int(key_seed))
            size = jnp.int32(min(written, capacity))
            infos = []
            t0 = time.time()
            for _ in range(n_updates // chunk):
                learner, key, info = train_chunk(learner, replay, key, size)
                infos.append(info)
            infos = jax.device_get(infos)
            mean = {k: float(np.mean([i[k] for i in infos])) for k in infos[0]}
            mean["train_seconds"] = time.time() - t0
            mean["skipped_updates"] = int(learner.skipped_updates)
            conn.send(mean)
        elif cmd == "params":
            conn.send(jax.device_get(learner.heads.params))
        elif cmd == "save":
            Path(payload).write_bytes(flax.serialization.msgpack_serialize(
                flax.serialization.to_state_dict(jax.device_get(learner))))
            conn.send("saved")
        elif cmd == "load":
            state = flax.serialization.msgpack_restore(Path(payload).read_bytes())
            learner = jax.device_put(flax.serialization.from_state_dict(jax.device_get(learner), state),
                                     jax.devices()[0])
            conn.send("loaded")
        elif cmd == "close":
            conn.close()
            return


def build_learner(jax, jnp, jg0, seed, cfg_overrides):
    """The full 4-head jg0 Learner, built on the CPU (same init in every process)."""
    cfg = jg0.get_config("best", **(cfg_overrides or {}))
    example = {k: jnp.zeros((2, d), jnp.float32) for k, d in
               (("observations", OBS_STATE_DIM), ("actions", ACTION_DIM), ("value_goals", GOAL_DIM),
                ("actor_goals", GOAL_DIM))}
    with jax.default_device(jax.devices("cpu")[0]):
        return jg0.Learner.create(int(seed), example, cfg)


# ----------------------------------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpus", default="0,1", help="two GPU ids: learner A (heads 0-1, also acting), learner B")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--resume", action="store_true", help="continue from <run-dir>/latest_*")
    p.add_argument("--total-steps", type=int, default=None)
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    total = int(a.total_steps or cfg["total_steps"])
    lanes, round_steps = int(cfg["lanes"]), int(cfg["round_steps"])
    upr, chunk = int(cfg["updates_per_round"]), int(cfg.get("update_chunk", 128))
    cfg_overrides = dict(cfg.get("jg0_overrides", {}) or {})
    cfg_overrides.setdefault("ensemble_collection_lanes", lanes)
    gpus = [g.strip() for g in a.gpus.split(",")]
    assert len(gpus) == 2
    run = Path(a.run_dir or f"runs/crl/{cfg['name']}_s{a.seed}")
    run.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.config, run / "config.yaml")
    log = open(run / "train.log", "a")

    def emit(d):
        line = json.dumps(d)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    # learners first (each sets its own CUDA_VISIBLE_DEVICES before importing jax)
    ctx = mp.get_context("spawn")
    heads = int(cfg_overrides.get("ensemble_heads", 4))
    half = heads // 2
    learners = []
    for gpu, hs in ((gpus[0], (0, half)), (gpus[1], (half, heads))):
        c1, c2 = ctx.Pipe()
        pr = ctx.Process(target=learner_proc, args=(c2, gpu, hs, a.seed, cfg_overrides, int(cfg["replay_size"]),
                                                     int(cfg["batch_size"]), float(cfg["gamma"]), chunk), daemon=True)
        pr.start()
        learners.append((c1, pr))
    for c, _ in learners:
        assert c.recv() == "ready"

    # the driver acts on the first GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import sys

    sys.path.insert(0, str(REPO))
    import jax
    import jax.numpy as jnp

    from scripts.goal_conditioned_rl import jg0

    actor = jax.device_put(build_learner(jax, jnp, jg0, a.seed, cfg_overrides), jax.devices()[0])

    def refresh_actor():
        for c, _ in learners:
            c.send(("params", None))
        parts = [c.recv() for c, _ in learners]
        params = jax.tree_util.tree_map(lambda x, y: np.concatenate([x, y], 0), parts[0], parts[1])
        return actor.replace(heads=actor.heads.replace(params=jax.device_put(params, jax.devices()[0])))

    pool = EnvPool(int(cfg.get("env_workers", 48)), lanes, a.seed, cfg.get("sim_overrides"),
                   int(cfg.get("action_latency", 2)), int(cfg.get("max_timesteps", 7000)))
    lane_ids = jnp.arange(lanes, dtype=jnp.int32)
    key = jax.random.PRNGKey(a.seed + 12345)
    rng = np.random.default_rng([a.seed, 99])

    # eval set: lattice goals x launches
    lattice = lattice_goals(int(cfg.get("lattice_rows", 5)), int(cfg.get("lattice_cols", 7)))
    launches = int(cfg.get("eval_launches", 5))
    eval_items = [(g, 900_000 + l) for l in range(launches) for g in lattice]
    eval_h = int(cfg.get("eval_horizon", 600))

    def act_eval(states, goals):
        return np.asarray(actor.sample_actions(jnp.asarray(states), jnp.asarray(goals)))

    state_file = run / "latest_state.json"
    rnd, steps, best = 0, 0, None
    if a.resume and state_file.exists():
        st = json.loads(state_file.read_text())
        rnd, steps, best = st["round"], st["steps"], st.get("best")
        for i, (c, _) in enumerate(learners):
            c.send(("load", str(run / f"latest_learner{i}.msgpack")))
            assert c.recv() == "loaded"
        actor = refresh_actor()
        emit(dict(resumed=True, round=rnd, steps=steps, note="replay starts empty; refilled by policy rounds"))
    t_start = time.time()
    while steps < total:
        t0 = time.time()
        random_round = steps == 0
        states, goals = pool.start_round()
        for _ in range(round_steps):
            if random_round:
                actions = rng.uniform(-1, 1, (lanes, ACTION_DIM)).astype(np.float32)
            else:
                key, sub = jax.random.split(key)
                actions = np.asarray(actor.sample_actions(jnp.asarray(states), jnp.asarray(goals), sub, 1.0,
                                                          lane_ids))
            states, goals = pool.step(actions)
        data, stats = pool.finish_round()
        t_collect = time.time() - t0
        for c, _ in learners:
            c.send(("insert", data))
        sizes = [c.recv() for c, _ in learners]
        steps += len(data[0])
        rnd += 1
        lens = np.array([s[0] for s in stats])
        mind = np.array([s[1] for s in stats])
        info = {}
        t1 = time.time()
        key_seed = int(rng.integers(2**31))
        for c, _ in learners:
            c.send(("train", (upr, key_seed)))
        infos = [c.recv() for c, _ in learners]
        info = {k: float(np.mean([i[k] for i in infos])) for k in infos[0]}
        actor = refresh_actor()
        t_train = time.time() - t1
        rec = dict(round=rnd, steps=steps, replay=sizes[0], episodes=len(stats), ep_len=float(lens.mean()),
                   ep_len_max=int(lens.max()), drop_frac=float(np.mean([s[3] for s in stats])),
                   collect_hit10=float(np.mean(mind < RADIUS)), collect_stop=float(np.mean([s[2] for s in stats])),
                   collect_min_dist=float(np.mean(mind)), random=random_round,
                   t_collect=round(t_collect, 1), t_train=round(t_train, 1),
                   hours=round((time.time() - t_start) / 3600, 3),
                   **{k: round(v, 4) for k, v in info.items()})
        emit(rec)

        if rnd % int(cfg.get("eval_every_rounds", 4)) == 0 or steps >= total:
            te = time.time()
            r = pool.evaluate(act_eval, eval_items, eval_h)
            n0 = len(lattice)
            ev = dict(eval=True, round=rnd, steps=steps, hit10=float(np.mean(r["min_d"] < RADIUS)),
                      hit5=float(np.mean(r["min_d"] < 0.05)), stop=float(np.mean(r["stop"])),
                      min_dist=float(np.mean(r["min_d"])), ep_steps=float(np.mean(r["steps"])),
                      dropped=float(np.mean(r["dropped"])), frac_within=float(np.mean(r["within"] / eval_h)),
                      launch0_hits=int(np.sum(r["min_d"][:n0] < RADIUS)), launch0_of=n0,
                      seconds=round(time.time() - te, 1))
            emit(ev)
            score = (ev["hit10"], -ev["min_dist"])
            if best is None or score > tuple(best["score"]):
                best = dict(score=list(score), round=rnd, steps=steps, hit10=ev["hit10"], stop=ev["stop"])
                actor.save(run / "best_policy.msgpack")
                emit(dict(new_best=best))
        if rnd % int(cfg.get("checkpoint_every_rounds", 20)) == 0 or steps >= total:
            actor.save(run / f"policy_{steps // 1_000_000:04d}M.msgpack")
        # full learner state (with Adam) for --resume, every round
        for i, (c, _) in enumerate(learners):
            c.send(("save", str(run / f"latest_learner{i}.msgpack")))
            assert c.recv() == "saved"
        actor.save(run / "latest_policy.msgpack")
        state_file.write_text(json.dumps(dict(round=rnd, steps=steps, best=best)))
    emit(dict(done=True, steps=steps, rounds=rnd, hours=round((time.time() - t_start) / 3600, 2), best=best))
    pool.close()
    for c, pr in learners:
        c.send(("close", None))
        pr.join(timeout=30)


if __name__ == "__main__":
    main()
