"""Y3H4 in one file: z-conditioned contrastive RL with four independent heads, on four-rooms push.

The algorithm is the R2CRL phase-1 arm `y3h4-converge` (agents/exploration/r2crl.py with
critic_code=commanded, critic_code_grad=true, actor_condition=embedding, ensemble_heads=4).
Everything the algorithm computes is written out below. README.md walks through this file line by
line, and parity_check.py compares it with the repository code.

Standalone: importing this file and running --demo need only
    pip install jax flax optax distrax numpy
    python y3h4.py --demo            # point-mass stand-in for four rooms, CPU, a few minutes
The four-rooms benchmark run additionally needs the repository's C simulator and lattice evaluator
(rlmini-envs importable), which load_harness() imports only on that path:
    python y3h4.py --repo /path/to/unsupervised-rl --seed 50 --save_dir /path/to/run
"""
import argparse
import csv
import json
import os
import pickle
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

REPO = Path(__file__).resolve().parents[2]  # Default --repo when this file sits in <repo>/standalone/y3h4/.


def load_harness(repo):
    """The four-rooms benchmark (C simulator, observation view, lattice evaluator), from the repository."""
    # deferred: optional dependency. Only the four-rooms run needs the repository harness; the algorithm and --demo
    # use nothing outside jax/flax/optax/distrax/numpy, so importing this file must not require the repository.
    sys.path.insert(0, str(repo))
    from envs.c_rooms_vector import CRoomsVectorEnv
    from envs.observation import ObservationView
    from envs.online_env_utils import make_online_env
    from scripts.eval.online_url import (complete_lattice_goals, evaluate_suites, evaluation_config, load_pairs,
                                         prepare_goal_pairs, prepare_lattice, task_goal_mode)
    from utils.goal_conditioning import GoalReadoutPolicy
    return SimpleNamespace(CRoomsVectorEnv=CRoomsVectorEnv, ObservationView=ObservationView,
                           make_online_env=make_online_env, complete_lattice_goals=complete_lattice_goals,
                           evaluate_suites=evaluate_suites, evaluation_config=evaluation_config, load_pairs=load_pairs,
                           prepare_goal_pairs=prepare_goal_pairs, prepare_lattice=prepare_lattice,
                           task_goal_mode=task_goal_mode, GoalReadoutPolicy=GoalReadoutPolicy)


# ---------------------------------------------------------------------------------------------------------------
# 1. Configuration: every value the y3h4-converge runs used.
# ---------------------------------------------------------------------------------------------------------------
C = dict(
    env_name='box2dmaze-fourrooms-c-push-rand-v0',
    goal_coordinates=(0, 1, 2, 3),  # A goal is the ball's (x, y, vx, vy).
    rest_coordinates=(2, 3),        # Commanded goals have zero velocity.
    heads=4,                        # Independent CRL agents trained on the same batches.
    commit_fraction=0.25,           # Share of collection lanes that act on the heads' vote.
    lanes=500,                      # Parallel collection episodes per round.
    episode_length=1000,            # Steps per episode; also steps per round per lane.
    buffer_size=10_000_000,         # FIFO replay capacity (transitions).
    updates_per_round=16_000,
    update_chunk=128,               # Updates per compiled scan call.
    batch_size=256,                 # Rows per update; also the InfoNCE negatives.
    width=256, depth=8, repr_dim=64,
    lr=3e-4,
    future=0.99,                    # Hindsight offset ~ Geometric(1 - 0.99), truncated at the episode end.
    logsumexp_penalty=0.1,
    goal_jitter=0.02,               # Gaussian noise on the critic's goals (goal units).
    target_entropy=-0.5 * 2,        # -0.5 x action dimension.
    diagnostic_interval=16,         # Rolled-code accuracy on every 16th update.
    eval_horizon=1000, eval_batch_size=2640,
    stop_min_transitions=200_000_000, stop_window=20, stop_lag=100, stop_min_gain=0.01,
)
SMOKE = dict(lanes=8, episode_length=20, buffer_size=30_000, updates_per_round=16, update_chunk=8,
             batch_size=8, width=16, depth=4, repr_dim=8, eval_horizon=50)
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

# ---------------------------------------------------------------------------------------------------------------
# 2. Networks: phi(s, a, z), psi(g), the tanh-Gaussian actor pi(a | s, z) and the entropy temperature.
# ---------------------------------------------------------------------------------------------------------------
_lecun_uniform = nn.initializers.variance_scaling(1 / 3, 'fan_in', 'uniform')


class ResidualMLP(nn.Module):
    """Dense -> LayerNorm -> swish, then depth // 4 residual blocks of four such layers."""
    width: int
    depth: int
    out_dim: Any

    @nn.compact
    def __call__(self, x):
        def layer(h):
            h = nn.Dense(self.width, kernel_init=_lecun_uniform, bias_init=nn.initializers.zeros)(h)
            return nn.swish(nn.LayerNorm()(h))

        x = layer(x)
        for _ in range(self.depth // 4):
            identity = x
            for _ in range(4):
                x = layer(x)
            x = x + identity
        if self.out_dim is None:
            return x
        return nn.Dense(self.out_dim, kernel_init=_lecun_uniform, bias_init=nn.initializers.zeros)(x)


class EuclideanHead(nn.Module):
    """Projection head: affine map of [1, h] to D - 1 coordinates, with a zero coordinate prepended."""

    @nn.compact
    def __call__(self, tangent):
        x = jnp.concatenate([jnp.ones_like(tangent[..., :1]), tangent], -1)
        spatial = nn.Dense(tangent.shape[-1] - 1, name='htc')(x)
        return jnp.concatenate([jnp.zeros_like(spatial[..., :1]), spatial], -1)


class Encoder(nn.Module):
    """phi when called with (observations, [actions, z]); psi when called with goals alone."""
    width: int
    depth: int
    repr_dim: int

    @nn.compact
    def __call__(self, observations, actions=None):
        x = observations if actions is None else jnp.concatenate([observations, actions], axis=-1)
        return EuclideanHead()(ResidualMLP(self.width, self.depth, self.repr_dim)(x))


class Actor(nn.Module):
    """pi(a | s, z): tanh-squashed diagonal Gaussian with log-std in [-5, 2]."""
    width: int
    depth: int
    action_dim: int

    @nn.compact
    def __call__(self, observations, code, temperature=1.0):
        h = ResidualMLP(self.width, self.depth, None)(jnp.concatenate([observations, code], axis=-1))
        means = nn.Dense(self.action_dim, kernel_init=_lecun_uniform, bias_init=nn.initializers.zeros)(h)
        log_stds = nn.Dense(self.action_dim, kernel_init=_lecun_uniform, bias_init=nn.initializers.zeros)(h)
        log_stds = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (nn.tanh(log_stds) + 1)
        base = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        return distrax.Transformed(base, distrax.Block(distrax.Tanh(), ndims=1))


class Temperature(nn.Module):
    """alpha = exp(log_value), initialised at 1."""

    @nn.compact
    def __call__(self):
        return jnp.exp(self.param('log_value', lambda key: jnp.full((), jnp.log(1.0))))


class Networks(nn.Module):
    """All four modules of one head under one parameter tree (`modules_<name>`)."""
    modules: dict

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        if name is None:  # Initialisation: one argument tuple per module.
            return {key: self.modules[key](*value) for key, value in kwargs.items()}
        return self.modules[name](*args, **kwargs)


NETWORKS = OPTIMIZER = None


def configure(overrides=None):
    """Apply config overrides and build the network/optimizer singletons (before anything is traced)."""
    global NETWORKS, OPTIMIZER
    C.update(overrides or {})
    shape = dict(width=C['width'], depth=C['depth'])
    NETWORKS = Networks(dict(sa_encoder=Encoder(repr_dim=C['repr_dim'], **shape),
                             g_encoder=Encoder(repr_dim=C['repr_dim'], **shape),
                             actor=Actor(action_dim=2, **shape),
                             alpha=Temperature()))
    OPTIMIZER = optax.adam(C['lr'])


def phi(params, observations, actions, z):
    return NETWORKS.apply({'params': params}, observations, jnp.concatenate([actions, z], -1), name='sa_encoder')


def psi(params, goals):
    return NETWORKS.apply({'params': params}, goals, name='g_encoder')


def policy(params, observations, code, temperature=1.0):
    return NETWORKS.apply({'params': params}, observations, code, temperature=temperature, name='actor')


def temperature_value(params):
    return NETWORKS.apply({'params': params}, name='alpha')


def l2_logits(x, y):
    """All pairs: logits[i, j] = -||x_i - y_j||."""
    return -jnp.sqrt(jnp.sum(jnp.square(x[:, None, :] - y[None, :, :]), axis=-1) + 1e-12)


def l2_pair(x, y):
    """Matched rows: -||x_i - y_i||."""
    return -jnp.sqrt(jnp.sum(jnp.square(x - y), axis=-1) + 1e-12)


# ---------------------------------------------------------------------------------------------------------------
# 3. Heads: four independently initialised agents, stacked along a leading axis of size 4.
# ---------------------------------------------------------------------------------------------------------------
def init_head(seed, example):
    rng, init_rng = jax.random.split(jax.random.PRNGKey(seed))
    params = NETWORKS.init(init_rng, **example)['params']
    return dict(params=params, opt_state=OPTIMIZER.init(params), rng=rng)


def init_heads(seed, observations, actions, goals):
    example = dict(sa_encoder=(observations, jnp.concatenate([actions, jnp.zeros((len(actions), C['repr_dim']))], -1)),
                   g_encoder=(goals,),
                   actor=(observations, jnp.zeros((len(observations), C['repr_dim']))),
                   alpha=())
    seeds = [seed] + [int(np.random.SeedSequence([seed, 421, i]).generate_state(1)[0]) for i in range(1, C['heads'])]
    heads = [init_head(s, example) for s in seeds]
    return jax.tree_util.tree_map(lambda *x: jnp.stack(x), *heads)


# ---------------------------------------------------------------------------------------------------------------
# 4. One head's loss: the critic's InfoNCE with z = psi(commanded goal), plus the actor and temperature losses.
# ---------------------------------------------------------------------------------------------------------------
def head_loss(params, batch, rng, step):
    frozen = jax.lax.stop_gradient(params)  # The head as it was before this update.
    critic_rng, actor_rng = jax.random.split(rng)
    obs, actions = batch['observations'], batch['actions']
    with jax.default_matmul_precision('highest'):
        # Critic: row i's positive is its own (jittered) hindsight goal, the other 255 rows' goals are negatives.
        noise = jax.random.normal(jax.random.fold_in(critic_rng, 2), batch['value_goals'].shape)
        goals = batch['value_goals'] + C['goal_jitter'] * noise
        z = psi(params, batch['commanded_goals'])  # Y3: the code carries gradient into psi.
        sa = phi(params, obs, actions, z)
        g = psi(params, goals)
        logits = l2_logits(sa, g)
        lse = jax.nn.logsumexp(logits, axis=1)
        critic_loss = -jnp.mean(jnp.diagonal(logits) - lse)
        critic_loss += C['logsumexp_penalty'] * jnp.mean(jnp.square(jax.nn.logsumexp(logits + 1e-6, axis=1)))

        def rolled_accuracy(_):  # Diagnostic: shift every row's z by one row; does phi still find the positive?
            rolled = phi(frozen, obs, actions, jnp.roll(jax.lax.stop_gradient(z), 1, axis=0))
            hits = jnp.argmax(l2_logits(rolled, jax.lax.stop_gradient(g)), axis=1) == jnp.arange(len(obs))
            return jnp.mean(hits).astype(jnp.float32)
        rolled = jax.lax.cond(step % C['diagnostic_interval'] == 0, rolled_accuracy,
                              lambda _: jnp.asarray(jnp.nan, jnp.float32), None)

    with jax.default_matmul_precision('highest'):
        # Actor: act on z = psi(g) for the row's own (unjittered) hindsight goal, scored by the pre-update critic.
        code = jax.lax.stop_gradient(psi(frozen, batch['actor_goals']))
        dist = policy(params, obs, code)
        new_actions, logp = dist.sample_and_log_prob(seed=actor_rng)
        score = l2_pair(phi(frozen, obs, new_actions, code), code)
        alpha = temperature_value(frozen)
        objective = jax.lax.stop_gradient(alpha) * logp - score
        entropy = -jax.lax.stop_gradient(logp).mean()
        actor_loss = objective.mean() + temperature_value(params) * (entropy - C['target_entropy'])

    info = dict(critic_loss=critic_loss, actor_loss=objective.mean(), entropy=entropy, alpha=alpha,
                accuracy=jnp.mean(jnp.argmax(logits, axis=1) == jnp.arange(len(obs))), rolled_code_accuracy=rolled)
    return critic_loss + actor_loss, info


# ---------------------------------------------------------------------------------------------------------------
# 5. Updates: one Adam step per head on the same batch (vmapped), skipped if anything is non-finite.
# ---------------------------------------------------------------------------------------------------------------
def head_update(head, batch, step):
    new_rng, rng = jax.random.split(head['rng'])
    grads, info = jax.grad(head_loss, has_aux=True)(head['params'], batch, rng, step)
    updates, opt_state = OPTIMIZER.update(grads, head['opt_state'], head['params'])
    info['grad_norm'] = optax.global_norm(grads)
    return dict(params=optax.apply_updates(head['params'], updates), opt_state=opt_state, rng=new_rng), info


def ensemble_update(heads, batch, step):
    new, info = jax.vmap(head_update, in_axes=(0, None, None))(heads, batch, step)
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for k, v in info.items() if k != 'rolled_code_accuracy']))
    finite &= jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(batch)]))
    heads = jax.lax.cond(finite, lambda: new, lambda: heads)
    return heads, {k: v.mean() for k, v in info.items()}


# ---------------------------------------------------------------------------------------------------------------
# 6. Replay: a FIFO ring of transitions with, per row, the steps left in its episode and its commanded goal.
# ---------------------------------------------------------------------------------------------------------------
class Replay:
    def __init__(self, capacity, observation_dim, action_dim, goal_dim):
        self.capacity, self.written = capacity, 0
        self.observations = np.zeros((capacity, observation_dim), np.float32)
        self.actions = np.zeros((capacity, action_dim), np.float32)
        self.next_goals = np.zeros((capacity, goal_dim), np.float32)       # Goal view of s_{t+1}, velocity kept.
        self.commanded_goals = np.zeros((capacity, goal_dim), np.float32)  # The episode's commanded goal.
        self.remaining = np.ones(capacity, np.int32)                       # Transitions from this row to episode end.

    @property
    def size(self):
        return min(self.written, self.capacity)

    def add(self, observations, actions, next_observations, episode_ids, commanded_goals):
        n = len(observations)
        last = np.flatnonzero(np.r_[episode_ids[1:] != episode_ids[:-1], True])  # Each episode's final row.
        rows = np.arange(n)
        slots = (self.written + rows) % self.capacity
        self.observations[slots] = observations
        self.actions[slots] = actions
        self.next_goals[slots] = goal_observation(next_observations)
        self.commanded_goals[slots] = commanded_goals
        episode = np.searchsorted(last, rows)
        first = np.r_[0, last[:-1] + 1][episode]
        first = first + (rows - first) // C['episode_length'] * C['episode_length']  # Cut into 1,000-step segments.
        end = np.minimum(last[episode], first + C['episode_length'] - 1)
        self.remaining[slots] = end - rows + 1
        self.written += n

    def device(self):
        return {k: jnp.asarray(getattr(self, k)) for k in
                ('observations', 'actions', 'next_goals', 'commanded_goals', 'remaining')}


def goal_observation(states):
    return states[..., list(C['goal_coordinates'])]


def commanded_goal(states):
    goals = goal_observation(states).copy()
    goals[..., [C['goal_coordinates'].index(i) for i in C['rest_coordinates']]] = 0.
    return goals


def sample_batch(replay, key, size):
    """Uniform rows; each row's positive is the goal view of the state Delta >= 1 steps later in its own episode."""
    key, rows_key, future_key = jax.random.split(key, 3)
    rows = jax.random.randint(rows_key, (C['batch_size'],), 0, size)
    remaining = replay['remaining'][rows]
    u = jax.random.uniform(future_key, (C['batch_size'],), minval=1e-8, maxval=1.)
    log_d = jnp.log(jnp.asarray(C['future'], jnp.float32))
    offset = jnp.floor(jnp.log1p(u * jnp.expm1(remaining * log_d)) / log_d)  # Inverse CDF, truncated geometric.
    offset = jnp.clip(offset.astype(jnp.int32), 0, remaining - 1)
    goals = replay['next_goals'][(rows + offset) % replay['observations'].shape[0]]
    return key, dict(observations=replay['observations'][rows], actions=replay['actions'][rows],
                     value_goals=goals, actor_goals=goals, commanded_goals=replay['commanded_goals'][rows])


@partial(jax.jit, static_argnames='steps')
def train_chunk(heads, replay, key, size, steps):
    def body(carry, step):
        heads, key = carry
        key, batch = sample_batch(replay, key, size)
        heads, info = ensemble_update(heads, batch, step)
        return (heads, key), info
    (heads, key), info = jax.lax.scan(body, (heads, key), jnp.arange(steps, dtype=jnp.int32))
    return heads, key, {k: jnp.nanmean(v) if k == 'rolled_code_accuracy' else v.mean() for k, v in info.items()}


# ---------------------------------------------------------------------------------------------------------------
# 7. Acting: each head proposes its mean action, all heads score all proposals, the best mean score wins.
# ---------------------------------------------------------------------------------------------------------------
def head_score(params, observations, actions, goals):
    z = jax.lax.stop_gradient(psi(params, goals))
    return l2_pair(phi(params, observations, actions, z), psi(params, goals))


def head_action(params, observations, goals, key, temperature):
    dist = policy(params, observations, jax.lax.stop_gradient(psi(params, goals)), temperature)
    if key is None:
        return jnp.clip(dist.bijector.forward(dist.distribution.mode()), -1, 1)
    return jnp.clip(dist.sample(seed=key), -1, 1)


@jax.jit
def act(params, observations, goals, key, temperature, lane_ids=None):
    with jax.default_matmul_precision('highest'):
        means = jax.vmap(head_action, (0, None, None, None, None))(params, observations, goals, None, 0.)
        scores = jax.vmap(lambda p: jax.vmap(lambda a: head_score(p, observations, a, goals))(means))(params)
        selected = jnp.argmax(scores.mean(0), axis=0)  # [scorer, proposer, lane] -> best proposer per lane.
        keys = jnp.stack([key] + [jax.random.fold_in(key, i) for i in range(1, C['heads'])])
        actions = jax.vmap(head_action, (0, None, None, 0, None))(params, observations, goals, keys, temperature)
    if lane_ids is not None:  # Collection: 25% of lanes follow the vote, the rest are pinned to one head each.
        commit = int(C['commit_fraction'] * C['lanes'])
        selected = jnp.where(lane_ids < commit, selected, (lane_ids - commit) % C['heads'])
    return actions[selected, jnp.arange(observations.shape[0])]


@partial(jax.jit, static_argnums=1)
def collection_keys(key, steps):
    return jax.lax.scan(lambda k, _: tuple(jax.random.split(k)), key, None, length=steps)[1]


class EvalPolicy(flax.struct.PyTreeNode):
    """The repository's lattice evaluator calls sample_actions(observations, goals, seed, temperature=0)."""
    params: Any

    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        return act(self.params, observations, goals, seed, temperature)


# ---------------------------------------------------------------------------------------------------------------
# 8. Collection: 500 lanes, one 1,000-step episode each per round, fresh native start, goal = a replay state at rest.
# ---------------------------------------------------------------------------------------------------------------
def spawn(env, rng, goal_pool):
    starts = env.reset(int(rng.integers(2**31)))
    goals = goal_pool[rng.integers(len(goal_pool), size=env.num_envs)].copy()
    return dict(starts=starts, goals=goals)


def collect(env, params, lanes, key, rng, goal_pool, first_episode):
    n, steps = env.num_envs, C['episode_length']
    observations = env.reset_at(lanes['starts'], lanes['goals']).copy()
    goals = jnp.asarray(commanded_goal(lanes['goals']))
    states = np.empty((steps, n, env.obs_dim), np.float32)
    next_states, actions = np.empty_like(states), np.empty((steps, n, env.action_dim), np.float32)
    lane_ids, keys = jnp.arange(n, dtype=jnp.int32), np.asarray(collection_keys(key, steps))
    for s in range(steps):
        states[s] = observations
        actions[s] = np.asarray(act(params, jnp.asarray(observations), goals, keys[s], 1.0, lane_ids))
        _, physical, _ = env.step(actions[s])
        observations = next_states[s] = np.asarray(physical, np.float32)
    lane_major = lambda x: x.swapaxes(0, 1).reshape((-1,) + x.shape[2:])
    batch = dict(observations=lane_major(states), actions=lane_major(actions), next_observations=lane_major(next_states),
                 episode_ids=np.repeat(first_episode + np.arange(n), steps),
                 commanded_goals=np.repeat(np.asarray(goals), steps, axis=0))
    return batch, spawn(env, rng, goal_pool)  # The next round's episodes; goals from this round's replay snapshot.


# ---------------------------------------------------------------------------------------------------------------
# 9. The loop: train on the seed data, then per round collect -> add -> 16,000 updates -> lattice evaluation.
# ---------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--save_dir', help='Run directory (four-rooms run; optional with --demo).')
    parser.add_argument('--rounds', type=int, help='Rounds after round 0 (default 1000; 20 with --demo).')
    parser.add_argument('--repo', default=str(REPO), help='unsupervised-rl checkout holding the four-rooms harness.')
    parser.add_argument('--inputs', help='Seed data and fixtures (default <repo>/runs/r2crl-p1-20261021/inputs).')
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--smoke', action='store_true', help='Tiny networks and lanes, for a quick check.')
    parser.add_argument('--demo', action='store_true', help='Point-mass stand-in for four rooms; needs no repository.')
    args = parser.parse_args()
    if args.demo:
        return demo(args)
    if args.save_dir is None:
        parser.error('--save_dir is required for the four-rooms run')
    args.rounds = 1000 if args.rounds is None else args.rounds
    h = load_harness(args.repo)
    args.inputs = args.inputs or os.path.join(args.repo, 'runs/r2crl-p1-20261021/inputs')
    configure(SMOKE if args.smoke else None)
    os.makedirs(args.save_dir, exist_ok=True)
    rng, key, sample_key = np.random.default_rng(args.seed), jax.random.PRNGKey(args.seed), jax.random.PRNGKey(args.seed + 1000)

    with np.load(os.path.join(args.inputs, 'seed.npz')) as data:
        seed = {k: data[k] for k in ('observations', 'actions', 'next_observations', 'episode_ids')}
    ids = seed['episode_ids']  # Seed episodes were not commanded: use the rest view of where each one ended.
    last = len(ids) - 1 - np.unique(ids[::-1], return_index=True)[1]
    final = dict(zip(np.unique(ids), seed['next_observations'][last]))
    replay = Replay(C['buffer_size'], seed['observations'].shape[1], seed['actions'].shape[1], len(C['goal_coordinates']))
    replay.add(seed['observations'], seed['actions'], seed['next_observations'], ids,
               commanded_goal(np.stack([final[i] for i in ids])))
    next_episode = int(ids.max()) + 1
    example_goal = jnp.asarray(commanded_goal(seed['observations'][:1]))
    heads = init_heads(args.seed, jnp.asarray(seed['observations'][:1]), jnp.asarray(seed['actions'][:1]), example_goal)

    env = h.CRoomsVectorEnv(C['env_name'], C['lanes'], horizon=2**30, seed=args.seed, num_threads=args.threads)
    view = h.ObservationView(goals=C['goal_coordinates'], zero_goal_coordinates=C['rest_coordinates'])
    eval_env = h.make_online_env(C['env_name'])
    eval_env.unwrapped.horizon = C['eval_horizon']
    eval_config = h.evaluation_config(eval_env, dict(eval_goal_dims=(0, 1), eval_goal_radius=0.16, eval_velocity_dims=(2, 3),
                                                     eval_stop_seconds=1.0, eval_velocity_scale=2.0, eval_rest_speed=0.02))
    pairs = h.prepare_goal_pairs(h.load_pairs(os.path.join(args.inputs, 'pairs.npz'), env.obs_dim, args.seed + 1),
                                 eval_config)
    lattice = h.load_pairs(os.path.join(args.inputs, 'lattice.npz'), env.obs_dim, args.seed + 1)
    lattice = h.prepare_lattice(h.complete_lattice_goals(lattice, eval_env, 'none'), eval_config)
    goal_mode = h.task_goal_mode(pairs, eval_config, 'multi_goal')

    lanes, collected, history = None, 0, []
    for iteration in range(args.rounds + 1):
        started = time.time()
        if iteration:  # Round 0 trains and evaluates on the seed data only.
            goal_pool = replay.observations[:replay.size]  # Snapshot view; rows are added after collection.
            lanes = lanes or spawn(env, rng, goal_pool)
            key, _, run_key = jax.random.split(key, 3)
            batch, lanes = collect(env, heads['params'], lanes, run_key, rng, goal_pool, next_episode)
            next_episode += C['lanes']
            replay.add(batch['observations'], batch['actions'], batch['next_observations'], batch['episode_ids'],
                       batch['commanded_goals'])
            collected += len(batch['observations'])
        device_replay, infos = replay.device(), []
        for done in range(0, C['updates_per_round'], C['update_chunk']):
            steps = min(C['update_chunk'], C['updates_per_round'] - done)
            heads, sample_key, info = train_chunk(heads, device_replay, sample_key, np.int32(replay.size), steps)
            infos.append(info)
        # The repository then samples one 4,096-row diagnostic batch, which advances the sampler key once.
        sample_key = jax.random.split(sample_key, 3)[0]
        with open(os.path.join(args.save_dir, 'params_latest.pkl'), 'wb') as file:
            pickle.dump(dict(iteration=iteration, params=jax.device_get(heads['params'])), file)
        result = h.evaluate_suites(h.GoalReadoutPolicy(EvalPolicy(heads['params']), view), eval_env, pairs, lattice,
                                   eval_config, C['eval_horizon'], goal_mode, batch_size=C['eval_batch_size'],
                                   env_name=C['env_name'], num_threads=args.threads)
        row = dict(iteration=iteration, collected_transitions=collected, lattice_coverage=result['lattice_coverage'],
                   lattice_hard_stop=result.get('lattice_hard_stop'), seconds=time.time() - started,
                   **{k: float(np.mean([float(i[k]) for i in infos])) for k in infos[0]})
        history.append(row['lattice_coverage'])
        write_row(os.path.join(args.save_dir, 'rounds.csv'), row)
        print(json.dumps(row), flush=True)
        if converged(history, collected):
            print('stop rule: converged', flush=True)
            break


def write_row(path, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def converged(history, collected):
    """From 200M transitions: the last 20 rounds gain under 1 point over the 20 rounds ending 100 rounds earlier."""
    w, lag = C['stop_window'], C['stop_lag']
    if collected < C['stop_min_transitions'] or len(history) < w + lag:
        return False
    return np.mean(history[-w:]) - np.mean(history[-w - lag:-lag]) < C['stop_min_gain']


# ---------------------------------------------------------------------------------------------------------------
# 10. --demo: the same learner, replay, voting and collection on a point mass, with no repository code.
# ---------------------------------------------------------------------------------------------------------------
DEMO = dict(lanes=64, episode_length=50, buffer_size=200_000, updates_per_round=500, update_chunk=100,
            batch_size=256, width=128, depth=4, repr_dim=32)


class PointMassEnv:
    """Stand-in for the four-rooms simulator with its interface and state layout, NOT the benchmark.

    A point mass in the box [-1, 1]^2. The state is [x, y, vx, vy, 0, ..., 0] (10 numbers, so the goal view
    (0, 1, 2, 3) and the rest coordinates (2, 3) mean the same as for the rooms ball), driven by 2-D accelerations.
    """
    obs_dim, action_dim = 10, 2

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.states = np.zeros((num_envs, self.obs_dim), np.float32)

    def reset(self, seed):
        states = np.zeros((self.num_envs, self.obs_dim), np.float32)
        states[:, :2] = np.random.default_rng(seed).uniform(-0.9, 0.9, (self.num_envs, 2))
        return states

    def reset_at(self, starts, goals):
        self.states = np.asarray(starts, np.float32).copy()
        return self.states.copy()

    def step(self, actions):
        velocity = np.clip(0.8 * self.states[:, 2:4] + 0.2 * np.clip(actions, -1, 1), -1, 1)
        position = self.states[:, :2] + 0.05 * velocity
        velocity[np.abs(position) > 1] = 0.  # Inelastic walls.
        self.states[:, :2], self.states[:, 2:4] = np.clip(position, -1, 1), velocity
        return self.states.copy(), self.states.copy(), np.zeros(self.num_envs, bool)


def demo_success(params, env, starts, goals, steps, radius=0.1):
    """Share of start/goal pairs where the deterministic voted policy brings the mass within `radius` of the goal."""
    observations, target = env.reset_at(starts, goals), jnp.asarray(commanded_goal(goals))
    reached, key = np.zeros(len(starts), bool), jax.random.PRNGKey(0)
    for _ in range(steps):
        key, step_key = jax.random.split(key)
        _, observations, _ = env.step(np.asarray(act(params, jnp.asarray(observations), target, step_key, 0.)))
        reached |= np.linalg.norm(observations[:, :2] - goals[:, :2], axis=1) < radius
    return float(reached.mean())


def demo(args):
    configure(DEMO)
    rounds = 20 if args.rounds is None else args.rounds
    rng, key, sample_key = np.random.default_rng(args.seed), jax.random.PRNGKey(args.seed), jax.random.PRNGKey(args.seed + 1000)
    env = PointMassEnv(C['lanes'])
    n, steps = env.num_envs, C['episode_length']
    lane_major = lambda x: np.stack(x).swapaxes(0, 1).reshape((-1,) + np.shape(x[0])[1:])
    replay = Replay(C['buffer_size'], env.obs_dim, env.action_dim, len(C['goal_coordinates']))
    next_episode = 0
    for _ in range(4):  # Seed data (the four-rooms run loads seed.npz): random actions, goal = where each episode ended.
        observations, states, actions, next_states = env.reset_at(env.reset(int(rng.integers(2**31))), None), [], [], []
        for _ in range(steps):
            states.append(observations)
            actions.append(rng.uniform(-1, 1, (n, env.action_dim)).astype(np.float32))
            _, observations, _ = env.step(actions[-1])
            next_states.append(observations)
        replay.add(lane_major(states), lane_major(actions), lane_major(next_states),
                   np.repeat(next_episode + np.arange(n), steps), np.repeat(commanded_goal(observations), steps, axis=0))
        next_episode += n
    heads = init_heads(args.seed, jnp.asarray(replay.observations[:1]), jnp.asarray(replay.actions[:1]),
                       jnp.asarray(commanded_goal(replay.observations[:1])))
    eval_env, eval_rng = PointMassEnv(256), np.random.default_rng(args.seed + 7)
    eval_starts, eval_goals = (eval_env.reset(int(eval_rng.integers(2**31))) for _ in range(2))
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    lanes, collected = None, 0
    for iteration in range(rounds + 1):
        started = time.time()
        if iteration:
            goal_pool = replay.observations[:replay.size]
            lanes = lanes or spawn(env, rng, goal_pool)
            key, _, run_key = jax.random.split(key, 3)
            batch, lanes = collect(env, heads['params'], lanes, run_key, rng, goal_pool, next_episode)
            next_episode += C['lanes']
            replay.add(batch['observations'], batch['actions'], batch['next_observations'], batch['episode_ids'],
                       batch['commanded_goals'])
            collected += len(batch['observations'])
        device_replay, infos = replay.device(), []
        for done in range(0, C['updates_per_round'], C['update_chunk']):
            heads, sample_key, info = train_chunk(heads, device_replay, sample_key, np.int32(replay.size),
                                                  min(C['update_chunk'], C['updates_per_round'] - done))
            infos.append(info)
        row = dict(iteration=iteration, collected_transitions=collected,
                   success=demo_success(heads['params'], eval_env, eval_starts, eval_goals, 2 * steps),
                   seconds=round(time.time() - started, 2),
                   **{k: float(np.mean([float(i[k]) for i in infos])) for k in infos[0]})
        print(json.dumps(row), flush=True)
        if args.save_dir:
            write_row(os.path.join(args.save_dir, 'rounds.csv'), row)


if __name__ == '__main__':
    main()
