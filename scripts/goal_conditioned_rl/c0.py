"""C0: plain goal-conditioned contrastive RL (CRL), standalone, Python 3.12.
Default preset 'c0' = the R2CRL baseline C0 learner: one actor/critic/temperature,
dot-product critic phi(s,a).psi(g), no projection heads, no goal jitter, no
ensemble, no prior, no residual head; width 256, depth 8, repr 64, batch 256.
Other presets (best/jg0/g0/full/incumbent) are kept as switches.
Install: pip install jax flax optax distrax ml-collections numpy
Run: python c0.py --demo --steps 2000  (small point-motion example, not four rooms)
Or: python c0.py --data replay.npz --steps 10000 --save learner.msgpack
Replay: observations [N,Ds], actions [N,Da] in [-1,1], next_goal_observations
[N,Dg], remaining [N] = transitions through episode end (inclusive).
Use chronological unwrapped rows and prepared goal coordinates. Future achieved
goals are positives; other batch rows are negatives. See get_config for switches.
API: Learner.create(seed,batch,cfg), update(batch), sample_actions(obs,goals).
Frozen source: rooms100-20261028 / 9d977723bffce9bccaefdcfe1ea86dcaf4e3b2907a03db61651bfd823bd3a09a.
The MIT License (MIT). Copyright (c) 2025 FQL Authors.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

_init = nn.initializers.variance_scaling(1 / 3, 'fan_in', 'uniform')

class ResidualMLP(nn.Module):
    width: int
    depth: int
    out_dim: int | None
    layer_norm: bool = True
    use_relu: bool = False
    @nn.compact
    def __call__(self, x):
        def layer(h):
            h = nn.Dense(self.width, kernel_init=_init, bias_init=nn.initializers.zeros)(h)
            if self.layer_norm:
                h = nn.LayerNorm()(h)
            return nn.relu(h) if self.use_relu else nn.swish(h)
        x = layer(x)
        for _ in range(self.depth // 4):
            identity = x
            for _ in range(4):
                x = layer(x)
            x = x + identity
        return x if self.out_dim is None else nn.Dense(
            self.out_dim, kernel_init=_init, bias_init=nn.initializers.zeros)(x)

class ResidualHead(nn.Module):
    layers: int
    width: int
    @nn.compact
    def __call__(self, h):
        for i in range(self.layers):
            z = nn.silu(nn.Dense(self.width, name=f'hidden_{i}')(h))
            raw = nn.Dense(2 * h.shape[-1], kernel_init=nn.initializers.zeros,
                           bias_init=nn.initializers.zeros, name=f'output_{i}')(z)
            u, v = jnp.split(raw, 2, axis=-1)
            h = h + (u + v) / jnp.sqrt(2.)
        return h

class EuclideanHead(nn.Module):
    @nn.compact
    def __call__(self, tangent):
        x = jnp.concatenate([jnp.ones_like(tangent[..., :1]), tangent], -1)
        spatial = nn.Dense(tangent.shape[-1] - 1, name='htc')(x)
        return jnp.concatenate([jnp.zeros_like(spatial[..., :1]), spatial], -1)

class ScalingCRLEncoder(nn.Module):
    config: Any
    residual: bool = False
    @nn.compact
    def __call__(self, observations, actions=None):
        c = self.config
        x = observations if actions is None else jnp.concatenate([observations, actions], -1)
        x = ResidualMLP(c['network_width'], c['network_depth'], c['repr_dim'],
                        c['layer_norm'], c['use_relu'])(x)
        if self.residual:
            x = ResidualHead(c['transform_layers'], c['transform_width'], name='transform')(x)
        return EuclideanHead()(x) if c['projection_head'] else x

class TransformedWithMode(distrax.Transformed):
    def mode(self):
        return self.bijector.forward(self.distribution.mode())

class ScalingCRLActor(nn.Module):
    config: Any
    action_dim: int
    @nn.compact
    def __call__(self, observations, goals, temperature=1.):
        c = self.config
        x = jnp.concatenate([observations, goals], -1)
        h = ResidualMLP(c['network_width'], c['network_depth'], None,
                        c['layer_norm'], c['use_relu'])(x)
        means = nn.Dense(self.action_dim, kernel_init=_init, bias_init=nn.initializers.zeros)(h)
        log_stds = nn.Dense(self.action_dim, kernel_init=_init, bias_init=nn.initializers.zeros)(h)
        log_stds = -5. + .5 * 7. * (jnp.tanh(log_stds) + 1.)
        dist = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        return TransformedWithMode(dist, distrax.Block(distrax.Tanh(), ndims=1))

class LogParam(nn.Module):
    @nn.compact
    def __call__(self):
        return jnp.exp(self.param('log_value', lambda key: jnp.full((), jnp.log(1.))))

class ModuleDict(nn.Module):
    modules: dict
    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        if name is None:
            return {k: self.modules[k](*v) for k, v in kwargs.items()}
        return self.modules[name](*args, **kwargs)

def energy(phi, psi, kind, paired=False):
    if kind == 'dot':
        return (phi * psi).sum(-1) if paired else phi @ psi.T
    diff = phi - psi if paired else phi[:, None, :] - psi[None, :, :]
    return -jnp.sqrt(jnp.square(diff).sum(-1) + 1e-12)

class _Head(flax.struct.PyTreeNode):
    params: Any
    prior_params: Any
    opt_state: Any
    rng: Any
    step: Any
    model: Any = flax.struct.field(pytree_node=False)
    config: Any = flax.struct.field(pytree_node=False)
    def apply(self, name, *args, params=None, **kwargs):
        return self.model.apply({'params': self.params if params is None else params},
                                *args, name=name, **kwargs)
    def score(self, params, obs, actions, goals, paired=False):
        def evaluate(p):
            phi = self.apply('sa_encoder', obs, actions, params=p)
            psi = self.apply('g_encoder', goals, params=p)
            return energy(phi, psi, self.config['energy'], paired)
        score = evaluate(params)
        beta = self.config['ensemble_prior_scale']
        return score + beta * evaluate(self.prior_params) if beta else score
    def loss(self, params, batch, key):
        with jax.default_matmul_precision('highest'):
            c = self.config
            critic_key, actor_key = jax.random.split(key)
            obs, actions, goals = batch['observations'], batch['actions'], batch['value_goals']
            if c['critic_goal_jitter']:
                goals = goals + c['critic_goal_jitter'] * jax.random.normal(
                    jax.random.fold_in(critic_key, 2), goals.shape)
            logits = self.score(params, obs, actions, goals)
            lse = jax.nn.logsumexp(logits, axis=1)
            weights = batch.get('weights', jnp.ones(len(obs)))
            nce = lse - jnp.diagonal(logits)
            penalty = jnp.square(jax.nn.logsumexp(logits + 1e-6, axis=1))
            critic = jnp.mean(weights * nce) + c['logsumexp_penalty_coef'] * jnp.mean(weights * penalty)
            goals = batch['actor_goals']  # Jitter never changes actor targets or acting.
            dist = self.apply('actor', obs, goals, params=params)
            actions, logp = dist.sample_and_log_prob(seed=actor_key)
            score = self.score(self.params, obs, actions, goals, paired=True)
            alpha = self.apply('alpha') if c['fixed_alpha'] is None else jnp.asarray(c['fixed_alpha'])
            objective = -score if c['disable_entropy'] else jax.lax.stop_gradient(alpha) * logp - score
            actor = jnp.mean(weights * objective)
            entropy = -jax.lax.stop_gradient(weights * logp).mean()
            dual = 0. if c['fixed_alpha'] is not None else self.apply('alpha', params=params) * (
                entropy - c['target_entropy'])
            return critic + actor + dual, dict(critic_loss=critic, actor_loss=actor,
                entropy=entropy, alpha=alpha, q=score.mean(), logsumexp=lse.mean())
    def update(self, batch):
        rng, key = jax.random.split(self.rng)
        (_, info), grads = jax.value_and_grad(self.loss, has_aux=True)(self.params, batch, key)
        updates, opt_state = optax.adam(self.config['lr']).update(grads, self.opt_state, self.params)
        return self.replace(params=optax.apply_updates(self.params, updates), opt_state=opt_state,
                            rng=rng, step=self.step + 1), info
    def act(self, obs, goals, seed=None, temperature=1.):
        dist = self.apply('actor', obs, goals, temperature=temperature)
        return jnp.clip(dist.mode() if seed is None else dist.sample(seed=seed), -1, 1)

class Learner(flax.struct.PyTreeNode):
    heads: _Head
    skipped_updates: Any
    @property
    def config(self):
        return self.heads.config
    @classmethod
    def create(cls, seed, example_batch, config=None):
        c = dict(get_config() if config is None else config)
        _validate_config(c)
        obs, actions, goals = (jnp.asarray(example_batch[k]) for k in
                              ('observations', 'actions', 'value_goals'))
        if any(x.ndim != 2 or x.shape[-1] < 1 for x in (obs, actions, goals)):
            raise ValueError('Example observations, actions and goals must be nonempty rank-2 arrays')
        c.update(observation_dim=obs.shape[-1], action_dim=actions.shape[-1], goal_dim=goals.shape[-1])
        if c['target_entropy'] is None:
            c['target_entropy'] = -c['entropy_param'] * actions.shape[-1]
        c = flax.core.FrozenDict(c)
        model = ModuleDict(dict(sa_encoder=ScalingCRLEncoder(c, c['phi_transform'] == 'residual'),
            g_encoder=ScalingCRLEncoder(c), actor=ScalingCRLActor(c, actions.shape[-1]), alpha=LogParam()))
        arguments = dict(sa_encoder=(obs, actions), g_encoder=(goals,), actor=(obs, goals), alpha=())
        heads = []
        for i in range(c['ensemble_heads']):
            head_seed = seed if i == 0 else int(np.random.SeedSequence([seed, 421, i]).generate_state(1)[0])
            prior_seed = int(np.random.SeedSequence([seed, 997, i]).generate_state(1)[0])
            rng, init_key = jax.random.split(jax.random.PRNGKey(head_seed))
            params = model.init(init_key, **arguments)['params']
            prior_key = jax.random.split(jax.random.PRNGKey(prior_seed))[1]
            prior = model.init(prior_key, **arguments)['params']
            prior = {k: prior[k] for k in ('modules_sa_encoder', 'modules_g_encoder')}
            heads.append(_Head(params, prior, optax.adam(c['lr']).init(params), rng,
                               jnp.asarray(1, jnp.int32), model, c))
        agent = cls(jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *heads), jnp.asarray(0, jnp.int32))
        agent.validate_batch(example_batch)
        return agent
    def validate_batch(self, batch):
        b = batch['observations'].shape[0]
        for key, width in [('observations', self.config['observation_dim']),
                           ('actions', self.config['action_dim']), ('value_goals', self.config['goal_dim']),
                           ('actor_goals', self.config['goal_dim'])]:
            x = batch[key]
            if b < 2 or x.shape != (b, width) or not jnp.issubdtype(x.dtype, jnp.floating):
                raise ValueError(f'{key} must be floating [{b}, {width}], with batch size >=2')
        if 'weights' in batch and batch['weights'].shape != (b,):
            raise ValueError('weights must have shape [batch]')
    @jax.jit
    def update(self, batch):
        self.validate_batch(batch)
        heads, metrics = jax.vmap(lambda h: h.update(batch))(self.heads)
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in
            jax.tree_util.tree_leaves((batch, heads.params, heads.opt_state, metrics))]))
        heads = jax.lax.cond(finite, lambda: heads, lambda: self.heads)
        info = {k: v.mean() for k, v in metrics.items()}
        info['skipped_update'] = (~finite).astype(jnp.float32)
        return self.replace(heads=heads, skipped_updates=self.skipped_updates + (~finite)), info
    @jax.jit
    def candidates(self, observations, goals):
        return jax.vmap(lambda h: h.act(observations, goals))(self.heads)
    @jax.jit
    def candidate_scores(self, observations, goals, candidates):
        return jax.vmap(lambda h: jax.vmap(lambda a: h.score(
            h.params, observations, a, goals, paired=True))(candidates))(self.heads)
    @jax.jit
    def goal_value(self, observations, actions, goals):
        return jax.vmap(lambda h: h.score(h.params, observations, actions, goals, paired=True))(self.heads).mean(0)
    @jax.jit
    def sample_actions(self, observations, goals, seed=None, temperature=1., lane_ids=None):
        """Batched [N,D] inputs; pass seed=None for evaluation, global lane_ids for collection."""
        c = self.config
        if observations.ndim != 2 or observations.shape[-1] != c['observation_dim']:
            raise ValueError('observations must have shape [N, observation_dim]')
        if goals.shape != (len(observations), c['goal_dim']):
            raise ValueError('goals must have shape [N, goal_dim]')
        means = self.candidates(observations, goals)
        selected = jnp.argmax(self.candidate_scores(observations, goals, means).mean(0), axis=0)
        if lane_ids is None and c['ensemble_eval_head'] >= 0:
            selected = jnp.full_like(selected, c['ensemble_eval_head'])
        actions = means
        if seed is not None:
            keys = jnp.stack([seed if i == 0 else jax.random.fold_in(seed, i)
                              for i in range(c['ensemble_heads'])])
            actions = jax.vmap(lambda h, k: h.act(observations, goals, k, temperature))(self.heads, keys)
        if lane_ids is not None:
            if lane_ids.shape != (len(observations),) or not jnp.issubdtype(lane_ids.dtype, jnp.integer):
                raise ValueError('lane_ids must be integer [N], stable global IDs before sharding')
            commit = int(c['ensemble_commit_fraction'] * c['ensemble_collection_lanes'])
            selected = jnp.where(lane_ids < commit, selected, (lane_ids - commit) % c['ensemble_heads'])
        return actions[selected, jnp.arange(len(observations))]
    def save(self, path):
        payload = dict(schema=1, config=dict(self.config), state=flax.serialization.to_state_dict(self))
        path = Path(path)
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_bytes(flax.serialization.msgpack_serialize(payload))
        temporary.replace(path)
    @classmethod
    def load(cls, path):
        payload = flax.serialization.msgpack_restore(Path(path).read_bytes())
        if payload['schema'] != 1:
            raise ValueError('Unknown standalone learner checkpoint schema')
        c = payload['config']
        batch = {k: jnp.zeros((2, c[d]), jnp.float32) for k, d in
                 [('observations', 'observation_dim'), ('actions', 'action_dim'),
                  ('value_goals', 'goal_dim'), ('actor_goals', 'goal_dim')]}
        return flax.serialization.from_state_dict(cls.create(0, batch, c), payload['state'])

@partial(jax.jit, static_argnames=('batch_size', 'future'))
def sample_replay(replay, key, size, batch_size=256, future=.99):
    """Episode-truncated geometric future sampler; next goals include the next state.
    Caller owns ring boundaries: valid rows [0,size) when partly filled,
    remaining>=1, never crossing an episode or an overwritten slot.
    """
    if batch_size < 2 or not 0 <= future <= 1:
        raise ValueError('batch_size >=2 and 0 <= future <=1 are required')
    key, rows_key, future_key = jax.random.split(key, 3)
    rows = jax.random.randint(rows_key, (batch_size,), 0, size)
    remaining = replay['remaining'][rows]
    u = jax.random.uniform(future_key, (batch_size,), minval=1e-8, maxval=1.)
    if future >= 1.:
        offset = jnp.floor(u * remaining)
    else:
        log_d = jnp.log(jnp.asarray(max(future, 1e-8), jnp.float32))
        offset = jnp.floor(jnp.log1p(u * jnp.expm1(remaining * log_d)) / log_d)
    offset = jnp.clip(offset.astype(jnp.int32), 0, remaining - 1)
    slots = (rows + offset) % replay['observations'].shape[0]
    goals = replay['next_goal_observations'][slots]
    batch = dict(observations=replay['observations'][rows], actions=replay['actions'][rows],
                 value_goals=goals, actor_goals=goals)
    if 'weights' in replay:
        batch['weights'] = replay['weights'][rows]
    return key, batch

def _validate_config(c):
    if c['energy'] not in ('dot', 'l2') or c['phi_transform'] not in ('none', 'residual'):
        raise ValueError('energy: dot/l2; phi_transform: none/residual')
    if c['projection_head'] and c['energy'] != 'l2':
        raise ValueError('The G0 projection requires L2; disable it for C0/dot')
    for name in ('ensemble_heads', 'network_width', 'repr_dim', 'transform_width', 'ensemble_collection_lanes'):
        if type(c[name]) is not int or c[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if type(c['network_depth']) is not int or c['network_depth'] < 4 or c['network_depth'] % 4:
        raise ValueError('network_depth counts four-layer residual blocks; use 4, 8, ...')
    if type(c['transform_layers']) is not int or c['transform_layers'] < 0:
        raise ValueError('transform_layers must be a nonnegative integer')
    if c['projection_head'] and c['repr_dim'] < 2:
        raise ValueError('G0 needs repr_dim >=2')
    for name in ('critic_goal_jitter', 'ensemble_prior_scale', 'logsumexp_penalty_coef', 'entropy_param'):
        if not np.isfinite(c[name]) or c[name] < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not np.isfinite(c['lr']) or c['lr'] <= 0 or not 0 <= c['ensemble_commit_fraction'] <= 1:
        raise ValueError('lr must be positive; collection commit fraction must be in [0,1]')
    if type(c['ensemble_eval_head']) is not int or not -1 <= c['ensemble_eval_head'] < c['ensemble_heads']:
        raise ValueError('ensemble_eval_head must be -1 or an existing head index')
    if c['target_entropy'] is not None and not np.isfinite(c['target_entropy']):
        raise ValueError('target_entropy must be None or finite')
    if c['fixed_alpha'] is not None and (not np.isfinite(c['fixed_alpha']) or c['fixed_alpha'] <= 0):
        raise ValueError('fixed_alpha must be None or finite and positive')

BEST = {'network_width': 512, 'ensemble_heads': 4, 'ensemble_prior_scale': 0.0, 'phi_transform': 'none', 'critic_goal_jitter': 0.02}

def get_config(mode='c0', **overrides):
    """All learner defaults in one place; overrides use campaign parameter names."""
    c = dict(network_width=256, network_depth=8, repr_dim=64, layer_norm=True,
        use_relu=False, energy='l2', projection_head=True, critic_goal_jitter=.02,
        ensemble_heads=4, ensemble_prior_scale=.1, phi_transform='residual',
        transform_layers=4, transform_width=128, lr=3e-4, logsumexp_penalty_coef=.1,
        entropy_param=.5, target_entropy=None, fixed_alpha=None, disable_entropy=False,
        ensemble_commit_fraction=.25, ensemble_collection_lanes=500, ensemble_eval_head=-1)
    if mode not in ('best', 'full', 'incumbent', 'jg0', 'g0', 'c0'):
        raise ValueError('preset must be best, full, incumbent, jg0, g0 or c0')
    if mode != 'full':
        c.update(ensemble_prior_scale=0., phi_transform='none')
    if mode in ('jg0', 'g0', 'c0'):
        c['ensemble_heads'] = 1
    if mode in ('g0', 'c0'):
        c['critic_goal_jitter'] = 0.
    if mode == 'c0':
        c.update(energy='dot', projection_head=False)
    if mode == 'best':
        c.update(BEST)
    unknown = set(overrides) - set(c)
    if unknown:
        raise ValueError(f'Unknown config fields: {sorted(unknown)}')
    c.update(overrides)
    _validate_config(c)
    return ml_collections.ConfigDict(c)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', help='Chronological replay NPZ with the four arrays above')
    p.add_argument('--demo', action='store_true', help='Small point-motion learning example')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save', default='learner.msgpack')
    p.add_argument('--preset', default='c0', choices=['best','full','incumbent','jg0','g0','c0'])
    p.add_argument('--config', default='{}', help='JSON overrides, e.g. {"ensemble_heads":1}')
    args = p.parse_args()
    if args.demo == bool(args.data) or args.steps < 1:
        p.error('Choose exactly one of --demo/--data, and steps >=1')
    overrides = dict(network_width=32, network_depth=4, repr_dim=16) if args.demo else {}
    overrides.update(json.loads(args.config))
    cfg = get_config(args.preset, **overrides)
    rng = np.random.default_rng(args.seed)
    if args.demo:
        # Real dynamics s[t+1]=clip(s[t]+0.15*a[t]); no rewards enter learning.
        episodes, horizon = 256, 32
        states = [rng.uniform(-1, 1, (episodes, 2)).astype(np.float32)]
        actions = rng.uniform(-1, 1, (horizon, episodes, 2)).astype(np.float32)
        for action in actions:
            states.append(np.clip(states[-1] + .15 * action, -1, 1))
        flatten = lambda x: np.asarray(x).swapaxes(0, 1).reshape(-1, 2)
        replay = dict(observations=flatten(states[:-1]), actions=flatten(actions),
            next_goal_observations=flatten(states[1:]),
            remaining=np.tile(np.arange(horizon, 0, -1), episodes).astype(np.int32))
    else:
        with np.load(args.data, allow_pickle=False) as data:
            replay = {k: data[k] for k in ('observations','actions','next_goal_observations','remaining')}
    size = len(replay['observations'])
    if size < 2 or any(len(v) != size for v in replay.values()):
        raise ValueError('Replay arrays need the same length >=2')
    ends = np.arange(size) + replay['remaining']
    if np.any(replay['remaining'] < 1) or np.any(ends > size):
        raise ValueError('Invalid episode ends; provide chronological, unwrapped replay')
    if not all(np.isfinite(v).all() for v in replay.values()):
        raise ValueError('Replay must be finite')
    replay = {k: jnp.asarray(v, jnp.int32 if k == 'remaining' else jnp.float32) for k,v in replay.items()}
    key, batch = sample_replay(replay, jax.random.PRNGKey(args.seed), size)
    agent = Learner.create(args.seed, batch, cfg)
    print('Configuration:', json.dumps(dict(cfg)), flush=True)
    for step in range(args.steps):
        key, batch = sample_replay(replay, key, size)
        agent, metrics = agent.update(batch)
        if (step + 1) % max(1, args.steps // 5) == 0:
            print(step + 1, {k: float(v) for k,v in metrics.items()}, flush=True)
    agent.save(args.save)
    print('Saved', args.save, flush=True)
    if args.demo:
        # Held-out starts/goals; reaching within .15 at any of 32 steps (toy metric).
        states = rng.uniform(-.8, .8, (128, 2)).astype(np.float32)
        goals = rng.uniform(-.8, .8, (128, 2)).astype(np.float32)
        hit = np.linalg.norm(states-goals, axis=1) < .15
        for _ in range(32):
            states = np.clip(states + .15*np.asarray(agent.sample_actions(states, goals)), -1, 1)
            hit |= np.linalg.norm(states-goals, axis=1) < .15
        print(f'Toy point-reaching success: {hit.mean():.1%}; NOT four-rooms.', flush=True)

if __name__ == '__main__':
    main()
