"""Numerical parity: our CRLAgent ``paper`` preset vs rafapi/contrastive_rl (contrastive_nce, state obs).

Reference side = rafapi contrastive/networks.py (_repr_fn / _critic_fn / _actor_fn, non-image, no twin q, no repr
norm) + Acme 0.4.0 NormalTanhDistribution (loc / softplus scale + min_scale), and contrastive/learning.py
critic_loss (MC, non-cpc branch) / actor_loss (random_goals 0.5, entropy_coefficient 0.0), optax.adam(3e-4, eps=1e-7).
The tanh-Gaussian sample is taken with shared noise (alpha = 0, so log_prob does not enter the loss).

Needs a venv with jax, optax, dm-haiku and torch (no Acme / reverb / TF needed); run from the repo root:
    <jaxenv>/bin/python scripts/goal_conditioned_rl/parity_paper.py
"""
import sys

sys.path.insert(0, "/data/rohanp/air-hockey-rl")
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch

from scripts.goal_conditioned_rl.crl_agent import CRLAgent, CRLConfig

jax.config.update("jax_enable_x64", False)
torch.set_num_threads(4)
S, A, G, B, H, R = 32, 2, 4, 256, (256, 256), 64
MIN_STD = 1e-6
rng = np.random.default_rng(0)


# ------------------------------------------------------------------ reference networks (rafapi networks.py)
def _repr_fn(obs, action):
    state, goal = obs[:, :S], obs[:, S:]
    sa_encoder = hk.nets.MLP(list(H) + [R], w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
                             activation=jax.nn.relu, name="sa_encoder")
    sa_repr = sa_encoder(jnp.concatenate([state, action], axis=-1))
    g_encoder = hk.nets.MLP(list(H) + [R], w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
                            activation=jax.nn.relu, name="g_encoder")
    return sa_repr, g_encoder(goal)


def _critic_fn(obs, action):
    sa, g = _repr_fn(obs, action)
    return jnp.einsum("ik,jk->ij", sa, g)


class NormalTanhDistribution(hk.Module):  # Acme 0.4.0, parameters only (distribution built below)
    def __init__(self, num_dimensions, min_scale=1e-3,
                 w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"), b_init=hk.initializers.Constant(0.)):
        super().__init__(name="Normal")
        self._min_scale = min_scale
        self._loc_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)
        self._scale_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)

    def __call__(self, inputs):
        loc = self._loc_layer(inputs)
        scale = jax.nn.softplus(self._scale_layer(inputs)) + self._min_scale
        return loc, scale


def _actor_fn(obs):
    net = hk.Sequential([
        hk.nets.MLP(list(H), w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
                    activation=jax.nn.relu, activate_final=True),
        NormalTanhDistribution(A, min_scale=MIN_STD)])
    return net(obs)


critic = hk.without_apply_rng(hk.transform(_critic_fn))
policy = hk.without_apply_rng(hk.transform(_actor_fn))
q_params = critic.init(jax.random.PRNGKey(1), jnp.zeros((1, S + G)), jnp.zeros((1, A)))
pi_params = policy.init(jax.random.PRNGKey(2), jnp.zeros((1, S + G)))


# ------------------------------------------------------------------ reference losses (rafapi learning.py)
def critic_loss(q_params, obs, action):
    I = jnp.eye(B)
    logits = critic.apply(q_params, obs, action)
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=I))
    correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
    return loss, dict(categorical_accuracy=jnp.mean(correct), logits_pos=jnp.sum(logits * I) / jnp.sum(I),
                      logits_neg=jnp.sum(logits * (1 - I)) / jnp.sum(1 - I))


def actor_loss(pi_params, q_params, obs, eps):
    state, goal = obs[:, :S], obs[:, S:]
    new_state = jnp.concatenate([state, state], axis=0)  # random_goals == 0.5
    new_goal = jnp.concatenate([goal, jnp.roll(goal, 1, axis=0)], axis=0)
    new_obs = jnp.concatenate([new_state, new_goal], axis=1)
    loc, scale = policy.apply(pi_params, new_obs)
    action = jnp.tanh(loc + scale * eps)  # Acme TanhTransformedDistribution reparameterised sample
    q_action = critic.apply(q_params, new_obs, action)
    alpha = 0.0  # entropy_coefficient 0.0 for state envs (lp_contrastive.py); alpha * log_prob == 0
    return jnp.mean(alpha * 0.0 - jnp.diag(q_action))


# ------------------------------------------------------------------ copy weights into our agent
agent = CRLAgent(CRLConfig.paper(), S, A, G)
assert agent.cfg.critic_loss == "nce" and agent.cfg.entropy == "off" and agent.cfg.random_goals == 0.5
assert agent.cfg.adam_eps == 1e-7 and agent.cfg.lr == 3e-4 and agent.cfg.actor_min_std == MIN_STD


def put(lin, p):
    with torch.no_grad():
        lin.weight.copy_(torch.as_tensor(np.asarray(p["w"]).T))
        lin.bias.copy_(torch.as_tensor(np.asarray(p["b"])))


for enc, name in ((agent.sa_encoder, "sa_encoder"), (agent.g_encoder, "g_encoder")):
    assert len(enc.layers) == 3
    for i, lin in enumerate(enc.layers):
        put(lin, q_params[f"{name}/~/linear_{i}"])
for i, lin in enumerate(agent.actor.trunk.layers):
    put(lin, pi_params[f"mlp/~/linear_{i}"])
put(agent.actor.loc, pi_params["Normal/~/linear"])
put(agent.actor.scale, pi_params["Normal/~/linear_1"])

# init statistics (independent draws, same distribution?)
ours0 = CRLAgent(CRLConfig.paper(), S, A, G)
for (n_ref, p_ref), lin in zip([("sa0", q_params["sa_encoder/~/linear_0"]), ("sa2", q_params["sa_encoder/~/linear_2"]),
                                ("pi0", pi_params["mlp/~/linear_0"]), ("loc", pi_params["Normal/~/linear"])],
                               [ours0.sa_encoder.layers[0], ours0.sa_encoder.layers[2], ours0.actor.trunk.layers[0],
                                ours0.actor.loc]):
    print(f"init {n_ref}: ref max|w| {np.abs(p_ref['w']).max():.4f} std {np.std(p_ref['w']):.4f} | ours max|w| "
          f"{lin.weight.abs().max():.4f} std {lin.weight.std():.4f}")

# ------------------------------------------------------------------ compare
s = rng.normal(size=(B, S)).astype(np.float32)
a = rng.uniform(-1, 1, size=(B, A)).astype(np.float32)
g = rng.normal(size=(B, G)).astype(np.float32)
eps = rng.normal(size=(2 * B, A)).astype(np.float32)
obs = jnp.asarray(np.concatenate([s, g], 1))

(c_ref, c_info), c_grads_ref = jax.value_and_grad(critic_loss, has_aux=True)(q_params, obs, jnp.asarray(a))
a_ref, a_grads_ref = jax.value_and_grad(actor_loss)(pi_params, q_params, obs, jnp.asarray(eps))

ts, ta, tg, te = (torch.as_tensor(x) for x in (s, a, g, eps))
c_t, info_t = agent.critic_loss(ts, ta, tg)
a_t, _, _ = agent.actor_loss(ts, tg, noise=te)
cg = torch.autograd.grad(c_t, list(agent.sa_encoder.parameters()) + list(agent.g_encoder.parameters()))
ag = torch.autograd.grad(a_t, list(agent.actor.parameters()))

worst = 0.0


def cmp(name, x_ref, x_t, rel=False):
    global worst
    x_ref, x_t = np.asarray(x_ref, np.float64), x_t.detach().double().numpy()
    d = np.abs(x_ref - x_t).max() / (np.abs(x_ref).max() + 1e-12 if rel else 1.0)
    worst = max(worst, d)
    print(f"{name:32s} {'rel' if rel else 'abs'} diff {d:.2e}")


cmp("critic loss", c_ref, c_t)
cmp("accuracy", c_info["categorical_accuracy"], info_t["accuracy"])
cmp("logits_pos", c_info["logits_pos"], info_t["logits_pos"])
cmp("logits_neg", c_info["logits_neg"], info_t["logits_neg"])
cmp("actor loss", a_ref, a_t)
names = [f"{e}/~/linear_{i}" for e in ("sa_encoder", "g_encoder") for i in range(3)]
for k, n in enumerate(names):
    cmp(f"dC/d {n} w", c_grads_ref[n]["w"].T, cg[2 * k], rel=True)
    cmp(f"dC/d {n} b", c_grads_ref[n]["b"], cg[2 * k + 1], rel=True)
pnames = ["mlp/~/linear_0", "mlp/~/linear_1", "Normal/~/linear", "Normal/~/linear_1"]
for k, n in enumerate(pnames):
    cmp(f"dA/d {n} w", a_grads_ref[n]["w"].T, ag[2 * k], rel=True)
    cmp(f"dA/d {n} b", a_grads_ref[n]["b"], ag[2 * k + 1], rel=True)

# one Adam step with these grads on both sides
opt = optax.adam(3e-4, eps=1e-7)
q_new = optax.apply_updates(q_params, opt.update(c_grads_ref, opt.init(q_params))[0])
pi_new = optax.apply_updates(pi_params, opt.update(a_grads_ref, opt.init(pi_params))[0])
for p, gr in zip(agent.critic_opt.param_groups[0]["params"], cg):
    p.grad = gr
for p, gr in zip(agent.actor_opt.param_groups[0]["params"], ag):
    p.grad = gr
agent.critic_opt.step(); agent.actor_opt.step()
cmp("after Adam: sa linear_0 w", q_new["sa_encoder/~/linear_0"]["w"].T, agent.sa_encoder.layers[0].weight)
cmp("after Adam: Normal linear w", pi_new["Normal/~/linear"]["w"].T, agent.actor.loc.weight)

print("PARITY", "PASS" if worst < 1e-4 else "FAIL", f"(worst {worst:.2e})")
