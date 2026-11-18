"""Parity check: the PyTorch c0 port (crl_agent.py, CRLConfig.c0) against the colleague's JAX c0.py.

Needs a separate venv with jax / flax / optax / distrax / ml-collections and CPU torch (not in .venv):
    <jaxenv>/bin/python scripts/goal_conditioned_rl/parity_c0.py
Builds c0.py's Learner at a small size, copies its Flax parameters into the torch agent, and compares on one
batch: critic logits, critic loss and its gradients; actor loss (same pre-tanh noise, own future goals =
random_goals 0, as c0.py) and its gradients; the deterministic action.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import c0  # noqa: E402  (the colleague's file, unchanged)
from scripts.goal_conditioned_rl.crl_agent import CRLAgent, CRLConfig  # noqa: E402

jax.config.update("jax_enable_x64", False)
W, D, R, B, DS, DA, DG = 32, 4, 8, 16, 6, 2, 4


def to_t(x):
    return torch.tensor(np.asarray(x), dtype=torch.float32)


def load_residual(tmod, fparams):
    """flax ResidualMLP (Dense_i / LayerNorm_i in creation order) -> crl_agent.ResidualMLP."""
    dense = [tmod.first] + [l for blk in tmod.blocks for l in blk] + ([tmod.out] if tmod.out is not None else [])
    lns = [tmod.first_ln] + [l for blk in tmod.block_lns for l in blk]
    for i, layer in enumerate(dense):
        p = fparams[f"Dense_{i}"]
        layer.weight.data = to_t(p["kernel"]).T.contiguous()
        layer.bias.data = to_t(p["bias"])
    for i, ln in enumerate(lns):
        p = fparams[f"LayerNorm_{i}"]
        ln.weight.data = to_t(p["scale"])
        ln.bias.data = to_t(p["bias"])


def main():
    rng = np.random.default_rng(0)
    batch = dict(observations=rng.normal(size=(B, DS)).astype(np.float32),
                 actions=rng.uniform(-1, 1, (B, DA)).astype(np.float32),
                 value_goals=rng.normal(size=(B, DG)).astype(np.float32))
    batch["actor_goals"] = batch["value_goals"]
    jb = {k: jnp.asarray(v) for k, v in batch.items()}
    cfg = c0.get_config("c0", network_width=W, network_depth=D, repr_dim=R)
    learner = c0.Learner.create(0, jb, cfg)
    head = jax.tree_util.tree_map(lambda x: x[0], learner.heads)  # ensemble_heads = 1
    fp = head.params

    agent = CRLAgent(CRLConfig.c0(width=W, depth=D, repr_dim=R, random_goals=0.0), DS, DA, DG)
    load_residual(agent.sa_encoder, fp["modules_sa_encoder"]["ResidualMLP_0"])
    load_residual(agent.g_encoder, fp["modules_g_encoder"]["ResidualMLP_0"])
    load_residual(agent.actor.trunk, fp["modules_actor"]["ResidualMLP_0"])
    for tl, name in ((agent.actor.loc, "Dense_0"), (agent.actor.scale, "Dense_1")):
        tl.weight.data = to_t(fp["modules_actor"][name]["kernel"]).T.contiguous()
        tl.bias.data = to_t(fp["modules_actor"][name]["bias"])
    agent.log_alpha.data = to_t(fp["modules_alpha"]["log_value"])

    s, a, g = (to_t(batch[k]) for k in ("observations", "actions", "value_goals"))
    results = {}

    # ---- critic
    with jax.default_matmul_precision("highest"):
        j_logits = head.score(fp, jb["observations"], jb["actions"], jb["value_goals"])

        def j_critic(p):
            logits = head.score(p, jb["observations"], jb["actions"], jb["value_goals"])
            lse = jax.nn.logsumexp(logits, axis=1)
            return jnp.mean(lse - jnp.diagonal(logits)) + 0.1 * jnp.mean(jnp.square(jax.nn.logsumexp(logits + 1e-6, 1)))

        j_closs, j_cgrad = jax.value_and_grad(j_critic)(fp)
    t_closs, _ = agent.critic_loss(s, a, g)
    results["logits max |diff|"] = float(np.abs(np.asarray(j_logits) - agent.logits(s, a, g).detach().numpy()).max())
    results["critic loss |diff|"] = abs(float(j_closs) - float(t_closs))
    agent.zero_grad()
    t_closs.backward()
    jg = j_cgrad["modules_sa_encoder"]["ResidualMLP_0"]["Dense_0"]["kernel"]
    results["critic grad (sa first Dense) max |diff|"] = float(
        np.abs(np.asarray(jg).T - agent.sa_encoder.first.weight.grad.numpy()).max())
    jg2 = j_cgrad["modules_g_encoder"]["ResidualMLP_0"][f"Dense_{D + 1}"]["kernel"]
    results["critic grad (g last Dense) max |diff|"] = float(
        np.abs(np.asarray(jg2).T - agent.g_encoder.out.weight.grad.numpy()).max())

    # ---- actor (same pre-tanh noise), as c0.py: alpha * logp - score with the pre-update critic
    eps = rng.normal(size=(B, DA)).astype(np.float32)
    with jax.default_matmul_precision("highest"):
        def j_actor(p):
            dist = head.apply("actor", jb["observations"], jb["actor_goals"], params=p)
            base = dist.distribution
            u = base.loc + base.scale_diag * jnp.asarray(eps)
            logp = base.log_prob(u) - dist.bijector.forward_log_det_jacobian(u)
            act = jnp.tanh(u)
            score = head.score(fp, jb["observations"], act, jb["actor_goals"], paired=True)
            alpha = head.apply("alpha", params=fp)
            return jnp.mean(jax.lax.stop_gradient(alpha) * logp - score)

        j_aloss, j_agrad = jax.value_and_grad(j_actor)(fp)
        j_mode = head.act(jb["observations"], jb["actor_goals"])
    agent.zero_grad()
    t_aloss, _, _ = agent.actor_loss(s, g, noise=to_t(eps))
    results["actor loss |diff|"] = abs(float(j_aloss) - float(t_aloss))
    t_aloss.backward()
    jga = j_agrad["modules_actor"]["ResidualMLP_0"]["Dense_0"]["kernel"]
    results["actor grad (trunk first Dense) max |diff|"] = float(
        np.abs(np.asarray(jga).T - agent.actor.trunk.first.weight.grad.numpy()).max())
    results["deterministic action max |diff|"] = float(
        np.abs(np.asarray(j_mode) - agent.act(s, g, deterministic=True).numpy()).max())

    scale = dict(critic=float(j_closs), actor=float(j_aloss))
    print(f"reference values: critic loss {scale['critic']:.6f}, actor loss {scale['actor']:.6f}")
    ok = True
    for k, v in results.items():
        flag = v < 1e-4
        ok &= flag
        print(f"{'OK ' if flag else 'BAD'} {k}: {v:.2e}")
    print("PARITY PASS" if ok else "PARITY FAIL")


if __name__ == "__main__":
    main()
