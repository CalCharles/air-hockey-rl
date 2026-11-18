"""Parity check: the PyTorch Y3H4 port (y3h4_agent.py) against the colleague's JAX y3h4.py.

Needs a venv with jax / flax / optax / distrax and CPU torch (not .venv), e.g. the one used for parity_c0.py:
    <jaxenv>/bin/python scripts/goal_conditioned_rl/parity_y3h4.py
Initialises y3h4.py's stacked heads at a small size, copies the Flax parameters into the torch agent and compares,
with the same random draws (goal jitter, actor noise, sampling noise reproduced from the JAX keys):
per-head critic / actor / temperature losses, their gradients, plain and rolled-code accuracy, and the voted
action (deterministic, with collection lanes, and sampled).
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
import y3h4  # noqa: E402  (the colleague's file, unchanged)
from scripts.goal_conditioned_rl.y3h4_agent import Y3H4Agent, Y3H4Config  # noqa: E402

H, W, D, R, B, DS, DA, DG = 3, 32, 4, 8, 16, 6, 2, 4


def to_t(x):
    return torch.tensor(np.asarray(x), dtype=torch.float32)


def load_mlp(tmod, fp):
    for i, layer in enumerate(list(tmod.dense) + ([tmod.out] if tmod.out is not None else [])):
        layer.weight.data = to_t(fp[f"Dense_{i}"]["kernel"])
        layer.bias.data = to_t(fp[f"Dense_{i}"]["bias"])
    for i, ln in enumerate(tmod.norms):
        ln.weight.data = to_t(fp[f"LayerNorm_{i}"]["scale"])
        ln.bias.data = to_t(fp[f"LayerNorm_{i}"]["bias"])


def load(agent, fp):
    for name in ("sa_encoder", "g_encoder"):
        enc, p = getattr(agent, name), fp[f"modules_{name}"]
        load_mlp(enc.mlp, p["ResidualMLP_0"])
        enc.htc.weight.data = to_t(p["EuclideanHead_0"]["htc"]["kernel"])
        enc.htc.bias.data = to_t(p["EuclideanHead_0"]["htc"]["bias"])
    p = fp["modules_actor"]
    load_mlp(agent.actor.trunk, p["ResidualMLP_0"])
    for tl, name in ((agent.actor.mean, "Dense_0"), (agent.actor.log_std, "Dense_1")):
        tl.weight.data = to_t(p[name]["kernel"])
        tl.bias.data = to_t(p[name]["bias"])
    agent.log_alpha.data = to_t(fp["modules_alpha"]["log_value"])


def main():
    y3h4.configure(dict(heads=H, width=W, depth=D, repr_dim=R, batch_size=B, lanes=B))
    rng = np.random.default_rng(0)
    batch = dict(observations=rng.normal(size=(B, DS)), actions=rng.uniform(-1, 1, (B, DA)),
                 value_goals=rng.normal(size=(B, DG)), commanded_goals=rng.normal(size=(B, DG)))
    batch["commanded_goals"][:, 2:] = 0.0
    batch["actor_goals"] = batch["value_goals"]
    batch = {k: v.astype(np.float32) for k, v in batch.items()}
    jb = {k: jnp.asarray(v) for k, v in batch.items()}
    heads = y3h4.init_heads(0, jb["observations"][:1], jb["actions"][:1], jb["commanded_goals"][:1])
    fp = heads["params"]

    agent = Y3H4Agent(Y3H4Config(heads=H, width=W, depth=D, repr_dim=R, batch_size=B), DS, DA, DG)
    load(agent, fp)
    t = {k: to_t(v) for k, v in batch.items()}

    # The random draws head_loss makes, per head, from its own key.
    jit_noise, act_noise, j_tot, j_info, j_grads = [], [], [], [], []
    for h in range(H):
        p_h = jax.tree_util.tree_map(lambda x: x[h], fp)
        rng_h = jax.random.split(heads["rng"][h])[1]  # head_update: new_rng, rng = split(head['rng'])
        critic_rng, actor_rng = jax.random.split(rng_h)
        jit_noise.append(np.asarray(jax.random.normal(jax.random.fold_in(critic_rng, 2), (B, DG))))
        act_noise.append(np.asarray(jax.random.normal(actor_rng, (B, DA))))
        (tot, info), g = jax.value_and_grad(y3h4.head_loss, has_aux=True)(p_h, jb, rng_h, 0)
        j_tot.append(float(tot)), j_info.append({k: float(v) for k, v in info.items()}), j_grads.append(g)
    jit_noise, act_noise = to_t(np.stack(jit_noise)), to_t(np.stack(act_noise))

    # Is the actor noise reproduced? distrax sample = loc + scale * normal(key): check one head's action.
    p0 = jax.tree_util.tree_map(lambda x: x[0], fp)
    code0 = y3h4.psi(p0, jb["actor_goals"])
    dist0 = y3h4.policy(p0, jb["observations"], code0)
    a_j, _ = dist0.sample_and_log_prob(seed=jax.random.split(jax.random.split(heads["rng"][0])[1])[1])
    loc, std = agent.policy(t["observations"], agent.psi(t["actor_goals"]).detach())
    a_t = torch.tanh(loc[0] + std[0] * act_noise[0])
    res = {"actor sample reproduced": float(np.abs(np.asarray(a_j) - a_t.detach().numpy()).max())}

    c_loss, a_loss, c_grads, a_grads, info = agent._losses(
        t["observations"], t["actions"], t["value_goals"], t["actor_goals"], t["commanded_goals"],
        jitter_noise=jit_noise, actor_noise=act_noise)
    res["total loss (sum over heads)"] = abs(sum(j_tot) - float(c_loss + a_loss))
    res["critic loss (mean over heads)"] = abs(np.mean([i["critic_loss"] for i in j_info]) - float(info["critic_loss"]))
    res["actor objective (mean)"] = abs(np.mean([i["actor_loss"] for i in j_info]) - float(info["actor_loss"]))
    res["entropy (mean)"] = abs(np.mean([i["entropy"] for i in j_info]) - float(info["entropy"]))
    res["accuracy (mean)"] = abs(np.mean([i["accuracy"] for i in j_info]) - float(info["accuracy"]))
    rolled = agent.rolled_code_accuracy(t["observations"], t["actions"], t["value_goals"], t["commanded_goals"],
                                        jitter_noise=jit_noise)
    res["rolled-code accuracy (mean)"] = abs(np.mean([i["rolled_code_accuracy"] for i in j_info]) - float(rolled))

    params = agent.critic_params + agent.actor_params + [agent.log_alpha]
    for p, gr in zip(params, list(c_grads) + list(a_grads)):
        p.grad = gr
    jg = {k: np.stack([np.asarray(_get(g, k)) for g in j_grads]) for k in GRAD_LEAVES}
    for k, tensor in (("sa first Dense", agent.sa_encoder.mlp.dense[0].weight),
                      ("sa htc", agent.sa_encoder.htc.weight),
                      ("g last Dense", agent.g_encoder.mlp.out.weight),
                      ("actor first Dense", agent.actor.trunk.dense[0].weight),
                      ("actor log-std head", agent.actor.log_std.weight),
                      ("alpha", agent.log_alpha)):
        res[f"grad {k}"] = float(np.abs(jg[k] - tensor.grad.numpy()).max())

    # Acting: deterministic vote, vote with collection lanes, and sampled actions.
    obs, goals = jb["observations"], jb["commanded_goals"]
    key = jax.random.PRNGKey(7)
    j_det = np.asarray(y3h4.act(fp, obs, goals, key, 0.0))
    res["voted action, temperature 0"] = float(np.abs(j_det - agent.act(t["observations"], t["commanded_goals"]).numpy()).max())
    lanes = jnp.arange(B, dtype=jnp.int32)
    keys = [key] + [jax.random.fold_in(key, i) for i in range(1, H)]
    s_noise = to_t(np.stack([np.asarray(jax.random.normal(k, (B, DA))) for k in keys]))
    j_lane = np.asarray(y3h4.act(fp, obs, goals, key, 1.0, lanes))
    t_lane = agent.act(t["observations"], t["commanded_goals"], temperature=1.0, lane_ids=torch.arange(B),
                       n_lanes=B, noise=s_noise).numpy()
    res["sampled action with lanes, temperature 1"] = float(np.abs(j_lane - t_lane).max())

    print(f"reference: total loss {sum(j_tot):.6f}, critic {np.mean([i['critic_loss'] for i in j_info]):.6f}, "
          f"accuracy {np.mean([i['accuracy'] for i in j_info]):.3f}")
    ok = True
    for k, v in res.items():
        flag = v < 1e-4
        ok &= flag
        print(f"{'OK ' if flag else 'BAD'} {k}: {v:.2e}")
    print("PARITY PASS" if ok else "PARITY FAIL")


GRAD_LEAVES = {
    "sa first Dense": ("modules_sa_encoder", "ResidualMLP_0", "Dense_0", "kernel"),
    "sa htc": ("modules_sa_encoder", "EuclideanHead_0", "htc", "kernel"),
    "g last Dense": ("modules_g_encoder", "ResidualMLP_0", f"Dense_{D + 1}", "kernel"),
    "actor first Dense": ("modules_actor", "ResidualMLP_0", "Dense_0", "kernel"),
    "actor log-std head": ("modules_actor", "Dense_1", "kernel"),
    "alpha": ("modules_alpha", "log_value"),
}


def _get(tree, key):
    for k in GRAD_LEAVES[key]:
        tree = tree[k]
    return tree


if __name__ == "__main__":
    main()
