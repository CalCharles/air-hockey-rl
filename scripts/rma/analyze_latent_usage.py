"""How much does the phase-1 base policy actually use the privileged latent?

For every checkpoint of a phase-1 run:
  * latent spread — std of z = mu(e) over e drawn uniformly in the DR box, mean over latent dims;
  * latent sensitivity — |mu(e_hi) - mu(e_lo)| for each physics parameter swept over its full DR range;
  * action sensitivity — mean |pi(x, a_prev, mu(e)) - pi(x, a_prev, mu(0))| on states visited by the
    final policy (one rollout set, reused for every checkpoint), relative to the mean |action|.

A base policy that has "washed out" the latent shows a shrinking latent spread
and an action sensitivity near zero, in which case privileged / adapted /
nominal agents behave alike and phase 2 has nothing to recover.

    .venv/bin/python -m scripts.rma.analyze_latent_usage --run-dir runs/rma/juggle_dr_phase1_seed0 [--out <csv>]
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List

import numpy as np
import torch

from airhockey import AirHockeyEnv
from scripts.rma.bundle import load_phase1
from scripts.rma.env_wrapper import raw_env_params
from scripts.rma.evaluate import PrivilegedRMAAgent


def collect_states(bundle, n_states: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = dict(bundle["config"]["air_hockey"])
    cfg["seed"] = int(seed)
    env = AirHockeyEnv(cfg)
    normalizer = bundle["normalizer"]
    agent = PrivilegedRMAAgent(bundle["actor"])
    obs_list, prev_list = [], []
    while len(obs_list) < n_states:
        obs, _ = env.reset()
        e = normalizer.normalize(raw_env_params(env, normalizer.random_variables))
        agent.reset(np.asarray(obs, dtype=np.float32), e)
        last = np.zeros(2, dtype=np.float32)
        done = False
        while not done and len(obs_list) < n_states:
            obs_list.append(np.asarray(obs, dtype=np.float32))
            prev_list.append(last.copy())
            action = agent.act(np.asarray(obs, dtype=np.float32), last)
            obs, _, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
            last = np.asarray(action, dtype=np.float32)
    env.close()
    return torch.as_tensor(np.stack(obs_list)), torch.as_tensor(np.stack(prev_list))


@torch.no_grad()
def analyze_actor(actor, obs: torch.Tensor, prev: torch.Tensor, random_variables: List[str], n_e: int = 4096, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    e = torch.rand((n_e, actor.env_param_dim), generator=g) * 2 - 1
    z = actor.encoder(e)
    spread = float(z.std(0).mean())
    sens = []
    for k in range(actor.env_param_dim):
        lo = torch.zeros(1, actor.env_param_dim)
        hi = torch.zeros(1, actor.env_param_dim)
        lo[0, k], hi[0, k] = -1.0, 1.0
        sens.append(float((actor.encoder(hi) - actor.encoder(lo)).norm()))
    e_states = torch.rand((obs.shape[0], actor.env_param_dim), generator=g) * 2 - 1
    z_states = actor.encoder(e_states)
    z0 = actor.encoder(torch.zeros(obs.shape[0], actor.env_param_dim))
    p = prev if actor.use_last_action else None
    a_e = actor.get_action_from_latent(obs, p, z_states)
    a_0 = actor.get_action_from_latent(obs, p, z0)
    shift = float((a_e - a_0).abs().mean())
    rel = shift / max(float(a_e.abs().mean()), 1e-6)
    return {"latent_std": spread, **{f"dz_{v}": s for v, s in zip(random_variables, sens)}, "action_shift": shift, "action_shift_rel": rel}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-states", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    cli = ap.parse_args()

    final = load_phase1(cli.run_dir)
    random_variables = final["normalizer"].random_variables
    obs, prev = collect_states(final, cli.n_states, cli.seed)
    ckpts = sorted(glob.glob(os.path.join(cli.run_dir, "checkpoint_*")), key=lambda p: int(p.rsplit("_", 1)[-1]))
    rows = []
    for ck in ckpts + [cli.run_dir]:
        step = int(ck.rsplit("_", 1)[-1]) if ck != cli.run_dir else int((final["args"] or {}).get("total_timesteps", 0))
        actor = load_phase1(ck)["actor"]
        rows.append({"step": step, **analyze_actor(actor, obs, prev, random_variables, seed=cli.seed)})
    keys = list(rows[0].keys())
    lines = [",".join(keys)] + [",".join(f"{r[k]:.5f}" if k != "step" else str(r[k]) for k in keys) for r in rows]
    text = "\n".join(lines)
    print(f"# {cli.run_dir}: latent usage per checkpoint (states from the final policy, n={obs.shape[0]})")
    for r in rows:
        if r["step"] % 250000 == 0 or r is rows[-1] or r["step"] == 25000:
            print(
                f"step {r['step']:>8d}: latent_std {r['latent_std']:.3f}  "
                + "  ".join(f"dz_{v} {r[f'dz_{v}']:.3f}" for v in random_variables)
                + f"  action_shift {r['action_shift']:.4f} ({100 * r['action_shift_rel']:.1f} % of |a|)"
            )
    if cli.out:
        with open(cli.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {cli.out}")


if __name__ == "__main__":
    main()
