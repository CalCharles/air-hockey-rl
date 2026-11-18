"""Paired goal eval of Y3H4 checkpoints: ``eval_crl.py eval`` with an extra policy spec ``y3h4:<ckpt.pt>``.

The episodes, metrics and output files are exactly eval_crl.py's (stop-goal success, paired counterfactual, ...);
this wrapper only teaches its policy factory the y3h4 spec (deterministic voted action, temperature 0), so
eval_crl.py itself is unchanged. All other specs (crl:, juggler:, random) still work and can be mixed in.

    python -m scripts.goal_conditioned_rl.eval_y3h4 eval --goals runs/crl/eval_goals.npz --zero-velocity \
        --policy y3h4:runs/crl/y3h4_s0/final.pt --label y3h4_s0 --horizon 7000 --workers 30 --out runs/crl/y3h4_eval
"""

from __future__ import annotations

import numpy as np

from scripts.goal_conditioned_rl import eval_crl
from scripts.goal_conditioned_rl.crl_env import policy_state

_base_make_policy = eval_crl._make_policy


def _make_policy(spec: str, env, device="cpu"):
    kind, _, path = spec.partition(":")
    if kind != "y3h4":
        return _base_make_policy(spec, env, device)
    import torch

    from scripts.goal_conditioned_rl.y3h4_agent import Y3H4Agent

    agent = Y3H4Agent.from_checkpoint(torch.load(path, map_location=device, weights_only=False), device)
    agent.eval()

    def act(obs, last, goal):
        s = torch.as_tensor(policy_state(obs, last), device=device).unsqueeze(0)
        g = torch.as_tensor(np.asarray(goal, dtype=np.float32), device=device).unsqueeze(0)
        return agent.act(s, g, temperature=0.0).cpu().numpy().reshape(2)

    return (lambda obs: None), act


# eval_crl's workers look the factory up as a module global at call time; they are forked after this import.
eval_crl._make_policy = _make_policy

if __name__ == "__main__":
    eval_crl.main()
