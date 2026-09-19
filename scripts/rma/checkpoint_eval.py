"""Per-checkpoint evaluation of an RMA phase-1 policy (subprocess entry point).

``scripts.rma.train_base_policy`` launches this in the background at every
checkpoint (like ``scripts.td3.checkpoint_eval`` for canonical runs).  The
privileged agent pi(x, a_prev, mu(e)) (or the long-history agent for
``rma_mode: history`` runs) is rolled on the fixed DR eval-env set
(env-0 GIF + per-env metrics -> ``multi_env_eval.json``).

Usage:
    python -m scripts.rma.checkpoint_eval --checkpoint-dir <run>/checkpoint_25000 [--eval-call-index 1]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict


def evaluate_checkpoint_dir(ckpt_dir: str, eval_call_index: int = 1, eps_per_env: int | None = None) -> Dict[str, Any]:
    from scripts.rma.bundle import load_phase1
    from scripts.rma.evaluate import HistoryAgent, PrivilegedRMAAgent, evaluate_multi_env

    bundle = load_phase1(ckpt_dir)
    args = bundle["args"] or {}
    config = bundle["config"]
    if config is None:
        raise FileNotFoundError(f"config.yaml not found next to {ckpt_dir}")
    eval_param_seed = args.get("eval_param_seed")
    if eval_param_seed is None:
        raise ValueError("RMA phase-1 runs need `eval_param_seed` in their args (fixed DR eval-env set).")
    actor = bundle["actor"]
    if bundle["mode"] == "history":
        agent_name, factory = "history", (lambda: HistoryAgent(actor))
    else:
        agent_name, factory = "privileged", (lambda: PrivilegedRMAAgent(actor))
    return evaluate_multi_env(
        config["air_hockey"],
        {agent_name: factory},
        bundle["normalizer"],
        eval_param_seed=int(eval_param_seed),
        n_envs=int(args.get("eval_n_envs", 5)),
        eps_per_env=int(eps_per_env if eps_per_env is not None else args.get("eval_eps_per_env", 4)),
        call_index=int(eval_call_index),
        save_dir=ckpt_dir,
        primary=agent_name,
        gif_agents=(agent_name,),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--eval-call-index", type=int, default=1)
    parser.add_argument("--eps-per-env", type=int, default=None)
    cli = parser.parse_args()
    evaluate_checkpoint_dir(os.path.abspath(cli.checkpoint_dir), cli.eval_call_index, cli.eps_per_env)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
