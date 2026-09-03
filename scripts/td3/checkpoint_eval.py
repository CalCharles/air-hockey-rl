"""Standalone per-checkpoint evaluation, runnable as a subprocess.

`td3_training.py` launches this in the background at every checkpoint (when
`checkpoint_eval_async: true`) so evaluation rollouts + GIF encoding no
longer block the training loop. It reproduces exactly what the in-process
eval did:

* plain runs -> `scripts.td3.evaluate.evaluate_agent` (4 episodes, 1 GIF)
* DR runs (args.yaml has `eval_param_seed`) -> the multi-env eval from
  `scripts.td3.td3_training_dr` (env-0 GIF + per-env metric rollouts,
  `multi_env_eval.json`)

Usage:
    python -m scripts.td3.checkpoint_eval --checkpoint-dir <run>/checkpoint_25000 \
        [--eval-call-index 1]

Reads `<checkpoint-dir>/args.yaml` + `config.yaml` (written by the trainer's
`save_full_checkpoint`). `--eval-call-index` reproduces the DR wrapper's
per-checkpoint eval-seed shift (it would otherwise restart at 1 in every
subprocess, replaying identical start states at every checkpoint).
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--eval-call-index", type=int, default=1)
    cli = parser.parse_args()

    ckpt_dir = os.path.abspath(cli.checkpoint_dir)
    with open(os.path.join(ckpt_dir, "args.yaml"), "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(ckpt_dir, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model_path = os.path.join(ckpt_dir, "model.pth")

    common = dict(
        n_eps=4,
        n_gifs=1,
        action_scale=1,
        agent_hidden_layer_size=int(args["agent_hidden_layer_size"]),
        agent_num_hidden_layers=int(args["agent_num_hidden_layers"]),
        use_last_action_in_policy_state=bool(args["use_last_action_in_policy_state"]),
    )

    if args.get("eval_param_seed") is not None:
        from scripts.td3 import td3_training_dr as dr

        dr._EVAL_PARAM_SEED = int(args["eval_param_seed"])
        dr._EVAL_N_ENVS = int(args.get("eval_n_envs", 1))
        dr._EVAL_EPS_PER_ENV = int(args.get("eval_eps_per_env", 4))
        dr._LOG_PARENT_DIR = os.path.dirname(ckpt_dir)
        dr._EVAL_CALL_COUNT = int(cli.eval_call_index) - 1  # incremented inside
        dr._evaluate_agent_multi_env(model_path, ckpt_dir, config["air_hockey"], **common)
    else:
        from scripts.td3.evaluate import evaluate_agent

        evaluate_agent(model_path, ckpt_dir, config["air_hockey"], **common)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
