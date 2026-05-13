"""
GAT (Grounded Action Transform) wrapper for TD3 training.

Execution model
---------------
This script is a thin wrapper around td3_training._entrypoint() that inserts
the GAT grounding phase before training and monkey-patches the training loop's
env factory and evaluation function.

Grounding phase (runs once at startup):
  1. Roll the pretrained source policy in the NOMINAL source sim
     (sysid_best_params_hist2.yaml) to collect transitions.
  2. Fit SourceDynamicsModel f_S on those transitions.
  3. Roll the same policy in the TARGET sim (sim2sim_warp075_p30.yaml) to
     collect grounding transitions.
  4. Fit ActionTransformer τ to minimise ||f_S(s, τ(s,a)) − s'_target||².

Fine-tuning phase (standard TD3):
  • td3_training.make_env is replaced so every source env is wrapped with
    GATEnvWrapper(τ), making source transitions look like target transitions.
  • td3_training.evaluate_agent is replaced so checkpoints are evaluated in
    the TARGET sim (no τ at deployment — standard GAT protocol).

Usage
-----
    python -m scripts.td3.td3_training_gat \\
        --gat-config configs/td3/gat/td3_gat_sim2sim.yaml \\
        --args-file  configs/td3/gat/td3_gat_td3_args.yaml

The ``--gat-config`` flag is consumed before tyro sees sys.argv, so all
standard td3_training flags (including ``--args-file``) work unchanged.

GAT config keys (configs/td3/gat/td3_gat_sim2sim.yaml)
-------------------------------------------------------
  source_config           : path to nominal source sim YAML
  target_config           : path to target sim YAML (used for grounding + eval)
  grounding_model_path    : path to pretrained actor checkpoint (.pth)
  use_last_action_in_policy_state : bool (must match checkpoint training)
  agent_hidden_layer_size : int
  agent_num_hidden_layers : int
  dynamics_collect_steps  : int   steps in source for dynamics model
  grounding_collect_steps : int   steps in target for transformer
  dynamics_epochs         : int
  transformer_epochs      : int
  hidden_size             : int   hidden dim for both networks
  batch_size              : int
  dynamics_lr             : float
  transformer_lr          : float
  device                  : str   e.g. "cuda:0" or "cpu"
  gat_save_dir            : str | null  where to write model checkpoints
"""

from __future__ import annotations

import argparse
import os
import sys

import gymnasium as gym
import torch
import yaml

from airhockey import AirHockeyEnv
from scripts.td3 import td3_training
from scripts.td3.eval_utils import build_policy_env_view, load_policy_for_evaluation
from scripts.td3.gat_trainer import (
    ActionTransformer,
    GATEnvWrapper,
    SourceDynamicsModel,
    collect_transitions,
    train_action_transformer,
    train_dynamics_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_single_env(config_path: str) -> gym.Env:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return AirHockeyEnv(cfg["air_hockey"])


def _load_grounding_policy(gat_cfg: dict, device: str):
    """Load the pretrained source policy used for collecting grounding data."""
    model_path = gat_cfg["grounding_model_path"]
    use_last_action = bool(gat_cfg.get("use_last_action_in_policy_state", True))
    hidden = int(gat_cfg.get("agent_hidden_layer_size", 64))
    n_layers = int(gat_cfg.get("agent_num_hidden_layers", 2))

    # Build a temporary vectorised env view so build_policy_env_view works.
    source_cfg_path = gat_cfg["source_config"]
    tmp_env = gym.vector.SyncVectorEnv([lambda: _make_single_env(source_cfg_path)])
    policy_env_view = build_policy_env_view(tmp_env, use_last_action)
    tmp_env.close()

    policy = load_policy_for_evaluation(
        model_path=model_path,
        policy_env_view=policy_env_view,
        action_scale=1.0,
        agent_hidden_layer_size=hidden,
        agent_num_hidden_layers=n_layers,
        map_location=device,
    )
    policy.to(device)
    return policy, use_last_action


# ---------------------------------------------------------------------------
# GAT grounding phase
# ---------------------------------------------------------------------------


def run_gat_setup(gat_cfg: dict) -> ActionTransformer:
    """Train dynamics model + action transformer from the GAT config.

    Returns the trained ActionTransformer (eval mode, CPU-resident copy
    suitable for GATEnvWrapper).
    """
    device = gat_cfg.get("device", "cpu")
    hidden = int(gat_cfg.get("hidden_size", 256))
    batch_size = int(gat_cfg.get("batch_size", 256))

    source_cfg_path = gat_cfg["source_config"]
    target_cfg_path = gat_cfg["target_config"]

    print("[GAT] Loading grounding policy...")
    policy, use_last_action = _load_grounding_policy(gat_cfg, device)

    # --- Step 1: source dynamics model ---
    print(f"[GAT] Collecting {gat_cfg.get('dynamics_collect_steps', 20000)} source transitions...")
    source_env = _make_single_env(source_cfg_path)
    src_obs, src_act, src_nobs = collect_transitions(
        source_env,
        policy,
        n_steps=int(gat_cfg.get("dynamics_collect_steps", 20000)),
        use_last_action=use_last_action,
        device=device,
    )
    source_env.close()

    dynamics_model = SourceDynamicsModel(hidden=hidden).to(device)
    dyn_opt = torch.optim.Adam(
        dynamics_model.parameters(), lr=float(gat_cfg.get("dynamics_lr", 1e-3))
    )
    print(f"[GAT] Training dynamics model ({gat_cfg.get('dynamics_epochs', 100)} epochs)...")
    dyn_loss = train_dynamics_model(
        obs=src_obs,
        actions=src_act,
        next_obs=src_nobs,
        model=dynamics_model,
        optimizer=dyn_opt,
        epochs=int(gat_cfg.get("dynamics_epochs", 100)),
        batch_size=batch_size,
        device=device,
    )
    print(f"[GAT] Dynamics model final MSE: {dyn_loss:.6f}")

    # --- Step 2: action transformer ---
    print(f"[GAT] Collecting {gat_cfg.get('grounding_collect_steps', 10000)} target transitions...")
    target_env = _make_single_env(target_cfg_path)
    tgt_obs, tgt_act, tgt_nobs = collect_transitions(
        target_env,
        policy,
        n_steps=int(gat_cfg.get("grounding_collect_steps", 10000)),
        use_last_action=use_last_action,
        device=device,
    )
    target_env.close()

    transformer = ActionTransformer(hidden=hidden).to(device)
    trf_opt = torch.optim.Adam(
        transformer.parameters(), lr=float(gat_cfg.get("transformer_lr", 1e-3))
    )
    print(f"[GAT] Training action transformer ({gat_cfg.get('transformer_epochs', 100)} epochs)...")
    trf_loss = train_action_transformer(
        target_obs=tgt_obs,
        target_actions=tgt_act,
        target_next_obs=tgt_nobs,
        dynamics_model=dynamics_model,
        transformer=transformer,
        optimizer=trf_opt,
        epochs=int(gat_cfg.get("transformer_epochs", 100)),
        batch_size=batch_size,
        device=device,
    )
    print(f"[GAT] Action transformer final MSE: {trf_loss:.6f}")

    save_dir = gat_cfg.get("gat_save_dir")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(dynamics_model.state_dict(), os.path.join(save_dir, "dynamics_model.pth"))
        torch.save(transformer.state_dict(), os.path.join(save_dir, "action_transformer.pth"))
        print(f"[GAT] Saved models to {save_dir}/")

    return transformer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Strip --gat-config from sys.argv so tyro (inside _entrypoint) does not
    # see it as an unknown argument.
    gat_parser = argparse.ArgumentParser(add_help=False)
    gat_parser.add_argument(
        "--gat-config",
        type=str,
        required=True,
        help="Path to the GAT config YAML (see module docstring for keys).",
    )
    gat_ns, remaining_argv = gat_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv

    with open(gat_ns.gat_config) as f:
        gat_cfg: dict = yaml.safe_load(f)

    # --- Grounding phase ---
    transformer = run_gat_setup(gat_cfg)

    # --- Patch 1: wrap source env with GATEnvWrapper ---
    _original_make_env = td3_training.make_env

    def _gat_make_env(env_id):  # noqa: ANN001
        original_thunk = _original_make_env(env_id)

        def _wrapped_thunk():
            env = original_thunk()
            return GATEnvWrapper(env, transformer)

        return _wrapped_thunk

    td3_training.make_env = _gat_make_env

    # --- Patch 2: evaluate in target env (GAT: no τ at deployment) ---
    with open(gat_cfg["target_config"]) as f:
        _target_config = yaml.safe_load(f)
    _target_params: dict = _target_config["air_hockey"]

    _orig_evaluate_agent = td3_training.evaluate_agent

    def _gat_evaluate_agent(model_path, save_dir, air_hockey_params, **kwargs):  # noqa: ANN001
        # Ignore the source env params; always evaluate in target env.
        return _orig_evaluate_agent(model_path, save_dir, _target_params, **kwargs)

    td3_training.evaluate_agent = _gat_evaluate_agent

    # --- Fine-tuning phase (standard TD3 training loop) ---
    td3_training._entrypoint()
