"""Minimal policy runner for the air-hockey env (real or sim).

What this is NOT:
  * NOT a trainer (no learner, no replay, no checkpointing).
  * NOT an evaluator (no reward / juggle / return logging, no JSONL/HDF5).
  * NOT a reset FSM (no programmatic puck-reset on the real robot).

What this IS: load a policy, reset the env once, then loop
``obs -> agent(obs) -> env.step(action)`` indefinitely. ``terminated`` /
``truncated`` flags are ignored — the policy just keeps running across
episode boundaries.

Human control: press SPACE to pause. On pause, ``env.reset()`` returns
the robot to its home pose via the standard mechanism (servoStop +
forceModeStop + moveL to reset_pose). Press SPACE again to resume.
Ctrl-C exits cleanly from either state.

Agent interface is intentionally trivial so different policies can be
swapped in without touching the loop. Implement a class that satisfies
the ``PolicyAgent`` protocol and register a constructor in
``build_agent``.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
import yaml

from airhockey import AirHockeyEnv
from airhockey.sims.real.multiprocessing import NonBlockingConsole
from scripts.real.sgcrl_policy import load_sgcrl_deterministic_policy
from scripts.real.iwr_policy import load_iwr_deterministic_policy
from scripts.real.gcrl_variant_policies import (
    load_crtr_deterministic_policy,
    load_ppo_gcrl_deterministic_policy,
    load_sac_gcrl_deterministic_policy,
    load_sac_her_deterministic_policy,
    load_sac_weighted_her_deterministic_policy,
)
from scripts.td3.deterministic_agent import DeterministicAgent
from scripts.td3.helper.real_td3_runtime import (
    _load_train_args,
    augment_policy_observation,
    build_policy_env_view,
    deterministic_actor_action,
)


# ---------------------------------------------------------------------------
# Agent interface.
# ---------------------------------------------------------------------------


@runtime_checkable
class PolicyAgent(Protocol):
    """Minimal interface every agent must satisfy.

    Implementations are responsible for any obs preprocessing
    (e.g. last-action augmentation, history stacking) and must return a
    numpy action of shape ``(act_dim,)`` already in the env's action
    range (``[-1, 1]`` for the air-hockey env).
    """

    def __call__(self, obs: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None:
        """Optional hook called at the start of every episode."""


def _maybe_reset_agent(agent: PolicyAgent) -> None:
    reset_fn = getattr(agent, "reset", None)
    if callable(reset_fn):
        reset_fn()


# ---------------------------------------------------------------------------
# Concrete agents.
# ---------------------------------------------------------------------------


class TD3DeterministicPolicy:
    """Wraps a frozen ``DeterministicAgent`` + last-action augmentation."""

    def __init__(
        self,
        actor: DeterministicAgent,
        *,
        act_dim: int,
        use_last_action_in_policy_state: bool,
        device: torch.device,
    ) -> None:
        self._actor = actor
        self._act_dim = int(act_dim)
        self._use_last_action = bool(use_last_action_in_policy_state)
        self._device = device
        self._last_action_t = torch.zeros(
            (1, self._act_dim), dtype=torch.float32, device=device
        )

    def reset(self) -> None:
        self._last_action_t.zero_()

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)
        policy_obs = augment_policy_observation(
            obs_t, self._last_action_t, self._use_last_action
        )
        action_t = deterministic_actor_action(self._actor, policy_obs).detach()
        self._last_action_t = action_t
        return action_t.squeeze(0).cpu().numpy().astype(np.float32)


class ZeroPolicy:
    """Outputs an all-zero action every step. Useful sanity-check baseline."""

    def __init__(self, act_dim: int) -> None:
        self._zero = np.zeros(int(act_dim), dtype=np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self._zero


class RandomPolicy:
    """Uniform random action in ``[-1, 1]^act_dim``."""

    def __init__(self, act_dim: int, seed: int = 0) -> None:
        self._act_dim = int(act_dim)
        self._rng = np.random.default_rng(int(seed))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=(self._act_dim,)).astype(np.float32)


# ---------------------------------------------------------------------------
# Loaders.
# ---------------------------------------------------------------------------


def load_td3_deterministic_policy(
    *,
    model_path: str | Path,
    args_yaml_path: str | Path | None,
    env_obs_dim: int,
    env_act_dim: int,
    device: torch.device,
) -> TD3DeterministicPolicy:
    """Build a ``TD3DeterministicPolicy`` from a saved checkpoint.

    Accepts either:
      * a raw actor state_dict (e.g. ``latest_models/canonical/<name>/model.pth``), or
      * a ``training_state.pth`` checkpoint dict (uses the ``"actor"`` entry).

    ``args_yaml_path`` defaults to ``<model_dir>/args.yaml`` — the
    architecture (``agent_hidden_layer_size``, ``agent_num_hidden_layers``,
    ``action_scale``, ``use_last_action_in_policy_state``) must come from
    the training run, NOT the env config.
    """
    model_path = Path(model_path)
    if args_yaml_path is None:
        args_yaml_path = model_path.parent / "args.yaml"
    train_args = _load_train_args(str(args_yaml_path))

    policy_obs_dim = (
        env_obs_dim + env_act_dim
        if train_args.use_last_action_in_policy_state
        else env_obs_dim
    )
    policy_env_view = build_policy_env_view(policy_obs_dim, env_act_dim)
    actor = DeterministicAgent(
        policy_env_view,
        action_scale=train_args.action_scale,
        action_bias=0.0,
        hidden_layer_size=train_args.agent_hidden_layer_size,
        num_hidden_layers=train_args.agent_num_hidden_layers,
    ).to(device)

    loaded = torch.load(str(model_path), map_location="cpu", weights_only=False)
    if isinstance(loaded, dict) and "actor" in loaded and not any(
        k.startswith("actor.") for k in loaded.keys()
    ):
        state_dict = loaded["actor"]
    else:
        state_dict = loaded
    actor.load_state_dict(state_dict, strict=False)
    actor.eval()

    print(
        f"[run_policy] loaded td3 actor from {model_path} "
        f"(hidden={train_args.agent_hidden_layer_size}x{train_args.agent_num_hidden_layers}, "
        f"action_scale={train_args.action_scale}, "
        f"use_last_action={train_args.use_last_action_in_policy_state})"
    )
    return TD3DeterministicPolicy(
        actor,
        act_dim=env_act_dim,
        use_last_action_in_policy_state=train_args.use_last_action_in_policy_state,
        device=device,
    )


def build_agent(
    kind: str,
    *,
    model_path: str | None,
    args_yaml_path: str | None,
    env_obs_dim: int,
    env_act_dim: int,
    device: torch.device,
    seed: int = 0,
) -> PolicyAgent:
    """Dispatch a ``--agent`` string to a concrete ``PolicyAgent``.

    Add new agents here. Each branch should return an object satisfying
    the ``PolicyAgent`` protocol (callable obs -> action numpy array,
    optional ``reset()``).
    """
    from scripts.real.agent_kinds import normalize_agent_kind

    kind = normalize_agent_kind(kind)
    if kind == "td3":
        if model_path is None:
            raise SystemExit("--agent td3 requires --model <path/to/model.pth>")
        return load_td3_deterministic_policy(
            model_path=model_path,
            args_yaml_path=args_yaml_path,
            env_obs_dim=env_obs_dim,
            env_act_dim=env_act_dim,
            device=device,
        )
    if kind == "sgcrl":
        if model_path is None:
            raise SystemExit("--agent sgcrl requires --model <path/to/sgcrl_checkpoint.pkl>")
        return load_sgcrl_deterministic_policy(
            model_path=model_path,
            env_obs_dim=env_obs_dim,
            env_act_dim=env_act_dim,
            device=device,
        )
    if kind == "iwr":
        if model_path is None:
            raise SystemExit(
                "--agent iwr requires --model "
                "<path/to/interaction_weighted_sampling_checkpoint.pkl>"
            )
        return load_iwr_deterministic_policy(
            model_path=model_path,
            env_obs_dim=env_obs_dim,
            env_act_dim=env_act_dim,
            device=device,
        )
    if kind == "crtr":
        if model_path is None:
            raise SystemExit("--agent crtr requires --model <path/to/crtr_checkpoint.pkl>")
        return load_crtr_deterministic_policy(
            model_path=model_path,
            env_obs_dim=env_obs_dim,
            env_act_dim=env_act_dim,
            device=device,
        )
    if kind == "sac-gcrl":
        if model_path is None:
            raise SystemExit(
                "--agent sac-gcrl requires --model <path/to/sac_gcrl_checkpoint.pkl>"
            )
        return load_sac_gcrl_deterministic_policy(
            model_path=model_path,
            env_obs_dim=env_obs_dim,
            env_act_dim=env_act_dim,
            device=device,
        )
    if kind == "sac-her":
        if model_path is None:
            raise SystemExit(
                "--agent sac-her requires --model <path/to/sac_her_checkpoint.pkl>"
            )
        return load_sac_her_deterministic_policy(
            model_path=model_path,
            env_obs_dim=env_obs_dim,
            env_act_dim=env_act_dim,
            device=device,
        )
    if kind == "sac-weighted-her":
        if model_path is None:
            raise SystemExit(
                "--agent sac-weighted-her requires --model "
                "<path/to/sac_weighted_her_checkpoint.pkl>"
            )
        return load_sac_weighted_her_deterministic_policy(
            model_path=model_path,
            env_obs_dim=env_obs_dim,
            env_act_dim=env_act_dim,
            device=device,
        )
    if kind == "ppo-gcrl":
        if model_path is None:
            raise SystemExit(
                "--agent ppo-gcrl requires --model <path/to/ppo_gcrl_checkpoint.pkl>"
            )
        return load_ppo_gcrl_deterministic_policy(
            model_path=model_path,
            env_obs_dim=env_obs_dim,
            env_act_dim=env_act_dim,
            device=device,
        )
    if kind == "zero":
        return ZeroPolicy(act_dim=env_act_dim)
    if kind == "random":
        return RandomPolicy(act_dim=env_act_dim, seed=seed)
    raise SystemExit(
        f"Unknown --agent {kind!r}; choose from: "
        "td3, sgcrl, iwr, crtr, sac-gcrl, sac-her, sac-weighted-her, ppo-gcrl, zero, random"
    )


# ---------------------------------------------------------------------------
# Env + run loop.
# ---------------------------------------------------------------------------


def _load_env(config_path: str | Path) -> AirHockeyEnv:
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    ah = dict(cfg["air_hockey"])
    if "seed" not in ah:
        seed_cfg = cfg.get("seed", 0)
        if isinstance(seed_cfg, (list, tuple)):
            seed_cfg = seed_cfg[0] if len(seed_cfg) > 0 else 0
        ah["seed"] = int(seed_cfg)
    if "n_training_steps" not in ah:
        ah["n_training_steps"] = int(cfg.get("n_training_steps", 1))
    ah.setdefault("return_goal_obs", False)
    # NonBlockingConsole's cbreak mode hides line-buffered stdout, so the
    # config's "Press space to start" prompt is invisible. Force-disable
    # the wait — same override the async-real eval entrypoint applies
    # (scripts/td3/extras/async_td3_real_eval.py:322).
    sim_params = ah.get("simulator_params", {})
    if isinstance(sim_params, dict):
        sim_params["wait_for_space_to_start"] = False
    return AirHockeyEnv(ah)


def run(
    env: AirHockeyEnv,
    agent: PolicyAgent,
    *,
    max_steps: int = 0,
    print_every: int = 50,
) -> None:
    """Continuous policy rollout with SPACE pause/resume.

    The loop ignores ``terminated`` / ``truncated`` and just keeps stepping.
    Press SPACE to pause: ``env.reset()`` brings the robot back to its
    home pose, then we idle. Press SPACE again to resume; the agent's
    internal state is cleared and stepping continues. Ctrl-C exits.
    """
    obs, _info = env.reset()
    _maybe_reset_agent(agent)
    step = 0
    print("[run_policy] running. SPACE = pause/return-to-home, Ctrl-C = exit.")
    try:
        with NonBlockingConsole() as nbc:
            while True:
                if nbc.get_data() == " ":
                    print("[run_policy] SPACE: pausing; returning robot to home pose...")
                    obs, _info = env.reset()
                    print("[run_policy] paused. SPACE to resume, Ctrl-C to exit.")
                    while True:
                        if nbc.get_data() == " ":
                            print("[run_policy] SPACE: resuming.")
                            _maybe_reset_agent(agent)
                            break
                        time.sleep(0.02)

                action = agent(obs)
                obs, _reward, _terminated, _truncated, _info = env.step(action)
                step += 1
                if print_every > 0 and step % print_every == 0:
                    print(f"[run_policy] step={step}")
                if max_steps > 0 and step >= max_steps:
                    print(f"[run_policy] reached --max-steps={max_steps}; exiting.")
                    return
    except KeyboardInterrupt:
        print("\n[run_policy] interrupted; exiting.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal policy runner (no reset FSM, no reward tracking, no logging).",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to AirHockey env config YAML (real or sim).",
    )
    parser.add_argument(
        "--agent", type=str, default="td3",
        help=(
            "Agent kind. Built-ins: td3 | sgcrl | iwr | crtr | sac-gcrl | sac-her | "
            "sac-weighted-her | ppo-gcrl | zero | random (underscore/hyphen equivalent). "
            "Extend in build_agent()."
        ),
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=(
            "Path to a checkpoint file. For --agent td3: model.pth (raw actor state_dict) "
            "or training_state.pth. For --agent sgcrl: a .pkl produced by the SGCRL trainer. "
            "For --agent iwr: a .pkl with algorithm_name=interaction_weighted_sampling."
        ),
    )
    parser.add_argument(
        "--args-yaml", type=str, default=None,
        help="Path to training-run args.yaml (architecture spec). Defaults to <model_dir>/args.yaml.",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--max-steps", type=int, default=0,
        help="Stop after this many env steps. 0 = run until interrupted.",
    )
    parser.add_argument(
        "--print-every", type=int, default=50,
        help="Print a heartbeat line every N steps. 0 disables.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    env = _load_env(args.config)
    obs_dim = int(env.single_observation_space.shape[0])
    act_dim = int(env.single_action_space.shape[0])
    print(f"[run_policy] env config={args.config} obs_dim={obs_dim} act_dim={act_dim}")

    agent = build_agent(
        args.agent,
        model_path=args.model,
        args_yaml_path=args.args_yaml,
        env_obs_dim=obs_dim,
        env_act_dim=act_dim,
        device=device,
        seed=args.seed,
    )
    run(env, agent, max_steps=args.max_steps, print_every=args.print_every)


if __name__ == "__main__":
    main()
