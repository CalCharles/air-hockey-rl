"""
TD3 training entrypoint with multi-environment evaluation across a fixed
seed-sampled set of dynamics-parameter dicts.

This wraps `td3_training` (no behavior change to the training loop) by
monkey-patching the per-checkpoint `evaluate_agent` call so that, in
addition to the usual single-env GIF + eval, the trainer also rolls
`eval_eps_per_env` episodes against each of `eval_n_envs` fixed
parameter overlays (sampled once at startup with `eval_param_seed`).

Multi-env eval semantics:
- The fixed env-param dicts are sampled at startup using
  `np.random.RandomState(eval_param_seed)` from the same
  `random_variable_ranges` block in the air_hockey config that drives
  per-reset randomization during training. Reproducible across runs.
- Per-env stats (mean episode return, success rate, mean episode length)
  and aggregate stats are written to <ckpt_dir>/multi_env_eval.json
  every checkpoint, and the aggregate is printed to stdout so the run
  log shows the trajectory.
- Env 0's GIF is preserved (for visual sanity); envs 1..N-1 don't save
  GIFs to keep disk usage from exploding (5x growth across 28+ ckpts).
- `eval_envs.json` is written once at training start in `log_parent_dir`
  for full reproducibility of which 5 dicts were used.

Run: identical to td3_training.py.
    python -m scripts.td3.td3_training_dr \
      --args-file <path/to/td3_args.yaml>

Activated when the YAML sets `eval_param_seed: <int>` (default None
disables the multi-env path; with the wrapper but seed=None the
behavior collapses to identical-to-canonical single-env eval).
"""

from __future__ import annotations

import copy
import json
import os
import sys
from typing import Any

import numpy as np
import torch
import yaml

# Import the canonical trainer module so we can monkey-patch its
# `evaluate_agent` callable before the training loop runs.
from scripts.td3 import td3_training
from scripts.td3.eval_utils import (
    augment_policy_observation,
    build_policy_env_view,
    load_policy_for_evaluation,
)
from airhockey import AirHockeyEnv
import gymnasium as gym
from types import SimpleNamespace
from scripts.transformer.context_encoder import ContextEncoder
from scripts.transformer.history_buffer import HistoryBuffer
from scripts.td3.encoder import AdaptationModule


# Module-level state populated at startup from the args YAML so the
# monkey-patched evaluate function can read them without a closure.
_EVAL_PARAM_SEED: int | None = None
_EVAL_N_ENVS: int = 1
_EVAL_EPS_PER_ENV: int = 4
_EVAL_ENV_OVERRIDES: list[dict[str, float]] | None = None  # cached on first call
_LOG_PARENT_DIR: str | None = None  # for one-time eval_envs.json dump
# Incremented on every multi-env eval call. Used to shift the eval env's
# `seed` so episodes vary across checkpoints (otherwise each fresh
# SyncVectorEnv builds with the YAML's seed=0 and replays identical
# trajectories — masking real policy improvement). The dynamics-parameter
# overrides (paddle_density / puck_damping / gravity) stay fixed across
# calls because they're sampled from `_EVAL_PARAM_SEED` once and cached
# in `_EVAL_ENV_OVERRIDES`.
_EVAL_CALL_COUNT: int = 0

# Keep a handle to the original `evaluate_agent` so we can call it for
# the GIF-saving path on env 0.
_original_evaluate_agent = td3_training.evaluate_agent


def _sample_eval_env_overrides(
    seed: int,
    n_envs: int,
    random_variable_ranges: dict[str, list[float]],
    random_variables: list[str],
) -> list[dict[str, float]]:
    """Draw `n_envs` independent parameter dicts using a fixed RandomState.

    Order of variables matters for reproducibility: this iterates
    `random_variables` (preserving the YAML's order). Each variable is
    sampled `n_envs` times in sequence, so swapping the order of two
    variables would change the sampled values. Keep the YAML's order
    stable across runs.
    """
    rng = np.random.RandomState(int(seed))
    overrides: list[dict[str, float]] = [{} for _ in range(n_envs)]
    for var in random_variables:
        if var not in random_variable_ranges:
            raise KeyError(
                f"random_variables lists '{var}' but random_variable_ranges "
                f"has no entry for it (keys: {list(random_variable_ranges.keys())})"
            )
        low, high = random_variable_ranges[var]
        for env_idx in range(n_envs):
            overrides[env_idx][var] = float(rng.uniform(float(low), float(high)))
    return overrides


def _apply_overrides_to_air_hockey_params(
    air_hockey_params: dict[str, Any],
    overrides: dict[str, float],
) -> dict[str, Any]:
    """Return a deep copy of `air_hockey_params` with the dynamics
    overrides applied to its `simulator_params` block. Also disables
    `domain_random` for the eval copy so the eval env is exactly the
    overridden dict (no per-reset re-randomization)."""
    cfg = copy.deepcopy(air_hockey_params)
    sim_params = cfg.setdefault("simulator_params", {})
    for var, value in overrides.items():
        sim_params[var] = value
    # Eval should run on the FIXED overridden env, so turn the per-reset
    # rebuild off for the eval copy regardless of training-side setting.
    cfg["domain_random"] = False
    return cfg


def _rollout_returns(
    air_hockey_params: dict[str, Any],
    model_path: str,
    n_eps: int,
    action_scale: float,
    agent_hidden_layer_size: int,
    agent_num_hidden_layers: int,
    use_last_action_in_policy_state: bool,

    use_history=False,
    use_transformer=False,
    context_vector_dim=8,
    context_len=7,
    HISTORY_ENTRY_DIM=6,
    use_rma=False,
    rma_latent_dim=8,
    rma_include_action_history=True,
    rma_adaptation_hidden_sizes=(256, 128),
    adaptation_module_path=None,

) -> dict[str, Any]:
    """Build env from `air_hockey_params`, load actor from `model_path`,
    run `n_eps` episodes, return per-episode returns + aggregate stats.

    Mirrors the rollout structure used by `_save_task_gif_with_last_action`
    in scripts/td3/evaluate.py but skips frame capture and
    returns metrics instead of writing a GIF."""

    def _make_eval_env():
        return AirHockeyEnv(air_hockey_params)

    envs = gym.vector.SyncVectorEnv([_make_eval_env])
    action_dim = int(np.prod(envs.single_action_space.shape))

    # TODO: Checked
    if use_rma:
        raw_obs_dim = int(np.prod(envs.single_observation_space.shape))
        augmented_obs_dim = raw_obs_dim + rma_latent_dim
        if use_last_action_in_policy_state:
            augmented_obs_dim += action_dim
        policy_env_view = SimpleNamespace(
            single_observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(augmented_obs_dim,), dtype=np.float32
            ),
            single_action_space=envs.single_action_space,
        )
    elif use_history:
        raw_obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))

        if use_transformer:
            augmented_obs_dim = raw_obs_dim + act_dim if use_last_action_in_policy_state else raw_obs_dim
            augmented_obs_dim += context_vector_dim
        else:
            # context_len only
            augmented_obs_dim = (context_len * HISTORY_ENTRY_DIM)
            if use_last_action_in_policy_state:
                augmented_obs_dim += act_dim

        policy_env_view = SimpleNamespace(
            single_observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(augmented_obs_dim,), dtype=np.float32
            ),
            single_action_space=envs.single_action_space,
        )
    else:
        policy_env_view = build_policy_env_view(envs, use_last_action_in_policy_state)


    model = load_policy_for_evaluation(
        model_path=model_path,
        policy_env_view=policy_env_view,
        action_scale=action_scale,
        agent_hidden_layer_size=agent_hidden_layer_size,
        agent_num_hidden_layers=agent_num_hidden_layers,
    )

    transformer = None

    history_buf = HistoryBuffer(
        context_len=context_len,
        include_action=rma_include_action_history if use_rma else False,
        action_dim=action_dim,
    )

    if use_transformer:
        
        # raw_obs_dim = int(np.prod(envs.single_observation_space.shape))

        transformer = ContextEncoder(
            obs_dim=HISTORY_ENTRY_DIM,
            context_dim=context_vector_dim,
            context_len=context_len,
        )
        transformer_path = os.path.join(os.path.dirname(model_path), "transformer.pth")
        if os.path.exists(transformer_path):
            transformer.load_state_dict(torch.load(transformer_path, map_location="cpu"))
            transformer.eval()
        else:
            print(f"Warning: transformer.pth not found at {transformer_path}, using random weights")

    adaptation_module = None
    if use_rma:
        path = adaptation_module_path or os.path.join(
            os.path.dirname(model_path), "adaptation_module.pth"
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"RMA adaptation checkpoint not found: {path}")
        adaptation_module = AdaptationModule(
            history_input_dim=context_len * (
                HISTORY_ENTRY_DIM + (action_dim if rma_include_action_history else 0)
            ),
            latent_dim=rma_latent_dim,
            hidden_size=rma_adaptation_hidden_sizes,
        )
        adaptation_module.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=False)
        )
        adaptation_module.eval()
        

    env = envs.envs[0]
    env.max_timesteps = 200  # match the GIF eval's truncation budget

    returns: list[float] = []
    successes: list[int] = []
    episode_lengths: list[int] = []

    for _ in range(n_eps):
        obs, _ = env.reset()
        history_buf.reset_env()
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        last_action = torch.zeros((1, action_dim), dtype=torch.float32)
        done = False
        cum_rew = 0.0
        steps = 0
        while not done:

            history_buf.add(obs, action=last_action.numpy().squeeze(0))

            # TODO: Checked
            if use_history or use_rma:

                state_history = history_buf.sample()

                if use_rma:
                    with torch.no_grad():
                        latent = adaptation_module(state_history)
                    obs_with_context = torch.cat(
                        [obs_tensor.unsqueeze(0), latent], dim=-1
                    )
                    policy_obs = augment_policy_observation(
                        obs_with_context,
                        last_action,
                        use_last_action_in_policy_state,
                    )
                elif transformer is not None:
                    with torch.no_grad():
                        context_vector = transformer(state_history)  # (1, context_dim)

                    obs_with_context = torch.cat([obs_tensor.unsqueeze(0), context_vector], dim=-1)
                    policy_obs = augment_policy_observation(
                        obs_with_context, last_action, use_last_action_in_policy_state
                    )
                else:
                    # context_len only
                    context = state_history.view(1, -1)

                    policy_obs = augment_policy_observation(
                        context, last_action, use_last_action_in_policy_state
                    )
            else:
                policy_obs = augment_policy_observation(
                    obs_tensor.unsqueeze(0), last_action, use_last_action_in_policy_state
                )

            with torch.no_grad():
                action = model(policy_obs).numpy().squeeze()
            obs, rew, term, trunc, info = env.step(action)
            cum_rew += float(rew)
            steps += 1
            done = bool(term or trunc)
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            last_action = torch.tensor(action, dtype=torch.float32).reshape(1, -1)
            if done:
                last_action.zero_()
        returns.append(cum_rew)
        # Use env's episode-length-based success heuristic to match training stats:
        # an episode with full max_timesteps without termination is a "success"
        # in the juggle task (the puck stayed alive). Fall back to info if present.
        success = int(info.get("success", steps >= env.max_timesteps and not term)) if info is not None else 0
        successes.append(success)
        episode_lengths.append(steps)

    envs.close()
    return {
        "returns": returns,
        "successes": successes,
        "episode_lengths": episode_lengths,
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
        "mean_success_rate": float(np.mean(successes)) if successes else float("nan"),
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
    }


def _evaluate_agent_multi_env(
    model_path,
    save_dir,
    air_hockey_params,
    air_hockey_config_path=None,
    n_eps=5,
    n_gifs=3,
    base_reward_scaling=1.0,
    reference_states=None,
    ref_max_episode_steps=None,
    action_scale=0.02,
    agent_hidden_layer_size=64,
    agent_num_hidden_layers=2,
    agent_hidden_size=None,
    use_last_action_in_policy_state=False,
    policy_type=None,

    HISTORY_ENTRY_DIM=6,
    use_transformer=False,
    use_history=False,
    context_vector_dim=8,
    context_len=7,
    use_rma=False,
    rma_latent_dim=8,
    rma_include_action_history=True,
    rma_adaptation_hidden_sizes=(256, 128),
    adaptation_module_path=None,
):
    """Drop-in replacement for `td3_training.evaluate_agent`.

    Behavior:
    1. Lazy-init the cached list of `eval_n_envs` parameter overrides on
       first call (uses module globals set up by `_entrypoint_dr`).
    2. For env 0: call the original `evaluate_agent` with the overridden
       config so the GIF is saved (visual sanity for one env).
    3. For envs 1..N-1: run a metric-only rollout (no GIF) at the same
       per-env episode count.
    4. Aggregate per-env stats + grand mean, dump to multi_env_eval.json
       in `save_dir`, print one summary line so it lands in the run log.
    """
    global _EVAL_ENV_OVERRIDES

    if _EVAL_PARAM_SEED is None:
        # Fall back to canonical single-env eval if the seed wasn't set.
        return _original_evaluate_agent(
            model_path=model_path,
            save_dir=save_dir,
            air_hockey_params=air_hockey_params,
            air_hockey_config_path=air_hockey_config_path,
            n_eps=n_eps,
            n_gifs=n_gifs,
            base_reward_scaling=base_reward_scaling,
            reference_states=reference_states,
            ref_max_episode_steps=ref_max_episode_steps,
            action_scale=action_scale,
            agent_hidden_layer_size=agent_hidden_layer_size,
            agent_num_hidden_layers=agent_num_hidden_layers,
            agent_hidden_size=agent_hidden_size,
            use_last_action_in_policy_state=use_last_action_in_policy_state,
            policy_type=policy_type,
            HISTORY_ENTRY_DIM=HISTORY_ENTRY_DIM,
            use_history=use_history,
            use_transformer=use_transformer,
            context_vector_dim=context_vector_dim,
            context_len=context_len,
            use_rma=use_rma,
            rma_latent_dim=rma_latent_dim,
            rma_include_action_history=rma_include_action_history,
            rma_adaptation_hidden_sizes=rma_adaptation_hidden_sizes,
            adaptation_module_path=adaptation_module_path,
        )

    if _EVAL_ENV_OVERRIDES is None:
        random_variables = list(air_hockey_params.get("random_variables", []))
        random_variable_ranges = dict(air_hockey_params.get("random_variable_ranges", {}))
        if not random_variables or not random_variable_ranges:
            raise ValueError(
                "td3_training_dr requires `random_variables` and "
                "`random_variable_ranges` in the air_hockey config to sample "
                "the multi-env eval set."
            )
        _EVAL_ENV_OVERRIDES = _sample_eval_env_overrides(
            seed=_EVAL_PARAM_SEED,
            n_envs=_EVAL_N_ENVS,
            random_variable_ranges=random_variable_ranges,
            random_variables=random_variables,
        )
        # One-time dump of the eval set for reproducibility.
        if _LOG_PARENT_DIR is not None:
            os.makedirs(_LOG_PARENT_DIR, exist_ok=True)
            with open(os.path.join(_LOG_PARENT_DIR, "eval_envs.json"), "w") as f:
                json.dump(
                    {
                        "eval_param_seed": _EVAL_PARAM_SEED,
                        "n_envs": _EVAL_N_ENVS,
                        "eps_per_env": _EVAL_EPS_PER_ENV,
                        "random_variables": random_variables,
                        "random_variable_ranges": {
                            k: list(v) for k, v in random_variable_ranges.items()
                        },
                        "overrides": _EVAL_ENV_OVERRIDES,
                    },
                    f,
                    indent=2,
                )
            print(
                f"[td3_training_dr] Sampled {_EVAL_N_ENVS} eval envs (seed={_EVAL_PARAM_SEED}) "
                f"-> {os.path.join(_LOG_PARENT_DIR, 'eval_envs.json')}"
            )

    global _EVAL_CALL_COUNT
    _EVAL_CALL_COUNT += 1

    per_env_results: list[dict[str, Any]] = []
    eps_per_env = _EVAL_EPS_PER_ENV

    for env_idx, override in enumerate(_EVAL_ENV_OVERRIDES):
        eval_cfg = _apply_overrides_to_air_hockey_params(air_hockey_params, override)
        # Per-call seed shift: each fresh env_test starts with this seed,
        # then env.reset() advances internally for the 4 in-call episodes.
        # Across calls, _EVAL_CALL_COUNT bumps the seed so we sample new
        # start states each checkpoint while the dynamics overrides stay
        # fixed (the actual ablation control). Reproducible across reruns
        # of the same training run because the formula is deterministic.
        eval_cfg["seed"] = int(
            (_EVAL_PARAM_SEED * 100000) + (env_idx * 1000) + _EVAL_CALL_COUNT
        )
        if env_idx == 0:
            # Save the GIF for env 0 via the original eval path so the
            # existing checkpoint_<step>/eval_*.gif convention is preserved.
            try:
                _original_evaluate_agent(
                    model_path=model_path,
                    save_dir=save_dir,
                    air_hockey_params=eval_cfg,
                    air_hockey_config_path=air_hockey_config_path,
                    n_eps=eps_per_env,
                    n_gifs=1,
                    base_reward_scaling=base_reward_scaling,
                    reference_states=reference_states,
                    ref_max_episode_steps=ref_max_episode_steps,
                    action_scale=action_scale,
                    agent_hidden_layer_size=agent_hidden_layer_size,
                    agent_num_hidden_layers=agent_num_hidden_layers,
                    agent_hidden_size=agent_hidden_size,
                    use_last_action_in_policy_state=use_last_action_in_policy_state,
                    policy_type=policy_type,

                    HISTORY_ENTRY_DIM=HISTORY_ENTRY_DIM,
                    use_history=use_history,
                    use_transformer=use_transformer,       
                    context_vector_dim=context_vector_dim,       
                    context_len=context_len,
                    use_rma=use_rma,
                    rma_latent_dim=rma_latent_dim,
                    rma_include_action_history=rma_include_action_history,
                    rma_adaptation_hidden_sizes=rma_adaptation_hidden_sizes,
                    adaptation_module_path=adaptation_module_path,
                )
            except Exception as e:
                print(f"[td3_training_dr] env0 GIF eval failed (continuing): {e}")
        # Always also collect metrics for env_idx (including 0).
        try:
            stats = _rollout_returns(
                air_hockey_params=eval_cfg,
                model_path=model_path,
                n_eps=eps_per_env,
                action_scale=action_scale,
                agent_hidden_layer_size=agent_hidden_layer_size,
                agent_num_hidden_layers=agent_num_hidden_layers,
                use_last_action_in_policy_state=use_last_action_in_policy_state,

                use_history=use_history,
                use_transformer=use_transformer,
                HISTORY_ENTRY_DIM=HISTORY_ENTRY_DIM,
                context_vector_dim=context_vector_dim,
                context_len=context_len,
                use_rma=use_rma,
                rma_latent_dim=rma_latent_dim,
                rma_include_action_history=rma_include_action_history,
                rma_adaptation_hidden_sizes=rma_adaptation_hidden_sizes,
                adaptation_module_path=adaptation_module_path,
            )
            stats["env_idx"] = env_idx
            stats["override"] = override
            per_env_results.append(stats)
        except Exception as e:
            print(f"[td3_training_dr] env{env_idx} metric rollout failed (continuing): {e}")
            per_env_results.append({"env_idx": env_idx, "override": override, "error": str(e)})

    valid = [r for r in per_env_results if "error" not in r]
    aggregate = {
        "n_envs_used": len(valid),
        "eps_per_env": eps_per_env,
        "mean_return_across_envs": (
            float(np.mean([r["mean_return"] for r in valid])) if valid else float("nan")
        ),
        "mean_success_across_envs": (
            float(np.mean([r["mean_success_rate"] for r in valid])) if valid else float("nan")
        ),
        "mean_ep_length_across_envs": (
            float(np.mean([r["mean_episode_length"] for r in valid])) if valid else float("nan")
        ),
        "per_env_mean_return": [r.get("mean_return", float("nan")) for r in per_env_results],
        "per_env_mean_success": [r.get("mean_success_rate", float("nan")) for r in per_env_results],
    }

    out_path = os.path.join(save_dir, "multi_env_eval.json")
    try:
        with open(out_path, "w") as f:
            json.dump({"aggregate": aggregate, "per_env": per_env_results}, f, indent=2)
    except Exception as e:
        print(f"[td3_training_dr] Failed to write {out_path}: {e}")

    per_env_returns_str = ", ".join(f"{x:.1f}" for x in aggregate["per_env_mean_return"])
    print(
        f"[td3_training_dr] Multi-env eval (n_envs={_EVAL_N_ENVS}, eps/env={eps_per_env}): "
        f"mean_return={aggregate['mean_return_across_envs']:.2f}, "
        f"mean_success={aggregate['mean_success_across_envs']:.3f}, "
        f"per_env_returns=[{per_env_returns_str}]"
    )


def _entrypoint_dr():
    """Read the args YAML up front to extract eval_param_seed/n_envs/eps_per_env
    + log_parent_dir, stash them in module globals, monkey-patch
    `td3_training.evaluate_agent`, then invoke the canonical trainer."""
    global _EVAL_PARAM_SEED, _EVAL_N_ENVS, _EVAL_EPS_PER_ENV, _LOG_PARENT_DIR

    args_file = None
    if "--args-file" in sys.argv:
        idx = sys.argv.index("--args-file")
        if idx + 1 < len(sys.argv):
            args_file = sys.argv[idx + 1]
    if args_file is None:
        raise RuntimeError(
            "td3_training_dr requires --args-file <path> so the wrapper can "
            "extract eval_param_seed before the trainer starts."
        )
    with open(args_file, "r") as f:
        args_dict = yaml.load(f, Loader=yaml.FullLoader)
    _EVAL_PARAM_SEED = args_dict.get("eval_param_seed", None)
    _EVAL_N_ENVS = int(args_dict.get("eval_n_envs", 1))
    _EVAL_EPS_PER_ENV = int(args_dict.get("eval_eps_per_env", 4))
    _LOG_PARENT_DIR = args_dict.get("log_parent_dir", None)
    # Honor a CLI override of --log-parent-dir so eval_envs.json lands next
    # to the actual trainer outputs rather than the YAML default.
    if "--log-parent-dir" in sys.argv:
        idx = sys.argv.index("--log-parent-dir")
        if idx + 1 < len(sys.argv):
            _LOG_PARENT_DIR = sys.argv[idx + 1]

    print(
        f"[td3_training_dr] eval_param_seed={_EVAL_PARAM_SEED}, "
        f"eval_n_envs={_EVAL_N_ENVS}, eval_eps_per_env={_EVAL_EPS_PER_ENV}, "
        f"log_parent_dir={_LOG_PARENT_DIR}"
    )

    # Monkey-patch the trainer's evaluate_agent to our multi-env version.
    td3_training.evaluate_agent = _evaluate_agent_multi_env

    # Hand off to the canonical trainer.
    td3_training._entrypoint()


if __name__ == "__main__":
    _entrypoint_dr()
