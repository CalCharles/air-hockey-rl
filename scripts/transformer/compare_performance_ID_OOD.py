"""
new_eval_id_ood_comparison.py

Pure analysis module — no arg parsing, no model loading.

Evaluates a single (already-loaded) actor [+ optional transformer/history_buf]
on a fixed ID and OOD set of dynamics-parameter dicts. The ID/OOD param sets
are loaded from `params_cache_path` if present, otherwise sampled and saved
there for reuse across runs.

Entry point:
    compare_performance_ID_OOD(
        actor=actor,
        air_hockey_base=config["air_hockey"],
        raw_obs_dim=raw_obs_dim,
        act_dim=act_dim,
        use_last_action=args.use_last_action_in_policy_state,
        use_history=args.use_history,
        use_transformer=args.use_transformer,
        transformer=transformer,            # required if use_transformer=True
        context_len=args.context_len,
        n_envs=args.eval_id_ood_n_envs,
        n_eps=args.eval_id_ood_n_eps,
        out_dir=args.eval_id_ood_out_dir,
        device=args.device,
        seed=args.seed,
        model_path=args.model_path or "",
        params_cache_path=args.params_cache_path,
    )

Outputs written to out_dir:
    summary.json          — per-condition stats + aggregates
    comparison_table.txt  — human-readable table
    bar_chart.png         — mean return + success rate bar chart
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from airhockey import AirHockeyEnv
from scripts.td3.eval_utils import augment_policy_observation
from scripts.transformer.history_buffer import HistoryBuffer


# ---------------------------------------------------------------------------
# Dynamics sampling
# ---------------------------------------------------------------------------

def _sample_id_params(random_variable_ranges, random_variables, rng):
    return {
        var: float(rng.uniform(*random_variable_ranges[var]))
        for var in random_variables
    }


def _build_env_config(base_config, param_overrides, seed=None):
    cfg = copy.deepcopy(base_config)
    sim_params = cfg.setdefault("simulator_params", {})
    for k, v in param_overrides.items():
        sim_params[k] = v
    cfg["domain_random"] = False
    if seed is not None:
        cfg["seed"] = int(seed)
    return cfg


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------

def _rollout_episodes(
    air_hockey_params: dict,
    actor,
    n_eps: int,
    act_dim: int,
    use_last_action: bool,
    use_history: bool = False,
    use_transformer: bool = False,
    transformer=None,
    history_buf: HistoryBuffer | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Roll out n_eps episodes, building the policy observation exactly as
    td3_training.py does (raw obs [+ history/context] [+ last action])."""
    env = AirHockeyEnv(air_hockey_params)
    env.max_timesteps = 200

    returns, successes, episode_lengths = [], [], []

    for _ in range(n_eps):
        obs, _ = env.reset()

        if history_buf is not None:
            history_buf.reset_env()

        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        last_action = torch.zeros((1, act_dim), dtype=torch.float32)
        done = False
        cum_rew = 0.0
        steps = 0

        while not done:
            if use_history:
                history_buf.add(obs)
                state_history = history_buf.sample().to(device)  # (1, T, 4)

                if use_transformer:
                    with torch.no_grad():
                        context = transformer(state_history)      # (1, context_dim)
                else:
                    context = state_history.view(1, -1)           # (1, T*4)

                obs_with_ctx = torch.cat(
                    [obs_tensor.unsqueeze(0).to(device), context], dim=-1
                )
                policy_obs = augment_policy_observation(
                    obs_with_ctx, last_action.to(device), use_last_action
                )
            else:
                policy_obs = augment_policy_observation(
                    obs_tensor.unsqueeze(0).to(device), last_action.to(device), use_last_action
                )

            with torch.no_grad():
                action = actor(policy_obs).cpu().numpy().squeeze()

            obs, rew, term, trunc, info = env.step(action)
            cum_rew += float(rew)
            steps += 1
            done = bool(term or trunc)
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            last_action = torch.tensor(action, dtype=torch.float32).reshape(1, -1)
            if done:
                last_action.zero_()

        returns.append(cum_rew)
        success = int(
            info.get("success", steps >= env.max_timesteps and not term)
            if info is not None else 0
        )
        successes.append(success)
        episode_lengths.append(steps)

    env.close()
    return {
        "returns": returns,
        "successes": successes,
        "episode_lengths": episode_lengths,
        "mean_return": float(np.mean(returns)),
        "mean_success_rate": float(np.mean(successes)),
        "mean_episode_length": float(np.mean(episode_lengths)),
    }


# ---------------------------------------------------------------------------
# Aggregation + output helpers
# ---------------------------------------------------------------------------

def _agg(stat_list):
    returns   = [s["mean_return"]       for s in stat_list]
    successes = [s["mean_success_rate"] for s in stat_list]
    lengths   = [s["mean_episode_length"] for s in stat_list]
    return {
        "mean_return":        float(np.mean(returns)),
        "std_return":         float(np.std(returns)),
        "mean_success_rate":  float(np.mean(successes)),
        "std_success_rate":   float(np.std(successes)),
        "mean_ep_length":     float(np.mean(lengths)),
        "per_env_returns":    returns,
        "per_env_successes":  successes,
    }


def _print_and_save_table(aggregates, out_dir, model_path, n_envs, n_eps):
    lines = []
    lines.append(f"ID/OOD Comparison  (n_envs={n_envs}, n_eps={n_eps})")
    lines.append(f"model : {model_path or '(in-memory)'}")
    lines.append("")
    header = f"{'Cond':>4}  {'MeanReturn':>12}  {'±Std':>8}  {'SuccessRate':>11}"
    lines.append(header)
    lines.append("-" * 44)
    for cond_name, agg in aggregates.items():
        lines.append(
            f"{cond_name.upper():>4}  "
            f"{agg['mean_return']:>12.2f}  "
            f"±{agg['std_return']:>7.2f}  "
            f"{agg['mean_success_rate']:>11.3f}"
        )

    id_ret  = aggregates["id"]["mean_return"]
    ood_ret = aggregates["ood"]["mean_return"]
    ratio   = (ood_ret / id_ret) if id_ret != 0 else float("nan")
    lines.append("")
    lines.append(
        f"  OOD/ID return ratio = {ratio:.3f}  (OOD={ood_ret:.2f}, ID={id_ret:.2f})"
    )

    for line in lines:
        print(line)

    table_path = os.path.join(out_dir, "comparison_table.txt")
    with open(table_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved comparison table → {table_path}")


def _plot_bar_chart(aggregates: dict, out_dir: str):
    conditions  = ["id", "ood"]
    cond_labels = {"id": "In-Distribution", "ood": "Out-of-Distribution"}
    colors      = {"id": "#2196F3", "ood": "#F44336"}

    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    for ax, metric, std_key, ylabel in zip(
        axes,
        ["mean_return",       "mean_success_rate"],
        ["std_return",        "std_success_rate"],
        ["Mean Episode Return", "Mean Success Rate"],
    ):
        vals = [aggregates[c][metric]  for c in conditions]
        stds = [aggregates[c][std_key] for c in conditions]
        bar_colors = [colors[c] for c in conditions]
        bars = ax.bar(
            [cond_labels[c] for c in conditions], vals,
            color=bar_colors, alpha=0.8, yerr=stds, capsize=4,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9,
            )
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("ID vs OOD Performance", fontsize=12)
    plt.tight_layout()
    chart_path = os.path.join(out_dir, "bar_chart.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Saved bar chart → {chart_path}")


# ---------------------------------------------------------------------------
# ID and OOD Environment Parameter save / load helper functions
# ---------------------------------------------------------------------------

def _load_params(cache_path: str) -> tuple[list[dict], list[dict]]:
    """Load ID and OOD param sets from a previously saved cache file."""
    with open(cache_path) as f:
        cached = json.load(f)
    id_params  = cached["id"]
    ood_params = cached["ood"]
    print(f"  Loaded {len(id_params)} ID + {len(ood_params)} OOD param sets from {cache_path}")
    return id_params, ood_params


def _save_params(cache_path: str, id_params: list[dict], ood_params: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"id": id_params, "ood": ood_params}, f, indent=2)
    print(f"  Saved param sets → {cache_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compare_performance_ID_OOD(
    actor,
    air_hockey_base: dict,
    raw_obs_dim: int,
    act_dim: int,
    use_last_action: bool,
    use_history: bool = False,
    use_transformer: bool = False,
    transformer=None,
    context_len: int = 0,
    n_envs: int = 10,
    n_eps: int = 8,
    out_dir: str = "results/id_ood_comparison",
    device: str = "cpu",
    seed: int = 42,
    model_path: str = "",
    params_cache_path: str | None = None,
):
    """Evaluate `actor` (+ optional `transformer`/history) on a fixed ID and
    OOD set of dynamics-parameter dicts.

    Args
    ----
    actor                : Loaded DeterministicAgent (eval mode, on device).
                           Expected to accept policy obs matching
                           use_history/use_transformer/use_last_action.
    air_hockey_base      : The air_hockey sub-dict from the sim config YAML.
                           Must contain random_variables, random_variable_ranges,
                           and random_variable_ranges_OOD.
    raw_obs_dim          : Raw env observation dim (e.g. 30).
    act_dim              : Action dim (e.g. 2).
    use_last_action      : Whether policy obs includes last action.
    use_history          : Whether policy obs includes history/context.
    use_transformer      : If use_history, whether context is transformer
                           output (True) or flat-concat history (False).
    transformer          : Loaded ContextEncoder. Required if use_transformer.
    context_len          : History length. Required if use_history.
    n_envs               : Number of distinct dynamics configs per condition.
    n_eps                : Episodes per dynamics config.
    out_dir              : Directory to write outputs.
    device               : torch device string.
    seed                 : RNG seed for reproducibility / param sampling.
    model_path           : Path string used in table header only.
    params_cache_path    : Path to load/save the ID+OOD param sets. If the
                           file exists, param sets are loaded from it
                           (ignoring n_envs/seed); otherwise n_envs sets are
                           sampled per condition and saved there.
    """
    os.makedirs(out_dir, exist_ok=True)

    if use_history and context_len <= 0:
        raise ValueError("use_history=True requires context_len > 0.")
    if use_history and use_transformer and transformer is None:
        raise ValueError("use_transformer=True requires a loaded `transformer`.")

    random_variables           = list(air_hockey_base.get("random_variables", []))
    random_variable_ranges     = dict(air_hockey_base.get("random_variable_ranges", {}))
    # random_variable_ranges_OOD = dict(air_hockey_base.get("random_variable_ranges_OOD", {}))

    if not random_variables:
        raise ValueError(
            "air_hockey config has no random_variables / random_variable_ranges. "
            "Use a paramrand config (e.g. sim_paramrand_pm25.yaml)."
        )
    # if not random_variable_ranges_OOD:
    #     raise ValueError(
    #         "air_hockey config has no random_variable_ranges_OOD. "
    #         "Use a config with an OOD range block (e.g. sim_paramrand_pm25_OOD.yaml)."
    #     )

    print(f"\n{'='*60}")
    print(f"  compare_performance_ID_OOD")
    print(f"  ID  ranges : {random_variable_ranges}")
    # print(f"  OOD ranges : {random_variable_ranges_OOD}")
    print(f"  n_envs={n_envs}, n_eps={n_eps}, seed={seed}")
    print(f"  use_history={use_history}, use_transformer={use_transformer}")
    print(f"{'='*60}\n")

    actor.eval()
    if transformer is not None:
        transformer.eval()

    history_buf = None
    if use_history:
        history_buf = HistoryBuffer(context_len=context_len, device=device)

    # --- Load or generate ID/OOD param sets ---
    if params_cache_path is not None and os.path.exists(params_cache_path):
        id_param_sets, ood_param_sets = _load_params(params_cache_path)
    else:
        rng = np.random.RandomState(seed)
        id_param_sets  = [_sample_id_params(random_variable_ranges,     random_variables, rng) for _ in range(n_envs)]
        # ood_param_sets = [_sample_id_params(random_variable_ranges_OOD, random_variables, rng) for _ in range(n_envs)]
        if params_cache_path is not None:
            _save_params(params_cache_path, id_param_sets, ood_param_sets)

    raw_results = {"id": [], "ood": []}

    for cond_name, param_sets in [("id", id_param_sets)]:   # , ("ood", ood_param_sets)
        for env_i, params in enumerate(param_sets):
            env_cfg = _build_env_config(
                air_hockey_base, params,
                seed=int(seed * 100000 + env_i * 1000 + (0 if cond_name == "id" else 500)),
            )
            label = f"[{cond_name.upper()} {env_i+1}/{len(param_sets)}]"

            stats = _rollout_episodes(
                air_hockey_params=env_cfg,
                actor=actor,
                n_eps=n_eps,
                act_dim=act_dim,
                use_last_action=use_last_action,
                use_history=use_history,
                use_transformer=use_transformer,
                transformer=transformer,
                history_buf=history_buf,
                device=device,
            )
            stats["params"]  = params
            stats["env_idx"] = env_i
            raw_results[cond_name].append(stats)

            print(
                f"  {label} mean_return={stats['mean_return']:7.2f}  "
                f"success={stats['mean_success_rate']:.2f}  "
                f"params={params}"
            )

    aggregates = {
        "id":  _agg(raw_results["id"]),
        "ood": _agg(raw_results["ood"]),
    }

    print(f"\n{'='*60}")
    _print_and_save_table(aggregates, out_dir, model_path, len(id_param_sets), n_eps)

    summary = {
        "config": {
            "model_path": model_path,
            "n_envs": len(id_param_sets), "n_eps": n_eps, "seed": seed,
            "use_history": use_history, "use_transformer": use_transformer,
            "context_len": context_len,
            "random_variables":           random_variables,
            "random_variable_ranges_ID":     random_variable_ranges,
            # "random_variable_ranges_OOD": random_variable_ranges_OOD,
        },
        "aggregates": aggregates,
        "per_env":    raw_results,
    }
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON → {json_path}")

    _plot_bar_chart(aggregates, out_dir)
    print(f"\n[compare_performance_ID_OOD] Done. Results in {out_dir}/\n")