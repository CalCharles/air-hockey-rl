"""
eval_id_ood_comparison.py

Pure analysis module — no arg parsing, no model loading.

Called from td3_training._entrypoint() after all models are set up, mirroring
the pattern used by context_vector_analysis.py.

Entry point:
    eval_id_ood_comparison(
        baseline_actor=actor,           # plain TD3 actor (no context)
        air_hockey_base=config["air_hockey"],
        raw_obs_dim=raw_obs_dim,
        act_dim=act_dim,
        use_last_action=args.use_last_action_in_policy_state,
        n_envs=args.eval_id_ood_n_envs,
        n_eps=args.eval_id_ood_n_eps,
        ood_scale=args.eval_id_ood_ood_scale,
        out_dir=args.eval_id_ood_out_dir,
        device=args.device,
        seed=args.seed,
        # context-vector model (optional):
        context_actor=actor,
        context_transformer=transformer,
        context_history_buf=history_buf,
    )

Outputs written to out_dir:
    summary.json          — per-env stats + aggregates
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
# Dynamics sampling  (mirrors context_vector_analysis.py exactly)
# ---------------------------------------------------------------------------

def _sample_id_params(random_variable_ranges, random_variables, rng):
    return {
        var: float(rng.uniform(*random_variable_ranges[var]))
        for var in random_variables
    }



def _sample_ood_params(random_variable_ranges, random_variables, rng, ood_scale=2.0, ood_gap=0.5):
    params = {}
    for var in random_variables:
        low, high = random_variable_ranges[var]
        id_width = high - low

        high_ood_start = high + ood_gap * id_width
        high_ood_end   = high_ood_start + ood_scale * id_width

        low_ood_end   = low - ood_gap * id_width
        low_ood_start = low_ood_end - ood_scale * id_width

        # A side is valid if its range doesn't cross zero in the wrong direction.
        # For positive variables (low >= 0): low side must stay >= 0.
        # For negative variables (high <= 0): high side must stay <= 0.
        # For mixed-sign variables: both sides are always valid.
        if low >= 0:
            # e.g. puck_damping, paddle_density — must stay non-negative
            low_side_valid  = low_ood_start >= 0.0
            high_side_valid = True
        elif high <= 0:
            # e.g. gravity — must stay non-positive
            low_side_valid  = True
            high_side_valid = high_ood_end <= 0.0
        else:
            # mixed sign — no physical constraint, both sides valid
            low_side_valid  = True
            high_side_valid = True

        both_valid = low_side_valid and high_side_valid
        if both_valid:
            if rng.rand() < 0.5:
                params[var] = float(rng.uniform(low_ood_start, low_ood_end))
            else:
                params[var] = float(rng.uniform(high_ood_start, high_ood_end))
        elif low_side_valid:
            params[var] = float(rng.uniform(low_ood_start, low_ood_end))
        else:
            params[var] = float(rng.uniform(high_ood_start, high_ood_end))

    return params




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
    transformer=None,
    history_buf: HistoryBuffer | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Roll out n_eps episodes. Works for both baseline and context models."""
    env = AirHockeyEnv(air_hockey_params)
    env.max_timesteps = 200

    returns, successes, episode_lengths = [], [], []

    for _ in range(n_eps):
        obs, _ = env.reset()

        if history_buf is not None:
            history_buf._reset_env()

        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        last_action = torch.zeros((1, act_dim), dtype=torch.float32)
        done = False
        cum_rew = 0.0
        steps = 0

        while not done:
            if transformer is not None and history_buf is not None:
                history_buf.add(obs, done=False)
                with torch.no_grad():
                    state_history = history_buf.sample()          # (1, T, obs_dim)
                    context_vec = transformer(state_history)      # (1, context_dim)
                obs_with_ctx = torch.cat(
                    [obs_tensor.unsqueeze(0), context_vec.cpu()], dim=-1
                )
                policy_obs = augment_policy_observation(obs_with_ctx, last_action, use_last_action)
            else:
                policy_obs = augment_policy_observation(
                    obs_tensor.unsqueeze(0), last_action, use_last_action
                )

            with torch.no_grad():
                action = actor(policy_obs.to(device)).cpu().numpy().squeeze()

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


def _print_and_save_table(aggregates, out_dir, baseline_model_path, context_model_path, n_envs, n_eps, ood_scale):
    lines = []
    lines.append(f"ID/OOD Comparison  (n_envs={n_envs}, n_eps={n_eps}, ood_scale={ood_scale})")
    lines.append(f"baseline : {baseline_model_path}")
    lines.append(f"context  : {context_model_path or '(none)'}")
    lines.append("")
    header = f"{'Model':>12}  {'Cond':>4}  {'MeanReturn':>12}  {'±Std':>8}  {'SuccessRate':>11}"
    lines.append(header)
    lines.append("-" * 56)
    for model_name, conds in aggregates.items():
        for cond_name, agg in conds.items():
            lines.append(
                f"{model_name:>12}  {cond_name.upper():>4}  "
                f"{agg['mean_return']:>12.2f}  "
                f"±{agg['std_return']:>7.2f}  "
                f"{agg['mean_success_rate']:>11.3f}"
            )
    lines.append("")
    for model_name, conds in aggregates.items():
        id_ret  = conds["id"]["mean_return"]
        ood_ret = conds["ood"]["mean_return"]
        ratio   = (ood_ret / id_ret) if id_ret != 0 else float("nan")
        lines.append(
            f"  {model_name}: OOD/ID return ratio = {ratio:.3f}  "
            f"(OOD={ood_ret:.2f}, ID={id_ret:.2f})"
        )

    for line in lines:
        print(line)

    table_path = os.path.join(out_dir, "comparison_table.txt")
    with open(table_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved comparison table → {table_path}")


def _plot_bar_chart(aggregates: dict, out_dir: str):
    model_names = list(aggregates.keys())
    conditions  = ["id", "ood"]
    cond_labels = {"id": "In-Distribution", "ood": "Out-of-Distribution"}
    colors      = {"id": "#2196F3", "ood": "#F44336"}

    x     = np.arange(len(model_names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, metric, std_key, ylabel in zip(
        axes,
        ["mean_return",       "mean_success_rate"],
        ["std_return",        "std_success_rate"],
        ["Mean Episode Return", "Mean Success Rate"],
    ):
        for ci, cond in enumerate(conditions):
            vals = [aggregates[m][cond][metric]  for m in model_names]
            stds = [aggregates[m][cond][std_key] for m in model_names]
            offset = (ci - 0.5) * width
            bars = ax.bar(
                x + offset, vals, width,
                label=cond_labels[cond], color=colors[cond],
                alpha=0.8, yerr=stds, capsize=4,
            )
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in model_names])
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Baseline vs Context-Vector TD3: ID vs OOD Performance", fontsize=12)
    plt.tight_layout()
    chart_path = os.path.join(out_dir, "bar_chart.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Saved bar chart → {chart_path}")


# ---------------------------------------------------------------------------
# Main entry point — called from td3_training._entrypoint()
# ---------------------------------------------------------------------------

def compare_performance_ID_OOD(
    baseline_actor,
    air_hockey_base: dict,
    raw_obs_dim: int,
    act_dim: int,
    use_last_action: bool,
    n_envs: int = 10,
    n_eps: int = 8,
    ood_scale: float = 2.0,
    out_dir: str = "results/id_ood_comparison",
    device: str = "cpu",
    seed: int = 42,
    # Optional context-vector model — omit for baseline-only run
    context_actor=None,
    context_transformer=None,
    context_history_buf: HistoryBuffer | None = None,
    # For table header only
    baseline_model_path: str = "",
    context_model_path: str = "",
):
    """Compare baseline vs context-vector TD3 on ID and OOD dynamics.

    Args
    ----
    baseline_actor       : Loaded DeterministicAgent (eval mode, on device).
    air_hockey_base      : The air_hockey sub-dict from the sim config YAML.
                           Must contain random_variables + random_variable_ranges.
    raw_obs_dim          : Raw env observation dim (e.g. 30).
    act_dim              : Action dim (e.g. 2).
    use_last_action      : Whether policy obs includes last action.
    n_envs               : Number of distinct dynamics configs per condition.
    n_eps                : Episodes per dynamics config.
    ood_scale            : OOD range multiplier (see _sample_ood_params).
    out_dir              : Directory to write outputs.
    device               : torch device string.
    seed                 : RNG seed for reproducibility.
    context_actor        : Context-vector actor (optional).
    context_transformer  : Loaded ContextEncoder (optional).
    context_history_buf  : HistoryBuffer for context model (optional).
    baseline_model_path  : Path string used in table header only.
    context_model_path   : Path string used in table header only.
    """
    os.makedirs(out_dir, exist_ok=True)

    random_variables      = list(air_hockey_base.get("random_variables", []))
    random_variable_ranges = dict(air_hockey_base.get("random_variable_ranges", {}))

    if not random_variables:
        raise ValueError(
            "air_hockey config has no random_variables / random_variable_ranges. "
            "Use a paramrand config (e.g. sim_paramrand_pm25.yaml)."
        )

    print(f"\n{'='*60}")
    print(f"  eval_id_ood_comparison")
    print(f"  ID ranges : {random_variable_ranges}")
    print(f"  OOD scale : {ood_scale}x outside ID bounds")
    print(f"  n_envs={n_envs}, n_eps={n_eps}, seed={seed}")
    print(f"{'='*60}\n")

    baseline_actor.eval()
    if context_actor is not None:
        context_actor.eval()
    if context_transformer is not None:
        context_transformer.eval()

    # Sample fixed ID/OOD dynamics configs
    rng = np.random.RandomState(seed)
    id_param_sets  = [_sample_id_params (random_variable_ranges, random_variables, rng) for _ in range(n_envs)]
    ood_param_sets = [_sample_ood_params(random_variable_ranges, random_variables, rng, ood_scale) for _ in range(n_envs)]

    # Which models to evaluate
    models = [("baseline", baseline_actor, None, None)]
    if context_actor is not None:
        models.append(("context", context_actor, context_transformer, context_history_buf))

    raw_results = {name: {"id": [], "ood": []} for name, *_ in models}

    for cond_name, param_sets in [("id", id_param_sets), ("ood", ood_param_sets)]:
        for env_i, params in enumerate(param_sets):
            env_cfg = _build_env_config(
                air_hockey_base, params,
                seed=int(seed * 100000 + env_i * 1000 + (0 if cond_name == "id" else 500)),
            )
            label = f"[{cond_name.upper()} {env_i+1}/{n_envs}]"

            for model_name, actor, transformer, history_buf in models:
                stats = _rollout_episodes(
                    air_hockey_params=env_cfg,
                    actor=actor,
                    n_eps=n_eps,
                    act_dim=act_dim,
                    use_last_action=use_last_action,
                    transformer=transformer,
                    history_buf=history_buf,
                    device=device,
                )
                stats["params"]  = params
                stats["env_idx"] = env_i
                raw_results[model_name][cond_name].append(stats)

                print(
                    f"  {label} {model_name:>10} | "
                    f"mean_return={stats['mean_return']:7.2f}  "
                    f"success={stats['mean_success_rate']:.2f}  "
                    f"params={params}"
                )

    aggregates = {
        name: {"id": _agg(raw_results[name]["id"]), "ood": _agg(raw_results[name]["ood"])}
        for name, *_ in models
    }

    print(f"\n{'='*60}")
    _print_and_save_table(
        aggregates, out_dir,
        baseline_model_path, context_model_path,
        n_envs, n_eps, ood_scale,
    )

    summary = {
        "config": {
            "baseline_model_path":   baseline_model_path,
            "context_model_path":    context_model_path,
            "n_envs": n_envs, "n_eps": n_eps,
            "ood_scale": ood_scale, "seed": seed,
            "random_variables":       random_variables,
            "random_variable_ranges": random_variable_ranges,
        },
        "aggregates": aggregates,
        "per_env":    raw_results,
    }
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON → {json_path}")

    _plot_bar_chart(aggregates, out_dir)
    print(f"\n[eval_id_ood_comparison] Done. Results in {out_dir}/\n")