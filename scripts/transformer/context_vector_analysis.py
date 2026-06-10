'''
context_vector_analysis.py

Goal with this script is to use the pre-trained transformer model to produce context vectors given a set of observations
from in-distribution environments (i.e. "environments spun up using same env params" as that during training the transformer)
and out-of-distribution environments (i.e. "..." differing from that seen during training the transformer)

We save the data used to produce the context vectors as well as the context vectors themselves.

We then plot the two sets of context vectors on the same graph using t-SNE. 
The hope is that we see the two sets produce distinct pattern observed via t-SNE to indicate that the context vector learns something useful to predict optimal actions.

'''



"""
Context vector analysis: collect context vectors across ID and OOD dynamics,
then plot with t-SNE.

This module is called directly from td3_training._entrypoint() after all
models and env configs are already set up. It receives live objects — no
file loading or arg parsing needed here.

Called via:
    python -m scripts.td3.td3_training \
        --args-file configs/... \
        --model-path runs/.../checkpoint_50000/model.pth \
        --analyze-context-vectors \
        --context-analysis-out-dir results/context_tsne
"""

import copy
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from airhockey import AirHockeyEnv
from scripts.transformer.history_buffer import HistoryBuffer
from scripts.td3.eval_utils import augment_policy_observation
from sklearn.manifold import TSNE


# ---------------------------------------------------------------------------
# Env parameter sampling
# ---------------------------------------------------------------------------

def _sample_id_params(random_variable_ranges, random_variables, rng):
    return {
        var: float(rng.uniform(*random_variable_ranges[var]))
        for var in random_variables
    }


def _sample_ood_params(random_variable_ranges, random_variables, rng, ood_scale=1.0, ood_gap=0.5):
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


def _build_env_config(base_config, param_overrides):
    cfg = copy.deepcopy(base_config)
    sim_params = cfg.setdefault("simulator_params", {})
    for k, v in param_overrides.items():
        sim_params[k] = v
    cfg["domain_random"] = False  # fix params for this episode set
    return cfg


# ---------------------------------------------------------------------------
# Rollout + context vector collection
# ---------------------------------------------------------------------------

def _collect_context_vectors_for_params(
    air_hockey_params,
    actor,
    transformer,
    raw_obs_dim,
    act_dim,
    context_len,
    use_last_action,
    n_eps,
    device,
    rng,
):
    """Run n_eps episodes, return array of context vectors (one per step)."""
    air_hockey_params = copy.deepcopy(air_hockey_params)
    air_hockey_params["seed"] = int(rng.randint(0, int(1e8)))

    env = AirHockeyEnv(air_hockey_params)
    env.max_timesteps = 200

    history_buf = HistoryBuffer(
        obs_dim=raw_obs_dim,
        context_len=context_len,
        device=device,
    )

    context_vectors = []

    for _ in range(n_eps):
        obs, _ = env.reset()
        if hasattr(history_buf, "reset"):
            history_buf.reset()
        last_action = torch.zeros((1, act_dim), dtype=torch.float32)
        done = False

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            history_buf.add(obs, done=False)

            with torch.no_grad():
                state_history = history_buf.sample()          # (1, T, obs_dim)
                context_vector = transformer(state_history)   # (1, context_dim)

            context_vectors.append(context_vector.squeeze(0).cpu().numpy())

            obs_with_context = torch.cat([obs_tensor.unsqueeze(0), context_vector.cpu()], dim=-1)
            policy_obs = augment_policy_observation(obs_with_context, last_action, use_last_action)

            with torch.no_grad():
                action = actor.get_action(policy_obs.to(device)).cpu().numpy().squeeze()

            obs, _, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
            last_action = torch.tensor(action, dtype=torch.float32).reshape(1, -1)
            if done:
                last_action.zero_()

    env.close()
    return np.array(context_vectors)  # (N_steps, context_dim)


# ---------------------------------------------------------------------------
# t-SNE plotting
# ---------------------------------------------------------------------------

def _plot_tsne(id_cvs, ood_cvs, out_dir, seed, perplexity=30.0, subsample=2000):

    rng = np.random.RandomState(seed)

    all_cvs = np.concatenate([id_cvs, ood_cvs], axis=0)
    all_labels = np.array([0] * len(id_cvs) + [1] * len(ood_cvs))

    if len(all_cvs) > subsample:
        idx = rng.choice(len(all_cvs), size=subsample, replace=False)
        all_cvs = all_cvs[idx]
        all_labels = all_labels[idx]
        print(f"  Subsampled to {subsample} points for t-SNE")

    print(f"  Running t-SNE on {len(all_cvs)} points (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, max_iter=1000)
    embedded = tsne.fit_transform(all_cvs)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {0: "#2196F3", 1: "#F44336"}
    labels_str = {0: "In-distribution", 1: "Out-of-distribution"}
    for label in [0, 1]:
        mask = all_labels == label
        ax.scatter(
            embedded[mask, 0], embedded[mask, 1],
            c=colors[label], label=labels_str[label],
            alpha=0.5, s=10, linewidths=0,
        )
    ax.set_title("t-SNE: context vectors — ID vs OOD dynamics")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(handles=[
        mpatches.Patch(color=colors[k], label=labels_str[k]) for k in [0, 1]
    ])
    plt.tight_layout()
    path = os.path.join(out_dir, "tsne_id_vs_ood.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return embedded, all_labels


# ---------------------------------------------------------------------------
# Main entry point called from td3_training._entrypoint()
# ---------------------------------------------------------------------------

def context_vector_analysis(
    actor,
    transformer,
    air_hockey_base: dict,
    raw_obs_dim: int,
    act_dim: int,
    context_len: int,
    context_vector_dim: int,
    use_last_action: bool,
    n_eps: int = 20,
    n_envs: int = 10,
    ood_scale: float = 1.0,
    ood_gap: float = 0.5,
    out_dir: str = "results/context_tsne",
    device: str = "cpu",
    seed: int = 42,
):
    """Collect ID and OOD context vectors, run t-SNE, save plots + raw arrays.

    Args:
        actor:           Loaded DeterministicAgent (already on device, eval mode).
        transformer:     Loaded ContextEncoder (already on device, eval mode).
        air_hockey_base: The air_hockey sub-dict from the sim config yaml
                         (must contain random_variables + random_variable_ranges).
        raw_obs_dim:     Raw env observation dim (e.g. 30).
        act_dim:         Action dim (e.g. 2).
        context_len:     Transformer context window length.
        context_vector_dim: Transformer output dim.
        use_last_action: Whether policy obs includes last action.
        n_eps:           Episodes per sampled dynamics config.
        n_envs:          Number of distinct dynamics configs to sample per condition.
        ood_scale:       How far outside ID range to sample OOD (multiplier on half-width).
        out_dir:         Directory to write plots + .npy arrays.
        device:          torch device string.
        seed:            RNG seed for reproducibility.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    random_variables = list(air_hockey_base.get("random_variables", []))
    random_variable_ranges = dict(air_hockey_base.get("random_variable_ranges", {}))

    if not random_variables:
        raise ValueError(
            "air_hockey config has no random_variables — cannot define ID/OOD ranges. "
            "Make sure you're using a paramrand config (e.g. sim_paramrand_pm25.yaml)."
        )

    print(f"\n[context_vector_analysis] Starting analysis")
    print(f"  ID ranges:  {random_variable_ranges}")
    print(f"  OOD scale:  {ood_scale}x (samples {ood_scale}x the half-width outside ID bounds)")
    print(f"  n_envs={n_envs}, n_eps_per_env={n_eps}, out_dir={out_dir}\n")

    actor.eval()
    transformer.eval()

    id_cvs_list = []
    id_param_list = []   # (n_envs * n_steps_per_env, n_params) for per-param coloring
    ood_cvs_list = []

    for env_i in range(n_envs):
        # --- ID ---
        id_params = _sample_id_params(random_variable_ranges, random_variables, rng)
        id_cfg = _build_env_config(air_hockey_base, id_params)
        print(f"  [ID  {env_i+1}/{n_envs}] {id_params}")
        cvs = _collect_context_vectors_for_params(
            air_hockey_params=id_cfg,
            actor=actor,
            transformer=transformer,
            raw_obs_dim=raw_obs_dim,
            act_dim=act_dim,
            context_len=context_len,
            use_last_action=use_last_action,
            n_eps=n_eps,
            device=device,
            rng=rng,
        )
        id_cvs_list.append(cvs)
        param_vals = [id_params[k] for k in sorted(id_params.keys())]
        id_param_list.extend([param_vals] * len(cvs))

        # --- OOD ---
        ood_params = _sample_ood_params(random_variable_ranges, random_variables, rng, ood_scale, ood_gap)
        ood_cfg = _build_env_config(air_hockey_base, ood_params)
        print(f"  [OOD {env_i+1}/{n_envs}] {ood_params}")
        cvs = _collect_context_vectors_for_params(
            air_hockey_params=ood_cfg,
            actor=actor,
            transformer=transformer,
            raw_obs_dim=raw_obs_dim,
            act_dim=act_dim,
            context_len=context_len,
            use_last_action=use_last_action,
            n_eps=n_eps,
            device=device,
            rng=rng,
        )
        ood_cvs_list.append(cvs)

    id_cvs = np.concatenate(id_cvs_list, axis=0)
    ood_cvs = np.concatenate(ood_cvs_list, axis=0)
    print(f"\n  Collected {len(id_cvs)} ID and {len(ood_cvs)} OOD context vectors")

    # Save raw arrays
    np.save(os.path.join(out_dir, "id_context_vectors.npy"), id_cvs)
    np.save(os.path.join(out_dir, "ood_context_vectors.npy"), ood_cvs)
    print(f"  Saved raw arrays to {out_dir}/")

    # t-SNE: ID vs OOD
    _plot_tsne(id_cvs, ood_cvs, out_dir=out_dir, seed=seed)

    print(f"\n[context_vector_analysis] Done. Results in {out_dir}/\n")


if __name__ == "__main__":
    context_vector_analysis()
