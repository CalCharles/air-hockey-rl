import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml


ORDERED_ENV_KEYS = [
    "paddle_density",
    "paddle_damping",
    "puck_density",
    "puck_damping",
    "force_scaling",
]


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class EnvEncoder(nn.Module):
    """Local copy to avoid ROS 'scripts' package import collisions."""

    def __init__(self, env_var_dim, latent_dim=8, hidden_size=(128, 128)):
        super().__init__()
        if env_var_dim <= 0:
            raise ValueError(f"env_var_dim must be positive, got {env_var_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if isinstance(hidden_size, int):
            hidden_dims = [hidden_size]
        else:
            hidden_dims = [int(x) for x in hidden_size]
        if len(hidden_dims) == 0:
            raise ValueError("EnvEncoder hidden_size must have at least one layer.")
        if any(h <= 0 for h in hidden_dims):
            raise ValueError(f"EnvEncoder hidden sizes must be positive, got {hidden_dims}")

        layers = []
        in_dim = env_var_dim
        for h in hidden_dims:
            layers.append(layer_init(nn.Linear(in_dim, h)))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(layer_init(nn.Linear(in_dim, latent_dim), std=1.0))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, env_vars):
        return self.net(env_vars)


def _load_yaml(path):
    with open(path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError:
            f.seek(0)
            return yaml.load(f, Loader=yaml.FullLoader)


def _extract_all_low_high_ranges(edge_eval_specs_path):
    raw = _load_yaml(edge_eval_specs_path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected list in edge_eval_specs.yaml, got {type(raw)}")

    all_low = None
    all_high = None
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "")
        if name == "all_low":
            all_low = entry
        elif name == "all_high":
            all_high = entry

    if all_low is None or all_high is None:
        raise ValueError(
            "Could not find both 'all_low' and 'all_high' entries in edge_eval_specs.yaml."
        )

    excluded = {"env_id", "name"}
    common_numeric_keys = []
    for key in sorted(set(all_low.keys()).intersection(set(all_high.keys()))):
        if key in excluded:
            continue
        try:
            float(all_low[key])
            float(all_high[key])
        except (TypeError, ValueError):
            continue
        common_numeric_keys.append(key)

    if len(common_numeric_keys) == 0:
        raise ValueError("No shared numeric keys found between all_low and all_high entries.")

    ranges = {}
    for key in common_numeric_keys:
        low = float(all_low[key])
        high = float(all_high[key])
        if high < low:
            low, high = high, low
        if np.isclose(high, low):
            high = low + 1e-6
        ranges[key] = (low, high)
    return ranges


def _sample_env_specs(num_samples, seed, env_ranges):
    rng = np.random.default_rng(seed)
    sampled = {}
    for key, (low, high) in env_ranges.items():
        sampled[key] = rng.uniform(low, high, size=num_samples).astype(np.float32)
    return sampled


def _normalize_env_vars(sampled_specs, env_ranges, env_var_dim):
    num_samples = len(next(iter(sampled_specs.values())))
    vec = np.zeros((num_samples, env_var_dim), dtype=np.float32)

    base = []
    for key in ORDERED_ENV_KEYS:
        if key not in sampled_specs or key not in env_ranges:
            raise ValueError(
                f"Missing key '{key}' in sampled specs or ranges. "
                "Expected keys include the canonical RMA parameters."
            )
        values = sampled_specs[key].astype(np.float32)
        low, high = env_ranges[key]
        mean = 0.5 * (low + high)
        std = (high - low) / np.sqrt(12.0)
        if std <= 1e-8:
            std = 1.0
        base.append(((values - mean) / std).reshape(-1, 1))

    base_arr = np.concatenate(base, axis=1).astype(np.float32)
    copy_len = min(env_var_dim, base_arr.shape[1])
    vec[:, :copy_len] = base_arr[:, :copy_len]
    return vec


def _run_encoder(encoder, env_vars, device, batch_size):
    latents = []
    with torch.no_grad():
        for i in range(0, env_vars.shape[0], batch_size):
            j = min(i + batch_size, env_vars.shape[0])
            batch = torch.from_numpy(env_vars[i:j]).to(device=device, dtype=torch.float32)
            z = encoder(batch).cpu().numpy()
            latents.append(z.astype(np.float32))
    return np.concatenate(latents, axis=0)


def _summarize_vector(v):
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "p05": float(np.percentile(v, 5)),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
    }


def _pairwise_distance_stats(latents, seed, subset_size):
    n = latents.shape[0]
    if n < 2:
        return {"num_pairs": 0, "mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    m = min(n, subset_size)
    rng = np.random.default_rng(seed + 1009)
    idx = rng.choice(n, size=m, replace=False)
    sample = latents[idx]
    diffs = sample[:, None, :] - sample[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    triu_i, triu_j = np.triu_indices(m, k=1)
    vals = dists[triu_i, triu_j]
    return {
        "num_pairs": int(vals.size),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "p50": float(np.percentile(vals, 50)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(np.max(vals)),
    }


def _analyze_latents(latents, seed):
    n, d = latents.shape
    centroid = np.mean(latents, axis=0)
    centered = latents - centroid.reshape(1, -1)
    norms = np.linalg.norm(latents, axis=1)
    centered_norms = np.linalg.norm(centered, axis=1)

    if n > 1:
        cov = np.cov(latents, rowvar=False, ddof=1)
    else:
        cov = np.zeros((d, d), dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)

    eig_sum = float(np.sum(eigvals))
    if eig_sum > 0:
        explained_ratio = eigvals / eig_sum
    else:
        explained_ratio = np.zeros_like(eigvals)
    explained_cum = np.cumsum(explained_ratio)

    p = explained_ratio[explained_ratio > 1e-12]
    effective_rank = float(np.exp(-np.sum(p * np.log(p)))) if p.size > 0 else 0.0
    participation_ratio = (
        float((np.sum(eigvals) ** 2) / np.sum(eigvals**2))
        if np.sum(eigvals**2) > 1e-12
        else 0.0
    )

    corr = np.corrcoef(latents, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    off_diag = np.abs(corr - np.eye(d, dtype=np.float64))
    off_diag_count = d * d - d
    off_diag_abs_mean = float(np.sum(off_diag) / max(1, off_diag_count))
    off_diag_abs_max = float(np.max(off_diag[np.triu_indices(d, k=1)])) if d > 1 else 0.0

    per_dim_mean = np.mean(latents, axis=0)
    per_dim_std = np.std(latents, axis=0)
    z = (latents - per_dim_mean.reshape(1, -1)) / np.maximum(per_dim_std.reshape(1, -1), 1e-8)
    skew = np.mean(z**3, axis=0)
    excess_kurtosis = np.mean(z**4, axis=0) - 3.0

    pca2 = centered @ eigvecs[:, : min(2, d)]
    if pca2.shape[1] < 2:
        pca2 = np.concatenate([pca2, np.zeros((n, 2 - pca2.shape[1]))], axis=1)

    return {
        "centroid": centroid.astype(np.float32),
        "covariance": cov.astype(np.float32),
        "correlation": corr.astype(np.float32),
        "pca_components": eigvecs.astype(np.float32),
        "pca_eigenvalues": eigvals.astype(np.float32),
        "pca_explained_ratio": explained_ratio.astype(np.float32),
        "pca_explained_cumulative": explained_cum.astype(np.float32),
        "pca_projection_2d": pca2.astype(np.float32),
        "summary": {
            "num_samples": int(n),
            "latent_dim": int(d),
            "norm_stats": _summarize_vector(norms),
            "centered_norm_stats": _summarize_vector(centered_norms),
            "effective_rank": effective_rank,
            "participation_ratio": participation_ratio,
            "offdiag_abs_corr_mean": off_diag_abs_mean,
            "offdiag_abs_corr_max": off_diag_abs_max,
            "skew_abs_mean": float(np.mean(np.abs(skew))),
            "kurtosis_abs_mean": float(np.mean(np.abs(excess_kurtosis))),
            "pairwise_distance_stats": _pairwise_distance_stats(
                latents=latents,
                seed=seed,
                subset_size=1500,
            ),
        },
    }


def _build_summary_payload(args, env_ranges, analysis, latents):
    centroid = analysis["centroid"]
    eigvals = analysis["pca_eigenvalues"]
    explained = analysis["pca_explained_ratio"]
    cum = analysis["pca_explained_cumulative"]

    dim_stats = []
    for i in range(latents.shape[1]):
        v = latents[:, i]
        dim_stats.append(
            {
                "dim": int(i),
                **_summarize_vector(v),
            }
        )

    payload = {
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "encoder_path": str(Path(args.encoder_path).resolve()),
        "edge_eval_specs_path": str(Path(args.edge_eval_specs_path).resolve()),
        "seed": int(args.seed),
        "num_samples": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "device": str(args.device),
        "env_ranges_from_all_low_high": {k: [float(v[0]), float(v[1])] for k, v in env_ranges.items()},
        "ordered_env_keys_for_normalization": list(ORDERED_ENV_KEYS),
        "env_var_dim": int(args.env_var_dim),
        "env_latent_dim": int(args.env_latent_dim),
        "latent_centroid": [float(x) for x in centroid.tolist()],
        "analysis_summary": analysis["summary"],
        "pca": {
            "eigenvalues": [float(x) for x in eigvals.tolist()],
            "explained_variance_ratio": [float(x) for x in explained.tolist()],
            "cumulative_explained_variance_ratio": [float(x) for x in cum.tolist()],
        },
        "per_dimension_stats": dim_stats,
    }
    return payload


def _resolve_paths(args):
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint_dir does not exist: {checkpoint_dir}")

    if args.encoder_path:
        encoder_path = Path(args.encoder_path).resolve()
    else:
        encoder_path = checkpoint_dir / "encoder.pth"
    if not encoder_path.exists():
        raise FileNotFoundError(f"encoder path does not exist: {encoder_path}")

    if args.edge_eval_specs_path:
        edge_specs_path = Path(args.edge_eval_specs_path).resolve()
    else:
        edge_specs_path = checkpoint_dir / "edge_eval_specs.yaml"
    if not edge_specs_path.exists():
        raise FileNotFoundError(f"edge_eval_specs path does not exist: {edge_specs_path}")

    if args.args_path:
        args_path = Path(args.args_path).resolve()
    else:
        args_path = checkpoint_dir / "args.yaml"
    if not args_path.exists():
        raise FileNotFoundError(f"args path does not exist: {args_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = checkpoint_dir / "latent_analysis"

    return checkpoint_dir, encoder_path, edge_specs_path, args_path, output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Sample random env specs from all_low/all_high ranges and analyze RMA encoder latents."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="ex_model/large_models/checkpoint_2500",
        help="Checkpoint directory containing encoder.pth/args.yaml/edge_eval_specs.yaml.",
    )
    parser.add_argument("--encoder-path", type=str, default=None, help="Optional explicit encoder.pth path.")
    parser.add_argument(
        "--edge-eval-specs-path",
        type=str,
        default=None,
        help="Optional explicit edge_eval_specs.yaml path.",
    )
    parser.add_argument("--args-path", type=str, default=None, help="Optional explicit args.yaml path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for analysis artifacts.")
    parser.add_argument("--num-samples", type=int, default=200000, help="Number of random env samples.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Encoder forward-pass batch size.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device for encoder inference.",
    )
    parser.add_argument(
        "--save-latents",
        action="store_true",
        help="If set, save full latent matrix to latent_vectors.npy (large file).",
    )
    args = parser.parse_args()

    checkpoint_dir, encoder_path, edge_specs_path, args_path, output_dir = _resolve_paths(args)
    os.makedirs(output_dir, exist_ok=True)
    args.encoder_path = str(encoder_path)
    args.edge_eval_specs_path = str(edge_specs_path)

    model_args = _load_yaml(args_path) or {}
    if not isinstance(model_args, dict):
        raise ValueError(f"Expected dict in args file: {args_path}")

    args.env_var_dim = int(model_args.get("env_var_dim", 8))
    args.env_latent_dim = int(model_args.get("env_latent_dim", 8))
    args.env_encoder_hidden_size = model_args.get("env_encoder_hidden_size", [128, 128])

    env_ranges = _extract_all_low_high_ranges(edge_specs_path)
    sampled_specs = _sample_env_specs(
        num_samples=args.num_samples,
        seed=args.seed,
        env_ranges=env_ranges,
    )
    env_vars = _normalize_env_vars(
        sampled_specs=sampled_specs,
        env_ranges=env_ranges,
        env_var_dim=args.env_var_dim,
    )

    device = torch.device(args.device)
    encoder = EnvEncoder(
        env_var_dim=args.env_var_dim,
        latent_dim=args.env_latent_dim,
        hidden_size=args.env_encoder_hidden_size,
    ).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()

    latents = _run_encoder(
        encoder=encoder,
        env_vars=env_vars,
        device=device,
        batch_size=args.batch_size,
    )
    analysis = _analyze_latents(latents=latents, seed=args.seed)
    summary_payload = _build_summary_payload(
        args=args,
        env_ranges=env_ranges,
        analysis=analysis,
        latents=latents,
    )

    summary_path = output_dir / "latent_distribution_summary.yaml"
    with open(summary_path, "w") as f:
        yaml.dump(summary_payload, f, sort_keys=False)

    np.save(output_dir / "latent_centroid.npy", analysis["centroid"])
    np.save(output_dir / "latent_covariance.npy", analysis["covariance"])
    np.save(output_dir / "latent_correlation.npy", analysis["correlation"])
    np.save(output_dir / "latent_pca_projection_2d.npy", analysis["pca_projection_2d"])
    np.save(output_dir / "latent_pca_components.npy", analysis["pca_components"])
    np.save(output_dir / "latent_pca_eigenvalues.npy", analysis["pca_eigenvalues"])
    np.save(output_dir / "latent_pca_explained_ratio.npy", analysis["pca_explained_ratio"])
    np.save(output_dir / "latent_pca_explained_cumulative.npy", analysis["pca_explained_cumulative"])
    if args.save_latents:
        np.save(output_dir / "latent_vectors.npy", latents)

    centroid_str = ", ".join(f"{x:.8f}" for x in analysis["centroid"].tolist())
    print("=== RMA Latent Distribution Analysis ===")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Encoder path:   {encoder_path}")
    print(f"Ranges source:  {edge_specs_path}")
    print(f"Samples:        {args.num_samples}")
    print(f"Latent dim:     {args.env_latent_dim}")
    print(f"Output dir:     {output_dir}")
    print("")
    print("Centroid vector (copy into rollout_new.py):")
    print(f"[{centroid_str}]")
    print("")
    print(f"Saved summary:  {summary_path}")


if __name__ == "__main__":
    main()
