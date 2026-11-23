"""Measure reward-inferred latent sampling error against frozen full replay.

Inputs: folders containing policy/, run.json, continuation.pt, source.json.
CPU only; does not mutate training runs. Sampling matches inference (replacement).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.fb.fb_training import Args, task_rewards
from scripts.fb.models import load_policy


@torch.no_grad()
def analyze(folder, sizes, repeats, seed):
    model = load_policy(folder / 'policy', device='cpu')
    args = Args(**json.loads((folder / 'run.json').read_text())['args'])
    state = torch.load(folder / 'continuation.pt', map_location='cpu', weights_only=True)
    batch = state['replay']
    obs = batch['next']['observation']
    tasks = ['juggle', 'touch', 'reach']
    weights = torch.cat([task_rewards(batch, t, args) for t in tasks], dim=1).double()
    embeddings = torch.cat([model.backward_map(x) for x in obs.split(8192)]).double()
    assert torch.isfinite(embeddings).all() and torch.isfinite(weights).all()
    # Sum or mean gives the same projected direction.
    reference = weights.T @ embeddings
    norms = reference.norm(dim=1)
    assert (norms > 0).all(), 'Full replay has undefined task direction'
    directions = reference / norms[:, None]
    # Check cached-embedding computation against the actual inference method.
    for t, task in enumerate(tasks):
        expected = model.reward_inference(obs[:1000], weights[:1000, t:t+1].float())
        actual = model.project_z((weights[:1000, t:t+1].T @ embeddings[:1000]).float())
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    rng = torch.Generator().manual_seed(seed)
    angles = torch.full((repeats, len(sizes), len(tasks)), float('nan'), dtype=torch.float64)
    raw_relative_error = torch.empty_like(angles)
    supports = torch.empty_like(angles)
    ess = torch.empty_like(angles)
    estimates = torch.zeros((repeats, len(sizes), len(tasks), embeddings.shape[1]), dtype=torch.float64)
    for rep in range(repeats):
        indices = torch.randint(len(obs), (max(sizes),), generator=rng)
        sums = torch.zeros_like(reference)
        weight_sum = torch.zeros(len(tasks), dtype=torch.float64)
        square_sum = weight_sum.clone()
        count = weight_sum.clone()
        previous = 0
        for j, size in enumerate(sizes):
            ids = indices[previous:size]
            w = weights[ids]
            sums += w.T @ embeddings[ids]
            weight_sum += w.sum(0)
            square_sum += w.square().sum(0)
            count += (w > 0).sum(0)
            norm = sums.norm(dim=1)
            valid = norm > 0
            direction = sums / norm.clamp_min(1e-300)[:, None]
            estimates[rep, j] = direction
            angles[rep, j, valid] = torch.rad2deg(torch.acos((direction[valid] * directions[valid]).sum(1).clamp(-1, 1)))
            raw_relative_error[rep, j] = ((sums / size - reference / len(obs)).norm(dim=1) / (norms / len(obs)))
            supports[rep, j] = count
            ess[rep, j] = weight_sum.square() / square_sum.clamp_min(1e-300)
            previous = size
        print(f'{folder.name}: repeat {rep+1}/{repeats}', flush=True)
    rows = []
    for j, size in enumerate(sizes):
        for t, task in enumerate(tasks):
            values = angles[:, j, t].numpy()
            valid = np.isfinite(values)
            dirs = estimates[valid, j, t]
            pair = torch.rad2deg(torch.acos((dirs @ dirs.T).clamp(-1, 1)))
            pair = pair[torch.triu(torch.ones_like(pair, dtype=torch.bool), diagonal=1)]
            rows.append(dict(model=folder.name, task=task, n=size,
                angle_mean_deg=float(np.nanmean(values)) if valid.any() else None,
                angle_std_deg=float(np.nanstd(values, ddof=1)) if valid.sum()>1 else None,
                angle_p95_deg=float(np.nanpercentile(values, 95)) if valid.any() else None,
                pairwise_angle_mean_deg=float(pair.mean()) if pair.numel() else None,
                undefined_repeats=int((~valid).sum()),
                relative_mean_error=float(raw_relative_error[:, j, t].mean()),
                positive_samples_mean=float(supports[:, j, t].mean()),
                effective_samples_mean=float(ess[:, j, t].mean())))
    torch.save(dict(angles=angles, estimates=estimates, reference_direction=directions,
                    raw_relative_error=raw_relative_error), folder/'sampling_results.pt')
    return rows, dict(size=len(obs), source=json.loads((folder/'source.json').read_text()),
                     positive_fraction={t:float((weights[:, i]>0).double().mean()) for i,t in enumerate(tasks)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--seed', type=int, default=41000)
    parser.add_argument('--threads', type=int, default=2)
    opts = parser.parse_args()
    torch.set_num_threads(opts.threads)
    sizes = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
    rows, sources = [], {}
    for variant in ['full', 'current']:
        result, source = analyze(opts.root/variant, sizes, opts.repeats, opts.seed)
        rows.extend(result)
        sources[variant] = source
    (opts.root/'results.json').write_text(json.dumps(dict(repeats=opts.repeats, seed=opts.seed,
        sampling='uniform with replacement; nested sizes within each repeat', sources=sources, rows=rows), indent=2))
    with (opts.root/'results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, task in zip(axes, ['juggle', 'touch', 'reach']):
        for variant in ['full', 'current']:
            data = [r for r in rows if r['model']==variant and r['task']==task]
            line, = ax.plot(sizes, [r['angle_mean_deg'] for r in data], marker='o', label=variant+' mean')
            ax.plot(sizes, [r['angle_p95_deg'] for r in data], linestyle='--', color=line.get_color(), label=variant+' 95th percentile')
        ax.axhline(5, color='gray', linewidth=1, linestyle=':')
        ax.set(xscale='log', yscale='log', title=task, xlabel='Sampled transitions (with replacement)')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Angle to full-buffer latent (degrees)')
    axes[-1].legend(fontsize=8)
    fig.suptitle(f'Fixed 8.75M checkpoints; 1M-state reference; {opts.repeats} repeats')
    fig.tight_layout()
    fig.savefig(opts.root/'convergence.png', dpi=180)
    fig.savefig(opts.root/'convergence.pdf')


if __name__ == '__main__':
    main()
