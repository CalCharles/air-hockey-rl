"""Plot existing history-trial metrics; never load policies or run environments.

Reproduce: .venv/bin/python -m scripts.analysis.juggle_training_plots
"""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = {
    'noisy_history_20261029_seed0': 'History seed 0',
    'noisy_history_20261029_seed1': 'History seed 1',
    'noisy_history_long_discount_20261029_seed0': 'Longer discount seed 0',
    'noisy_history_low_actor_lr_20261029_seed0': 'Lower actor LR seed 0',
}
OUT = Path('docs/assets/juggling')

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evaluation, training, provenance = [], [], {}
    for run, label in RUNS.items():
        root = Path('runs/td3') / run / 'train'
        sources = [(int(p.parents[1].name.split('_')[-1]), p) for p in root.glob('checkpoint_*/survival_eval/results.json')]
        sources.append((6000000, root / 'survival_eval/results.json'))
        for step, path in sorted(sources):
            payload = json.loads(path.read_text())
            provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            for episode in payload['scenarios']['training']['episodes']:
                evaluation.append(dict(run=run, training_step=step, seed=episode['seed'],
                    horizon=payload['horizon'], steps=episode['steps'], reward=episode['return'],
                    sensed_juggles=episode['juggles'], physical_juggles=episode['true_state_juggles'], source=str(path)))
        accumulator = EventAccumulator(str(root), size_guidance={'scalars': 0})
        accumulator.Reload()
        rewards = accumulator.Scalars('charts/episodic_return')
        lengths = accumulator.Scalars('charts/episodic_length')
        assert len(rewards) == len(lengths)
        for reward, length in zip(rewards, lengths):
            assert reward.step == length.step
            training.append(dict(run=run, training_step=reward.step, reward=reward.value, steps=length.value))
        for event in root.glob('events.out.tfevents.*'):
            provenance[str(event)] = hashlib.sha256(event.read_bytes()).hexdigest()
    write_csv(OUT / 'history-checkpoint-episodes.csv', evaluation)
    write_csv(OUT / 'history-training-episodes.csv', training)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for run, label in RUNS.items():
        rows = [r for r in evaluation if r['run'] == run]
        steps = sorted({r['training_step'] for r in rows})
        for ax, metric, title in zip(axes.flat, ['physical_juggles', 'sensed_juggles', 'steps', 'reward'],
                ['Physical juggles per episode', 'Sensed juggles per episode', 'Episode length (steps)', 'Episode reward (sum)']):
            means = [np.mean([r[metric] for r in rows if r['training_step'] == s]) for s in steps]
            ax.plot(np.array(steps)/1e6, means, marker='.', markersize=3, label=label)
            ax.set(xlabel='Training environment steps (millions)', ylabel=title)
            ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Saved checkpoint evaluations: 8 reused starts per point, 5,000-step limit\nUnsmoothed episode means; online actor weights')
    fig.savefig(OUT / 'history-checkpoint-curves.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for run, label in RUNS.items():
        rows = [r for r in training if r['run'] == run]
        for ax, metric, title in zip(axes, ['reward', 'steps'], ['Training episode reward (sum)', 'Training episode length (steps)']):
            x, y = [], []
            for start in range(0, 6000000, 100000):
                values = [r[metric] for r in rows if start <= r['training_step'] < start + 100000]
                if values:
                    x.append((start + 50000)/1e6)
                    y.append(np.mean(values))
            ax.plot(x, y, marker='.', markersize=3, label=label)
            ax.set(xlabel='Training environment steps (millions)', ylabel=title)
            ax.grid(alpha=.25)
    axes[0].legend(fontsize=9)
    fig.suptitle('Exploratory training episodes: means in 100,000-step bins\nEpisodes assigned by logged completion step; raw episode data in CSV')
    fig.savefig(OUT / 'history-training-reward-curves.png', dpi=160)
    plt.close(fig)
    (OUT / 'history-plot-provenance.json').write_text(json.dumps(dict(
        generated_utc=datetime.now(timezone.utc).isoformat(), runs=RUNS,
        evaluation_rows=len(evaluation), training_rows=len(training), source_sha256=provenance), indent=2)+'\n')
    for run in RUNS:
        rows = [r for r in evaluation if r['run']==run]
        final = [r for r in rows if r['training_step']==6000000]
        print(run, 'points', len({r['training_step'] for r in rows}), 'final n',len(final),
              {k: float(np.mean([r[k] for r in final])) for k in ['physical_juggles','sensed_juggles','steps','reward']})

if __name__ == '__main__':
    main()
