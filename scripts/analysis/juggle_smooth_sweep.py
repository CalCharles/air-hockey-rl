"""Publish descriptive raw metrics from the active smoothness sweep.

Reproduce: .venv/bin/python -m scripts.analysis.juggle_smooth_sweep
Never runs training, environments or acceptance gates.
"""
from pathlib import Path
import csv
from datetime import datetime, timezone
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    report = Path('docs/juggling-smooth-sweep.md')
    launch = json.loads(Path('docs/juggling-smooth-sweep-launch.json').read_text())
    rows, statuses = [], []
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for run_index, run in enumerate(launch['launches']):
        root = Path(run['root'])
        status_path = root/'status.json'
        status = json.loads(status_path.read_text())['status'] if status_path.exists() else 'submitted'
        statuses.append((run['name'], run['job_id'], status))
        results = sorted((root/'train').glob('checkpoint_*/survival_eval/results.json'), key=lambda p:int(p.parents[1].name.split('_')[-1]))
        if (root/'train/survival_eval/results.json').exists():
            results.append(root/'train/survival_eval/results.json')
        points = []
        for p in results:
            payload = json.loads(p.read_text())
            summary = payload['scenarios']['training']['summary']
            step = 6000000 if p.parents[1].name == 'train' else int(p.parents[1].name.split('_')[-1])
            points.append((step, summary))
            rows.append(dict(run=run['name'], phase='checkpoint_validation', step=step, weights='online',
                episodes=payload['episodes_per_scenario'], horizon=payload['horizon'],
                physical_juggles=summary['mean_true_state_juggles'], sensed_juggles=summary['mean_juggles'],
                episode_steps=summary['restricted_mean_steps'], reward=summary['mean_return'],
                action_delta_rms=summary['mean_action_delta_rms'], source=str(p)))
        for ax, key, ylabel in zip(axes.flat,
                ['mean_true_state_juggles','restricted_mean_steps','mean_return','mean_action_delta_rms'],
                ['Mean physical juggles','Mean episode steps','Mean episode reward','Mean action-change RMS']):
            if points:
                ax.plot([s/1e6 for s,_ in points], [v[key] for _,v in points], marker='.', color=f'C{run_index}', label=f"λ={ {'0005':'0.0005','002':'0.002','008':'0.008'}[run['name'].split('_')[3]]}, seed {run['manifest']['seed']}")
            ax.set(xlabel='Training steps (millions)', ylabel=ylabel)
            ax.grid(alpha=.25)
        for p in sorted(root.glob('paired_*/results.json')):
            payload = json.loads(p.read_text()); summary = payload['scenarios']['training']['summary']
            parts = p.parent.name.split('_')
            rows.append(dict(run=run['name'], phase='32_reused_starts', step=int(parts[1]), weights=parts[2],
                episodes=payload['episodes_per_scenario'], horizon=payload['horizon'],
                physical_juggles=summary['mean_true_state_juggles'], sensed_juggles=summary['mean_juggles'],
                episode_steps=summary['restricted_mean_steps'], reward=summary['mean_return'],
                action_delta_rms=summary['mean_action_delta_rms'], source=str(p)))
    figure = report.with_suffix('.png')
    if rows:
        axes[0,0].legend(fontsize=8)
    fig.suptitle('Fresh smoothness sweep: unsmoothed checkpoint means\n8 reused starts per point, 5k recording limit; no binary gates')
    if rows:
        fig.savefig(figure,dpi=160)
    plt.close(fig)
    table = report.with_suffix('.csv')
    fields=['run','phase','step','weights','episodes','horizon','physical_juggles','sensed_juggles','episode_steps','reward','action_delta_rms','source']
    with table.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    lines=['# Smoothness sweep progress','', 'Generated '+datetime.now(timezone.utc).isoformat()+' from saved artifacts; no new rollouts.', '',
        'Each coefficient (0.0005, 0.002, 0.008) has two independent fresh training seeds. Simulator, sensing, control and DR remain unchanged. GPU ceiling: 16 total user allocations, including other jobs. Training seeds remain separate curves; starts within one run are not training-seed replications.', '',
        '| Run | Slurm job | Saved runner status |','|---|---|---|']
    lines += [f'| {n} | {j} | {s} |' for n,j,s in statuses]
    lines += ['',f'![Checkpoint metrics]({figure.name})' if rows else 'Checkpoint figures are unavailable until the first saved evaluation.', '',f'[Raw summary CSV]({table.name}); [launch provenance](juggling-smooth-sweep-launch.json).', '',
        'Plots use saved 8-start 5k online-actor validation only. The CSV separately labels 32-start 20k online/target tests; do not mix these protocols. Physical juggles use true simulated trajectories; reward is the environment episode sum. Recording limits truncate counts and durations. Action-change RMS is a command diagnostic, not measured energy or UR5 safety. Physical speed/acceleration/jerk distributions are stored in each matched test’s local `motion_metrics.json`, with first-250-step and whole-episode estimates. Short episodes still have different exposure and visited states.', '',
        'No new policy-quality result yet.' if not rows else 'Curves are descriptive repeated-validation results. Independent longer tests and qualitative review follow candidate comparison; no binary acceptance gate is used.']
    report.write_text('\n'.join(lines)+'\n')
    print('Saved summary rows:',len(rows))


if __name__ == '__main__':
    main()
