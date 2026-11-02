"""Publish descriptive raw juggling and physical-motion metrics from saved runs.

Reproduce: .venv/bin/python -m scripts.analysis.juggle_smooth_sweep
No policy loading, environments, training, acceptance gates or simulator changes.
"""
from pathlib import Path
import csv
from datetime import datetime, timezone
import json
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.analysis.juggle_motion_metrics import analyze

MOTION = {
    'median_episode_p95_speed_m_s': 'speed_m_s',
    'median_episode_p95_acceleration_m_s2': 'acceleration_m_s2',
    'median_episode_p95_jerk_m_s3': 'jerk_m_s3',
}


def motion_summary(result_path):
    """Cache read-only position diagnostics beside the original ignored results."""
    path = result_path.parent/'motion_metrics.json'
    if not path.exists():
        analyze(result_path)
    data = json.loads(path.read_text())
    rows = [r for r in data['episodes'] if r['scenario'] == 'training']
    for row in rows:
        for metric, order in [('speed_m_s', 1), ('acceleration_m_s2', 2), ('jerk_m_s3', 3)]:
            if row['whole_episode'][metric]['samples'] != max(0, row['steps']-order):
                raise ValueError('Motion sample count does not match episode steps')
    output = {}
    for field, metric in MOTION.items():
        values = [r['whole_episode'][metric]['p95'] for r in rows if r['whole_episode'][metric]['p95'] is not None]
        output[field] = float(np.median(values)) if values else None
    return output, rows


def collect(result_path, run, phase, step, weights, episode_rows):
    payload = json.loads(result_path.read_text())
    summary = payload['scenarios']['training']['summary']
    if phase.startswith('32_reused_starts') and (payload['episodes_per_scenario'],payload['seed_base'],payload['horizon']) != (32,700000,20000):
        raise ValueError('Unexpected fixed-comparison protocol')
    motion, details = motion_summary(result_path)
    indexed = {r['seed']: r for r in details}
    for ep in payload['scenarios']['training']['episodes']:
        m = indexed[ep['seed']]
        episode_rows.append(dict(run=run, phase=phase, step=step, weights=weights, seed=ep['seed'],
            steps=ep['steps'], physical_juggles=ep['true_state_juggles'], sensed_juggles=ep['juggles'], reward=ep['return'],
            position_p95_speed_m_s=m['whole_episode']['speed_m_s']['p95'],
            position_p95_acceleration_m_s2=m['whole_episode']['acceleration_m_s2']['p95'],
            position_p95_jerk_m_s3=m['whole_episode']['jerk_m_s3']['p95'], source=str(result_path),
            trajectory=str(result_path.parent/ep['trajectory'])))
    return dict(run=run, phase=phase, step=step, weights=weights,
        episodes=payload['episodes_per_scenario'], seed_base=payload['seed_base'], horizon=payload['horizon'],
        physical_juggles=summary['mean_true_state_juggles'], sensed_juggles=summary['mean_juggles'],
        episode_steps=summary['restricted_mean_steps'], reward=summary['mean_return'],
        action_delta_rms=summary['mean_action_delta_rms'], **motion, source=str(result_path))


def main():
    report = Path('docs/juggling-smooth-sweep.md')
    launches = []
    for filename in ['docs/juggling-smooth-sweep-launch.json', 'docs/juggling-smooth-longdiscount-launch.json']:
        if Path(filename).exists():
            launches.extend(json.loads(Path(filename).read_text())['launches'])
    rows, statuses, curves, episode_rows = [], [], [], []
    run_discounts = {}
    for run_index, run in enumerate(launches):
        cfg = yaml.safe_load(Path(run['config']).read_text())
        run_discounts[run['name']] = float(cfg['gamma'])
        label = f"λ={cfg['actor_action_delta_penalty']:g}, γ={cfg['gamma']:g}, seed {cfg['seed']}"
        root = Path(run['root'])
        status_path = root/'status.json'
        status = json.loads(status_path.read_text())['status'] if status_path.exists() else 'submitted'
        statuses.append((run['name'], run['job_id'], status))
        results = sorted((root/'train').glob('checkpoint_*/survival_eval/results.json'), key=lambda p:int(p.parents[1].name.split('_')[-1]))
        if (root/'train/survival_eval/results.json').exists():
            results.append(root/'train/survival_eval/results.json')
        points = []
        for path in results:
            step = 6000000 if path.parents[1].name == 'train' else int(path.parents[1].name.split('_')[-1])
            row = collect(path, run['name'], 'checkpoint_validation', step, 'online', episode_rows)
            if (row['episodes'], row['seed_base'], row['horizon']) != (8, 10000, 5000):
                raise ValueError('Unexpected checkpoint protocol; cannot combine curves')
            rows.append(row); points.append(row)
        curves.append((label, points, dict(color=f'C{run_index}', marker='.')))
        for path in sorted(root.glob('paired_*/results.json')):
            parts = path.parent.name.split('_')
            rows.append(collect(path, run['name'], '32_reused_starts', int(parts[1]), parts[2], episode_rows))
    controls = [
        ('noisy_history_20261029_seed0', .995, 'juggle_history_baseline_1m_20261101', 'black', '--'),
        ('noisy_history_long_discount_20261029_seed0', .999, 'juggle_history_longdiscount_baseline_1m_20261102', 'dimgray', ':'),
    ]
    for run_name, discount, archive_name, color, linestyle in controls:
        archived = Path('runs/td3')/archive_name
        for role in ('online','target'):
            path = archived/f'paired_{role}_20k_eval/results.json'
            if path.exists():
                rows.append(collect(path, run_name, '32_reused_starts_historical', 1000000, role, episode_rows))
        # Each control stops at the newest checkpoint of its own discount group.
        limit = max((r['step'] for r in rows if r['phase']=='checkpoint_validation' and run_discounts[r['run']]==discount), default=0)
        baseline = Path('runs/td3')/run_name/'train'
        points = []
        paths = sorted(baseline.glob('checkpoint_*/survival_eval/results.json'), key=lambda p:int(p.parents[1].name.split('_')[-1]))
        if limit>=6000000 and (baseline/'survival_eval/results.json').exists():
            paths.append(baseline/'survival_eval/results.json')
        for path in paths:
            step = 6000000 if path.parents[1].name=='train' else int(path.parents[1].name.split('_')[-1])
            if step>limit:continue
            row=collect(path,run_name,'historical_checkpoint_validation',step,'online',episode_rows)
            if (row['episodes'],row['seed_base'],row['horizon'])!=(8,10000,5000):
                raise ValueError('Historical baseline does not match checkpoint protocol')
            rows.append(row);points.append(row)
        curves.append((f'Historical λ=0, γ={discount:g}, seed 0',points,dict(color=color,linestyle=linestyle)))
    panels = [
        (report.with_suffix('.png'), (2, 2), ['physical_juggles','episode_steps','reward','action_delta_rms'],
         ['Mean physical juggles','Mean episode steps','Mean episode reward','Mean action-change RMS'],
         'Fresh smoothness sweep: unsmoothed checkpoint means'),
        (report.with_name('juggling-smooth-sweep-motion.png'), (1, 3), list(MOTION),
         ['Median episode p95 displacement speed (m/s)','Median episode p95 acceleration (m/s²)','Median episode p95 jerk (m/s³)'],
         'Sampled physical paddle motion from true positions; whole episodes'),
    ]
    for figure, shape, fields, labels, title in panels:
        fig, axes = plt.subplots(*shape, figsize=(12, 7) if shape == (2, 2) else (15, 4), constrained_layout=True)
        for label, points, style in curves:
            for ax, field, ylabel in zip(axes.flat, fields, labels):
                available = [r for r in points if r[field] is not None]
                if available:
                    ax.plot([r['step']/1e6 for r in available], [r[field] for r in available], label=label, **style)
                ax.set(xlabel='Training steps (millions)', ylabel=ylabel); ax.grid(alpha=.25)
        if rows:
            axes.flat[0].legend(fontsize=6)
        fig.suptitle(title+'\n8 reused starts per point, 5k recording limit; different episode lengths and visited states')
        if rows:
            fig.savefig(figure,dpi=160)
        plt.close(fig)
    table = report.with_suffix('.csv')
    fields=['run','phase','step','weights','episodes','seed_base','horizon','physical_juggles','sensed_juggles','episode_steps','reward','action_delta_rms',*MOTION,'source']
    with table.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    episode_table = report.with_name('juggling-smooth-sweep-episodes.csv')
    if episode_rows:
        with episode_table.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(episode_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(episode_rows)
    lines=['# Smoothness sweep progress','', 'Generated '+datetime.now(timezone.utc).isoformat()+' from saved artifacts; no new rollouts.', '',
        'Original sweep: each coefficient (0.0005, 0.002, 0.008) has two fresh training seeds at gamma 0.995. An additional pair uses coefficient 0.0005 and gamma 0.999. Original checkpoints are every 100k steps; the additional pair saves every 500k for storage. All budgets are 6M. Simulator, sensing, control and DR remain unchanged. GPU ceiling: 16 total user allocations, including other jobs. Training seeds remain separate curves; starts within one run are not training-seed replications.', '',
        '| Run | Slurm job | Saved runner status |','|---|---|---|']
    lines += [f'| {n} | {j} | {s} |' for n,j,s in statuses]
    if rows:
        lines += ['', '![Checkpoint juggling, steps, reward and commands](juggling-smooth-sweep.png)', '',
                  '![Sampled physical speed, acceleration and jerk](juggling-smooth-sweep-motion.png)']
    else:
        lines += ['', 'Checkpoint figures are unavailable until the first saved evaluation.']
    lines += ['',f'[Raw summary CSV]({table.name}); [per-episode metrics]({episode_table.name}); [original launch provenance](juggling-smooth-sweep-launch.json); [longer-discount pair](juggling-smooth-longdiscount-launch.json); [matched longer-discount baseline](juggling-longdiscount-baseline-1m-launch.json).', '',
        'Curves use saved 8-start 5k online-actor validation only, including historical unregularized seed-0 controls at gamma 0.995 and 0.999 on exactly the same seeds/horizon. Each control is shown only through the latest checkpoint in its own discount group. The CSV separately labels historical checkpoints, 32-start 20k online/target tests, and the archived unregularized 1M comparisons at both discounts; do not mix protocols. Archived 1M comparisons use the same 32 starts and are not plotted as 8-start curves. No interpolation supplies missing checkpoints. Latest points can be at different training budgets; compare the same x-coordinate, not each curve’s last point. Physical juggles use true simulated trajectories; reward is the environment episode sum. Recording limits truncate counts and durations.', '',
        'Motion curves show the median across episodes of each episode’s p95 speed, acceleration or jerk, using finite differences of true paddle positions at the control rate. These are whole-episode distributions, with different lengths and visited states. They exclude the reset-position sample and do not capture all substep spikes, joint torques or UR5 controller behavior. Position-difference displacement speed is not cached body velocity; solver/workspace position corrections can contribute, so do not equate these estimates to the simulator velocity cap. They are not robot-safety thresholds. Per-episode diagnostics remain beside the original local results in `motion_metrics.json`. Action-change RMS remains a separate command diagnostic, not measured motor energy.', '',
        'Curves are descriptive repeated-validation results. Independent longer tests and qualitative review follow candidate comparison; no binary acceptance gate is used.', '',
        'Measurement clarification: [actual PID-target replay audit](../notes/scratch/experiments/2026-11-01_22-59_pid-target-replay-audit.md). Legacy `target_delta_rms_m` is a scaled displacement-change proxy, not direct PID-target telemetry.']
    report.write_text('\n'.join(lines)+'\n')
    print('Saved summary rows:',len(rows))


if __name__ == '__main__':
    main()
