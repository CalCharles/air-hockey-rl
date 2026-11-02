"""Publish a completed fixed-policy long test from saved, verified trajectories."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--verified-trajectories', required=True, type=Path)
    parser.add_argument('--out-prefix', required=True, type=Path)
    parser.add_argument('--policy-label', help='Display label; defaults to the frozen bundle directory name')
    args = parser.parse_args()
    primary = json.loads(args.results.read_text())
    verified = json.loads(args.verified_trajectories.read_text())
    group = primary['scenarios']['training']
    if len(primary['scenarios']) != 1 or group['config']['simulator_params'].get('step_frequency', 20) != 20:
        raise ValueError('Expected one canonical training scenario at20Hz')
    episodes = group['episodes']
    by_seed = {e['seed']: e for e in verified['episodes']}
    count = primary['episodes_per_scenario']
    seeds = set(range(primary['seed_base'], primary['seed_base']+count))
    if (len(episodes) != count or len(by_seed) != len(verified['episodes']) or
            {e['seed'] for e in episodes} != seeds or set(by_seed) != seeds):
        raise ValueError('Incomplete or duplicate episode cohort')
    model = Path(primary['run_dir'])/'model.pth'
    policy_label = args.policy_label or model.parent.name
    if digest(model) != verified['model_sha256']:
        raise ValueError('Frozen policy hash changed')
    rows = []
    for e in sorted(episodes, key=lambda x: x['seed']):
        r = by_seed[e['seed']]
        if (e['steps'], e['true_state_juggles'], e['juggles']) != (r['steps'], r['physical_juggles'], r['sensed_juggles']):
            raise ValueError('Primary and trajectory-derived outcomes differ')
        if not np.isclose(e['return'], r['reward'], rtol=0, atol=1e-8):
            raise ValueError('Primary and trajectory-derived reward differ')
        if digest(r['trajectory']) != r['trajectory_sha256']:
            raise ValueError('Recorded trajectory changed')
        if (Path(r['trajectory']).resolve() != (args.results.parent/e['trajectory']).resolve() or
                not 1 <= e['steps'] <= primary['horizon']):
            raise ValueError('Wrong trajectory source or duration')
        row = {k: r[k] for k in ('seed', 'steps', 'physical_juggles', 'sensed_juggles', 'reward',
                                'action_delta_rms', 'terminated', 'truncated', 'trajectory', 'trajectory_sha256')}
        row['recorded_end_reasons'] = json.dumps(e['end_reasons'])
        for metric in ('speed_m_s', 'acceleration_m_s2', 'jerk_m_s3'):
            row['position_p95_'+metric] = r['position_motion'][metric]['p95']
        rows.append(row)
    now = datetime.now(timezone.utc)
    summary = {key: dict(mean=statistics.mean(r[key] for r in rows),
                         quantiles=dict(zip(('min', 'p25', 'median', 'p75', 'max'),
                                            np.percentile([r[key] for r in rows], [0,25,50,75,100]).tolist())))
               for key in ('steps', 'physical_juggles', 'sensed_juggles', 'reward', 'action_delta_rms')}
    for key in ('position_p95_speed_m_s', 'position_p95_acceleration_m_s2', 'position_p95_jerk_m_s3'):
        summary[key] = dict(median=statistics.median(r[key] for r in rows if r[key] is not None))
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_prefix.with_suffix('.csv')
    with csv_path.open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    output = dict(generated_utc=now.isoformat(), policy_label=policy_label, episodes=count, seed_base=primary['seed_base'],
                  recording_limit_steps=primary['horizon'], model_sha256=verified['model_sha256'],
                  sources={str(p.resolve()): digest(p) for p in (args.results, args.verified_trajectories, model, Path(__file__))},
                  published_csv_sha256=digest(csv_path), summary=summary,
                  limitations='One fixed training seed and policy; new reset starts. Recording limits censor counts and durations. Whole-episode motion uses20Hz position differences, excludes the reset sample, and involves unequal durations/visited states. No joint loads, substep spikes, infinite reliability or hardware-stop prediction.')
    args.out_prefix.with_suffix('.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    x = np.arange(count)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0,0].bar(x, [r['steps'] for r in rows], color='#4c78a8')
    axes[0,0].set_ylabel('Episode steps')
    axes[0,0].axhline(primary['horizon'], color='gray', linestyle=':', label='Recording limit')
    axes[0,0].legend(fontsize=8)
    axes[0,1].bar(x-.2, [r['physical_juggles'] for r in rows], width=.4, label='Physical')
    axes[0,1].bar(x+.2, [r['sensed_juggles'] for r in rows], width=.4, label='Sensed')
    axes[0,1].set_ylabel('Juggles'); axes[0,1].legend(fontsize=8)
    for ax, key, label in [(axes[1,0], 'position_p95_acceleration_m_s2', 'Episode p95 acceleration (m/s²)'),
                           (axes[1,1], 'position_p95_jerk_m_s3', 'Episode p95 jerk (m/s³)')]:
        ax.bar(x, [r[key] for r in rows], color='#54a24b'); ax.set_ylabel(label)
    for ax in axes.flat:
        ax.set_xlabel(f'Reset seed − {primary["seed_base"]:,}')
        ax.set_xticks(x[::4]); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle(f'{policy_label}: {count} new reset starts, {primary["horizon"]:,}-step recording limit\n'
                 f'Generated {now:%Y-%m-%d}; sampled position motion is not a robot-stop model')
    fig.savefig(args.out_prefix.with_suffix('.png'), dpi=160); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
