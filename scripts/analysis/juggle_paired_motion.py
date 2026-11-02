"""Compare saved paddle motion over equal per-start prefixes; no rollouts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from scripts.analysis.juggle_motion_metrics import sampled_motion


def compare(baseline: Path, candidate: Path, prefix_steps: int = 250):
    if prefix_steps < 4:
        raise ValueError('At least four positions are required for jerk')
    sources = [Path(baseline), Path(candidate)]
    payloads = [json.loads(p.read_text()) for p in sources]
    if any(p['horizon'] != payloads[0]['horizon'] for p in payloads):
        raise ValueError('Recording horizons differ')
    scenarios = [p['scenarios'] for p in payloads]
    if scenarios[0].keys() != scenarios[1].keys():
        raise ValueError('Scenario sets differ')
    rows = []
    for scenario in scenarios[0]:
        groups = [s[scenario] for s in scenarios]
        # Match the effective environment, including sensing and randomization.
        if groups[0]['config'] != groups[1]['config']:
            raise ValueError('Effective simulator configurations differ')
        rate = float(groups[0]['config']['simulator_params'].get('step_frequency', 20))
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError('Invalid control rate')
        indexed = []
        for group in groups:
            by_seed = {r['seed']: r for r in group['episodes']}
            if len(by_seed) != len(group['episodes']):
                raise ValueError('Duplicate reset seeds')
            indexed.append(by_seed)
        if indexed[0].keys() != indexed[1].keys() or not indexed[0]:
            raise ValueError('Nonempty identical reset seed sets are required')
        for seed in sorted(indexed[0]):
            episode = [g[seed] for g in indexed]
            positions = []
            for source, record in zip(sources, episode):
                with np.load(source.parent / record['trajectory']) as data:
                    xy = np.asarray(data['true_paddle_xy'], dtype=np.float64)
                if xy.shape != (record['steps'], 2) or not np.isfinite(xy).all():
                    raise ValueError('Trajectory shape/values disagree with episode record')
                positions.append(xy)
            count = min(prefix_steps, *(len(x) for x in positions))
            row = dict(scenario=scenario, seed=seed, paired_steps=count)
            for label, xy, record in zip(('baseline', 'candidate'), positions, episode):
                source = sources[0 if label == 'baseline' else 1].parent / record['trajectory']
                row[label] = dict(episode_steps=record['steps'],
                    physical_juggles=record['true_state_juggles'],
                    sensed_juggles=record['juggles'],
                    trajectory=str(source.resolve()),
                    trajectory_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    motion=sampled_motion(xy[:count], 1/rate))
            rows.append(row)
    summary = {}
    for scenario in scenarios[0]:
        paired = [r for r in rows if r['scenario'] == scenario]
        summary[scenario] = dict(episodes=len(paired), metrics={})
        for metric in ('speed_m_s', 'acceleration_m_s2', 'jerk_m_s3'):
            valid = [r for r in paired if all(r[l]['motion'][metric]['p95'] is not None
                                            for l in ('baseline', 'candidate'))]
            ratios = [r['candidate']['motion'][metric]['p95'] / r['baseline']['motion'][metric]['p95']
                      for r in valid if r['baseline']['motion'][metric]['p95'] > 0]
            item = dict(paired_episodes=len(valid), ratio_episodes=len(ratios),
                        median_paired_ratio=statistics.median(ratios) if ratios else None)
            for label in ('baseline', 'candidate'):
                item[label+'_median_episode_p95'] = statistics.median(
                    r[label]['motion'][metric]['p95'] for r in valid) if valid else None
            summary[scenario]['metrics'][metric] = item
    return dict(method='Each reset seed uses the first min(baseline steps, candidate steps, requested prefix) positions; no reset-position sample.',
                limitations='Paired reset seeds and equal durations do not ensure equal visited states. Control-rate position differences omit substep spikes, joint torque and hardware stop behavior. Short cases lack higher derivatives; zero baseline denominators are excluded from ratios.',
                requested_prefix_steps=prefix_steps,
                sources={label: dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                         for label, p in zip(('baseline', 'candidate'), sources)},
                analysis_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__), Path(__file__).with_name('juggle_motion_metrics.py'))},
                summary=summary, episodes=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--prefix-steps', default=250, type=int)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate, args.prefix_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
