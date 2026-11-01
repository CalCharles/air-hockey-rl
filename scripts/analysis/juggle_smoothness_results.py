"""Export and plot existing smoothness-trial results without policy rollouts.

Reproduce: .venv/bin/python -m scripts.analysis.juggle_smoothness_results
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('docs/assets/juggling')
ROOT = Path('runs/td3/noisy_history_smooth_20261031_seed0')
BASE = Path('runs/td3/noisy_history_20261029_seed0')


def main():
    sources = [(f'smooth_{step}_{role}', ROOT / f'paired_{step}_{role}_20k_eval/results.json')
               for step in (1000000, 3000000, 6000000) for role in ('online', 'target')]
    sources += [('baseline_6000000_online', BASE / 'final_6m_paired_20k_eval/results.json'),
                ('baseline_6000000_target', BASE / 'target_actor_final_6m_paired_20k_eval/results.json'),
                ('smooth_final_10k', ROOT / 'final_eval/results.json')]
    summaries, episodes, hashes = [], [], {}
    keys = ['mean_true_state_juggles', 'mean_juggles', 'restricted_mean_steps', 'median_observed_steps',
            'mean_return', 'mean_action_delta_rms', 'mean_target_delta_rms_m',
            'worst_peak_paddle_speed_m_s', 'worst_peak_paddle_acceleration_m_s2']
    for label, path in sources:
        payload = json.loads(path.read_text())
        scenario = payload['scenarios']['training']
        rows = scenario['episodes']
        assert len(rows) == payload['episodes_per_scenario'] == 32
        for metric, field in [('mean_true_state_juggles', 'true_state_juggles'),
                              ('mean_juggles', 'juggles'), ('restricted_mean_steps', 'steps'),
                              ('mean_return', 'return'), ('mean_action_delta_rms', 'action_delta_rms'),
                              ('mean_target_delta_rms_m', 'target_delta_rms_m')]:
            assert abs(sum(r[field] for r in rows) / len(rows) - scenario['summary'][metric]) < 1e-8
        summaries.append(dict(policy=label, episodes=len(rows), horizon=payload['horizon'],
                              seed_base=payload['seed_base'], **{k: scenario['summary'][k] for k in keys}, source=str(path)))
        for row in rows:
            episodes.append(dict(policy=label, seed=row['seed'], steps=row['steps'],
                physical_juggles=row['true_state_juggles'], sensed_juggles=row['juggles'],
                reward=row['return'], action_delta_rms=row['action_delta_rms'],
                target_delta_rms_m=row['target_delta_rms_m'],
                peak_speed_m_s=row['peak_paddle_speed_m_s'],
                peak_acceleration_m_s2=row['peak_paddle_acceleration_m_s2'], source=str(path)))
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in [('smooth-results-summary', summaries), ('smooth-results-episodes', episodes)]:
        with (OUT / (name+'.csv')).open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            w.writeheader()
            w.writerows(rows)
    selected = [next(r for r in summaries if r['policy'] == name) for name in
                ['baseline_6000000_online', 'smooth_6000000_online',
                 'baseline_6000000_target', 'smooth_6000000_target']]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    labels = ['Baseline\nonline', 'Smooth\nonline', 'Baseline\ntarget', 'Smooth\ntarget']
    for ax, metric, title in zip(axes.flat,
            ['mean_true_state_juggles', 'restricted_mean_steps', 'mean_action_delta_rms', 'worst_peak_paddle_acceleration_m_s2'],
            ['Mean physical juggles per episode', 'Mean episode length (steps)',
             'Mean action-change RMS (normalized)', 'Largest recorded paddle acceleration (m/s²)']):
        values = [r[metric] for r in selected]
        ax.bar(labels, values, color=['#4c78a8', '#f58518', '#4c78a8', '#f58518'])
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:,.3f}' if metric == 'mean_action_delta_rms' else f'{v:,.2f}', ha='center', va='bottom', fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(values)*1.2)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Final 6M weights: baseline vs action-change penalty 0.05\nSame 32 reused starts; 20k recording limit; episode durations differ')
    fig.savefig(OUT/'smooth-final-comparison.png', dpi=160)
    plt.close(fig)
    (OUT/'smooth-results-provenance.json').write_text(json.dumps(dict(
        generated_utc=datetime.now(timezone.utc).isoformat(), source_sha256=hashes,
        summary_rows=len(summaries), episode_rows=len(episodes)), indent=2)+'\n')
    print(f'Validated/exported {len(episodes)} saved episodes and {len(summaries)} summaries; no new rollouts.')


if __name__ == '__main__':
    main()
