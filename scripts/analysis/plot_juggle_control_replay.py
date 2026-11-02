"""Publish equal-prefix cached PID-target/body-velocity replay diagnostics."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--out-prefix', required=True, type=Path)
    parser.add_argument('--title', default='Equal-prefix cached controller replay')
    args = parser.parse_args()
    sources = [args.baseline, args.candidate]
    payloads = [json.loads(p.read_text()) for p in sources]
    if any(payloads[i]['source_sha256'] != payloads[1-i]['paired_source_sha256'] for i in (0,1)):
        raise ValueError('Replay sources are not each other’s paired counterpart')
    indexed = [{(r['scenario'],r['seed']): r for r in d['episodes']} for d in payloads]
    if (not indexed[0] or indexed[0].keys() != indexed[1].keys() or
            any(len(d['episodes']) != len(g) for d,g in zip(payloads,indexed)) or
            len({key[0] for key in indexed[0]}) != 1):
        raise ValueError('Expected one nonempty matching scenario with unique reset seeds')
    keys = sorted(indexed[0])
    rows = []
    for key in keys:
        pair = [g[key] for g in indexed]
        if not all(r['exact_match'] for r in pair) or pair[0]['replayed_steps'] != pair[1]['replayed_steps']:
            raise ValueError('Replays must be exact and have equal per-start exposure')
        rows.extend(dict(policy=label, **row) for label,row in zip(('baseline','candidate'),pair))
    args.out_prefix.parent.mkdir(parents=True,exist_ok=True)
    csv_path = args.out_prefix.with_suffix('.csv')
    with csv_path.open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    specifications = [('pid_target_change_rms_m','mean','m'), ('pid_target_change_p99_m','median','m'),
        ('pid_target_change_maximum_m','max','m'), ('body_speed_maximum_m_s','max','m/s'),
        ('velocity_acceleration_p95_m_s2','median','m/s²'), ('velocity_acceleration_p99_m_s2','median','m/s²'),
        ('velocity_acceleration_maximum_m_s2','max','m/s²'), ('velocity_jerk_p95_m_s3','median','m/s³'),
        ('velocity_jerk_p99_m_s3','median','m/s³'), ('velocity_jerk_maximum_m_s3','max','m/s³')]
    summary=[]
    for metric,aggregate,unit in specifications:
        item=dict(metric=metric,aggregate=aggregate,unit=unit)
        for label,group in zip(('baseline','candidate'),indexed):
            values=[r[metric] for r in group.values() if r[metric] is not None]
            if not values or not np.isfinite(values).all():
                raise ValueError('Missing/nonfinite diagnostics for '+metric)
            item[label]={'mean':statistics.mean,'median':statistics.median,'max':max}[aggregate](values)
        item['candidate_baseline_ratio']=item['candidate']/item['baseline'] if item['baseline'] else None
        summary.append(item)
    summary_path=args.out_prefix.with_name(args.out_prefix.name+'-summary.csv')
    with summary_path.open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(summary[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(summary)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    metadata=dict(generated_utc=datetime.now(timezone.utc).isoformat(),episodes_per_policy=len(keys),
        replayed_steps_per_policy=sum(indexed[0][key]['replayed_steps'] for key in keys),
        paired_step_range=[min(indexed[0][key]['replayed_steps'] for key in keys),
                           max(indexed[0][key]['replayed_steps'] for key in keys)],
        sources={str(p.resolve()):digest(p) for p in sources},publisher_sha256=digest(__file__),
        published_csv_sha256={str(p):digest(p) for p in (csv_path,summary_path)},summary=summary,
        limitations='Equal-duration prefixes, different visited states. Final cached PID targets and20Hz body velocities omit complete substep traces and the reset-to-first-target edge; no UR5 joint/stop-risk model.')
    args.out_prefix.with_suffix('.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True);x=np.arange(len(keys))
    panels=[('pid_target_change_rms_m','PID-target change RMS (m)'),
            ('pid_target_change_maximum_m','Largest PID-target step in prefix (m)'),
            ('velocity_acceleration_p99_m_s2','Body-velocity acceleration p99 (m/s²)'),
            ('velocity_jerk_p99_m_s3','Body-velocity jerk p99 (m/s³)')]
    for ax,(metric,label) in zip(axes.flat,panels):
        for offset,color,legend,group in zip((-.2,.2),('gray','#54a24b'),('Baseline','Candidate'),indexed):
            ax.bar(x+offset,[group[key][metric] for key in keys],width=.4,color=color,label=legend)
        ax.set_ylabel(label);ax.set_xlabel(f'Reset seed − {keys[0][1]:,}')
        ax.set_xticks(x[::4],[keys[i][1]-keys[0][1] for i in range(0,len(keys),4)])
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(args.title+f'\nExact saved-action replays, equal per-start prefixes up to {metadata["paired_step_range"][1]} steps; no hardware stop model')
    fig.savefig(args.out_prefix.with_suffix('.png'),dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
