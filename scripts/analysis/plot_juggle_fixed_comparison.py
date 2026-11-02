"""Plot a saved fixed-budget comparison CSV, keeping each seed/weight separate."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--title',default='Fixed-budget comparison')
    args=p.parse_args();rows=list(csv.DictReader(args.summary.open()))
    if not rows:raise ValueError('No summary rows')
    protocol={(r['step'],r['episodes'],r['seed_base'],r['horizon']) for r in rows}
    if len(protocol)!=1:raise ValueError('Comparison must use one budget and evaluation protocol')
    labels,colors=[],[]
    for row in rows:
        if row['run']=='noisy_history_20261029_seed0':
            label='Baseline s0 '+row['weights'];color='black'
        elif row['run']=='noisy_history_long_discount_20261029_seed0':
            label='Baseline γ=.999 s0 '+row['weights'];color='black'
        else:
            slug=row['run'].split('smooth_',1)[1].split('_')[0]
            coefficient={'0005':'0.0005','002':'0.002','008':'0.008'}[slug]
            gamma=' γ=.999' if '_g0999_' in row['run'] else ''
            label=f"λ={coefficient}{gamma} s{row['run'].split('seed')[-1]} {row['weights']}"
            color={'0005':'#54a24b','002':'#4c78a8','008':'#f58518'}[slug]
        labels.append(label);colors.append(color)
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for ax,key,label in zip(axes.flat,['physical_juggles','episode_steps','action_delta_rms','median_episode_p95_jerk_m_s3'],
            ['Mean physical juggles','Mean episode steps','Mean action-change RMS','Median episode p95 position-derived jerk (m/s³)']):
        ax.barh(labels,[float(r[key]) for r in rows],color=colors);ax.invert_yaxis();ax.set_xlabel(label)
        ax.tick_params(axis='y',labelsize=8);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    _,episodes,_,horizon=next(iter(protocol))
    fig.suptitle(args.title+f'\nOne seed/weight variant per bar; {episodes} reused starts, {int(horizon):,}-step recording limit; training continues')
    args.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.out,dpi=160);plt.close(fig)


if __name__=='__main__':
    main()
