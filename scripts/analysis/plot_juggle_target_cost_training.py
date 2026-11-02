"""Publish original/shaped training reward and cached target-motion cost blocks.

This reads logs only. Physical-juggle performance comes from the separately
seeded checkpoint evaluator, not these exploratory training reward blocks.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch',type=Path,default=Path('docs/juggling-target-cost-launch.json'))
    parser.add_argument('--max-step',type=int)
    parser.add_argument('--out-prefix',type=Path,required=True)
    args=parser.parse_args()
    launch=json.loads(args.launch.read_text())
    rows=[];sources=[]
    for run in launch['launches']:
        path=Path(run['root'])/'target_cost_training.csv'
        if not path.exists():
            continue
        lines=path.read_bytes().splitlines(keepends=True)
        if lines and not lines[-1].endswith(b'\n'):
            lines=lines[:-1]  # A live writer's unfinished final row is not evidence.
        if len(lines)<2:
            continue
        parsed=list(csv.DictReader(io.StringIO(b''.join(lines).decode())))
        selected=[];previous=0
        for row in parsed:
            if any(value is None for value in row.values()):
                raise ValueError('Malformed complete CSV row')
            values={k:float(v) for k,v in row.items()}
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError('Non-finite log value')
            step=int(values['step']);count=int(values['block_steps'])
            if step!=values['step'] or count!=values['block_steps'] or count<=0 or step-previous!=count:
                raise ValueError('Non-contiguous training blocks')
            previous=step
            if not math.isclose(values['original_reward_sum']-values['shaped_reward_sum'],values['cost_sum'],abs_tol=1e-8):
                raise ValueError('Reward and cost accounting differ')
            if values['cost_sum']<0:
                raise ValueError('Negative motion cost')
            if args.max_step is not None and step>args.max_step:
                break
            selected.append(dict(run=run['name'],seed=run['manifest']['seed'],**row,
                original_reward_per_step=values['original_reward_sum']/count,
                shaped_reward_per_step=values['shaped_reward_sum']/count,cost_per_step=values['cost_sum']/count))
        if not selected:
            continue
        prefix=b''.join(lines[:len(selected)+1])
        sources.append(dict(path=str(path),captured_prefix_bytes=len(prefix),captured_prefix_sha256=hashlib.sha256(prefix).hexdigest()))
        rows.extend(selected)
    if not rows:
        raise ValueError('No complete reward blocks')
    args.out_prefix.parent.mkdir(parents=True,exist_ok=True)
    table=Path(str(args.out_prefix)+'.csv')
    with table.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for name in dict.fromkeys(r['run'] for r in rows):
        own=[r for r in rows if r['run']==name];x=[int(r['step']) for r in own]
        label='Seed '+str(own[0]['seed'])
        for ax,key,ylabel in zip(axes.flat,['original_reward_per_step','shaped_reward_per_step','cost_per_step','target_step_p95_m'],
            ['Original reward / step','Shaped reward / step','Target cost / step','Block p95 filtered-target change (m)']):
            ax.plot(x,[float(r[key]) for r in own],label=label)
            ax.set(xlabel='Training steps',ylabel=ylabel);ax.grid(alpha=.25)
    axes.flat[0].legend()
    learning_starts={yaml.safe_load(Path(r['config']).read_text())['learning_starts'] for r in launch['launches']}
    transition=''
    if len(learning_starts)==1:
        start=learning_starts.pop()
        for ax in axes.flat:
            ax.axvline(start,color='0.5',linestyle=':',linewidth=1)
        transition=f'\nRandom/primitive exploration before {start:,} steps; actor actions thereafter (dotted line)'
    fig.suptitle('Fresh PID-target reward-cost learners: exploratory training logs\nUnsmoothed block statistics; not fixed-policy juggling or hardware safety evidence'+transition)
    figure=Path(str(args.out_prefix)+'.png');fig.savefig(figure,dpi=160);plt.close(fig)
    launch_inputs=[dict(name=r['name'],root=r['root'],seed=r['manifest']['seed'],coefficient=r['coefficient'],
                        config=r['config'],config_sha256=r['manifest']['reviewed_sha256'][r['config']]) for r in launch['launches']]
    metadata=dict(generated_utc=datetime.now(timezone.utc).isoformat(),max_step=args.max_step,blocks=len(rows),
        sources=sources,launch_path=str(args.launch),launch_inputs=launch_inputs,
        launch_inputs_sha256=hashlib.sha256(json.dumps(launch_inputs,sort_keys=True).encode()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        output_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [table,figure]},
        limitations='Exploration, visited states and episode lengths differ. Includes random pre-learning exploration. No evaluation juggles inferred from reward.',
        per_run=[dict(run=name,steps=sum(int(r['block_steps']) for r in rows if r['run']==name),
                      original_reward=sum(float(r['original_reward_sum']) for r in rows if r['run']==name),
                      shaped_reward=sum(float(r['shaped_reward_sum']) for r in rows if r['run']==name),
                      cost=sum(float(r['cost_sum']) for r in rows if r['run']==name)) for name in dict.fromkeys(r['run'] for r in rows)])
    Path(str(args.out_prefix)+'.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Published',len(rows),'blocks',args.out_prefix)


if __name__=='__main__':
    main()
