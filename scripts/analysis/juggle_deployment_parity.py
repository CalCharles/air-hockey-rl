"""Compare saved simulator actions with the real-eval history adapter offline.

Reads noisy observations and known previous actions only; no env construction,
robot connection, physics changes, model training or live control invocation.
Normal calls/reset/rollover are tested; holds and recovery overrides are not.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts.real.evaluate_policy import resolve_train_args
from scripts.td3.helper.real_eval_agents import build_rma_eval_agent
from scripts.td3.helper.real_td3_runtime import Args


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=300)
    a=p.parse_args()
    if a.steps<1:raise ValueError('steps must be positive')
    torch.set_num_threads(1)
    meta=json.loads((a.bundle/'rma_meta.json').read_text())
    train_args,_=resolve_train_args('rma',str(a.bundle),None)
    factory=build_rma_eval_agent(args=Args(model_path=str(a.bundle)),train_args=train_args,
        obs_dim=int(meta['obs_dim']),act_dim=int(meta['act_dim']),
        action_low_np=-np.ones(int(meta['act_dim']),dtype=np.float32),
        action_high_np=np.ones(int(meta['act_dim']),dtype=np.float32),device=torch.device('cpu'))
    payload=json.loads(a.results.read_text());rows=[]
    for name,scenario in payload['scenarios'].items():
        for ep in scenario['episodes']:
            path=a.results.parent/ep['trajectory']
            with np.load(path) as saved:
                factory.actor.on_episode_start(saved['observation'][0])
                previous=np.zeros(int(meta['act_dim']),dtype=np.float32)
                count=min(a.steps,len(saved['action']))
                for index in range(count):
                    obs=np.asarray(saved['observation'][index],dtype=np.float32)
                    policy=np.concatenate((obs,previous)) if meta['use_last_action'] else obs
                    actual=factory.actor.get_action(torch.from_numpy(policy).reshape(1,-1)).detach().cpu().numpy().reshape(-1)
                    actual=np.clip(actual,-1.,1.).astype(np.float32)
                    expected=np.asarray(saved['action'][index],dtype=np.float32)
                    if not np.array_equal(actual,expected):
                        raise ValueError(f'Deployment action mismatch {name} seed={ep["seed"]} step={index}: max error {np.max(np.abs(actual-expected))}')
                    previous=actual.copy()
                rows.append(dict(scenario=name,seed=ep['seed'],episode_steps=ep['steps'],tested_steps=count,
                    exact_numeric_actions=True,trajectory=str(path),trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    out=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),bundle=str(a.bundle),model_sha256=hashlib.sha256((a.bundle/'model.pth').read_bytes()).hexdigest(),
        source_results=str(a.results),source_results_sha256=hashlib.sha256(a.results.read_bytes()).hexdigest(),
        episodes=len(rows),actions=sum(r['tested_steps'] for r in rows),prefix_limit=a.steps,
        deployment_source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),Path('scripts/td3/helper/real_eval_agents.py'),Path('scripts/real/evaluate_policy.py'),Path('scripts/rma/evaluate.py'),Path('scripts/rma/history.py'),Path('scripts/rma/networks.py')]},
        scope='Normal deployment-adapter calls, history rollovers and episode resets on saved noisy observations; no hardware or physics stepping.',
        limitations='Does not cover overridden/held actions, e-stop recovery, camera timing, real command timing or UR5 stop risk.',rows=rows)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n')
    print('Exact deployment-adapter actions:',out['actions'],'across',len(rows),'episodes; no robot connected.')


if __name__=='__main__':
    main()
