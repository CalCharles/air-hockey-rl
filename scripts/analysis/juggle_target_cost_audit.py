"""Verify a reward-only target cost against saved canonical noisy rollouts."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from airhockey import AirHockeyEnv
from scripts.td3.juggle_survival import diagnostic_body_positions
from scripts.td3.juggle_target_cost import TargetStepCost


def audit(results, out, prefix_steps=250, coefficient=100.0):
    if prefix_steps < 2:
        raise ValueError('Need at least two commands')
    payload = json.loads(results.read_text())
    rows = []
    for name, scenario in payload['scenarios'].items():
        for episode in scenario['episodes']:
            seed = episode['seed']
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            config = copy.deepcopy(scenario['config'])
            config.update(seed=seed, max_timesteps=payload['horizon']+1)
            config['simulator_params']['seed'] = seed
            env = TargetStepCost(AirHockeyEnv(config), coefficient)
            distances, costs, original_rewards = [], [], []
            path = results.parent/episode['trajectory']
            with np.load(path) as saved:
                try:
                    obs, _ = env.reset(seed=seed)
                    count = min(len(saved['action']), prefix_steps)
                    for index in range(count):
                        if not np.array_equal(np.asarray(obs, dtype=saved['observation'].dtype), saved['observation'][index]):
                            raise ValueError(f'Before observation mismatch: {seed}/{index}')
                        obs, reward, term, trunc, info = env.step(saved['action'][index].copy())
                        if not np.array_equal(np.asarray(obs, dtype=saved['next_observation'].dtype), saved['next_observation'][index]):
                            raise ValueError(f'After observation mismatch: {seed}/{index}')
                        if info['original_reward'] != float(saved['reward'][index]):
                            raise ValueError('Underlying reward changed')
                        if reward != info['original_reward'] - info['target_step_cost']:
                            raise ValueError('Incorrect shaped reward')
                        if bool(term) != bool(saved['terminated'][index]) or bool(trunc) != bool(saved['truncated'][index]):
                            raise ValueError('Termination changed')
                        paddle, puck = diagnostic_body_positions(env.simulator)
                        if not np.array_equal(paddle,saved['true_paddle_xy'][index]) or not np.array_equal(puck,saved['true_puck_xy'][index]):
                            raise ValueError('Physical trajectory changed')
                        distances.append(info['target_step_m'])
                        costs.append(info['target_step_cost'])
                        original_rewards.append(info['original_reward'])
                    row = dict(scenario=name, seed=seed, steps=count, exact_original_rollout=True,
                               trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               original_reward_sum=float(sum(original_rewards)), cost_sum=float(sum(costs)),
                               fourth_power_sum_m4=float(sum(d**4 for d in distances)),
                               target_step_p95_m=float(np.percentile(distances,95)),
                               target_step_p99_m=float(np.percentile(distances,99)),
                               target_step_max_m=float(max(distances)))
                    # Share of cost from the largest 1% of measured changes.
                    top = max(1, math.ceil((count-1)*.01))
                    measured = np.asarray(distances[1:])**4
                    row['top_one_percent_cost_share'] = float(np.sort(measured)[-top:].sum()/measured.sum()) if measured.sum() else 0.0
                    rows.append(row)
                finally:
                    env.close()
    if not rows:
        raise ValueError('No episodes audited')
    source_files = [Path(__file__), Path('scripts/td3/juggle_target_cost.py'),
                    Path('airhockey/airhockey_base.py'), Path('airhockey/sims/airhockey_box2d.py')]
    output = dict(source=str(results), source_sha256=hashlib.sha256(results.read_bytes()).hexdigest(),
                  coefficient=coefficient, prefix_steps=prefix_steps,
                  definition='Reward-only c*norm(change in final cached filtered PID target)^4; first target after each reset excluded.',
                  limitations='Fixed saved actions: validates reward-only parity and calibrates cost, does not measure learned-policy performance, motor energy or hardware safety.',
                  code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                  episodes=rows, total_steps=sum(r['steps'] for r in rows),
                  original_reward_sum=sum(r['original_reward_sum'] for r in rows),
                  cost_sum=sum(r['cost_sum'] for r in rows),
                  median_top_one_percent_cost_share=float(np.median([r['top_one_percent_cost_share'] for r in rows])))
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('episodes','source_sha256')}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prefix-steps',type=int,default=250)
    p.add_argument('--coefficient',type=float,default=100)
    a=p.parse_args();audit(a.results,a.out,a.prefix_steps,a.coefficient)
