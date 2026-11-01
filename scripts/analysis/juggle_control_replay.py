"""Replay saved actions exactly and read cached controller targets/velocities.

Does not change simulator code, policy state, action mapping or any live learner.
Checks every observation/reward/termination and physical position against saved
arrays. No calls to sensor-sampling get_current_state are added.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from airhockey import AirHockeyEnv
from scripts.td3.juggle_survival import diagnostic_body_positions


def replay(results, out, prefix_steps=250):
    payload = json.loads(results.read_text())
    if prefix_steps < 2:
        raise ValueError('Need at least two steps for target changes')
    rows = []
    for name, scenario in payload['scenarios'].items():
        for episode in scenario['episodes']:
            seed = episode['seed']
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            config = copy.deepcopy(scenario['config'])
            config.update(seed=seed, max_timesteps=payload['horizon']+1)
            config['simulator_params']['seed'] = seed
            env = AirHockeyEnv(config)
            targets, velocities = [], []
            with np.load(results.parent/episode['trajectory']) as saved:
                try:
                    obs, _ = env.reset(seed=seed)
                    count = min(len(saved['action']), prefix_steps)
                    for index in range(count):
                        if not np.array_equal(np.asarray(obs,dtype=saved['observation'].dtype),saved['observation'][index]):
                            raise ValueError(f'Before-observation mismatch: {seed} step {index}')
                        obs, reward, term, trunc, _ = env.step(saved['action'][index].copy())
                        if not np.array_equal(np.asarray(obs,dtype=saved['next_observation'].dtype),saved['next_observation'][index]):
                            raise ValueError(f'Next-observation mismatch: {seed} step {index}')
                        if float(reward) != float(saved['reward'][index]) or bool(term) != bool(saved['terminated'][index]) or bool(trunc) != bool(saved['truncated'][index]):
                            raise ValueError(f'Outcome mismatch: {seed} step {index}')
                        paddle, puck = diagnostic_body_positions(env.simulator)
                        if not np.array_equal(paddle,saved['true_paddle_xy'][index]) or not np.array_equal(puck,saved['true_puck_xy'][index]):
                            raise ValueError(f'True-body mismatch: {seed} step {index}')
                        targets.append(np.asarray(env.simulator.last_target_position,dtype=np.float64).copy())
                        velocity = env.simulator.paddles['paddle_ego'].linearVelocity
                        velocities.append(np.asarray([-velocity[1],velocity[0]],dtype=np.float64))
                    rate = float(env.simulator.step_frequency)
                    target = np.asarray(targets)
                    delta = np.diff(target,axis=0)
                    proxy_delta = np.diff(saved['action'][:count],axis=0)*[env.action_x_ratio,env.action_y_ratio]
                    speed = np.linalg.norm(np.asarray(velocities),axis=1)
                    accel = np.linalg.norm(np.diff(velocities,axis=0)*rate,axis=1)
                    jerk = np.linalg.norm(np.diff(velocities,n=2,axis=0)*rate**2,axis=1)
                    rows.append(dict(scenario=name,seed=seed,episode_steps=episode['steps'],replayed_steps=count,exact_match=True,
                        pid_target_change_rms_m=float(np.sqrt(np.mean(delta**2))) if len(delta) else None,
                        scaled_action_change_rms_m_excluding_reset=float(np.sqrt(np.mean(proxy_delta**2))) if len(proxy_delta) else None,
                        trajectory_sha256=hashlib.sha256((results.parent/episode['trajectory']).read_bytes()).hexdigest(),
                        pid_target_change_p95_m=float(np.percentile(np.linalg.norm(delta,axis=1),95)) if len(delta) else None,
                        body_speed_p95_m_s=float(np.percentile(speed,95)),
                        velocity_acceleration_p95_m_s2=float(np.percentile(accel,95)) if len(accel) else None,
                        velocity_jerk_p95_m_s3=float(np.percentile(jerk,95)) if len(jerk) else None))
                finally:
                    env.close()
    output=dict(source=str(results),source_sha256=hashlib.sha256(results.read_bytes()).hexdigest(),prefix_limit=prefix_steps,
        definition='Cached final filtered PID target after each control step; differences exclude reset target. Physical derivatives use cached body velocity at control rate.',
        limitations='Different visited states and exposure when an episode ends before prefix limit; no UR5 joint dynamics or stop-risk prediction.',episodes=rows)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,indent=2)+'\n')
    print(f'Exact replay matches: {len(rows)} episodes; {sum(r["replayed_steps"] for r in rows)} steps; {out}')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prefix-steps',type=int,default=250)
    a=p.parse_args();replay(a.results,a.out,a.prefix_steps)
