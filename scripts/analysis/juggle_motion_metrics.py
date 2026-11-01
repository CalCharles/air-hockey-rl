"""Read-only physical motion distributions from saved juggling trajectories.

Finite differences of true paddle positions at the control rate measure sampled
motion, not UR5 joint loads, substep contact spikes or emergency-stop likelihood.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import yaml


def sampled_motion(position, dt):
    position = np.asarray(position, dtype=np.float64)
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError('dt must be finite and positive')
    if position.ndim != 2 or position.shape[1] != 2 or not np.isfinite(position).all():
        raise ValueError('positions must be finite N x 2')
    values = {}
    for label, order in [('speed_m_s', 1), ('acceleration_m_s2', 2), ('jerk_m_s3', 3)]:
        array = np.linalg.norm(np.diff(position, n=order, axis=0) / dt**order, axis=1)
        values[label] = dict(samples=len(array), rms=float(np.sqrt(np.mean(array**2))) if len(array) else None,
                             p50=float(np.percentile(array, 50)) if len(array) else None,
                             p95=float(np.percentile(array, 95)) if len(array) else None,
                             p99=float(np.percentile(array, 99)) if len(array) else None,
                             maximum=float(array.max()) if len(array) else None)
    return values


def analyze(results):
    payload = json.loads(results.read_text())
    output = dict(source=str(results), method='finite differences of true body position at control rate; no initial reset sample',
                  limitations='Different episode lengths and visited states; no UR5 dynamics, joint torque or stop prediction.', episodes=[])
    for scenario, data in payload['scenarios'].items():
        rate = float(data['config']['simulator_params'].get('step_frequency', 20))
        for row in data['episodes']:
            with np.load(results.parent / row['trajectory']) as trajectory:
                position = trajectory['true_paddle_xy']
                output['episodes'].append(dict(scenario=scenario, seed=row['seed'], steps=row['steps'],
                    dt_seconds=1/rate, whole_episode=sampled_motion(position, 1/rate),
                    first_250_steps=sampled_motion(position[:250], 1/rate)))
    dest = results.parent/'motion_metrics.json'
    dest.write_text(json.dumps(output, indent=2)+'\n')
    return dest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', required=True, type=Path)
    print(analyze(p.parse_args().results))
