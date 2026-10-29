"""Run fresh TD3 in the canonical noisy simulator, with durable status and eval."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

import yaml

from scripts.td3.juggle_original_sensing import REPO, assert_original_sensing, original_sensing_config


def assert_canonical_scratch_env(config):
    """Allow explicit reward shaping while retaining the exact simulator/DR.

    These three existing base-env knobs affect only reward computation. Task,
    terminations, observation sampling and every physical/control key stay fixed.
    """
    normalized = dict(config)
    expected = original_sensing_config()['air_hockey']
    if 'enable_survival_bonus' in config and not isinstance(config['enable_survival_bonus'], bool):
        raise ValueError('enable_survival_bonus must be a boolean')
    for key in ('survival_bonus_per_step', 'puck_loss_penalty'):
        if key in config and (not math.isfinite(float(config[key])) or float(config[key]) < 0):
            raise ValueError(key + ' must be finite and nonnegative')
    for key in ('enable_survival_bonus', 'survival_bonus_per_step', 'puck_loss_penalty'):
        if key in expected:
            normalized[key] = expected[key]
        else:
            normalized.pop(key, None)
    assert_original_sensing(normalized)


def scratch_trainer(args):
    """Choose a deployable trainer while rejecting all policy warm starts."""
    if args.get('model_path') is not None or args.get('full_checkpoint_load') == 'residual':
        raise ValueError('Scratch training must not load a checkpoint or residual base')
    if args.get('history_warm_start_plain', False):
        raise ValueError('Scratch history policies must not warm-start an encoder or actor')
    if args.get('exploration_pre_learning_action_source') != 'random':
        raise ValueError('Scratch training requires random initial exploration')
    if args.get('rma_mode') == 'history':
        return 'scripts.rma.train_base_policy'
    if args.get('rma_mode') is not None:
        raise ValueError('Scratch juggling requires a deployable policy without privileged actor inputs')
    return 'scripts.td3.td3_training'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--args-file', default='configs/td3/sustained/noisy_scratch.yaml')
    cli = parser.parse_args()
    root = Path(cli.root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    state = {'status': 'preparing', 'started': datetime.now(timezone.utc).isoformat()}

    def status(value):
        state['status'] = value
        state['updated'] = datetime.now(timezone.utc).isoformat()
        temp = root / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(root / 'status.json')

    try:
        args = yaml.safe_load((REPO / cli.args_file).read_text())
        assert_canonical_scratch_env(yaml.safe_load((REPO / args['config']).read_text())['air_hockey'])
        trainer = scratch_trainer(args)
        # The trainer uses this exact directory, rather than choosing an rN suffix.
        args.update(log_parent_dir=str(root / 'train'), resume_in_place=True)
        args_path = root / 'launch_args.yaml'
        args_path.write_text(yaml.safe_dump(args, sort_keys=False))
        command = [sys.executable, '-u', '-m', trainer, '--args-file', str(args_path)]
        state['training_command'] = command
        status('training')
        with (root / 'training.log').open('w') as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        status('final_evaluation')
        command = [sys.executable, '-u', '-m', 'scripts.td3.juggle_survival',
                   '--run-dir', str(root / 'train'), '--scenario-set', 'training',
                   '--episodes', '32', '--horizon', '10000', '--seed-base', '200000',
                   '--out-dir', str(root / 'final_eval')]
        state['evaluation_command'] = command
        with (root / 'evaluation.log').open('w') as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        status('complete')
    except Exception as exc:
        state['error'] = repr(exc)
        status('failed')
        raise


if __name__ == '__main__':
    main()
