"""Isolated target-cost scratch trainer and frozen descriptive milestone tests.

Submit inside a one-GPU Slurm job; source hashes are fixed before submission.
An auxiliary evaluation failure is recorded without stopping the full learner.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml
from scripts.td3.juggle_noisy_scratch import assert_canonical_scratch_env, scratch_trainer

REPO = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--args-file', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    cli = p.parse_args()
    root = cli.root.resolve()
    manifest = json.loads(cli.manifest.read_text())

    def check():
        for name, expected in manifest['reviewed_sha256'].items():
            if hashlib.sha256((REPO/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Reviewed input changed: '+name)

    check()
    args = yaml.safe_load(cli.args_file.read_text())
    assert_canonical_scratch_env(yaml.safe_load((REPO/args['config']).read_text())['air_hockey'])
    scratch_trainer(args)
    if root.exists():
        raise FileExistsError(root)
    coefficient = float(manifest['target_cost_coefficient'])
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('Target cost coefficient must be finite and nonnegative')
    if args.get('actor_action_delta_penalty', 0) != 0:
        raise ValueError('Target reward trial must isolate the existing actor penalty')
    root.mkdir()
    args.update(log_parent_dir=str(root/'train'))
    launch_args = root/'launch_args.yaml'
    launch_args.write_text(yaml.safe_dump(args, sort_keys=False))
    command = [sys.executable, '-u', '-m', 'scripts.td3.train_juggle_target_cost',
               '--args-file', str(launch_args), '--coefficient', str(coefficient)]
    training_log = (root/'training.log').open('w')
    trainer = subprocess.Popen(command, cwd=REPO, stdout=training_log, stderr=subprocess.STDOUT)
    def training_status(status):
        payload = dict(status=status, training_command=command,
                       target_cost_coefficient=coefficient,
                       updated_utc=datetime.now(timezone.utc).isoformat())
        path = root/'status.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(payload, indent=2)+'\n')
        temp.replace(path)
    training_status('training')
    errors = []

    def save(**state):
        state.update(updated_utc=datetime.now(timezone.utc).isoformat(), errors=errors)
        path = root/'milestone_test_status.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(path)

    try:
        for step in (1000000, 3000000, 6000000):
            checkpoint = root/'train'/f'checkpoint_{step}' if step < 6000000 else root/'train'
            ready = checkpoint/'survival_eval/results.json'
            while True:
                valid = False
                if ready.exists():
                    try:
                        valid = json.loads(ready.read_text())['scenarios']['training']['summary']['episodes'] == 8
                    except json.JSONDecodeError:
                        pass
                code = trainer.poll()
                if code is not None:
                    training_status('complete' if code == 0 else 'failed')
                if valid and (step < 6000000 or code == 0):
                    break
                if code is not None and (code != 0 or not valid):
                    raise RuntimeError(f'Trainer exited {code} before required evaluation {step}')
                if root.exists():
                    save(status='waiting', training_step=step)
                time.sleep(20)
            for role in ('online', 'target'):
                try:
                    check()
                    cfg = yaml.safe_load((checkpoint/'config.yaml').read_text())['air_hockey']
                    assert_canonical_scratch_env(dict(cfg, seed=0))
                    bundle = root/f'frozen_{step}_{role}'
                    bundle.mkdir()
                    source = checkpoint/('model.pth' if role == 'online' else 'actor_target.pth')
                    shutil.copy2(source, bundle/'model.pth')
                    for name in ('args.yaml', 'config.yaml', 'rma_meta.json'):
                        shutil.copy2(checkpoint/name, bundle/name)
                    sha = hashlib.sha256((bundle/'model.pth').read_bytes()).hexdigest()
                    out = root/f'paired_{step}_{role}_20k_eval'
                    if out.exists():
                        raise FileExistsError(out)
                    save(status='evaluating', training_step=step, role=role, sha256=sha)
                    cmd = [sys.executable, '-u', '-m', 'scripts.td3.juggle_survival', '--run-dir', str(bundle),
                           '--scenario-set', 'training', '--episodes', '32', '--horizon', '20000', '--seed-base', '700000',
                           '--gif-steps', '200', '--out-dir', str(out)]
                    with (root/f'paired_{step}_{role}.log').open('w') as log:
                        subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
                    if hashlib.sha256((bundle/'model.pth').read_bytes()).hexdigest() != sha:
                        raise RuntimeError('Frozen evaluation weights changed')
                    (out/'frozen_provenance.json').write_text(json.dumps(dict(source=str(source), model_sha256=sha, command=cmd), indent=2)+'\n')
                    subprocess.run([sys.executable, '-m', 'scripts.analysis.juggle_motion_metrics', '--results', str(out/'results.json')], cwd=REPO, check=True)
                except Exception as exc:
                    errors.append(dict(step=step, role=role, error=repr(exc)))
                    save(status='auxiliary_test_failed', training_step=step, role=role)
        code = trainer.wait()
        training_log.close()
        training_status('complete' if code == 0 else 'failed')
        save(status='complete' if code == 0 and not errors else 'complete_with_errors', trainer_exit_code=code)
    except Exception as exc:
        errors.append(dict(error=repr(exc)))
        code = trainer.wait()
        training_log.close()
        training_status('failed')
        if root.exists():
            save(status='failed', trainer_exit_code=code)
        raise


if __name__ == '__main__':
    main()
