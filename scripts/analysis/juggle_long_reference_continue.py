"""Complete a fixed 32-start reference after its original Slurm job is terminal.

Preserves valid completed trajectories. Only missing/partial cases are evaluated
from their original reset seeds, with the same frozen policy/config and horizon.
No learner, simulator, sensing or control code is changed. CPU-only inference.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import numpy as np
from scripts.td3.helper.juggle_counter import count_juggles_from_arrays
from scripts.analysis.juggle_motion_metrics import sampled_motion


def trajectory_metrics(path, seed, horizon):
    """Read complete recorded episodes; reject a truncated ZIP or mid-episode data."""
    with np.load(path) as data:
        required = ('action','reward','terminated','truncated','observation','next_observation','true_paddle_xy','true_puck_xy')
        lengths = [len(data[k]) for k in required]
        if not lengths or len(set(lengths)) != 1 or not 1 <= lengths[0] <= horizon:
            raise ValueError('Invalid or inconsistent trajectory lengths')
        steps = lengths[0]
        for key in ('action','true_paddle_xy','true_puck_xy'):
            if data[key].shape != (steps,2):raise ValueError('Invalid two-dimensional trajectory shape')
        for key in ('observation','next_observation'):
            if data[key].shape != (steps,30):raise ValueError('Unexpected canonical observation shape')
        if steps < horizon and not (bool(data['terminated'][-1]) or bool(data['truncated'][-1])):
            raise ValueError('Incomplete episode has no recorded terminal outcome')
        if not all(np.isfinite(data[k]).all() for k in required):
            raise ValueError('Nonfinite trajectory data')
        physical = count_juggles_from_arrays(data['true_paddle_xy'],data['true_puck_xy'],np.zeros(steps))
        observed = data['next_observation']
        sensed = count_juggles_from_arrays(observed[:,12:14],observed[:,27:29],observed[:,29])
        position = sampled_motion(data['true_paddle_xy'], .05)
        action_delta = np.diff(np.vstack((np.zeros((1,2)),data['action'])),axis=0)
        row = dict(seed=seed, steps=steps, physical_juggles=physical.n_juggles, sensed_juggles=sensed.n_juggles,
            reward=float(np.sum(data['reward'],dtype=np.float64)), action_delta_rms=float(np.sqrt(np.mean(action_delta**2))),
            terminated=bool(data['terminated'][-1]), truncated=bool(data['truncated'][-1]),
            end_reason='not recorded in NPZ; do not infer cause', trajectory=str(path),
            trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), position_motion=position)
        standard = path.parent/'results.json'
        if standard.exists():
            try:
                payload = json.loads(standard.read_text())
                matching = [r for r in payload['scenarios']['training']['episodes'] if r['seed'] == seed and r['steps'] == steps]
                if len(matching) == 1:
                    row['recorded_end_reasons'] = matching[0]['end_reasons']
                    row['source_results_json'] = str(standard)
            except (json.JSONDecodeError, KeyError):
                pass
        # This is a descriptive recording boundary, not a success/acceptance gate.
        if steps == horizon and not (row['terminated'] or row['truncated']):
            row['end_reason'] = 'recording limit'
        return row


def verify_inputs(manifest, repo):
    for name, expected in manifest['reviewed_sha256'].items():
        if hashlib.sha256((repo/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Reviewed source/config changed: '+name)
    if hashlib.sha256(Path(manifest['bundle_model']).read_bytes()).hexdigest()!=manifest['model_sha256']:
        raise RuntimeError('Fixed actor changed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    args=p.parse_args();manifest=json.loads(args.manifest.read_text())
    repo=Path(manifest['repo']);root=Path(manifest['root']);parent=Path(manifest['parent_root'])
    terminal=('COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED')
    state=''
    for attempt in range(30):
        try:
            state=subprocess.run(['sacct','-X','-j',manifest['parent_job'],'-n','-o','State%40'],capture_output=True,text=True,check=True).stdout.strip()
        except subprocess.CalledProcessError:
            state='accounting query failed'
        if any(state.startswith(s) for s in terminal):break
        time.sleep(10)
    else:
        raise RuntimeError('Parent is not authoritatively terminal: '+repr(state))
    root.mkdir(exist_ok=True)
    def save(status,**details):
        payload=dict(status=status,parent_accounting_state=state,updated_utc=datetime.now(timezone.utc).isoformat(),**details)
        temp=root/'status.tmp';temp.write_text(json.dumps(payload,indent=2)+'\n');temp.replace(root/'status.json')
    verify_inputs(manifest,repo)
    saved,missing,invalid={},{},{}
    for seed in range(manifest['seed_base'],manifest['seed_base']+manifest['episodes']):
        path=parent/'independent_100k_eval'/f'training_seed{seed}.npz'
        candidates=[path]+sorted(root.glob(f'seed_{seed}/training_seed{seed}.npz'))+sorted(root.glob(f'seed_{seed}_retry*/training_seed{seed}.npz'))
        for candidate in candidates:
            try:
                saved[seed]=trajectory_metrics(candidate,seed,manifest['horizon'])
                break
            except (FileNotFoundError,ValueError,EOFError,OSError,KeyError,TypeError,zipfile.BadZipFile) as exc:
                if candidate.exists():invalid[str(candidate)]=repr(exc)
        if seed not in saved:missing[seed]=path
    save('continuing' if missing else 'collecting',reused_cases=len(saved),missing_seeds=list(missing),invalid_parent_cases=invalid)
    def evaluate(seed):
        verify_inputs(manifest,repo)
        out=root/f'seed_{seed}'
        retry=0
        while out.exists():
            retry+=1
            out=root/f'seed_{seed}_retry{retry}'
        cmd=[sys.executable,'-u','-m','scripts.td3.juggle_survival','--run-dir',manifest['bundle'],
             '--scenario-set','training','--episodes','1','--horizon',str(manifest['horizon']),
             '--seed-base',str(seed),'--gif-steps','0','--out-dir',str(out)]
        with (root/f'seed_{seed}.log').open('w') as log:
            subprocess.run(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
        verify_inputs(manifest,repo)
        row=trajectory_metrics(out/f'training_seed{seed}.npz',seed,manifest['horizon'])
        row['evaluation_command']=cmd
        return row
    try:
        with ThreadPoolExecutor(max_workers=manifest['parallel_cases']) as pool:
            futures={pool.submit(evaluate,seed):seed for seed in missing}
            for future in as_completed(futures):
                seed=futures[future];saved[seed]=future.result()
                # Persist each completed case for evidence if this worker is interrupted.
                (root/f'case_{seed}.json').write_text(json.dumps(saved[seed],indent=2)+'\n')
                save('continuing',completed_cases=len(saved),planned_cases=manifest['episodes'],reused_cases=manifest['episodes']-len(missing))
        rows=[saved[s] for s in sorted(saved)]
        if len(rows)!=manifest['episodes']:raise RuntimeError('Planned cohort incomplete')
        for row in rows:
            if row['seed'] not in missing:(root/f'case_{row["seed"]}.json').write_text(json.dumps(row,indent=2)+'\n')
        summary=dict(episodes=len(rows),recording_limit_steps=manifest['horizon'],
            mean_steps=float(np.mean([r['steps'] for r in rows])),median_steps=float(np.median([r['steps'] for r in rows])),
            mean_physical_juggles=float(np.mean([r['physical_juggles'] for r in rows])),
            mean_sensed_juggles=float(np.mean([r['sensed_juggles'] for r in rows])),
            mean_reward=float(np.mean([r['reward'] for r in rows])),
            mean_action_delta_rms=float(np.mean([r['action_delta_rms'] for r in rows])))
        output=dict(model_sha256=manifest['model_sha256'],parent_accounting_state=state,
            seed_base=manifest['seed_base'],summary=summary,episodes=rows,
            limitations='Same fixed policy/cohort; completed cases reused and missing cases reset after authoritative parent termination. Position derivatives are sampled diagnostics, not UR5 stop-risk predictions. Some terminal reasons were not stored in original NPZ.')
        (root/'raw_results.json').write_text(json.dumps(output,indent=2)+'\n')
        save('complete',summary=summary,reused_cases=manifest['episodes']-len(missing),new_cases=len(missing))
    except Exception as exc:
        save('failed',error=repr(exc),completed_cases=len(saved))
        raise


if __name__=='__main__':
    main()
