"""Prepare, never submit, a login-node Slurm handoff for the two 50M runs."""
import argparse
from pathlib import Path
import shlex
import sys
import yaml

ROOT=Path(__file__).absolute().parents[2]


def prepare(folder, after_job):
    if not str(after_job).isdigit():
        raise ValueError('after_job must be a numeric Slurm job ID')
    folder=Path(folder).absolute()
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'logs').mkdir()
    python=ROOT/'.venv/bin/python'
    commands=[]
    for mode in ('full','current'):
        source=ROOT/f'configs/fb/fb_50M_{mode}.yaml'
        config=yaml.safe_load(source.read_text())
        config['output_dir']=str((ROOT/config['output_dir']).absolute())
        config['config']=str((ROOT/config['config']).absolute())
        config['resume_latest']=True
        config['max_runtime_seconds']=47*3600+50*60
        snapshot=folder/f'{mode}.yaml'
        snapshot.write_text(yaml.safe_dump(config,sort_keys=False))
        command=shlex.join([str(python),'-u','-m','scripts.fb.fast_training','--args-file',str(snapshot)])
        commands.append(command+f' > {shlex.quote(str(folder/"logs"))}/{mode}_${{SLURM_JOB_ID}}.log 2>&1 &\npids+=("$!")')
    script='''#!/bin/bash
#SBATCH --job-name=fb-50M-continue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=gh
#SBATCH --account=ASC26008
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
set -euo pipefail
'''
    script+=f'cd {shlex.quote(str(ROOT))}\n'
    script+='''export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
pids=()
trap 'kill -TERM "${pids[@]}" 2>/dev/null || true; wait || true' TERM INT
'''
    script+='\n'.join(commands)+'\nstatus=0\nfor pid in "${pids[@]}"; do\n    wait "$pid" || status=1\ndone\nexit "$status"\n'
    (folder/'job.sbatch').write_text(script)
    wrapper='''#!/bin/bash
set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit only from a login node outside a Slurm allocation." >&2; exit 2
fi
case "$(hostname -s)" in
    login[0-9]*) ;;
    *) echo "Not a login node; refusing submission." >&2; exit 2 ;;
esac
'''
    wrapper+=f'exec sbatch --dependency=afterany:{after_job} --chdir={shlex.quote(str(folder))} {shlex.quote(str(folder/"job.sbatch"))}\n'
    (folder/'submit.sh').write_text(wrapper)
    print('Prepared only; NO job submitted.')
    print(f'On a login node: bash {shlex.quote(str(folder/"submit.sh"))}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',required=True)
    parser.add_argument('--after-job',required=True)
    args=parser.parse_args()
    prepare(args.folder,args.after_job)
