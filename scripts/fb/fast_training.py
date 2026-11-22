"""Block-collected online FB with CPU acting, full-update CUDA graphs and resume.

Run separately from the original trainer. Policies sync once per rollout block;
FB losses and the one-update-per-transition ratio remain unchanged.
"""
from collections import deque
import copy
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import tyro
import yaml
from metamotivo.buffers.buffers import DictBuffer
from scripts.fb.fb_training import (Args, create_agent, evaluate_tasks, isolated_rng, load_sim_config,
                                    make_env, policy_obs, transition)
from scripts.fb.fast_checkpoint import base_args, load_training, restore_rng, save_continuation
from scripts.fb.fast_update import FastFBUpdate


@dataclass
class FastArgs(Args):
    total_timesteps: int = 50_000_000
    output_dir: str = 'runs/fb/fast_50M'
    resume_from: str | None = None
    resume_latest: bool = False
    collect_steps: int = 64
    use_graph: bool = True
    matmul_precision: str = 'high'
    eval_every_steps: int = 250_000
    checkpoint_every_steps: int = 250_000
    log_every_steps: int = 10_000
    eval_episodes: int = 10
    gif_episodes: int = 2
    max_runtime_seconds: float | None = None
    allocation_safety_seconds: int = 300


def stack_transitions(rows):
    return {k: stack_transitions([r[k] for r in rows]) if isinstance(rows[0][k],dict)
            else np.concatenate([r[k] for r in rows],axis=0) for k in rows[0]}


def sync_actor(agent, rollout):
    # CPU inference never uses F/B. Only actor and independent observation stats
    # need refresh; update occurs at most once per collect_steps transitions.
    for name in ('_actor','_obs_normalizer'):
        source=getattr(agent._model,name).state_dict()
        getattr(rollout,name).load_state_dict({k:v.detach().cpu() for k,v in source.items()})


def _train_locked(args):
    args.validate()
    if args.collect_steps<1 or args.collect_steps>args.buffer_size:
        raise ValueError('collect_steps must be between 1 and buffer_size')
    if args.matmul_precision not in ('highest','high'):
        raise ValueError('matmul_precision must be highest or high')
    torch.set_num_threads(args.torch_threads)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(torch.device(args.device) if ':' in args.device else 0)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    output=Path(args.output_dir)
    resume_path=args.resume_from
    if args.resume_latest and (output/'latest_checkpoint.json').exists():
        resume_path=json.loads((output/'latest_checkpoint.json').read_text())['path']
    if (output/'args.yaml').exists() and not args.resume_latest:
        raise FileExistsError(f'Existing run: {output}; pass --resume-latest')
    output.mkdir(parents=True,exist_ok=True)
    step=updates=0
    resume=None
    if resume_path:
        agent,replay,run,resume=load_training(resume_path,args)
        sim_config=run['sim_config']
        step,updates=run['step'],run['updates']
    else:
        sim_config=load_sim_config(args.config)
        agent=None
        replay=DictBuffer(args.buffer_size,device=args.device)
    if step>=args.total_timesteps:
        print(f'Already at target: {step}',flush=True)
        return Path(resume_path)
    if args.backward_input=='current' and sim_config['air_hockey'].get('obs_type')!='history':
        raise ValueError('Current backward input requires history observations')
    (output/'args.yaml').write_text(yaml.safe_dump(asdict(args)))
    (output/'sim_config.yaml').write_text(yaml.safe_dump(sim_config))
    initial_step=step
    # Old 1M checkpoints have no replay. Refill from their policy before learning.
    learning_after=step if len(replay)>=args.batch_size else step+max(args.learning_starts,args.batch_size)
    env=make_env(sim_config,args,args.seed+step)
    tracker=writer=None
    stopped={'value':False}
    old_handlers={}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig]=signal.signal(sig,lambda *_:stopped.update(value=True))
    start=time.monotonic()
    deadline=float('inf')
    if args.max_runtime_seconds is not None:
        deadline=start+args.max_runtime_seconds
    if os.environ.get('SLURM_JOB_END_TIME'):
        remaining=float(os.environ['SLURM_JOB_END_TIME'])-time.time()-args.allocation_safety_seconds
        deadline=min(deadline,start+max(0,remaining))
    status='failed'
    (output/'status.json').write_text(json.dumps({'status':'running','start_step':step,
        'target_steps':args.total_timesteps,'pid':os.getpid(),'resume_from':resume_path},indent=2))
    try:
        if args.wandb:
            from scripts.fb.monitoring import WandbLogger
            tracker=WandbLogger(args,sim_config,output)
        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer=SummaryWriter(str(output/'tb'))
        obs,_=env.reset(seed=args.seed+step)
        last_action=np.zeros(env.action_space.shape,np.float32)
        state=policy_obs(obs,last_action,args.use_last_action)
        if agent is None: agent=create_agent(args,state.size,last_action.size)
        # Deepcopy must be isolated because constructing CPU inference nets can
        # otherwise change training randomness. B/F copies are unused for rollout.
        rollout=copy.deepcopy(agent._model).to('cpu').eval().requires_grad_(False)
        if resume: restore_rng(resume)
        if resume_path and resume is None:
            print('Warm continuation: weights/optimizers restored; refilling missing replay for '
                  f'{learning_after-step} steps. Environment resets on all resumes.',flush=True)
        update=None
        z=None; skill_age=0; ep_return=0.; ep_length=0
        returns=deque(maxlen=100); lengths=deque(maxlen=100)
        metrics={}; checkpoint=None
        with (output/'metrics.jsonl').open('a') as log:
            def emit(values):
                row={'step':step,'updates':updates,**values}
                log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush()
                print(json.dumps(row),flush=True)
                if tracker: tracker.log(values,step)
                if writer:
                    for key,value in values.items(): writer.add_scalar(key,value,step)

            emit({'resume/start_step':step,'resume/replay_size':len(replay),
                  'resume/replay_refill_steps':max(0,learning_after-step)})
            while step<args.total_timesteps:
                if stopped['value'] or time.monotonic()>=deadline:
                    status='paused_for_allocation' if not stopped['value'] else 'stopped_by_signal'
                    break
                sync_actor(agent,rollout)
                count=min(args.collect_steps,args.total_timesteps-step)
                for interval in (args.log_every_steps,args.eval_every_steps,args.checkpoint_every_steps):
                    if interval: count=min(count,interval-step%interval)
                if step<learning_after: count=min(count,learning_after-step)
                rows=[]
                for _ in range(count):
                    if not resume_path and step<args.learning_starts:
                        action=env.action_space.sample()
                    else:
                        with torch.no_grad():
                            if z is None or skill_age>=args.update_z_every_steps:
                                z=rollout.sample_z(1,device='cpu'); skill_age=0
                            action=rollout.act(torch.from_numpy(state[None]),z,mean=False).numpy()[0]
                        skill_age+=1
                    obs,reward,terminated,truncated,info=env.step(action)
                    next_state=policy_obs(obs,action,args.use_last_action)
                    rows.append(transition(state,action,next_state,reward,terminated,truncated,info,
                                           env.current_state['paddles']['paddle_ego']['position']))
                    state=next_state; ep_return+=float(reward); ep_length+=1; step+=1
                    if terminated or truncated:
                        returns.append(ep_return); lengths.append(ep_length)
                        obs,_=env.reset()
                        state=policy_obs(obs,np.zeros_like(action),args.use_last_action)
                        ep_return=0.; ep_length=0; z=None; skill_age=0
                replay.extend(stack_transitions(rows))
                # Exactly one update per eligible collected transition. Warmup
                # boundary itself is included, matching the original trainer.
                eligible=max(0,step-max(step-count,learning_after-1))
                if eligible and len(replay)>=args.batch_size:
                    if update is None: update=FastFBUpdate(agent,replay,use_graph=args.use_graph)
                    n_updates=eligible*args.updates_per_step
                    metrics=update.update(n_updates); updates+=n_updates
                    values=torch.stack([v.detach().mean() for v in metrics.values()])
                    if not torch.isfinite(values).all().item():
                        raise FloatingPointError(f'Non-finite learner metrics at step {step}')
                if step%args.log_every_steps==0 or step==args.total_timesteps:
                    emit({**{f'train/{k}':float(v.detach().mean()) for k,v in metrics.items()},
                          'replay/size':len(replay),'time/steps_per_second':(step-initial_step)/(time.monotonic()-start),
                          'rollout/return_mean':float(np.mean(returns)) if returns else 0.,
                          'rollout/length_mean':float(np.mean(lengths)) if lengths else 0.})
                if args.checkpoint_every_steps and step%args.checkpoint_every_steps==0:
                    checkpoint=save_continuation(agent,replay,args,sim_config,output,step,updates,'periodic')
                if args.eval_every_steps and step%args.eval_every_steps==0:
                    with isolated_rng(args.seed+200000):
                        # Evaluate a fresh CPU snapshot; graph-owned model tensors stay intact.
                        evaluator=copy.deepcopy(agent._model).to('cpu').eval().requires_grad_(False)
                        batch=replay.sample(min(len(replay),args.inference_samples))
                        from scripts.fb.fast_checkpoint import map_tensors
                        batch=map_tensors(batch,lambda t:t.cpu())
                        gif_dir=output/'gifs'/f'step_{step:09d}' if args.gif_episodes else None
                        emit(evaluate_tasks(evaluator,batch,sim_config,replace(base_args(args),device='cpu'),gif_dir,step))
                        if tracker and gif_dir: tracker.log_gifs(gif_dir,step)
                        del evaluator,batch
            if step>=args.total_timesteps: status='completed'
            if len(replay):
                checkpoint=save_continuation(agent,replay,args,sim_config,output,step,updates,status)
            (output/'status.json').write_text(json.dumps({'status':status,'step':step,'updates':updates,
                'target_steps':args.total_timesteps,'checkpoint':str(checkpoint) if checkpoint else None},indent=2))
            return checkpoint
    except Exception as exc:
        (output/'status.json').write_text(json.dumps({'status':'failed','step':step,
            'updates':updates,'error':str(exc)},indent=2))
        raise
    finally:
        env.close()
        for sig,handler in old_handlers.items(): signal.signal(sig,handler)
        if writer: writer.close()
        if tracker: tracker.finish(status!='failed')


def train(args):
    # An allocation handoff must never create two writers for one run.
    import fcntl
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.trainer.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another trainer owns {output}') from exc
        return _train_locked(args)


def main():
    import argparse
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--args-file')
    known,remaining=parser.parse_known_args()
    defaults=FastArgs()
    if known.args_file:
        with open(known.args_file) as f: defaults=FastArgs(**yaml.safe_load(f))
    train(tyro.cli(FastArgs,default=defaults,args=remaining))

if __name__=='__main__': main()
