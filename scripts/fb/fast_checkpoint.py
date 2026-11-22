"""Atomic continuation checkpoints with replay; reset the simulator on restart."""
from dataclasses import asdict, fields
import json
from pathlib import Path
import random
import shutil
import uuid

import numpy as np
import torch
import safetensors.torch
from metamotivo.buffers.buffers import DictBuffer
from scripts.fb.fb_training import Args, create_agent, save_checkpoint


def base_args(args):
    return Args(**{f.name: getattr(args, f.name) for f in fields(Args)})


def map_tensors(data, fn):
    return {k: map_tensors(v, fn) if isinstance(v, dict) else fn(v) for k,v in data.items()}


def load_training(checkpoint, args):
    checkpoint = Path(checkpoint)
    run = json.loads((checkpoint/'run.json').read_text())
    if run['args'].get('backward_input', 'full') != args.backward_input:
        raise ValueError('Checkpoint backward_input does not match requested architecture')
    for key in ('action_latency_steps', 'use_last_action', 'discount', 'batch_size', 'norm_obs', 'train_goal_ratio', 'actor_std'):
        if run['args'][key] != getattr(args,key):
            raise ValueError(f'Resume changes {key}; explicit migration is required')
    cfg = json.loads((checkpoint/'config.json').read_text())
    agent = create_agent(args, cfg['model']['obs_dim'], cfg['model']['action_dim'])
    safetensors.torch.load_model(agent._model, checkpoint/'model/model.safetensors', device=args.device)
    state = torch.load(checkpoint/'optimizers.pth', map_location=args.device, weights_only=True)
    for name in ('forward_optimizer','backward_optimizer','actor_optimizer'):
        getattr(agent,name).load_state_dict(state[name])
    replay = DictBuffer(args.buffer_size, device=args.device)
    resume = None
    if (checkpoint/'continuation.pt').exists():
        resume = torch.load(checkpoint/'continuation.pt', map_location='cpu', weights_only=True)
        if resume['capacity'] != args.buffer_size:
            raise ValueError('Cannot change replay capacity on continuation')
        replay.extend(resume['replay'])
        replay._idx, replay._is_full = resume['idx'], resume['full']
    return agent, replay, run, resume


def restore_rng(state):
    random.setstate(state['python_rng'])
    name, values, pos, has_gauss, cached = state['numpy_rng']
    np.random.set_state((name, np.asarray(values,dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state['torch_rng'])
    if torch.cuda.is_available() and state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])


def save_continuation(agent, replay, args, sim_config, output, step, updates, reason):
    output = Path(output)
    checkpoints = output/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    final = checkpoints/f'step_{step:09d}'
    if final.exists():
        return final
    temporary = checkpoints/f'.saving_{uuid.uuid4().hex}'
    batch = save_checkpoint(agent, replay, base_args(args), sim_config, temporary, step, updates)
    np_rng = np.random.get_state()
    state = {'capacity':replay.capacity, 'idx':replay._idx, 'full':replay._is_full,
             'replay':map_tensors(replay.get_full_buffer(), lambda t:t.detach().cpu()),
             'python_rng':random.getstate(), 'numpy_rng':(np_rng[0],np_rng[1].tolist(),*np_rng[2:]),
             'torch_rng':torch.get_rng_state(),
             'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    torch.save(state, temporary/'continuation.pt')
    (temporary/'fast_args.json').write_text(json.dumps(asdict(args),indent=2))
    (temporary/'COMPLETE').write_text(reason+'\n')
    temporary.rename(final)
    pointer = output/'latest_checkpoint.tmp'
    pointer.write_text(json.dumps({'path':str(final.absolute()),'step':step,'updates':updates,
                                   'reason':reason,'target_steps':args.total_timesteps},indent=2))
    pointer.replace(output/'latest_checkpoint.json')
    # Retain small inference exports at each checkpoint, but only the last two
    # full replay/optimizer snapshots. 50M steps must not create tens of GB of replay.
    policy = output/'policies'/f'step_{step:09d}'
    policy.mkdir(parents=True, exist_ok=False)
    shutil.copytree(final/'policy', policy/'policy')
    for filename in ('run.json','inference_samples.pt','reward_z.pt'):
        shutil.copyfile(final/filename, policy/filename)
    complete = sorted(p for p in checkpoints.glob('step_*') if (p/'COMPLETE').exists())
    for stale in complete[:-2]:
        shutil.rmtree(stale)
    return final
