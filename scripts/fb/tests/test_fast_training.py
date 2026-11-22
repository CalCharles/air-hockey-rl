import copy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch
from metamotivo.buffers.buffers import DictBuffer
from scripts.fb.fb_training import Args, create_agent
from scripts.fb.fast_update import FastFBUpdate
from scripts.fb.fast_training import FastArgs, train
from scripts.fb.fast_checkpoint import load_training


def sample_data(n=16):
    rng=np.random.default_rng(3)
    return {'observation':rng.normal(size=(n,32)).astype('float32'),
            'action':rng.uniform(-1,1,(n,2)).astype('float32'),
            'next':{'observation':rng.normal(size=(n,32)).astype('float32'),
                    'terminated':(rng.uniform(size=(n,1))>.8)}}


@pytest.mark.parametrize('mode',['full','current'])
@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required'))])
def test_fast_math_matches_upstream(mode,device):
    torch.set_num_threads(1)
    cfg=Args(device=device,backward_input=mode,batch_size=8,z_dim=8,hidden_dim=32,backward_hidden_dim=32)
    agent=create_agent(cfg,32,2)
    other=copy.deepcopy(agent)
    replay=DictBuffer(16,device=device); replay.extend(sample_data())
    batch=replay.sample(8)
    class Fixed:
        def sample(self,n): return batch
    updater=FastFBUpdate(other,replay,use_graph=False)
    torch.manual_seed(17)
    reference=agent.update({'train':Fixed()},0)
    torch.manual_seed(17)
    actual=updater.step_batch(batch)
    for k,v in reference.items(): torch.testing.assert_close(actual[k],v)
    for k,v in agent._model.state_dict().items(): torch.testing.assert_close(other._model.state_dict()[k],v)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('mode',['full','current'])
def test_graph_capture_restores_state_and_matches_deterministic_updates(mode):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    cfg=Args(device='cuda',backward_input=mode,batch_size=8,z_dim=8,hidden_dim=32,backward_hidden_dim=32,actor_std=0)
    eager=create_agent(cfg,32,2); graphed=create_agent(cfg,32,2)
    graphed._model.load_state_dict(eager._model.state_dict())
    replay=DictBuffer(16,device='cuda'); replay.extend(sample_data())
    batch=replay.sample(8)
    z=eager._model.sample_z(8,device='cuda')
    eager.sample_mixed_z=lambda **kwargs:z
    graphed.sample_mixed_z=lambda **kwargs:z
    a=FastFBUpdate(eager,replay,use_graph=False)
    b=FastFBUpdate(graphed,replay,use_graph=True)
    a.sample=lambda:batch; b.sample=lambda:batch
    before={k:v.clone() for k,v in graphed._model.state_dict().items()}
    b.capture()
    for k,v in before.items(): torch.testing.assert_close(graphed._model.state_dict()[k],v,rtol=0,atol=0)
    for opt in b.optimizers:
        assert all(state['step'].item()==0 for state in opt.state.values())
    a.update(3); b.update(3); torch.cuda.synchronize()
    for k,v in eager._model.state_dict().items():
        torch.testing.assert_close(graphed._model.state_dict()[k],v,atol=2e-5,rtol=2e-4)
    for opt in b.optimizers:
        assert all(state['step'].item()==3 for state in opt.state.values())


@pytest.mark.parametrize('mode',['full','current'])
def test_replay_and_optimizer_continuation(tmp_path,mode):
    args=FastArgs(device='cpu',use_graph=False,backward_input=mode,
                  output_dir=str(tmp_path/'run'),total_timesteps=40,collect_steps=4,
                  learning_starts=8,batch_size=8,buffer_size=24,hidden_dim=32,
                  backward_hidden_dim=32,z_dim=8,eval_every_steps=0,
                  checkpoint_every_steps=20,log_every_steps=20,tensorboard=False,wandb=False)
    checkpoint=train(args)
    agent,replay,run,state=load_training(checkpoint,args)
    assert run['step']==40 and run['updates']==33
    assert len(replay)==24 and replay._idx==16 and replay._is_full
    assert (checkpoint/'COMPLETE').is_file()
    assert agent.actor_optimizer.state[next(iter(agent._model._actor.parameters()))]['step']==33
    stored=state['replay']['observation'].clone()
    torch.testing.assert_close(replay.storage['observation'],stored)
    resumed=train(replace(args,resume_latest=True,total_timesteps=64))
    _,_,end,_=load_training(resumed,args)
    assert end['step']==64 and end['updates']==57
    assert len(list((tmp_path/'run/checkpoints').glob('step_*')))==2
    assert json.loads((tmp_path/'run/status.json').read_text())['status']=='completed'


def test_second_writer_is_refused(tmp_path):
    import fcntl
    output=tmp_path/'locked'
    output.mkdir()
    with (output/'.trainer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='Another trainer'):
            train(FastArgs(output_dir=str(output)))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_graph_sampling_uses_fresh_randomness_and_current_replay_size():
    torch.set_num_threads(1)
    cfg=Args(device='cuda',batch_size=8,z_dim=8,hidden_dim=32,backward_hidden_dim=32)
    agent=create_agent(cfg,32,2)
    replay=DictBuffer(16,device='cuda'); replay.extend(sample_data())
    update=FastFBUpdate(agent,replay)
    update.update()
    first=update.last_indices.clone()
    update.update()
    assert not torch.equal(first,update.last_indices)
    replay._idx=8; replay._is_full=False
    update.update()
    assert update.last_indices.max().item()<8


def test_allocation_stop_saves_replay(tmp_path):
    cfg=FastArgs(device='cpu',use_graph=False,output_dir=str(tmp_path/'source'),
                 total_timesteps=12,collect_steps=4,learning_starts=8,batch_size=8,
                 buffer_size=24,hidden_dim=32,backward_hidden_dim=32,z_dim=8,
                 eval_every_steps=0,checkpoint_every_steps=0,tensorboard=False)
    saved=train(cfg)
    paused=replace(cfg,output_dir=str(tmp_path/'paused'),resume_from=str(saved),
                   total_timesteps=100,max_runtime_seconds=0)
    checkpoint=train(paused)
    status=json.loads((tmp_path/'paused/status.json').read_text())
    assert status['status']=='paused_for_allocation' and status['step']==12
    assert (checkpoint/'continuation.pt').exists()
    _,replay,_,_=load_training(checkpoint,paused)
    assert len(replay)==12
