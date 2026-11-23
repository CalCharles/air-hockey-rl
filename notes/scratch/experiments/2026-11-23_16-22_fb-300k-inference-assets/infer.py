import json
from pathlib import Path
from dataclasses import replace
import torch
from scripts.fb.fb_training import Args, infer_task_z, task_rewards, evaluate_tasks
from scripts.fb.models import load_policy

torch.set_num_threads(2)
root=Path('runs/fb/reward_inference_300k_20261123')
summary={}
for variant in ['full','current']:
    run_dir=Path('runs/fb')/f'realplant_{variant}_50M'
    checkpoint=Path(json.loads((run_dir/'latest_checkpoint.json').read_text())['path'])
    assert (checkpoint/'COMPLETE').exists()
    run=json.loads((checkpoint/'run.json').read_text())
    args=replace(Args(**run['args']),device='cpu',inference_samples=300000,tensorboard=False,wandb=False,gif_episodes=0)
    model=load_policy(checkpoint/'policy',device='cpu')
    replay=torch.load(checkpoint/'continuation.pt',map_location='cpu',weights_only=True)['replay']
    generator=torch.Generator().manual_seed(args.seed+200000)
    indices=torch.randint(len(replay['next']['observation']),(300000,),generator=generator)
    batch={'next':{k:v[indices] for k,v in replay['next'].items()}}
    dest=root/variant
    dest.mkdir(exist_ok=True)
    with torch.no_grad():
        latents={t:infer_task_z(model,batch,t,args).cpu() for t in ['juggle','touch','reach']}
        for z in latents.values():
            assert z.shape==(1,50) and torch.isfinite(z).all()
            torch.testing.assert_close(z.norm(),torch.tensor(50**.5),atol=1e-5,rtol=1e-5)
        torch.save(latents,dest/'reward_z_by_task.pt')
        torch.save(batch,dest/'inference_samples.pt')
        metadata={'checkpoint':str(checkpoint),'step':run['step'],'inference_samples':len(indices),'seed':args.seed+200000,'sampling':'uniform with replacement from frozen replay','latent_norms':{t:float(z.norm()) for t,z in latents.items()},'positive_samples':{t:int((task_rewards(batch,t,args)>0).sum()) for t in latents}}
        (dest/'inference.json').write_text(json.dumps(metadata,indent=2))
        print(variant,'300k inference complete',json.dumps(metadata),flush=True)
        metrics=evaluate_tasks(model,batch,run['sim_config'],args)
        (dest/'evaluation.json').write_text(json.dumps(metrics,indent=2))
        summary[variant]={**metadata,'evaluation':metrics}
        print(variant,'evaluation complete',json.dumps(metrics),flush=True)
(root/'summary.json').write_text(json.dumps(summary,indent=2))
