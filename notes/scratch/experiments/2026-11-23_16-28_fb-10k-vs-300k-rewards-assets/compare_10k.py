import json
from pathlib import Path
from dataclasses import replace
import torch
from scripts.fb.fb_training import Args, infer_task_z, evaluate_tasks
from scripts.fb.models import load_policy

torch.set_num_threads(2)
root=Path('runs/fb/reward_inference_300k_20261123')
summary=json.loads((root/'summary.json').read_text())
comparison={}
for variant in ['full','current']:
    reference=summary[variant]
    checkpoint=Path(reference['checkpoint'])
    run=json.loads((checkpoint/'run.json').read_text())
    args=replace(Args(**run['args']),device='cpu',inference_samples=10000,tensorboard=False,wandb=False,gif_episodes=0)
    model=load_policy(checkpoint/'policy',device='cpu')
    saved=torch.load(root/variant/'inference_samples.pt',map_location='cpu',weights_only=True)
    batch={'next':{k:v[:10000] for k,v in saved['next'].items()}}
    with torch.no_grad():
        latents={t:infer_task_z(model,batch,t,args).cpu() for t in ['juggle','touch','reach']}
        prior=torch.load(root/variant/'reward_z_by_task.pt',map_location='cpu',weights_only=True)
        angles={t:float(torch.rad2deg(torch.acos(torch.nn.functional.cosine_similarity(latents[t],prior[t]).clamp(-1,1)))) for t in latents}
        metrics=evaluate_tasks(model,batch,run['sim_config'],args)
    torch.save(latents,root/variant/'reward_z_10k_by_task.pt')
    comparison[variant]={'checkpoint':str(checkpoint),'step':run['step'],'eval_episodes':args.eval_episodes,'eval_seed_start':args.seed+100000,'sampling_seed':reference['seed'],'sampling':'10k prefix of the saved 300k sample; uniform with replacement','latent_angle_degrees':angles,'10k':metrics,'300k':reference['evaluation']}
    print(variant,json.dumps(comparison[variant]),flush=True)
(root/'comparison_10k_300k.json').write_text(json.dumps(comparison,indent=2))
