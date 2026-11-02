"""Isolated fresh-history TD3 entrypoint with a reward-only PID-target cost.

The ordinary trainer, evaluator and simulator files remain unchanged. Only the
training process's environment factory is wrapped. Evaluators retain their own
unmodified environment factories and the original unshaped task reward.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import yaml
from airhockey import AirHockeyEnv
from scripts.td3.juggle_noisy_scratch import assert_canonical_scratch_env, scratch_trainer
from scripts.td3.juggle_target_cost import TargetStepCost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--args-file',type=Path,required=True)
    parser.add_argument('--coefficient',type=float,required=True)
    cli = parser.parse_args()
    args = yaml.safe_load(cli.args_file.read_text())
    if scratch_trainer(args) != 'scripts.rma.train_base_policy' or args.get('num_envs') != 1:
        raise ValueError('This experiment requires a single fresh history learner')
    if args.get('actor_action_delta_penalty',0) != 0:
        raise ValueError('Isolate the reward cost from the existing actor penalty')
    config = yaml.safe_load(Path(args['config']).read_text())['air_hockey']
    assert_canonical_scratch_env(config)
    if config.get('return_goal_obs',False):
        raise ValueError('Target cost is not implemented for HER goal relabeling')
    if not args.get('log_parent_dir'):
        raise ValueError('An explicit new log_parent_dir is required')
    train_dir = Path(args['log_parent_dir']).resolve()
    if train_dir.exists():
        raise FileExistsError(train_dir)
    # Validate the scalar before creating any output or constructing a simulator.
    TargetStepCost(None,cli.coefficient)
    train_dir.parent.mkdir(parents=True,exist_ok=True)
    metadata = dict(coefficient=cli.coefficient,definition='reward - coefficient * norm(cached filtered PID target change)^4',
                    reset_edge='First commanded target after reset has zero cost',
                    stationary=True, from_scratch=True, raw_evaluation_reward=True,
                    args_file=str(cli.args_file.resolve()),
                    args_sha256=hashlib.sha256(cli.args_file.read_bytes()).hexdigest(),
                    code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__),Path('scripts/td3/juggle_target_cost.py'))})
    with (train_dir.parent/'target_cost_recipe.json').open('x') as f:
        json.dump(metadata,f,indent=2); f.write('\n')
    fields=['step','block_steps','original_reward_sum','shaped_reward_sum','cost_sum','target_step_p95_m','target_step_max_m']
    with (train_dir.parent/'target_cost_training.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); stream.flush()

        class LoggedCost(TargetStepCost):
            def __init__(self,env):
                super().__init__(env,cli.coefficient)
                self.steps=0
                self.block=[]

            def flush(self):
                if not self.block:
                    return
                values=np.asarray(self.block)
                writer.writerow(dict(step=self.steps,block_steps=len(values),original_reward_sum=values[:,0].sum(),
                                     shaped_reward_sum=values[:,1].sum(),cost_sum=values[:,2].sum(),
                                     target_step_p95_m=np.percentile(values[:,3],95),target_step_max_m=values[:,3].max()))
                stream.flush(); self.block=[]

            def step(self,action):
                result=super().step(action)
                info=result[-1]; self.steps+=1
                self.block.append((info['original_reward'],result[1],info['target_step_cost'],info['target_step_m']))
                if self.steps%1000==0:
                    self.flush()
                return result

        from scripts.rma import train_base_policy as trainer
        original_factory=trainer.AirHockeyEnv
        instances=[]

        def make_training_env(cfg):
            env=LoggedCost(AirHockeyEnv(cfg)); instances.append(env)
            return env

        trainer.AirHockeyEnv=make_training_env
        old_argv=sys.argv
        sys.argv=[str(Path(trainer.__file__)), '--args-file',str(cli.args_file.resolve())]
        try:
            trainer._entrypoint()
        finally:
            for env in instances:
                env.flush()
            sys.argv=old_argv
            trainer.AirHockeyEnv=original_factory


if __name__ == '__main__':
    main()
