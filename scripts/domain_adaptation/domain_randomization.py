import os
import torch
import yaml
from torch import nn
import numpy as np
import time
import random
from airhockey import AirHockeyEnv
from airhockey.renderers.render import AirHockeyRenderer
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs, VecEnvObs, VecEnvStepReturn
from stable_baselines3 import HerReplayBuffer, SAC
from curriculum.classifier_curriculum import CurriculumCallback
from utils import EvalCallback
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union
from typing import Callable
from utils import EvalCallback, save_evaluation_gifs, save_tensorboard_plots
import argparse
import wandb
import shutil

class SubprocVecEnv_domain_random(SubprocVecEnv):
    def reset(self, seed=None, **kwargs) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        results = [remote.recv() for remote in self.remotes]
        obs, self.reset_infos = zip(*results)  # type: ignore[assignment]
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return _flatten_obs(obs, self.observation_space), self.reset_infos

class SubprocVecEnv_domain_random_eval(SubprocVecEnv_domain_random):
    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        super().__init__(env_fns, start_method)
        self.finished_envs = [False] * len(env_fns)
        self.results = [None] * len(env_fns)

    def step_async(self, actions: np.ndarray) -> None:
        for index, (remote, action) in enumerate(zip(self.remotes, actions)):
            if not self.finished_envs[index]:
                remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        for index, remote in enumerate(self.remotes):
            if not self.finished_envs[index]:
                self.results[index] = remote.recv()
                if self.results[index][2]:
                    self.finished_envs[index] = True
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*self.results)  # type: ignore[assignment]
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos  # type: ignore

def make_env(air_hockey_cfg):
    air_hockey_params = air_hockey_cfg['air_hockey']
    air_hockey_params['n_training_steps'] = air_hockey_cfg['n_training_steps']

    if 'sac' == air_hockey_cfg['algorithm']:
        if 'goal' in air_hockey_cfg['air_hockey']['task']:
            air_hockey_cfg['air_hockey']['return_goal_obs'] = True
        else:
            air_hockey_cfg['air_hockey']['return_goal_obs'] = False
    else:
        air_hockey_cfg['air_hockey']['return_goal_obs'] = False
    
    def get_airhockey_env_for_parallel():
        """
        Utility function for multiprocessed env.

        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environments you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            curr_seed = random.randint(0, int(1e8))
            air_hockey_params['seed'] = curr_seed
            # Note: With this seed, an individual rng is created for each env
            # It does not affect the global rng!
            env = AirHockeyEnv(air_hockey_params)
            return Monitor(env)
        return _init()

    # set seed for reproducibility
    seed = air_hockey_params['seed']
    random.seed(seed)
    # get number of threads
    n_threads = air_hockey_cfg['n_threads']
    env = SubprocVecEnv_domain_random([get_airhockey_env_for_parallel for _ in range(n_threads)])
    return env


def train(args, use_wandb, device, clear_prior_task_results, progress_bar=False):
    if args.cfg is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        air_hockey_cfg_fp = os.path.join(dir_path, '../configs', 'default_train_puck_vel.yaml')
    else:
        air_hockey_cfg_fp = args.cfg
    
    with open(air_hockey_cfg_fp, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)
        
    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            project="air_hockey_rl", 
            entity="maxrudolph",
            config=air_hockey_cfg,
            sync_tensorboard=True,
            save_code=True)

    file_path = os.path.dirname(os.path.realpath(__file__))
    wandb.run.log_code(os.path.join(file_path, '..'), name="Codebase", include_fn=lambda s: s.endswith('.py'))
        
    env = make_env(air_hockey_cfg)
    
    # if goal-conditioned use SAC
    if 'sac' == air_hockey_cfg['algorithm']:
        # SAC hyperparams:
        # Create 4 artificial transitions per real transition air_hockey_simulator
        import pdb; pdb.set_trace()
        n_sampled_goal = 4
        model = SAC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=n_sampled_goal,
                goal_selection_strategy="future",
            ),
            learning_starts=10000,
            verbose=1,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95,
            batch_size=512,
            tensorboard_log=log_parent_dir,
            seed=air_hockey_cfg['seed'],
            device=device,
        )
    else:
        # import pdb; pdb.set_trace()
        model = PPO("MlpPolicy", env, verbose=1, 
                tensorboard_log=log_parent_dir, 
                device=device, 
                seed=air_hockey_cfg['seed'],
                # batch_size=512,
                #n_epochs=5,
                gamma=air_hockey_cfg['gamma']) 
        
    eval_env = make_env(air_hockey_cfg)
    
    os.makedirs(air_hockey_cfg['tb_log_dir'], exist_ok=True)
    log_parent_dir = os.path.join(air_hockey_cfg['tb_log_dir'], air_hockey_cfg['air_hockey']['task'])
    if clear_prior_task_results and os.path.exists(log_parent_dir):
        shutil.rmtree(log_parent_dir)
    os.makedirs(log_parent_dir, exist_ok=True)
    
    # determine the actual log dir
    subdirs = [x for x in os.listdir(log_parent_dir) if os.path.isdir(os.path.join(log_parent_dir, x))]
    subdir_nums = [int(x.split(air_hockey_cfg['tb_log_name'] + '_')[1]) for x in subdirs]
    next_num = max(subdir_nums) + 1 if subdir_nums else 1
    log_dir = os.path.join(log_parent_dir, air_hockey_cfg['tb_log_name'] + f'_{next_num}')
        
    if 'curriculum' in air_hockey_cfg.keys() and len(air_hockey_cfg['curriculum']['model']) > 0:
        callback = CurriculumCallback(eval_env, 
                                        curriculum_config=air_hockey_cfg['curriculum'], 
                                        log_dir=log_dir, 
                                        n_eval_eps=air_hockey_cfg['n_eval_eps'], 
                                        eval_freq=air_hockey_cfg['eval_freq'])
    else:
        callback = EvalCallback(eval_env, 
                                log_dir=log_dir, 
                                n_eval_eps=air_hockey_cfg['n_eval_eps'], 
                                eval_freq=air_hockey_cfg['eval_freq'])
    
    # import pdb; pdb.set_trace()
    model.learn(total_timesteps=air_hockey_cfg['n_training_steps'],
                tb_log_name=air_hockey_cfg['tb_log_name'], 
                callback=callback,
                progress_bar=progress_bar)
    
    os.makedirs(log_parent_dir, exist_ok=True)
    
    # let's save model and vec normalize here too
    model_filepath = os.path.join(log_dir, air_hockey_cfg['model_save_filepath'])
    env_filepath = os.path.join(log_dir, air_hockey_cfg['vec_normalize_save_filepath'])
    # copy cfg to same folder
    cfg_filepath = os.path.join(log_dir, 'model_cfg.yaml')
    with open(cfg_filepath, 'w') as f:
        yaml.dump(air_hockey_cfg, f)

    model.save(model_filepath)
    # env.save(env_filepath)
    
    # let's also evaluate the policy and save the results!
    air_hockey_cfg['air_hockey']['max_timesteps'] = 200
    
    air_hockey_params = air_hockey_cfg['air_hockey']
    air_hockey_params['n_training_steps'] = air_hockey_cfg['n_training_steps']
    env_test = AirHockeyEnv(air_hockey_params)
    renderer = AirHockeyRenderer(env_test)
    
    env_test = DummyVecEnv([lambda : env_test])
    # env_test = VecNormalize.load(os.path.join(log_dir, air_hockey_cfg['vec_normalize_save_filepath']), env_test)
    
    # if goal-conditioned use SAC
    if 'sac' == air_hockey_cfg['algorithm']:
        model = SAC.load(model_filepath, env=env_test)
    else:
        model = PPO.load(model_filepath)

    # first let's create some videos offline into gifs
    print("Saving gifs...(this will tqdm for EACH gif to save)")
    save_evaluation_gifs(5, 3, env_test, model, renderer, log_dir, use_wandb, wandb_run)
    save_tensorboard_plots(log_dir, air_hockey_cfg)
    
    env_test.close()

    if use_wandb:
        wandb_run.finish()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain randomization for air hockey.')
    parser.add_argument('--cfg', type=str, default=None, help='Path to the configuration file.')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training.')
    
    args = parser.parse_args()
    
    
    use_wandb = args.wandb
    device = args.device
    clear_prior_task_results = args.clear
    train(args, use_wandb, device, clear_prior_task_results, progress_bar=False)
    
    
    