from stable_baselines3 import PPO 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3 import HerReplayBuffer, SAC
from airhockey import AirHockeyEnv
from airhockey.renderers.render import AirHockeyRenderer
import argparse
import yaml
import os
import random
import wandb
import argparse
import os
import yaml
from utils import CustomCallback, save_evaluation_gifs

            
def train_air_hockey_model(air_hockey_cfg, use_wandb=False, device='cpu'):
    """
    Train an air hockey paddle model using stable baselines.

    This script loads the configuration file, creates an AirHockey2D environment,
    wraps the environment with necessary components, trains the model,
    and saves the trained model and environment statistics.
    """
    
    air_hockey_params = air_hockey_cfg['air_hockey']
    air_hockey_params['n_training_steps'] = air_hockey_cfg['n_training_steps']
    
    air_hockey_params_cp = air_hockey_params.copy()
    air_hockey_params_cp['seed'] = 42
    air_hockey_params_cp['max_timesteps'] = 200
    eval_env = AirHockeyEnv.from_dict(air_hockey_params_cp)
    
    if type(air_hockey_cfg['seed']) is not list:
        seeds = [int(air_hockey_cfg['seed'])]
    else:
        seeds = [int(s) for s in air_hockey_cfg['seed']]
        del air_hockey_cfg['seed'] # otherwise it will be saved in the model cfg when copied over
        
    # Train different seeds. If one seed in config, this is just one iteration.
    for seed in seeds:
        air_hockey_cfg['seed'] = seed # since it it used as training seed
        air_hockey_params['seed'] = seed # and environment seed
        
        wandb_run = None
        if use_wandb:
            wandb_run = wandb.init(
                project="air-hockey",
                config=air_hockey_cfg,
                sync_tensorboard=True,
                save_code=True)
        
        if air_hockey_cfg['n_threads'] > 1:

            # set seed for reproducibility
            seed = air_hockey_params['seed']
            random.seed(seed)
            
            # get number of threads
            n_threads = air_hockey_cfg['n_threads']
            
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
                    env = AirHockeyEnv.from_dict(air_hockey_params)
                    return Monitor(env)
                return _init()

            # check_env(env)
            env = SubprocVecEnv([get_airhockey_env_for_parallel for _ in range(n_threads)])
            env = VecNormalize(env) # probably something to try when tuning
        else:
            env = AirHockeyEnv.from_dict(air_hockey_params)
            def wrap_env(env):
                wrapped_env = Monitor(env) # needed for extracting eprewmean and eplenmean
                wrapped_env = DummyVecEnv([lambda: wrapped_env]) # Needed for all environments (e.g. used for multi-processing)
                # wrapped_env = VecNormalize(wrapped_env) # probably something to try when tuning
                return wrapped_env
            env = wrap_env(env)

        os.makedirs(air_hockey_cfg['tb_log_dir'], exist_ok=True)
        log_parent_dir = os.path.join(air_hockey_cfg['tb_log_dir'], air_hockey_cfg['air_hockey']['task'])
        os.makedirs(log_parent_dir, exist_ok=True)
        
        # determine the actual log dir
        subdirs = [x for x in os.listdir(log_parent_dir) if os.path.isdir(os.path.join(log_parent_dir, x))]
        subdir_nums = [int(x.split(air_hockey_cfg['tb_log_name'] + '_')[1]) for x in subdirs]
        next_num = max(subdir_nums) + 1 if subdir_nums else 1
        log_dir = os.path.join(log_parent_dir, air_hockey_cfg['tb_log_name'] + f'_{next_num}')
        
        callback = CustomCallback(eval_env)
        
        # if goal-conditioned use SAC
        if 'goal' in air_hockey_cfg['air_hockey']['task']:
            # SAC hyperparams:
            # Create 4 artificial transitions per real transitionair_hockey_simulator
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
                seed=seed,
                device=device,
            )
        else:
            model = PPO("MlpPolicy", env, verbose=1, 
                    tensorboard_log=log_parent_dir, 
                    device=device, 
                    seed=seed,
                    # batch_size=512,
                    #n_epochs=5,
                    gamma=air_hockey_cfg['gamma']) 
        
        model.learn(total_timesteps=air_hockey_cfg['n_training_steps'],
                    tb_log_name=air_hockey_cfg['tb_log_name'], 
                    callback=callback,
                    progress_bar=True)
        
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
        env_test = AirHockeyEnv.from_dict(air_hockey_params)
        renderer = AirHockeyRenderer(env_test)
        
        env_test = DummyVecEnv([lambda : env_test])
        # env_test = VecNormalize.load(os.path.join(log_dir, air_hockey_cfg['vec_normalize_save_filepath']), env_test)
        
        # if goal-conditioned use SAC
        if 'goal' in air_hockey_cfg['air_hockey']['task']:
            model = SAC.load(model_filepath, env=env_test)
        else:
            model = PPO.load(model_filepath)

        # first let's create some videos offline into gifs
        print("Saving gifs...(this will tqdm for EACH gif to save)")
        save_evaluation_gifs(5, 3, env_test, model, renderer, log_dir, use_wandb, wandb_run)
        
        env_test.close()

        if use_wandb:
            wandb_run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
    parser.add_argument('--cfg', type=str, default=None, help='Path to the configuration file.')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training.')
    args = parser.parse_args()
    
    if args.cfg is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        air_hockey_cfg_fp = os.path.join(dir_path, '../configs', 'default_train_puck_vel.yaml')
    else:
        air_hockey_cfg_fp = args.cfg
    
    with open(air_hockey_cfg_fp, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)
    
    use_wandb = args.wandb
    device = args.device
    
    train_air_hockey_model(air_hockey_cfg, use_wandb, device)
