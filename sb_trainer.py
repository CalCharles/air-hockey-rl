from stable_baselines3 import PPO 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.env_checker import check_env
from matplotlib import pyplot as plt
from airhockey import AirHockeyEnv
from render import AirHockeyRenderer
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import argparse
import yaml
import os
import re
import time
import imageio
import cv2
import tqdm
import random
import wandb
from stable_baselines3.common.callbacks import BaseCallback


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, eval_env, eval_freq=5000, n_eval_eps=30, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_eps = n_eval_eps
        self.next_eval = 0
        self.best_success_so_far = 0.0
    
    def _eval(self):
        avg_undiscounted_return = 0.0
        avg_success_rate = 0.0
        for _ in range(self.n_eval_eps):
            obs, info = self.eval_env.reset()
            done = False
            undiscounted_return = 0.0
            success = False
            while not done:
                action, _ = self.model.predict(obs)
                obs, rew, done, truncated, info = self.eval_env.step(action)
                done = done or truncated
                undiscounted_return += rew
                assert 'success' in info
                assert (info['success'] is True) or (info['success'] is False)
                if info['success'] is True:
                    success = True
            avg_undiscounted_return += undiscounted_return
            avg_success_rate += 1.0 if success else 0.0
        avg_undiscounted_return /= self.n_eval_eps
        avg_success_rate /= self.n_eval_eps
        return avg_undiscounted_return, avg_success_rate

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        if self.num_timesteps >= self.next_eval:
            avg_undiscounted_return, avg_success_rate = self._eval()
            self.logger.record("eval/ep_return", avg_undiscounted_return)
            self.logger.record("eval/success_rate", avg_success_rate)
            if avg_success_rate > self.best_success_so_far:
                self.best_success_so_far = avg_success_rate
            self.logger.record("eval/best_success_rate", self.best_success_so_far)
            self.next_eval += self.eval_freq

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        return True

    # def _on_rollout_end(self) -> None:
    #     """
    #     This event is triggered before updating the policy.
    #     """
    #     pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        avg_undiscounted_return, avg_success_rate = self._eval()
        self.logger.record("eval/ep_return", avg_undiscounted_return)
        self.logger.record("eval/success_rate", avg_success_rate)

def train_air_hockey_model(air_hockey_cfg):
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
        
    for seed in seeds:
        air_hockey_cfg['seed'] = seed # since it it used as training seed
        air_hockey_params['seed'] = seed # and environment seed
        
        # wandb_run = wandb.init(
        #     project="air-hockey",
        #     config=air_hockey_cfg,
        #     sync_tensorboard=True,
        #     save_code=True)
        
        if air_hockey_cfg['n_threads'] > 1:

            # set seed for reproducibility
            seed = air_hockey_params['seed']
            random.seed(seed)
            
            # def get_env():
            #     # get random seed
            #     # since previous seed is fixed, this is also fixed too
            #     curr_seed = random.randint(0, 1e8)
            #     air_hockey_params['seed'] = curr_seed
            #     env = AirHockeyEnv.from_dict(air_hockey_params)
            #     return Monitor(env)
            
            def get_env(env_id=None, rank=None, seed=0):
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
                    env = AirHockeyEnv.from_dict(air_hockey_params)
                    return Monitor(env)
                # set_global_seeds(seed)
                return _init()
            
            # get number of threads
            n_threads = air_hockey_cfg['n_threads']

            # check_env(env)
            env = SubprocVecEnv([get_env for _ in range(n_threads)])
            # env = VecNormalize(env) # probably something to try when tuning
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
                device='cuda',
            )
        else:
            
            model = PPO("MlpPolicy", env, verbose=1, 
                    tensorboard_log=log_parent_dir, 
                    device="cpu", 
                    seed=seed,
                    # batch_size=512,
                    #n_epochs=5,
                    gamma=air_hockey_cfg['gamma']) 
        
        model.learn(total_timesteps=air_hockey_cfg['n_training_steps'],
                    tb_log_name=air_hockey_cfg['tb_log_name'], 
                    callback=callback,
                    progress_bar=True)
        
        os.makedirs(log_parent_dir, exist_ok=True)
        # get log dir ending with highest number
        # subdirs = [x for x in os.listdir(log_parent_dir) if os.path.isdir(os.path.join(log_parent_dir, x))]
        # subdirs.sort(key=lambda x: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', x)])
        # log_dir = os.path.join(log_parent_dir, subdirs[-1])
        
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

        # env_test.training = False
        # env_test.norm_reward = False
        
        # Initialize an event accumulator
        ea = event_accumulator.EventAccumulator(log_dir,
                                                size_guidance={  # see below regarding this argument
                                                    event_accumulator.COMPRESSED_HISTOGRAMS: 0,
                                                    event_accumulator.IMAGES: 0,
                                                    event_accumulator.AUDIO: 0,
                                                    event_accumulator.SCALARS: 0,
                                                    event_accumulator.HISTOGRAMS: 0,
                                                })

        # Load all events from the directory
        ea.Reload()

        # uncomment below to see the tags in the tensorboard log file, then you can add them to metrics
        # print("Available tags: ", ea.Tags()['scalars'])
        #
        
        if 'goal' in air_hockey_cfg['air_hockey']['task']:
            metrics = [
                'rollout/ep_rew_mean',
                'train/actor_loss',
                'train/ent_coef_loss',
                'train/learning_rate',
                'eval/ep_return',
                'eval/success_rate']
        else:
            metrics = [
                'rollout/ep_rew_mean',
                'train/approx_kl',
                'eval/success_rate',
                'eval/ep_return',
                'train/loss',
                'train/value_loss']
        
        def save_plot(metrics):
            # Create a 2x3 subplot
            fig, axs = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Air Hockey Training Summary')
        
            # Flatten the axs array for easy iteration
            axs = axs.flatten()
        
            for i, metric in enumerate(metrics):
                if metric in ea.Tags()['scalars']:
                    # Extract time steps and values for the metric
                    times, step_nums, values = zip(*ea.Scalars(metric))
        
                    # Plot on the i-th subplot
                    axs[i].plot(step_nums, values, label=metric)
                    axs[i].set_title(metric)
                    axs[i].set_xlabel("Steps")
                    axs[i].set_ylabel("Value")
                    axs[i].legend()
                else:
                    print(f"Metric {metric} not found in logs.")
                    axs[i].set_title(f"{metric} (not found)")
                    axs[i].set_xlabel("Steps")
                    axs[i].set_ylabel("Value")
        
            # Adjust layout for better readability
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            # let's save in same folder
            plot_fp = os.path.join(log_dir, 'training_summary.png')
            plt.savefig(plot_fp)
            plt.close()
        
        save_plot(metrics)

        obs = env_test.reset()
        start = time.time()
        done = False
        
        # first let's create some videos offline into gifs
        print("Saving gifs...(this will tqdm for EACH gif to save)")
        n_eps_viz = 5
        n_gifs = 5
        env_test.max_timesteps = 200
        for gif_idx in range(n_gifs):
            frames = []
            for i in tqdm.tqdm(range(n_eps_viz)):
                obs = env_test.reset()
                done = False
                while not done:
                    frame = renderer.get_frame()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # decrease width to 160 but keep aspect ratio
                    aspect_ratio = frame.shape[1] / frame.shape[0]
                    frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))
                    frames.append(frame)
                    action = model.predict(obs, deterministic=True)[0]
                    obs, rew, done, info = env_test.step(action)
            gif_savepath = os.path.join(log_dir, f'eval_{gif_idx}.gif')
            def fps_to_duration(fps):
                return int(1000 * 1/fps)
            imageio.mimsave(gif_savepath, frames, format='GIF', loop=0, duration=fps_to_duration(30))
            
        # upload last gif to wandb
        # wandb_run.log({"Evaluation Video": wandb.Video(gif_savepath, fps=30)})
        
        env_test.close()
        # wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
    parser.add_argument('--cfg', type=str, default=None, help='Path to the configuration file.')
    args = parser.parse_args()
    if args.cfg is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        air_hockey_cfg_fp = os.path.join(dir_path, 'configs', 'train_ppo.yaml')
    else:
        air_hockey_cfg_fp = args.cfg
    with open(air_hockey_cfg_fp, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)
    train_air_hockey_model(air_hockey_cfg)
