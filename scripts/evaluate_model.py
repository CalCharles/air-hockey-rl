import os
import cv2
import imageio
import numpy as np
import yaml
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, SAC
from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
import argparse
from utils import save_evaluation_gifs


def get_frames(renderer, env, model, n_eps_viz, n_eval_eps, cfg):
        
        dataset = {}
        observations = []
        actions = []
        rewards = []
        dones = []

        frames = []
        robosuite_frames = {}
        env = env_test.envs[0]
        for ep_idx in range(n_eval_eps):
            obs = env_test.reset()
            done = False
            while not done:
                if ep_idx < n_eps_viz:
                    frame = renderer.get_frame()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # decrease width to 160 but keep aspect ratio
                    aspect_ratio = frame.shape[1] / frame.shape[0]
                    frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))
                    frames.append(frame)
                    if cfg['air_hockey']['simulator'] == 'robosuite':
                        for key in env.current_state:
                            if 'image' not in key:
                                continue
                            current_img = env.current_state[key]
                            # flip upside down
                            current_img = cv2.flip(current_img, 0)
                            # concatenate with frame
                            current_img = cv2.resize(current_img, (160, int(160 / aspect_ratio)))
                            current_img = np.concatenate([frame, current_img], axis=1)
                            if key not in robosuite_frames:
                                robosuite_frames[key] = [current_img]
                            else:
                                robosuite_frames[key].append(current_img)
                observations.append(obs)
                # action, _ = model.predict(obs)
                action = np.random.uniform(-1, 1, size=(1,6))
                actions.append(action)
                obs, rew, done, info = env_test.step(action)
                rewards.append(rew)
                dones.append(done[0] or info[0]['TimeLimit.truncated'])
                done = done[0] or info[0]['TimeLimit.truncated']
        
        dataset['states'] = np.array(observations)
        dataset['actions'] = np.array(actions)
        dataset['rewards'] = np.array(rewards)
        dataset['dones'] = np.array(dones)

        # import pdb; pdb.set_trace()

        return frames, robosuite_frames, dataset

# Takes in a folder with a model zip and the config for the model, and uses it to generate evaluation gifs.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save an evaluation gif of a trained model.')
    parser.add_argument('--model', type=str, default="", help='Folder that contains model and model_cfg.')
    parser.add_argument('--save-dir', type=str, default="", help='Path to save the evaluation gifs to.')
    parser.add_argument('--seed', type=int, default=0, help='The random seed for the environment')
    args = parser.parse_args()

    with open(os.path.join(args.model, "model_cfg.yaml"), 'r') as f:
        model_cfg = yaml.safe_load(f)

    air_hockey_params = model_cfg['air_hockey']
    air_hockey_params['n_training_steps'] = model_cfg['n_training_steps']


    # Set the return_goal_obs parameter for sac
    model_cfg['air_hockey']['return_goal_obs'] = 'goal' in model_cfg['air_hockey']['task'] and 'sac' == model_cfg['algorithm']
    
    model_cfg['air_hockey']['max_timesteps'] = 200
            
    air_hockey_params = model_cfg['air_hockey']
    air_hockey_params['n_training_steps'] = model_cfg['n_training_steps']
    air_hockey_params['seed'] = args.seed

    env_test = AirHockeyEnv(air_hockey_params)
    renderer = AirHockeyRenderer(env_test)

    env_test = DummyVecEnv([lambda: env_test])

    # Load the correct model based on the algorithm specified in the config
    if model_cfg['algorithm'] == 'sac':
        model = SAC.load(os.path.join(args.model, "model.zip"))
    else:
        model = PPO.load(os.path.join(args.model, "model.zip"))

    frames, robosuite_frames, dataset = get_frames(renderer=renderer, env=env_test, model=model, n_eps_viz=5, n_eval_eps=3, cfg=model_cfg)

    # save dataset to disk
    dataset_savepath = os.path.join(args.save_dir, f'eval_dataset.npz')
    np.savez(dataset_savepath, **dataset)

    # make a subfolder in log dir for latest progress
    os.makedirs(args.save_dir, exist_ok=True)
    
    # save gif
    assert len(frames) > 0
    gif_savepath = os.path.join(args.save_dir, f'eval.gif')
    def fps_to_duration(fps):
        return int(1000 * 1/fps)
    fps = 30 # slightly faster than 20 fps (simulation time), but makes rendering smooth
    imageio.mimsave(gif_savepath, frames, format='GIF', loop=0, duration=fps_to_duration(fps))
    if len(robosuite_frames) > 0:
        for key in robosuite_frames:
            frames = robosuite_frames[key]
            gif_savepath = os.path.join(args.save_dir, f'feval_robosuite_{key}.gif')
            imageio.mimsave(gif_savepath, frames, format='GIF', loop=0, duration=fps_to_duration(fps))
