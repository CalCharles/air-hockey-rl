import os

import cv2
import imageio
import numpy as np
import yaml
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    air_hockey_cfg_fp = os.path.join(dir_path, '../configs', 'baseline_configs/puck_height_robosuite.yaml')

    with open(air_hockey_cfg_fp, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)

    air_hockey_params = air_hockey_cfg['air_hockey']
    air_hockey_params['n_training_steps'] = air_hockey_cfg['n_training_steps']

    if 'sac' == air_hockey_cfg['algorithm']:
        if 'goal' in air_hockey_cfg['air_hockey']['task']:
            air_hockey_cfg['air_hockey']['return_goal_obs'] = True
        else:
            air_hockey_cfg['air_hockey']['return_goal_obs'] = False
    else:
        air_hockey_cfg['air_hockey']['return_goal_obs'] = False

    air_hockey_params_cp = air_hockey_params.copy()
    air_hockey_params_cp['seed'] = 43
    air_hockey_params_cp['max_timesteps'] = 200

    eval_env = AirHockeyEnv(air_hockey_params_cp)


    def wrap_env(env):
        wrapped_env = Monitor(env)  # needed for extracting eprewmean and eplenmean
        wrapped_env = DummyVecEnv([lambda: wrapped_env])  # Needed for all environments (e.g. used for multi-processing)
        # wrapped_env = VecNormalize(wrapped_env) # probably something to try when tuning
        return wrapped_env


    # eval_env = wrap_env(eval_env)
    renderer = AirHockeyRenderer(eval_env)

    frames = []
    robosuite_frames = {}

    obs, info = eval_env.reset()
    done = False
    success = False
    cum_rew = 0
    step = 0
    while not done:
        step += 1
        frame = renderer.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # decrease width to 160 but keep aspect ratio
        aspect_ratio = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (480, int(480 / aspect_ratio)))

        # bird_view_image, sideview_image
        current_img = eval_env.current_state["sideview_image"]
        # flip sideways
        current_img = cv2.flip(current_img, 1)
        # concatenate with frame
        current_img = cv2.resize(current_img, (480, int(480 / aspect_ratio)))
        current_img = np.concatenate([frame, current_img], axis=1)

        frames.append(current_img)
        cv2.imshow("AirHockey", current_img)

        forward = -1 if step > 32 else 0.01
        action = np.array([forward, 0.0165])  # debugging!
        obs, rew, done, truncated, info = eval_env.step(action)
        print(rew, done)
        cum_rew += rew
        done = done or truncated
        cv2.waitKey(1)

    print(cum_rew)
    if not os.path.exists('../eval_gifs'):
        os.makedirs('../eval_gifs')

    gif_savepath = os.path.join('../eval_gifs', air_hockey_params['task'] + '.gif')

    def fps_to_duration(fps):
        return int(1000 * 1 / fps)


    imageio.mimsave(gif_savepath, frames, format='GIF', loop=0, duration=fps_to_duration(30))
