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
    air_hockey_cfg_fp = os.path.join(dir_path, '../configs', 'baseline_configs/robosuite/puck_height_robosuite.yaml')

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
    air_hockey_params_cp['max_timesteps'] = 60
    # Disable out-of-bounds truncation for the visualization run: spawn_paddle
    # in the robosuite sim only records the desired EEF pose, it never actually
    # teleports the EEF, so the UR5e starts at whatever robosuite's default
    # initial joint config produces — which may be outside paddle_bounds and
    # cause immediate truncation otherwise.
    air_hockey_params_cp['terminate_on_out_of_bounds'] = False
    air_hockey_params_cp.setdefault('simulator_params', {})['seed'] = 43
    air_hockey_params_cp['simulator_params']['has_renderer'] = False
    air_hockey_params_cp['simulator_params']['has_offscreen_renderer'] = True

    headless = os.environ.get('AIRHOCKEY_HEADLESS', '1') == '1'

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

        # birdview_image is the useful camera for an air-hockey scene (top-down
        # on the table). It comes out of MuJoCo's framebuffer already oriented
        # with the robot at the bottom of the image (matching the Box2D-style
        # top-down render on the left panel), so no flip is needed.
        # Note: sideview_image *is* upside down out of MuJoCo and needs
        # np.flipud to be right-side up — the previous code's
        # cv2.flip(img, axis=1) was the wrong axis (horizontal mirror).
        current_img = eval_env.current_state["birdview_image"]
        # concatenate with frame
        current_img = cv2.resize(current_img, (480, int(480 / aspect_ratio)))
        current_img = np.concatenate([frame, current_img], axis=1)

        frames.append(current_img)
        if not headless:
            cv2.imshow("AirHockey", current_img)
            cv2.waitKey(1)

        # No-op action so the paddle holds position via OSC and the puck's
        # initial +x velocity carries it down the table — gives us a visible
        # rollout to render. (Old action [-1, 0.0165] drove the OSC paddle
        # out of bounds in a couple of steps even after softening.)
        action = np.zeros(2)
        obs, rew, done, truncated, info = eval_env.step(action)
        print(rew, done)
        cum_rew += rew
        done = done or truncated

    print(cum_rew)
    gif_dir = os.path.abspath(os.path.join(dir_path, '..', 'eval_gifs'))
    os.makedirs(gif_dir, exist_ok=True)

    gif_savepath = os.path.join(gif_dir, air_hockey_params['task'] + '_robosuite.gif')

    def fps_to_duration(fps):
        return int(1000 * 1 / fps)


    imageio.mimsave(gif_savepath, frames, format='GIF', loop=0, duration=fps_to_duration(30))
