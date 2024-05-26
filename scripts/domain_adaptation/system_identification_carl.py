import copy
import os
import argparse
import imageio
import numpy as np
import yaml
from airhockey.renderers.render import AirHockeyRenderer
from planners import CEMPlanner# TODO: using https://cma-es.github.io/ for CMA-ES
from scripts.domain_adaptation.encode_env_params import assign_values, extract_value
from airhockey import AirHockeyEnv
import wandb
import cv2

def set_ipdb_debugger():
    import sys
    import ipdb
    import traceback

    def info(t, value, tb):
        if t == KeyboardInterrupt:
            traceback.print_exception(t, value, tb)
            return
        else:
            traceback.print_exception(t, value, tb)
            ipdb.pm()

    sys.excepthook = info

if True:
    set_ipdb_debugger()
    import faulthandler
    faulthandler.enable()

def get_value(param_vector, param_names, base_config, trajectories):
    new_config = assign_values(param_vector, param_names, base_config)
    eval_env = AirHockeyEnv(new_config)
    
    evaluated_states = list()
    for i in range(len(trajectories['states'])):
        eval_env.reset_from_state(trajectories['states'][i, 0]) # TODO: need to be able to reset from state
        for act in trajectories['actions'][i]:
            evaluated_state = eval_env.step(act)[0]
            evaluated_states.append(evaluated_state)
    
    evaluated_states = np.array(evaluated_states)
    return compare_trajectories(evaluated_states, trajectories['states'].reshape(-1, trajectories['states'].shape[-1]))

def compare_trajectories(a_traj, b_traj, comp_type="l2"):
    # TODO: Dynamic time warping for trajectory comparison
    if comp_type == "l2":
        return - np.linalg.norm(a_traj - b_traj)

def load_dataset(dataset_pth):
    dataset = np.load(dataset_pth)
    data = dict()
    for key in dataset.keys():
        if len(dataset[key].shape) == 3:
            data[key] = np.squeeze(dataset[key], axis=1)   # N x 1 x dim -> N x dim
        else:
            data[key] = dataset[key]
    return data

def get_frames(renderer, env, actions):
        
        frames = []
        robosuite_frames = {}
        i = 0
        while i < len(actions):
            obs = env.reset()
            done = False
            while not done:
                if i >= len(actions):
                    break
                frame = renderer.get_frame()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # decrease width to 160 but keep aspect ratio
                aspect_ratio = frame.shape[1] / frame.shape[0]
                frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))
                frames.append(frame)
                    
                action = actions[i]
                obs, rew, done, truncated, info = env.step(action)
                i += 1
        # import pdb; pdb.set_trace()

        return frames



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
    parser.add_argument('--cfg', type=str, default='scripts/domain_adaptation/custom_config/model_cfg.yaml', help='Path to the configuration file.')
    parser.add_argument('--dataset_pth', type=str, default='scripts/domain_adaptation/custom_config/eval_dataset.npz', help='Path to the dataset file.')
    args = parser.parse_args()


    ### dataset loading ### 
    data = load_dataset(args.dataset_pth)



    with open(args.cfg, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)
    # param_names = list(air_hockey_cfg['air_hockey']["simulator_params"].keys())

    param_names = ['force_scaling', 'max_force_timestep', 'paddle_damping', 'paddle_density'] # 'puck_damping', 'puck_density']
    param_names.sort()
    
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
    air_hockey_params_cp['seed'] = 42
    air_hockey_params_cp['max_timesteps'] = 200


    initial_params = extract_value(param_names, air_hockey_params_cp)
    print(initial_params, param_names)


    wandb.init(project="air_hockey_rl", entity="carltheq", config=air_hockey_cfg)
    wandb.run.name = "sysid_CEM_paddle_reach"


    new_config = assign_values(initial_params, param_names, air_hockey_params_cp)
    eval_env = AirHockeyEnv(new_config)
    renderer = AirHockeyRenderer(eval_env)
    frames = get_frames(renderer, eval_env, data['actions'])
    gif_savepath = os.path.join(os.path.split(args.dataset_pth)[0], f'pre_cem.gif')
    def fps_to_duration(fps):
        return int(1000 * 1/fps)
    fps = 30 # slightly faster than 20 fps (simulation time), but makes rendering smooth
    imageio.mimsave(gif_savepath, frames, format='GIF', loop=0, duration=fps_to_duration(fps))

    lower_bounds = np.array([900, 90, 2, 2000])
    upper_bounds = np.array([1100, 110, 4, 3000])
    initial_params = [1000, 100, 3, 2500]

    planner = CEMPlanner(eval_fn=lambda params, trajs: get_value(params, param_names, air_hockey_params_cp, trajs), 
                         trajectories=data, elite_frac=0.2, n_samples=500, n_iterations=10, variance=0.1, lower_bounds=None, upper_bounds=None, param_names=param_names)
    # TODO Implemetn CMA-ES planner also
    
    optimal_parameters = planner.optimize(initial_params)
    print(f"Optimal Parameters: {optimal_parameters}")


    new_config = assign_values(optimal_parameters, param_names, air_hockey_params_cp)
    eval_env = AirHockeyEnv(new_config)
    renderer = AirHockeyRenderer(eval_env)
    frames = get_frames(renderer, eval_env, data['actions'])
    gif_savepath = os.path.join(os.path.split(args.dataset_pth)[0], f'post_cem.gif')
    imageio.mimsave(gif_savepath, frames, format='GIF', loop=0, duration=fps_to_duration(fps))