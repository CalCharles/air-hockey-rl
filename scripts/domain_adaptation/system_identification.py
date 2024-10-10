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
from dataset_management.create_dataset import load_dataset
import wandb
import cv2
from scipy.spatial import distance
from scripts.train import init_params

# dynamic time warping
def DTW(a, b):
    # num_states x obs dimension
    an = len(a)
    bn = len(b)
    obs_dim = a.shape[-1]
    pointwise_distance = distance.cdist(a.reshape(-1,obs_dim),b.reshape(-1,obs_dim))
    cumdist = np.matrix(np.ones((an+1,bn+1)) * np.inf)
    cumdist[0,0] = 0
    
    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost

    return cumdist[an, bn]

def lcss(t0, t1,min_dist = 0.1):
    """
    Usage
    -----
    The Longuest-Common-Subsequence distance between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    min_dist : float, the minimum distance to consider two points close together

    Returns
    -------
    lcss : float
           The Longuest-Common-Subsequence distance between trajectory t0 and t1
    """
    n0 = len(t0)
    n1 = len(t1)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n1+1) for _ in range(n0+1)]
    for i in range(1, n0+1):
        for j in range(1, n1+1):
            if np.linalg.norm(t0[i-1],t1[j-1])<min_dist:
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = max(C[i][j-1], C[i-1][j])
    lcss = 1-float(C[n0][n1])/min([n0,n1])
    return lcss


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

if False:
    set_ipdb_debugger()
    import faulthandler
    faulthandler.enable()

def get_value(param_vector, param_names, base_config, trajectories, task_name):
    new_config = assign_values(param_vector, param_names, base_config)
    eval_env = AirHockeyEnv(new_config)
    
    evaluated_states = list()
    for i in range(len(trajectories['observations'])):
        if 'goal' in task_name:
            eval_env.reset_from_state_and_goal(trajectories['observations'][i, 0, :-2], trajectories['observations'][i, 0, -2:])  ## todo, make this more general
        else:
            eval_env.reset_from_state(trajectories['observations'][i, 0]) # TODO: need to be able to reset from state
        for j in range(trajectories['actions'].shape[1] - 1):
            act = trajectories['actions'][i, j]
            evaluated_state = eval_env.step(act)[0]
            evaluated_states.append(evaluated_state)
    
    evaluated_states = np.array(evaluated_states)
    target_states = np.concatenate(trajectories['observations'][:, 1:, :4], axis=0)
    value = compare_trajectories(evaluated_states[:, :4], target_states)
    return value

def compare_trajectories(a_traj, b_traj, comp_type="l2"):
    # TODO: Dynamic time warping for trajectory comparison
    if comp_type == "l2":
        return - np.linalg.norm(a_traj - b_traj)
    if comp_type == "last":
        return - np.linalg.norm(a_traj[-1] - b_traj[-1])
    if comp_type == "dtw":
        return - DTW(a_traj, b_traj)
    if comp_type == "lcss":
        return - lcss(a_traj, b_traj)

def load_custom_dataset(dataset_pth):
    dataset = np.load(dataset_pth)
    data = dict()
    for key in dataset.keys():
        if len(dataset[key].shape) == 3:
            data[key] = np.squeeze(dataset[key], axis=1)   # N x 1 x dim -> N x dim
        else:
            data[key] = dataset[key]
    return data

def get_frames(renderer, env, states, actions, terminals, task_name):
        
        frames = []
        robosuite_frames = {}
        i = 0
        while i < len(actions):
            if 'goal' in task_name:
                eval_env.reset_from_state_and_goal(states[i, :-2], states[i, -2:])
            else:
                eval_env.reset_from_state(states[i]) 
            done = terminals[i]
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
                done = terminals[i]
                i += 1
        # import pdb; pdb.set_trace()

        return frames

def load_sys_id_yaml(pth):
    
    with open(pth, 'r') as f:
        id_config = yaml.safe_load(f)
        param_names = list(id_config.keys())
        param_names.sort()

        lower_bounds = list()
        upper_bounds = list()
        initial_params = list()
        for n in param_names:
            lower_bounds.append(id_config[n]["lower"])
            upper_bounds.append(id_config[n]["upper"])
            initial_params.append(id_config[n]["start"])
    return param_names, lower_bounds, upper_bounds, initial_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
    parser.add_argument('--cfg', type=str, default='scripts/domain_adaptation/realworld_paddle_config/model_cfg.yaml', help='Path to the configuration file.')
    parser.add_argument('--sys-id-pth', type=str, default='scripts/domain_adaptation/sys_id_configs/paddle_id_params.yaml', help='Path to the range of parameters to modify.')
    parser.add_argument('--dataset-pth', type=str, default='scripts/domain_adaptation/paddle_reach_config/eval_dataset.npz', help='Path to the dataset file.')
    parser.add_argument('--wdb-entity', type=str, default='', help='who to log wdb to.')
    parser.add_argument('--run-id', type=str, default='', help="logging identification")
    args = parser.parse_args()

    # Dataset format:
        # dictionary with keys: images, observations, actions, terminals
        # observation with format: paddle pos vel, puck pos vel, goal pos (if goal based)
    # python scripts/domain_adaptation/system_identification.py --cfg scripts/domain_adaptation/realworld_paddle_config/model_cfg.yaml --dataset-pth /datastor1/calebc/public/data/mouse/state_data_all/ 



    with open(args.cfg, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)
    
    air_hockey_cfg['air_hockey'] = init_params(air_hockey_cfg)
    # param_names = list(air_hockey_cfg['air_hockey']["simulator_params"].keys())

    param_names, lower_bounds, upper_bounds, initial_params = load_sys_id_yaml(args.sys_id_pth)

    ### dataset loading ### 
    new_config = assign_values(initial_params, param_names, air_hockey_cfg['air_hockey'])
    eval_env = AirHockeyEnv(new_config)
    data = load_dataset(args.dataset_pth, "history", eval_env, 10)
    # data = load_dataset("/datastor1/calebc/public/data/mouse/state_data_all/", "history", eval_env)
    # print(list(data.keys()), data["observations"].shape)

    # magic number to shift the observations
    for observations in data["observations"]:
        observations[:, 0] += 1.2
    ###

    vis_before = True
    if vis_before:
        # gif_path = os.path.join(os.path.split(args.dataset_pth)[0], 'eval.gif')
        # saved_frames = imageio.mimread(gif_path)
        saved_frames = data['images'][0]
        new_config = assign_values(initial_params, param_names, air_hockey_cfg['air_hockey'])
        eval_env = AirHockeyEnv(new_config)

        renderer = AirHockeyRenderer(eval_env)
        frames = get_frames(renderer, eval_env, data['observations'][0], data['actions'][0], data['terminals'][0], air_hockey_cfg['air_hockey']['task'])
        gif_savepath = os.path.join(os.path.split(args.cfg)[0], f'pre_cem.gif')
        def fps_to_duration(fps):
            return int(1000 * 1/fps)
        fps = 30 # slightly faster than 20 fps (simulation time), but makes rendering smooth

        # rotate the saved_frames by -90 degrees and resize
        frameshape = frames[0].shape
        saved_frames = list(map(lambda f: cv2.resize(cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE), (frameshape[1], frameshape[0])), saved_frames))
        # plot the frame number on the image
        for i, frame in enumerate(saved_frames):
            cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        side_by_side = [np.concatenate([saved_frames[i], frames[i]], axis=1) for i in range(min(len(frames), len(saved_frames)))]
        imageio.mimsave(gif_savepath, side_by_side, format='GIF', loop=0, duration=fps_to_duration(fps))
    
    if args.wdb_entity:
        import datetime
        timestamp = datetime.datetime.now()
        formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        wandb.init(project="air_hockey_rl", 
                   entity=args.wdb_entity, 
                   config=air_hockey_cfg,
                   name=os.path.basename(args.dataset_pth) + "_" + os.path.basename(args.sys_id_pth),
                   id=args.run_id + '_' + formatted_timestamp.replace(":", "_"))

    planner = CEMPlanner(eval_fn=lambda params, trajs: get_value(params, param_names, air_hockey_cfg['air_hockey'], trajs, air_hockey_cfg['air_hockey']['task']), 
                         trajectories=data, elite_frac=0.2, n_samples=100, n_iterations=50, variance=0.2, 
                         lower_bounds=lower_bounds, upper_bounds=upper_bounds, param_names=param_names,
                         wdb_logging=len(args.wdb_entity) > 0)
    # TODO Implemetn CMA-ES planner also
    
    optimal_parameters = planner.optimize(initial_params)
    optimal_parameters = optimal_parameters.tolist()
    print(f"Optimal Parameters: {optimal_parameters}")


    vis_after = True
    if vis_after:

        new_config = assign_values(optimal_parameters, param_names, air_hockey_cfg['air_hockey'])
        ## save the new config with the optimal parameters
        cfg_folder = os.path.split(args.cfg)[0]
        with open(os.path.join(cfg_folder, 'optimal_config.yaml'), 'w') as f:
            new_air_hockey_cfg = copy.deepcopy(air_hockey_cfg)
            new_air_hockey_cfg['air_hockey']["simulator_params"] = new_config
            yaml.dump(new_config, f)
        eval_env = AirHockeyEnv(new_config)

        renderer = AirHockeyRenderer(eval_env)
        post_frames = get_frames(renderer, eval_env, data['observations'][0], data['actions'][0], data['terminals'][0], air_hockey_cfg['air_hockey']['task'])
        gif_savepath = os.path.join(os.path.split(args.cfg)[0], f'post_cem.gif')

        side_by_side = [np.concatenate([saved_frames[i], post_frames[i]], axis=1) for i in range(min(len(post_frames), len(saved_frames)))]
        imageio.mimsave(gif_savepath, side_by_side, format='GIF', loop=0, duration=fps_to_duration(fps))