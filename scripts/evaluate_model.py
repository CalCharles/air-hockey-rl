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
import h5py
from utils import save_evaluation_gifs

def save_pose_dataset(pose_dataset, target_path):
    # TODO: do this properly
    try:
        os.makedirs(target_path)
    except OSError as e:
        pass
    for i in range(pose_dataset):
        with h5py.File(os.path.join(target_path, 'trajectory_data' + str(i) + '.hdf5'), 'w') as hf:
            for k in pose_dataset[i].keys():
                hf.create_dataset(k,
                                shape=pose_dataset[i][k].shape,
                                compression="gzip",
                                compression_opts=9,
                                data = pose_dataset[i][k])

def register_pose_dataset(obs, action, done, image, obs_type="vel"):
    if obs_type == "vel":
        pose = obs[...,:2]
        speed = obs[...,2:4]
        desired_pose = action+pose # TODO: may need to transform action space
        puck = obs[...,4:6]
        image = image
        done = done
    else:
        raise NotImplementedError("Obs type: "+ obs_type + " not supported")
    return pose, speed, desired_pose, puck, image, done

def get_frames(renderer, env_test, model, n_eps_viz, n_eval_eps, cfg, get_pose_dataset=False):
        
        dataset = {}
        observations = []
        actions = []
        rewards = []
        dones = []

        frames = []
        robosuite_frames = {}
        env = env_test.envs[0]
        successes = 0

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
                elif get_pose_dataset:
                    frames.append(np.zeros(frame.shape)) # assumes at least one render
                observations.append(obs)
                action, _ = model.predict(obs)
                # action = np.random.uniform(-1, 1, size=(1,6))
                actions.append(action)
                obs, rew, done, info = env_test.step(action)
                rewards.append(rew)
                dones.append(done[0] or info[0]['TimeLimit.truncated'])
                if done:
                    print(info[0]["success"])
                    successes += 1 if info[0]["success"] else 0
        
        dataset['states'] = np.array(observations)
        dataset['actions'] = np.array(actions)
        dataset['rewards'] = np.array(rewards)
        dataset['dones'] = np.array(dones)
        dataset["frames"] = np.array(frames)
        
        success_rate = successes / n_eval_eps
        print("Run complete with success rate: " + str(success_rate))

        pose_dataset = list()
        if get_pose_dataset:
            pose, speed, desired_pose, puck, image, done = register_pose_dataset(dataset["states"], dataset["actions"], dataset["dones"], dataset["frames"])
            traj_idxes = np.nonzero(done.astype(int))
            sidx = 0
            for i in range(traj_idxes):
                eidx = traj_idxes[i]
                pose_traj = dict(pose=pose[sidx:eidx], speed=speed[sidx:eidx], desired_pose=desired_pose[sidx:eidx], puck=puck[sidx:eidx], image=image[sidx:eidx])
                pose_dataset.append(pose_traj)

        # import pdb; pdb.set_trace()

        return frames, robosuite_frames, dataset, pose_dataset

# Takes in a folder with a model zip and the config for the model, and uses it to generate evaluation gifs.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save an evaluation gif of a trained model.')
    parser.add_argument('--model', type=str, default="", help='Folder that contains model and model_cfg.')
    parser.add_argument('--save-dir', type=str, default="", help='Path to save the evaluation gifs to.')
    parser.add_argument('--save-pose-dir', type=str, default="", help='Path to save the pose dataset.')
    parser.add_argument('--seed', type=int, default=42, help='The random seed for the environment')
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

    frames, robosuite_frames, dataset, pose_dataset = get_frames(renderer=renderer, env_test=env_test, model=model, n_eps_viz=5, n_eval_eps=30, cfg=model_cfg)

    # save dataset to disk
    dataset_savepath = os.path.join(args.save_dir, f'eval_dataset.npz')
    np.savez(dataset_savepath, **dataset)

    # save pose dataset to disk
    if len(args.save_pose_dataset) > 0:
        save_pose_dataset(pose_dataset, args.save_pose_dir)

    # make a subfolder in log dir for latest progress
    os.makedirs(args.save_dir, exist_ok=True)
    
    # save gif, sample onl yfrom the live frames
    assert len(frames) > 0
    gif_savepath = os.path.join(args.save_dir, f'eval.gif')
    def fps_to_duration(fps):
        return int(1000 * 1/fps)
    fps = 30 # slightly faster than 20 fps (simulation time), but makes rendering smooth
    control_freq = air_hockey_params["simulator_params"].get("control_freq", 20)
    imageio.mimsave(gif_savepath, frames[::int(control_freq/20)], format='GIF', loop=0, duration=fps_to_duration(fps))
    if len(robosuite_frames) > 0:
        for key in robosuite_frames:
            frames = robosuite_frames[key]
            gif_savepath = os.path.join(args.save_dir, f'feval_robosuite_{key}.gif')
            imageio.mimsave(gif_savepath, frames[::int(control_freq/20)], format='GIF', loop=0, duration=fps_to_duration(fps))