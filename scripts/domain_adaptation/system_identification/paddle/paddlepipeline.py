import argparse
import os
import cv2
import imageio
import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import gymnasium as gym
from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from scripts.smooth_policy.agent import Agent
from scripts.smooth_policy.running.run_single_episode import *
from scripts.domain_adaptation.planners import CMAPlanner


# ──────────────────────────────────────────────
# Standalone utility functions
# ──────────────────────────────────────────────

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

def compare_trajectories(a_traj, b_traj, comp_type="l2"):
	# TODO: Dynamic time warping for trajectory comparison
	diffs = a_traj - b_traj
	if comp_type == "l2":
		step_l2 = np.linalg.norm(diffs, axis=2)
		traj_loss = np.mean(step_l2, axis=1)
		paddle_batch_loss = np.mean(traj_loss)
		return -paddle_batch_loss
	if comp_type == "posl2":
		return - np.linalg.norm(a_traj[...,:2] - b_traj[...,:2])
	if comp_type == "l1":
		return - np.linalg.norm(a_traj - b_traj, ord=1)
	if comp_type == "posl1":
		return - np.linalg.norm(a_traj[...,:2] - b_traj[...,:2], ord=1)
	if comp_type == "last":
		return - np.linalg.norm(a_traj[-1] - b_traj[-1])
	if comp_type == "dtw":
		return - DTW(a_traj, b_traj)
	if comp_type == "lcss":
		return - lcss(a_traj, b_traj)

def plot_trajectories(ground_truth_results, evaluated_states, object_name="paddle", comp_type="l2", new_config=None):
    """
    Plots the 2D (x, y) trajectories of the specified object for the ground truth
    and the evaluated run.

    Parameters
    ----------
    ground_truth_results : dict
        The result dictionary from the ground truth run (containing 'observations').
    evaluated_states : list or numpy.ndarray
        The list/array of evaluated states from the evaluate_reset/run_single_episode.
    object_name : str
        The object to plot: "paddle", "puck", or "all".
    comp_type : str
        The comparison type (unused for plotting, kept for consistency).
    new_config : dict, optional
        The new config dictionary from the evaluated run, used for offset, if any.
    """
    if isinstance(evaluated_states, list):
        evaluated_states = np.array(evaluated_states)

    # 1. Determine which part of the state vector corresponds to (x, y) position
    # The 'all' object is often the full state, but we need to extract positions.
    # Air hockey state typically is: [paddle_x, paddle_y, paddle_vx, paddle_vy,
    #                                  puck_x, puck_y, puck_vx, puck_vy, ...]

    if object_name == "paddle":
        # Paddle position is usually the first 2 elements (indices 0, 1)
        gt_positions = ground_truth_results['observations'][:, :2]
        eval_positions = evaluated_states[:, :2]
        title_obj = "Paddle"
    elif object_name == "puck":
        # Puck position is usually the 5th and 6th elements (indices 4, 5)
        # Note: ground_truth_results['observations'] is a 3D array (traj_idx, step, obs_dim)
        # The main code structure seems to use a single trajectory, so we take the first index (0).
        gt_positions = ground_truth_results['observations'][0, :, 4:6]
        eval_positions = evaluated_states[:, 4:6]
        title_obj = "Puck"
    else: # object_name == "all"
        # For 'all', we'll plot the paddle's position by default for a simple 2D plot
        gt_positions = ground_truth_results['observations'][0, :, :2]
        eval_positions = evaluated_states[:, :2]
        title_obj = "Paddle (Observed for 'all' object)"

    # 2. Apply any offset if provided in the new_config (as seen in compute_comparison)
    if new_config and "simulator_params" in new_config:
        offset_x = new_config["simulator_params"].get("x_offset", 0.0)
        offset_y = new_config["simulator_params"].get("y_offset", 0.0)

        # Apply offset to the evaluated trajectory's position data (x, y)
        if object_name == "all":
            # If plotting paddle position for 'all', apply paddle offset
             eval_positions[:, 0] += offset_x
             eval_positions[:, 1] += offset_y
        elif object_name in ["paddle", "puck"]:
             eval_positions[:, 0] += offset_x
             eval_positions[:, 1] += offset_y

    # 3. Plotting
    plt.figure(figsize=(10, 6))

    # Plot ground truth trajectory
    plt.plot(gt_positions[:, 0], gt_positions[:, 1],
             label='Ground Truth Trajectory', color='blue', linestyle='-')
    plt.scatter(gt_positions[0, 0], gt_positions[0, 1], color='cyan', marker='o', label='Ground Truth Start', s=100)

    # Plot evaluated trajectory
    plt.plot(eval_positions[:, 0], eval_positions[:, 1],
             label='Evaluated Trajectory', color='red', linestyle='--')
    plt.scatter(eval_positions[0, 0], eval_positions[0, 1], color='magenta', marker='x', label='Evaluated Start', s=100)

    # 4. Final Touches
    plt.title(f'{title_obj} Trajectory Comparison (2D Position)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Keep aspect ratio for spatial plot
    plt.show()

def plot_loss_history(planner, save_path):
	if hasattr(planner, 'best_values_history'):
		history = planner.best_values_history
		if not history:
			print("Planner history is empty. Cannot plot loss.")
			return
		loss_values = -np.array(history)
		iterations = range(1, len(loss_values) + 1)

		plt.figure(figsize=(10, 6))
		plt.plot(iterations, loss_values, marker='o', linestyle='-', color='darkred')

		plt.title('Optimization Loss (Trajectory Distance) over Iterations')
		plt.xlabel('Optimization Iteration')
		plt.ylabel(f'Loss / Distance (L2 Norm)')
		plt.grid(True)
		if save_path:
			plt.savefig(save_path, dpi=150, bbox_inches='tight')
			print(f"\nPlot saved to: {save_path}")

		plt.show()
	else:
		print("Planner object does not have 'best_values_history' attribute. Ensure your planner implementation logs this data.")
	#save the plot graph into the directory

def save_actions_gif_plain(env, renderer, log_dir, actions, gif_name, fps=20):
    """
    Generate a GIF by replaying a fixed sequence of actions in the environment.

    Args:
        env: Environment with reset() and step() methods.
        renderer: Object with get_frame() -> np.ndarray.
        log_dir (str): Directory to save the GIF.
        actions (np.ndarray | list): Sequence of actions to replay (shape: [T, action_dim]).
        fps (int): Playback speed (frames per second).
        gif_name (str): Name of the saved GIF.
    """
    os.makedirs(log_dir, exist_ok=True)

    obs, _ = env.reset()
    frames = []
    total_reward = 0

    for t, action in enumerate(tqdm.tqdm(actions, desc="Replaying actions")):
        # Render current frame
        frame = renderer.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        aspect_ratio = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))

        # Overlay reward
        cv2.putText(frame, f"Reward: {total_reward:.2f}",
                    (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        frames.append(frame)

        # Step environment with the next action
        obs, rew, term, trunc, info = env.step(action)
        total_reward += rew
		#stops the trajectory if the puck falls
        # if term or trunc:
        #     break

    # Save as GIF
    gif_savepath = os.path.join(log_dir, gif_name)
    duration = int(1000 * 1 / fps)
    imageio.mimsave(gif_savepath, frames, format='GIF', loop=0, duration=duration)

    print(f"Saved replay GIF: {gif_savepath}")

def evaluate_iterative_smoothing(model_path, save_dir, air_hockey_params, actions, gif_name, n_eps=5, n_gifs=3):
    envs = gym.vector.SyncVectorEnv([lambda : AirHockeyEnv(air_hockey_params)])
    model = Agent(envs)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    env = envs.envs[0]
    renderer = AirHockeyRenderer(env)

    # create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    save_actions_gif_plain(env, renderer, save_dir, actions, gif_name)

def produce_synchronized_gifs(path_to_source_gif, path_to_target_gif, path_for_synchronization):
    gif1 = imageio.mimread(path_to_source_gif)
    gif2 = imageio.mimread(path_to_target_gif)

    gif1 = [np.array(f) for f in gif1]
    gif2 = [np.array(f) for f in gif2]
    title_left = "Source Env"
    title_right = "Target Env"
    font_size = 26
    title_bar_height = 60
    background_color = (0, 0, 0)
    text_color = "White"

    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("Arial Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()

    combined_frames = []

    for f1, f2 in zip(gif1, gif2):
        combined = np.hstack((f1, f2))
        gif_img = Image.fromarray(combined)
        W, H = gif_img.size
        half_W = W // 2
        new_img = Image.new("RGB", (W, H + title_bar_height), background_color)
        new_img.paste(gif_img, (0, 0))

        draw = ImageDraw.Draw(new_img)

        bbox_left = draw.textbbox((0, 0), title_left, font=font)
        tw_left = bbox_left[2] - bbox_left[0]
        th_left = bbox_left[3] - bbox_left[1]

        bbox_right = draw.textbbox((0, 0), title_right, font=font)
        tw_right = bbox_right[2] - bbox_right[0]
        th_right = bbox_right[3] - bbox_right[1]
        y_pos = H + (title_bar_height - th_left) // 2
        draw.text(
            ((half_W - tw_left) // 2, y_pos),
            title_left,
            fill=text_color,
            font=font
        )
        draw.text(
            (half_W + (half_W - tw_right) // 2, y_pos),
            title_right,
            fill=text_color,
            font=font
        )

        combined_frames.append(np.array(new_img))
    imageio.mimsave(path_for_synchronization, combined_frames, loop=0, fps=20)

    print(f"Produced synchronized GIF: {path_for_synchronization}")

def divide_into_non_interaction_subarrays_for_puck_trajectories(array):
        # find the subarrays with contiguous zeroes until you get a 1

        # store the start and end indices into a tuple

        # combine these tuples into an array
        prev = -1
        start = 0
        end = 0
        tuple_array = []
        for i in range(len(array)):
            if prev == 0 and array[i] == 0:
                end += 1
            elif prev == 0 and array[i] == 1:
                end = i
                tuple_array.append((start, end))
            elif prev == 1 and array[i] == 1:
                continue
            elif prev == 1 and array[i] == 0:
                start = i
                end = i
            prev = array[i]
        return tuple_array


# ──────────────────────────────────────────────
# PaddlePipeline class
# ──────────────────────────────────────────────

class PaddlePipeline:
	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)

		self.source_env_params = self.set_source_env_cfg()
		self.param_names, self.lower_bounds, self.upper_bounds, self.param_start_values = self.load_sys_id_yaml()
		self.dataframe_columns = self.make_dataframe_columns(self.param_names, self.object_name)
		self.df = pd.DataFrame(columns=self.dataframe_columns)
		self.ground_truth_results = None

	def set_source_env_cfg(self):
		with open(self.cfg, 'r') as f:
			air_hockey_cfg = yaml.load(f, Loader=yaml.FullLoader)
		return air_hockey_cfg['air_hockey']

	def load_sys_id_yaml(self):
		with open(self.sys_id_path_paddle, 'r') as f:
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

	def make_dataframe_columns(self, array, object_name):
		columns = []
		for s in array:
			columns.append(f'initial_{s}_param_value')
		for s in array:
			columns.append(f'best_{s}_param_value')
		columns.append(f'{object_name}_traj_loss')
		return columns

	def perturb_initial_params(self, initial_params, lower, upper):
		initial_params = np.array(initial_params)
		lower = np.array(lower)
		upper = np.array(upper)
		d = len(initial_params)
		values = []

		for i in range(d):
			base = initial_params[i]
			max_delta = min(base - lower[i], upper[i] - base)
			perturbation = np.random.uniform(-max_delta, max_delta)
			perturbed_value = base + perturbation
			values.append(perturbed_value)

		return np.array(values)

	def evaluate_reset(self, i, task_name, eval_env, trajectories):
		# start the trajectory at a random time before num_steps from the end if rand_start, 0 otherwise
		start = 0

		evaluated_states = list()
		if 'goal' in task_name:
			eval_env.reset_from_state_and_goal(trajectories['observations'][i, 0, :-2], trajectories['observations'][i, 0, -2:])  ## todo, make this more general
		else:
			eval_env.reset_from_state(trajectories['observations'][i, start])
		images = list()
		# renderer = AirHockeyRenderer(eval_env)
		for j in range(start, trajectories['actions'].shape[1] - start - 1):
			# images.append(renderer.get_frame())
			act = trajectories['actions'][i, j]
			evaluated_state = eval_env.step(act)[0]
			evaluated_states.append(evaluated_state)
		# print("start", trajectories['observations'][i, start, :2], np.concatenate([trajectories['observations'][i, :10, :2],trajectories['observations'][i, :10, 4:6], np.stack(evaluated_states[:10], axis=0)[:, :2]], axis=-1))
		# cv2.imshow("frame", trajectories['images'][i,start])
		# cv2.imshow("img", frames[0])
		# cv2.waitKey(10000)
		# error
		# print("start", trajectories['observations'][i, start, 4:8], np.concatenate([trajectories['observations'][i, :10, 4:8], np.stack(evaluated_states[:10], axis=0)[:, 4:8]], axis=-1))
		return evaluated_states

	def compute_comparison(self, evaluated_states, trajectories, object_name, comp_type, new_config):
		# modify states by the offset
		if object_name == "paddle":
			evaluated_states = evaluated_states[:, :,:4]
			target_states = trajectories['observations'][:, :,:4]
			if "x_offset" in new_config["simulator_params"]: evaluated_states[:,0] += new_config["simulator_params"]["x_offset"]
			if "y_offset" in new_config["simulator_params"]: evaluated_states[:,1] += new_config["simulator_params"]["y_offset"]
		elif object_name == "puck":
			target_states = trajectories['observations'][:, :, 4:8]
			evaluated_states = evaluated_states[:, :, 4:8]
			if "x_offset" in new_config["simulator_params"]: evaluated_states[:,0] += new_config["simulator_params"]["x_offset"]
			if "y_offset" in new_config["simulator_params"]: evaluated_states[:,1] += new_config["simulator_params"]["y_offset"]
		elif object_name == "all":
			target_states = trajectories['observations']
			# target_states = trajectories['observations'][:, :8]
			# evaluated_states = evaluated_states[:, :8]
			if "x_offset" in new_config["simulator_params"]: evaluated_states[:,0] += new_config["simulator_params"]["x_offset"]
			if "y_offset" in new_config["simulator_params"]: evaluated_states[:,1] += new_config["simulator_params"]["y_offset"]
			if "x_offset" in new_config["simulator_params"]: evaluated_states[:,4] += new_config["simulator_params"]["x_offset"]
			if "y_offset" in new_config["simulator_params"]: evaluated_states[:,5] += new_config["simulator_params"]["y_offset"]
		value = compare_trajectories(evaluated_states, target_states, comp_type=comp_type)
		return value

	def get_value(self, param_vector, trajectories):
		evaluated_states = run_single_episode(param_vector=param_vector, param_names=self.param_names, base_config=self.source_env_params, action_sequence=self.ground_truth_results['actions'], model_path=self.model_path, ground_truth=False)
		states = []
		for s in trajectories['start_points']:
			states.append(evaluated_states['observations'][s: s + trajectories['traj_length']])
		sample_eval_states = np.stack(states, axis=0)
		value = self.compute_comparison(sample_eval_states, trajectories, self.object_name, self.comp_type, evaluated_states['new_config'])
		return value, evaluated_states

	def run_source_episode(self):
		#Ground truth trajectories
		self.ground_truth_results = run_single_episode(
			model_path=self.model_path,
			param_vector=None,
			param_names=None,
			action_sequence=None,
			base_config=self.source_env_params,
			render=self.render,
			device=self.device,
			save_plot=not self.no_plot,
			plot_dir=self.plot_dir
		)
		#testing the puck_sample_hit stuff
		# indices = list(range(0, len(self.ground_truth_results['hits_array'])))
		# zip_hits_indices = zip(self.ground_truth_results['hits_array'], indices)
		# # print(list(zip_hits_indices))
		# t_array = divide_into_non_interaction_subarrays_for_puck_trajectories(self.ground_truth_results['hits_array'])

	def write_episode_to_run_dir(self):
		run_dir = f'{self.run_dir}/{self.run_id}'
		os.makedirs(run_dir, exist_ok=True)

		action_sequence_dir = f'{self.run_dir}/{self.run_id}/action_sequence.npy'
		np.save(Path(action_sequence_dir), self.ground_truth_results['actions'])
		data = np.load(action_sequence_dir)
		print(f'The length is: {len(data)}')
		print(data)

	def write_cma_params_to_run_dir(self):
		run_dir = f'{self.run_dir}/{self.run_id}'
		cma_yaml_file = os.path.join(run_dir, 'cma_params.yaml')
		cma_dict = {
			'num_resample': self.num_resamples,
			'traj_length': self.traj_length,
			'num_populations': self.num_population,
			'num_iterations': self.num_iterations,
			'variance': self.variance,
			'simulation_iterations': self.simulation_iterations
		}
		with open(cma_yaml_file, 'w') as file:
			yaml.dump(cma_dict, file, default_flow_style=False)

	def run_simulation(self):
		# Need to get the best running params from the simulation
		# Need to plot the loss function on the same graph over all the iterations
		# Need to plot a distribution of the initialization with their respective losses
		# Need to make a config file with different CMA parameters to test for

		#loss graph to output
		fig, ax = plt.subplots(figsize=(8, 5))

		#This variable is not being used
		all_losses = []

		run_dir = f'{self.run_dir}/{self.run_id}'

		for i in range(self.simulation_iterations):
			#perturbs initial starting parameters
			print(f"Iteration: {i}")
			df = pd.DataFrame(columns=self.dataframe_columns)
			initial_params = self.perturb_initial_params(self.param_start_values, self.lower_bounds, self.upper_bounds)
			planner = CMAPlanner(
				eval_fn=lambda params, trajs: self.get_value(params, trajs),
				trajectories=self.ground_truth_results,
				n_samples=self.num_population,
				n_iterations=self.num_iterations,
				variance=self.variance,
				n_starts=self.num_resamples,
				traj_length=self.traj_length,
				lower_bounds=self.lower_bounds,
				upper_bounds=self.upper_bounds,
				param_names=self.param_names,
				wdb_logging=len(self.wdb_entity) > 0
			)
			optimal_parameters, optimal_parameters_iteration_num, best_traj_error, best_traj_error_index = planner.optimize(initial_params)

			assert optimal_parameters_iteration_num == best_traj_error_index, "ASSERTION NOT SATISFIED"

			df_values = []
			df_values.extend(initial_params)
			df_values.extend(optimal_parameters)
			df_values.append(float(best_traj_error))

			#adding the row of initial parameters, final parameters, and loss to my dataframe
			df.loc[0] = df_values

			curr_losses = planner.best_values_history

			# initial_evaluated_results = run_single_episode(
			# 	param_vector=initial_params,
			# 	param_names=self.param_names,
			# 	action_sequence=self.ground_truth_results['actions'],
			# 	base_config=self.source_env_params,
			# 	model_path=self.model_path,
			# 	ground_truth=False
			# )

			# final_evaluated_results = run_single_episode(
			# 	param_vector=optimal_parameters,
			# 	param_names=self.param_names,
			# 	action_sequence=self.ground_truth_results['actions'],
			# 	base_config=self.source_env_params,
			# 	model_path=self.model_path,
			# 	ground_truth=False
			# )

			ax.plot(curr_losses, alpha=0.5)
			# Improve readability
			ax.set_title("Loss Curves Across Trials")
			ax.set_xlabel("Iteration")
			ax.set_ylabel("Loss")
			plot_path = os.path.join(run_dir, 'loss_curves.png')
			fig.savefig(plot_path, dpi=150, bbox_inches="tight")

			#continuously writing to the csv file
			if i == 0:
				df.to_csv(os.path.join(run_dir, "data.csv"), index=False, mode='w', header=True)
			else:
				df.to_csv(os.path.join(run_dir, "data.csv"), index=False, mode='a', header=False)
		plt.close(fig)

		# initial_evaluated_states = initial_evaluated_results['observations']
		# initial_evaluated_config = initial_evaluated_results['new_config']
		# final_evaluated_states = final_evaluated_results['observations']
		# final_evaluated_config = final_evaluated_results['new_config']

		# for k in final_evaluated_config['simulator_params']:
		# 	final_evaluated_config['simulator_params'][k] = float(final_evaluated_config['simulator_params'][k])
		# 	initial_evaluated_config['simulator_params'][k] = float(initial_evaluated_config['simulator_params'][k])

		# with open(self.optimal_args_path, 'w') as file:
		# 	yaml.dump(final_evaluated_config, file, default_flow_style=False)
		# with open(self.initial_args_path, 'w') as file:
		# 	yaml.dump(initial_evaluated_config, file, default_flow_style=False)
		# evaluate_iterative_smoothing(self.model_path, self.save_dir, initial_evaluated_config, self.ground_truth_results['actions'], "initial.gif")
		# evaluate_iterative_smoothing(self.model_path, self.save_dir, self.source_env_params, self.ground_truth_results['actions'], "source.gif")
		# evaluate_iterative_smoothing(self.model_path, self.save_dir, final_evaluated_config, self.ground_truth_results['actions'], "target.gif")

		# source_path_gif = 'scripts/domain_adaptation/system_identification/sim2sim/agent/evals/source.gif'
		# target_path_gif = 'scripts/domain_adaptation/system_identification/sim2sim/agent/evals/target.gif'
		# synchronized_path_gif = 'scripts/domain_adaptation/system_identification/sim2sim/agent/evals/synchronized.gif'
		# produce_synchronized_gifs(source_path_gif, target_path_gif, synchronized_path_gif)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
	parser.add_argument('--cfg', type=str, default='agents/static_scaling/config.yaml', help='Path to the configuration file.')
	parser.add_argument('--paddle_configs', type=str, default='scripts/domain_adaptation/system_identification/sim2sim/agent/paddle_runs/1/optimal_final_args.yaml')
	parser.add_argument('--sys-id-path-paddle', type=str, default='scripts/domain_adaptation/sys_id_configs/sim2simagent/paddle_id_params_to_estimate.yaml', help='Path to the configuration file.')
	parser.add_argument('--model-path', type=str, default='agents/static_scaling/iterative_smoothing_model.pth', help='Path to the saved model state dict')
	parser.add_argument('--render', action='store_true', help='Render the episode')
	parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (cpu/cuda)')
	parser.add_argument('--no-plot', action='store_true', help='Skip creating action plot')
	parser.add_argument('--plot-dir', type=str, default=None, help='Directory to save the action plot (default: current directory)')
	parser.add_argument('--wdb-entity', type=str, default='', help='who to log wdb to.')
	parser.add_argument('--comp-type', type=str, default='l2', help='how to do comparisons.')
	parser.add_argument('--optimizer', type=str, default='cma', choices=['cma', 'cem'], help='Optimizer to use (CMA-ES or CEM)')
	parser.add_argument('--object-name', type=str, default='paddle', help='which object to do comparisons on. "all" compares the whole state')
	parser.add_argument('--run-id', type=str, default='9', help="logging identification")

	#cma parameters
	parser.add_argument('--num-resamples', type=int, default=100, help="number of subsequences to sample")
	parser.add_argument('--traj-length', type=int, default=200, help="maximum length of a subsequence")
	parser.add_argument('--num-population', type=int, default=20, help="population size for CMA")
	parser.add_argument('--num-iterations', type=int, default=100, help="number of steps for learning")
	parser.add_argument('--variance', type=float, default=0.2, help="variance of the perturbation")
	parser.add_argument('--simulation-iterations', type=int, default=100, help="simulation iterations number")

	parser.add_argument('--run-dir', type=str, default="scripts/domain_adaptation/system_identification/sim2sim/agent/paddle_runs", help='Path to save a run experiment')
	parser.add_argument('--save-dir', type=str, default="scripts/domain_adaptation/system_identification/sim2sim/agent/evals", help='Path to save the evaluation gifs to.')
	parser.add_argument('--total-loss-curve', type=str, default="scripts/domain_adaptation/system_identification/sim2sim/agent/paddle_runs")
	parser.add_argument('--optimal-args-path', type=str, default='scripts/domain_adaptation/system_identification/sim2sim/agent/evals/optimal_final_args.yaml')
	parser.add_argument('--initial-args-path', type=str, default='scripts/domain_adaptation/system_identification/sim2sim/agent/evals/initial_args.yaml')
	parser.add_argument('--test_config', type=str, default='scripts/domain_adaptation/system_identification/sim2sim/agent/test_stuff/test_params.yaml', help='Path to the configuration file.')
	args = parser.parse_args()

	paddle_pipeline = PaddlePipeline(**vars(args))
	paddle_pipeline.run_source_episode()
	paddle_pipeline.write_episode_to_run_dir()
	paddle_pipeline.write_cma_params_to_run_dir()
	paddle_pipeline.run_simulation()
