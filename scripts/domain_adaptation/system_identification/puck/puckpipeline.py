import argparse
import yaml
import torch
import cv2
import imageio
import pandas as pd
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont
import random
import gymnasium as gym
from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from scripts.smooth_policy.agent import Agent
from scripts.smooth_policy.running.run_single_episode import *
from scripts.domain_adaptation.planners import CMAPlanner


# TODO: Implement the Pipelines as follows below
# Run multiple episodes from the source environment
# Use trajectory sampling to then extract the segments and then compute the loss with the ones from the target
# Use reset_from_state to mimic starting at a state and then rollout the trajectory
class PuckPipeline:
	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)
		
		self.source_env_params = self.set_source_env_cfg()
		self.episode_data = self.create_episode_data()

		# Target Environment Variables
		self.param_names, self.lower_bounds, self.upper_bounds, self.param_start_values = self.load_sys_id_yaml()
		
		# Dataframe to keep track of the iterations
		self.dataframe_columns = self.make_dataframe_columns(self.param_names, self.object_name)
		self.df = pd.DataFrame(columns = self.dataframe_columns)
		# a collection of the partial trajectories observed
		# one for the real and one for the simulation ones collected throughout the process
		self.partial_trajectories_base_path = os.path.join(self.run_dir, str(self.run_id), "partial_trajectories")

	
	def make_dataframe_columns(self, array, object_name: str):
		columns = []
		for s in array:
			columns.append(f'initial_{s}_param_value')
		for s in array:
			columns.append(f'best_{s}_param_value')
		columns.append(f'{object_name}_traj_loss')
		return columns
	
	def set_source_env_cfg(self):
		with open(self.cfg, 'r') as f:
			source_env_cfg = yaml.load(f, Loader = yaml.FullLoader)
		if 'air_hockey' in source_env_cfg:
			return source_env_cfg['air_hockey']
		return source_env_cfg
	def create_episode_data(self):
		episode_data = dict()
		for i in range(self.num_episodes):
			key = f'ep_{i}'
			episode_data[key] = {}
		return episode_data
	def run_episodes(self):
		"""
		Runs the source experiments and then assigns the data to the episode_data parameter
		"""
		for i in range(self.num_episodes):
			ground_truth_results = run_single_episode(
				model_path=self.model_path,
				param_vector = None, 
				param_names = None,
				action_sequence = None,
				base_config = self.source_env_params,
				render=self.render,
				device=self.device,
				save_plot=not self.no_plot,
				plot_dir=self.plot_dir
			)

			#assigning the data to the self.episode_data here
			key = f'ep_{i}'
			self.episode_data[key]['actions'] = ground_truth_results['actions']
			self.episode_data[key]['hits_array'] = ground_truth_results['hits_array']
			self.episode_data[key]['observations'] = ground_truth_results['observations']	
	
	def write_episodes_to_run_dir(self):
		"""
		Writes the actions sequences for the various episodes into the run directories
		"""
		run_base_path = os.path.join(self.run_dir, str(self.run_id))

		if os.path.exists(run_base_path):
			print(f"Cleaning up old data at: {run_base_path}")
			shutil.rmtree(run_base_path)
		os.makedirs(run_base_path, exist_ok=True)
		for i in self.episode_data:
			episode_folder = os.path.join(run_base_path, f'episode_{i}')
			os.makedirs(episode_folder, exist_ok=True)
			file_path = os.path.join(episode_folder, 'action_sequence.npy')
			np.save(file_path, self.episode_data[i]['actions'])
			data = np.load(file_path)
			print(f'Episode {i} saved. Data length: {len(data)}')
	def create_partial_trajectories_folder_for_simulation(self):
		if os.path.exists(self.partial_trajectories_base_path):
			print(f'Cleaning up old partial trajectories')
			shutil.rmtree(self.partial_trajectories_base_path)
		os.makedirs(self.partial_trajectories_base_path, exist_ok=True)
	def save_partial_trajectory_to_path(self, state_vector, action_vector, env_params, iteration, sample_num, subfolder):
		"""
		Save all the contents needed to create the partial trajectory gifs without generating the gif
		Will save the partial trajectory and the source trajectory for the puck in the SAME instance
		To Save:
			- state_vector:
			- obs:
				- Both Source and Current Simulation Environment
			- action_vector
			- env_params:
			- sample_num: the different params tested by cma on this iteration for evolution
		"""
		if os.path.exists(self.partial_trajectories_base_path):
			curr_iteration_folder = os.path.join(self.partial_trajectories_base_path, f'it_{iteration}')
			if not os.path.exists(curr_iteration_folder):
				os.makedirs(curr_iteration_folder)
			curr_sample_folder = os.path.join(curr_iteration_folder, f'sample_{sample_num}')
			if not os.path.exists(curr_sample_folder):
				os.makedirs(curr_sample_folder)
			subfolder_path = os.path.join(curr_sample_folder,f"pt_{subfolder}")
			os.makedirs(subfolder_path, exist_ok = True)
			action_vector_path = os.path.join(subfolder_path, 'action_vector')
			np.save(action_vector_path, action_vector)
			target_env_params_file = os.path.join(subfolder_path, 'target_env_params.yaml')
			with open(target_env_params_file, 'w') as file:
				yaml.dump(env_params, file, default_flow_style = False)
			state_vector_path = os.path.join(subfolder_path, 'state_vector')
			np.save(state_vector_path, state_vector)


		
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
			yaml.dump(cma_dict, file, default_flow_style= False)

	def load_sys_id_yaml(self):
		with open(self.sys_id_puck_paddle, 'r') as f:
			id_config = yaml.safe_load(f)
			param_names = list(id_config.keys())
			initial_params = param_names.sort()

			lower_bounds = list()
			upper_bounds = list()
			initial_params = list()
			for n in param_names:
				lower_bounds.append(id_config[n]["lower"])
				upper_bounds.append(id_config[n]["upper"])
				initial_params.append(id_config[n]["start"])
		return param_names, lower_bounds, upper_bounds, initial_params 			

	def run_simulation(self):
		fig, ax = plt.subplots(figsize=(8, 5))
		for i in range(args.simulation_iterations):
		#perturbs initial starting parameters
			print(f"Iteration: {i}")
			df = pd.DataFrame(columns = self.dataframe_columns)
			initial_params = self.perturb_initial_params(self.param_start_values, self.lower_bounds, self.upper_bounds)
			print("Initial Params Here: ", initial_params)
			planner = CMAPlanner(
				eval_fn=lambda params, trajs,sample_num, it: self.get_value(
					params, trajs, sample_num, it
				),
				trajectories=puck_pipeline.episode_data,
				n_samples= self.num_population, 
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
			run_dir = f'{self.run_dir}/{self.run_id}'
			curr_losses = planner.best_values_history

			ax.plot(curr_losses, alpha= 0.5)
			# Improve readability
			ax.set_title("Loss Curves Across Trials")
			ax.set_xlabel("Iteration")
			ax.set_ylabel("Loss")
			plot_path = os.path.join(run_dir, 'loss_curves.png')
			fig.savefig(plot_path, dpi=150, bbox_inches="tight")

			#continuously writing to the csv file
			if i == 0:
				df.to_csv(os.path.join(run_dir, "data.csv"), index = False, mode = 'w', header = True)
			else:
				df.to_csv(os.path.join(run_dir, "data.csv"), index = False, mode = 'a', header = False)

	def evaluate_reset(self):
		pass

	def get_value(self, param_vector, trajectories, sample_num, iteration):
		"""
		trajectories: sampled coming from the planners.py sample_trajectories file
		"""
		
		# TODO
		# For run_single_episodes, use the starting spots for the trajectories to get the loss until the puck falls
		# This way, we will get the proper evaluated states
		# From there, compute the loss only on the trajectory with the smaller length, most cases will be the evaluated state
		# Make sure to sample from multiple episodes and denote which episode/actions the trajectory is coming from (logic to be done in planners.py)
		evaluated_states = []
		source_states = []
		losses = []
		#curr_target_simulation_params = assign_values(param_vector, self.param_names, self.source_env_params)
		#generating 4 subgroups from this sampled parameter config to observe
		random_subgroup_sample = random.sample(range(0,len(trajectories)), 4)
		for i in range(len(trajectories)):
			curr_episode_id = trajectories['episode_ids'][i]
			state_start_index = trajectories['start_points'][i]
			state_start_vector = self.episode_data[curr_episode_id]['observations'][state_start_index]
			action_vector = trajectories['actions'][i]
			curr_traj_length = trajectories['traj_length']
			assert curr_traj_length == len(action_vector)
			# running an episode with the no contact zones - so the "done" parameter in run_single_episode should never sound
			curr_traj_evaluated_state = run_single_episode(param_vector = param_vector, param_names = self.param_names, 
												  base_config = self.source_env_params, action_sequence = action_vector, 
												  model_path = self.model_path, ground_truth = False, state_vector = state_start_vector,
												  traj_length = curr_traj_length)
			# if i in random_subgroup_sample:
			# 	for k in curr_traj_evaluated_state['new_config']['simulator_params']:
			# 		curr_traj_evaluated_state['new_config']['simulator_params'][k] = float(curr_traj_evaluated_state['new_config']['simulator_params'][k])
			# 	self.save_partial_trajectory_to_path(state_start_vector, action_vector, curr_traj_evaluated_state['new_config'], iteration, sample_num, i)
			episode_length = curr_traj_evaluated_state['episode_length']
			evaluated_states.append(curr_traj_evaluated_state['observations'])
			comp_length = min(episode_length, curr_traj_length)
			source_episode_observation = trajectories['observations'][i][0: comp_length]
			source_states.append(source_episode_observation)
			assert len(curr_traj_evaluated_state['observations'][0:comp_length]) == len(source_episode_observation)
			s = np.asarray(source_episode_observation[:, 4:8])
			t = np.asarray(curr_traj_evaluated_state['observations'][:, 4:8])
			diff = s - t
			step_l2 = np.linalg.norm(diff, axis=1)
			traj_loss = np.mean(step_l2)
			losses.append(traj_loss)
		#from CMA planner, contains the episode aligned observations
		#value = self.compute_comparison(evaluated_states, source_states, self.object_name, self.comp_type, new_config = None)
		value = -np.mean(losses)
		return value, evaluated_states
	
	def compare_trajectories(self, a_traj, b_traj, comp_type="l2"):
		"""
		a_traj, b_traj: list of state vectors (lists)
		returns: scalar fitness (higher is better)
		"""
		a = np.asarray(a_traj, dtype=np.float64)
		b = np.asarray(b_traj, dtype=np.float64)
		T = min(len(a), len(b))
		a = a[:T]
		b = b[:T]
		diffs = a - b  # (T, D)

		if self.comp_type == "l2":
			step_l2 = np.linalg.norm(diffs, axis=1)
			return -np.mean(step_l2)

		elif self.comp_type == "posl2":
			step_l2 = np.linalg.norm(diffs[:, :2], axis=1)
			return -np.mean(step_l2)

		elif self.comp_type == "l1":
			step_l1 = np.linalg.norm(diffs, ord=1, axis=1)
			return -np.mean(step_l1)

		elif self.comp_type == "posl1":
			step_l1 = np.linalg.norm(diffs[:, :2], ord=1, axis=1)
			return -np.mean(step_l1)

		elif self.comp_type == "last":
			return -np.linalg.norm(a[-1] - b[-1])

		else:
			raise ValueError(f"Unknown comp_type: {self.comp_type}")

	def compute_comparison(self, evaluated_states, trajectories, object_name, comp_type, new_config):
		# modify states by the offset
		if object_name == "paddle":
			evaluated_states = evaluated_states[:,:,:4]
			target_states = trajectories['observations'][:,:,:4]
			if "x_offset" in new_config["simulator_params"]: evaluated_states[:,0] += new_config["simulator_params"]["x_offset"]
			if "y_offset" in new_config["simulator_params"]: evaluated_states[:,1] += new_config["simulator_params"]["y_offset"]
		elif object_name == "puck":
			target_states = trajectories[:, 4:8]
			evaluated_states = evaluated_states[:, 4:8]
			# if "x_offset" in new_config["simulator_params"]: evaluated_states[:,0] += new_config["simulator_params"]["x_offset"]
			# if "y_offset" in new_config["simulator_params"]: evaluated_states[:,1] += new_config["simulator_params"]["y_offset"]
		elif object_name == "all":
			target_states = trajectories['observations']
			# target_states = trajectories['observations'][:, :8]
			# evaluated_states = evaluated_states[:, :8]
			if "x_offset" in new_config["simulator_params"]: evaluated_states[:,0] += new_config["simulator_params"]["x_offset"]
			if "y_offset" in new_config["simulator_params"]: evaluated_states[:,1] += new_config["simulator_params"]["y_offset"]
			if "x_offset" in new_config["simulator_params"]: evaluated_states[:,4] += new_config["simulator_params"]["x_offset"]
			if "y_offset" in new_config["simulator_params"]: evaluated_states[:,5] += new_config["simulator_params"]["y_offset"]
		value = self.compare_trajectories(evaluated_states, target_states, comp_type=comp_type)
		return value

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

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
	
	# Original arguments
	parser.add_argument('--cfg', type=str, default='agents/static_scaling/config.yaml', help='Path to the configuration file.')
	parser.add_argument('--paddle-configs', type=str, default='scripts/domain_adaptation/system_identification/sim2sim/agent/paddle_runs/1/optimal_final_args.yaml')
	parser.add_argument('--sys-id-puck-paddle', type=str, default='scripts/domain_adaptation/sys_id_configs/sim2simagent/puck_id_params_to_estimate.yaml', help='Path to the configuration file.')
	parser.add_argument('--model-path', type=str, default='agents/static_scaling/iterative_smoothing_model.pth', help='Path to the saved model state dict')
	parser.add_argument('--render', action='store_true', help='Render the episode')
	parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (cpu/cuda)')
	parser.add_argument('--no-plot', action='store_true', help='Skip creating action plot')
	parser.add_argument('--plot-dir', type=str, default=None, help='Directory to save the action plot (default: current directory)')
	parser.add_argument('--wdb-entity', type=str, default='', help='who to log wdb to.')
	parser.add_argument('--comp-type', type=str, default='l2', help='how to do comparisons.')
	parser.add_argument('--optimizer', type=str, default='cma', choices=['cma', 'cem'], help='Optimizer to use (CMA-ES or CEM)')
	parser.add_argument('--object-name', type=str, default='puck', help='which object to do comparisons on. "all" compares the whole state')
	parser.add_argument('--run-id', type=str, default='21', help="logging identification")
	
	# CMA / optimization parameters
	parser.add_argument('--num-resamples', type=int, default=200, help="number of subsequences to sample")
	parser.add_argument('--traj-length', type=int, default=18, help="maximum length of a subsequence")
	parser.add_argument('--num-population', type=int, default=20, help="population size for CMA")
	parser.add_argument('--num-iterations', type=int, default=100, help="number of steps for learning")
	parser.add_argument('--variance', type=float, default=0.2, help="variance of the perturbation")
	parser.add_argument('--simulation-iterations', type=int, default=100, help="simulation iterations number")

	# Paths
	parser.add_argument('--run-dir', type=str, default="scripts/domain_adaptation/system_identification/sim2sim/agent/puck_runs", help='Path to save a run experiment')
	parser.add_argument('--save-dir', type=str, default="scripts/domain_adaptation/system_identification/sim2sim/agent/evals", help='Path to save the evaluation gifs to.')
	parser.add_argument('--total-loss-curve', type=str, default="scripts/domain_adaptation/system_identification/sim2sim/agent/puck_runs")
	parser.add_argument('--optimal-args-path', type=str, default='scripts/domain_adaptation/system_identification/sim2sim/agent/evals/optimal_final_args.yaml')
	parser.add_argument('--initial-args-path', type=str, default='scripts/domain_adaptation/system_identification/sim2sim/agent/evals/initial_args.yaml')
	parser.add_argument('--test_config', type=str, default='scripts/domain_adaptation/system_identification/sim2sim/agent/test_stuff/test_params.yaml', help='Path to the configuration file.')

	# New argument
	parser.add_argument('--num_episodes', type=int, default='3', help='Number of episodes to run')
	args = parser.parse_args()

	# Instantiate PuckPipeline directly from argparse using **vars(args)
	puck_pipeline = PuckPipeline(**vars(args))
	puck_pipeline.run_episodes()

	assert(len(puck_pipeline.episode_data) == args.num_episodes)

	puck_pipeline.write_episodes_to_run_dir()
	puck_pipeline.write_cma_params_to_run_dir()
	puck_pipeline.create_partial_trajectories_folder_for_simulation()
	puck_pipeline.run_simulation()
	







# TODO
# Add plotting logic for the loss_cuves
# Change the planners.py file for sampling from the puck trajectories
# Incorporate multiple episodes in the sampling of the directories
# Write tests to ensure everything is working properly
# Make the gif plots at different points in the trajectory
# Make the code production ready to be used in the demo for the Sim2Sim
# Make the PuckPipeLine Class (DONE)
# Establish the episode functions (run_episodes, write_episodes_to_run_dir)(DONE)
# Wall Collisions (DONE)
