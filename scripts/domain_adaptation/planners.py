import numpy as np
from tqdm import tqdm
import wandb
import time
from multiprocessing import Pool

from scripts.domain_adaptation.normalization import MinMaxNormalizer

class CEMPlanner:
    def __init__(self, eval_fn, trajectories, elite_frac=0.2, n_samples=100, n_iterations=10, variance=0.1, n_starts=20, traj_length=50, lower_bounds=None, upper_bounds=None, param_names=None, wdb_logging=False):
        # Existing initialization
        self.eval_fn = eval_fn
        self.data = trajectories
        self.elite_frac = elite_frac
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.variance = variance
        # Initialize distribution parameters
        self.mean = None
        self.std = None
        self.param_names = param_names
        # New: Parameter bounds
        self.lower_bounds = np.array(lower_bounds) if lower_bounds is not None else None
        self.upper_bounds = np.array(upper_bounds) if upper_bounds is not None else None
        self.normalizer = MinMaxNormalizer(min_val=self.lower_bounds, max_val=self.upper_bounds)
        self.n_starts = n_starts
        self.traj_length = traj_length
        self.wdb_logging = wdb_logging

    def initialize(self, initial_guess):
        """Initialize the distribution parameters based on the initial guess."""
        initial_guess = np.array(initial_guess)
        initial_guess = self.normalizer.normalize(initial_guess)

        self.mean = initial_guess
        self.std = np.ones_like(self.mean) * self.variance

    def sample_actions(self):
        """Sample a set of actions based on the current distribution parameters and clip them to the bounds."""
        samples = np.random.randn(self.n_samples, self.mean.shape[0]) * self.std + self.mean
        # Clip the samples if bounds are defined
        if self.lower_bounds is not None and self.upper_bounds is not None:
            samples = self.normalizer.denormalize(samples)
            samples = np.clip(samples, self.lower_bounds, self.upper_bounds)
        return samples
    
    def sample_trajectories(self, num_samples=20, traj_length=20):
        N = len(self.data['observations'])
        random_idx = np.random.randint(0, N, num_samples)
        random_starts = [np.random.randint(0, len(self.data["observations"][idx]) - traj_length) for idx in random_idx]

        sampled_states = np.concatenate([self.data['observations'][idx][random_start: random_start + traj_length][None, :] 
                                         for idx, random_start in zip(random_idx, random_starts)], axis=0)
        sampled_actions = np.concatenate([self.data['actions'][idx][random_start: random_start + traj_length][None, :] 
                                          for idx, random_start in zip(random_idx, random_starts)], axis=0)
        sampled_dones = np.concatenate([self.data['terminals'][idx][random_start: random_start + traj_length][None, :]
                                        for idx, random_start in zip(random_idx, random_starts)], axis=0)
        sampled_images = np.concatenate([self.data['images'][idx][random_start: random_start + traj_length][None, :]
                                        for idx, random_start in zip(random_idx, random_starts)], axis=0)
        # Create a dictionary for the sampled trajectory
        sampled_traj = {
            'observations': sampled_states,
            'actions': sampled_actions,
            'terminals': sampled_dones,
            'images': sampled_images
        }
            
        return sampled_traj

    def update_plan(self, rewards, samples):
        """Update the distribution parameters based on the sampled rewards."""
        # Rank the samples based on the rewards
        elite_idx = np.argsort(rewards)[-int(self.n_samples * self.elite_frac):]
        if self.lower_bounds is not None and self.upper_bounds is not None:
            elite_samples = self.normalizer.normalize(samples[elite_idx])
        else:
            elite_samples = samples[elite_idx]

        # Update the distribution parameters
        self.mean = np.mean(elite_samples, axis=0)
        self.std = np.std(elite_samples, axis=0) + 1e-6  # Add a small value to avoid division by zero

        return self.mean  # This could be the new guess for the next iteration

    def optimize(self, initial_guess):
        """Run the CEM optimization process."""
        self.initialize(initial_guess)
        pbar = tqdm(range(self.n_iterations), desc='Initializing...')

        total_sampling_time = 0
        total_evaluation_time = 0
        total_update_time = 0

        for iteration in pbar:
            start_time = time.time()
            samples = self.sample_actions()
            for dim in range(len(self.param_names)):
                denormed_mean = self.normalizer.denormalize(self.mean)
                if self.wdb_logging: wandb.log({self.param_names[dim]: denormed_mean[dim]}, step=iteration)
            total_sampling_time += time.time() - start_time

            trajs = self.sample_trajectories(num_samples=20, traj_length=50)

            start_time = time.time()
            rewards = np.array([self.eval_fn(sample, trajs) for sample in samples])
            # with Pool(processes=self.n_samples) as pool:  # Adjust the number of processes based on your CPU
                # rewards = np.array(pool.map(self.eval_fn, samples))
            total_evaluation_time += time.time() - start_time
            avg_reward = np.mean(rewards)
            min_reward = np.min(rewards)
            max_reward = np.max(rewards)
            pbar.set_description(f"Iteration {iteration}, Avg Return: {avg_reward:.2f}, Min Return: {min_reward:.2f}, Max Return: {max_reward:.2f}")
            if self.wdb_logging: wandb.log({"Average Return": avg_reward, "Min Return": min_reward, "Max Return": max_reward}, step=iteration)
            
            start_time = time.time()
            self.update_plan(rewards, samples)
            total_update_time += time.time() - start_time

            if self.wdb_logging: wandb.log({"Total Sampling Time": total_sampling_time, "Total Evaluation Time": total_evaluation_time, "Total Update Time": total_update_time}, step=iteration)
        
        
        
        return self.normalizer.denormalize(self.mean)