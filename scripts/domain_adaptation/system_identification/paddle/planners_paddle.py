import numpy as np
from tqdm import tqdm
import wandb
import time
import cma
from multiprocessing import Pool

from scripts.domain_adaptation.normalization import MinMaxNormalizer

def _evaluate_params_parallel(args_tuple):
    """Wrapper function for multiprocessing pool map."""
    # args_tuple: (sample, trajs, eval_fn)
    sample, trajs, eval_fn = args_tuple
    return eval_fn(sample, trajs)

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
            # 1. Prepare arguments: Create a list of tuples, one for each sample
            args_list = [(sample, trajs, self.eval_fn) for sample in samples]
            
            # 2. Parallel Evaluation
            # Use a pool with a suitable number of processes (e.g., the population size or number of CPU cores)
            num_processes = 16 # Adjust this based on your machine's CPU count
            with Pool(processes=num_processes) as pool:  
                rewards = np.array(pool.map(_evaluate_params_parallel, args_list))
            # rewards = np.array([self.eval_fn(sample, trajs) for sample in samples])

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
class CMAPlanner:
    """
    CMA-ES based planner for parameter optimization.
    Compatible with CEMPlanner interface.
    """
    
    def __init__(self, eval_fn, trajectories, elite_frac=0.2, n_samples=100, 
                 n_iterations=10, variance=0.2, n_starts=20, traj_length=50,
                 lower_bounds=None, upper_bounds=None, param_names=None, 
                 wdb_logging=False):
        """
        Args:
            eval_fn: Evaluation function that takes (params, trajectories) and returns reward
            trajectories: Dataset of trajectories
            elite_frac: Fraction of elite samples (not used in CMA-ES, kept for compatibility)
            n_samples: Population size for CMA-ES
            n_iterations: Maximum number of iterations
            variance: Initial step size (sigma0) for CMA-ES
            n_starts: Number of trajectory samples to evaluate on
            traj_length: Length of trajectory samples
            lower_bounds: Lower bounds for parameters
            upper_bounds: Upper bounds for parameters
            param_names: Names of parameters (for logging)
            wdb_logging: Whether to log to Weights & Biases
        """
        self.eval_fn = eval_fn
        #these are the ground truth results from running the single run of the episode for the source environment
        self.data = trajectories

        self.n_samples = n_samples  # Population size
        self.n_iterations = n_iterations
        self.variance = variance  # sigma0
        self.n_starts = n_starts
        self.traj_length = traj_length
        self.param_names = param_names
        self.wdb_logging = wdb_logging
        
        # Parameter bounds
        self.lower_bounds = np.array(lower_bounds) if lower_bounds is not None else None
        self.upper_bounds = np.array(upper_bounds) if upper_bounds is not None else None
        self.normalizer = MinMaxNormalizer(min_val=self.lower_bounds, max_val=self.upper_bounds)
        
        # CMA-ES state
        self.es = None
        self.mean = None
        self.best_values_history = []
        # self.tuple_non_hit_array_start_end = self.divide_into_non_interaction_subarrays_for_puck_trajectories()
        
    def initialize(self, initial_guess):
        """Initialize CMA-ES with initial guess."""
        initial_guess = np.array(initial_guess)
        eps = 1e-3  # Instead of 1e-8, so CMA has room to explore
        initial_guess_normalized = self.normalizer.normalize(initial_guess)
        initial_guess_normalized = np.clip(initial_guess_normalized, eps, 1 - eps)
        print("="*60)
        print("The normalized params are")
        print(initial_guess_normalized)
        print("="*60)
        self.mean = initial_guess_normalized

        sigma0 = 0.1

        LB = np.zeros(len(initial_guess_normalized))
        UB = np.ones(len(initial_guess_normalized))

        # CMA-ES options
        opts = {
            'maxiter': self.n_iterations,
            'popsize': self.n_samples,
            'bounds': [LB,UB],  # Normalized bounds
            'verbose': -9,  # Suppress CMA-ES output (we use tqdm)
            'tolfun': 1e-8,
            'tolx': 1e-8,
        }

        self.es = cma.CMAEvolutionStrategy(initial_guess_normalized, sigma0, opts)
    def divide_into_non_interaction_subarrays_for_puck_trajectories(self):
        # find the subarrays with contiguous zeroes until you get a 1

        # store the start and end indices into a tuple

        # combine these tuples into an array
        prev = -1
        start = 0
        end = 0
        tuple_array = []
        for i in range(self.data['hits_array']):
            if prev == 0 and self.data['hits_array'][i] == 0:
                end += 1
            elif prev == 0 and self.data['hits_array'][i] == 1:
                end = i
                tuple_array.append((start, end))
            elif prev == 1 and self.data['hits_array'][i] == 1:
                continue
            elif prev == 1 and self.data['hits_array'][i] == 0:
                start = i
                end = i
            prev = self.data['hits_array'][i]
        return tuple_array
    def sample_trajectories(self, iteration, num_samples=20, traj_length=200):
        observations = self.data['observations']  # shape: [N, obs_dim]
        actions = self.data['actions']            # shape: [N, act_dim]

        N = len(observations)
        if iteration == 0:
            print("="*60)
            print(f'The number of observations is {N}')
            print(f'The length of the trajecotry is {traj_length}')
            print("="*60)
        sampled_obs_segments = []
        sampled_act_segments = []
        start_points = []
        for _ in range(num_samples):
            start = np.random.randint(0, N - traj_length)
            start_points.append(start)
            sampled_obs_segments.append(observations[start: start + traj_length])
            sampled_act_segments.append(actions[start: start + traj_length])

        sampled_states = np.stack(sampled_obs_segments, axis=0)
        sampled_actions = np.stack(sampled_act_segments, axis=0)
        return {'observations': sampled_states, 'actions': sampled_actions, 'start_points': start_points, 'traj_length': traj_length}
    
    def optimize(self, initial_guess):
        """Run CMA-ES optimization."""
        self.initialize(initial_guess)

        pbar = tqdm(range(self.n_iterations), desc='Initializing...')
        total_sampling_time = 0
        total_evaluation_time = 0
        total_update_time = 0

        iteration = 0
        while not self.es.stop() and iteration < self.n_iterations:
            start_time = time.time()

            samples_normalized = self.es.ask()

            samples = np.array([self.normalizer.denormalize(s) for s in samples_normalized])
            samples = np.clip(samples, self.lower_bounds, self.upper_bounds)

            denormed_mean = self.normalizer.denormalize(self.es.mean)
            if self.wdb_logging and self.param_names is not None:
                for dim in range(len(self.param_names)):
                    wandb.log({self.param_names[dim]: denormed_mean[dim]}, step=iteration)

            total_sampling_time += time.time() - start_time

            trajs = self.sample_trajectories(iteration, num_samples=self.n_starts, traj_length=self.traj_length,)
            start_time = time.time()
            rewards = np.array([self.eval_fn(sample, trajs)[0] for sample in samples])
            total_evaluation_time += time.time() - start_time

            start_time = time.time()
            self.es.tell(samples_normalized, -rewards)
            total_update_time += time.time() - start_time

            avg_reward = np.mean(np.abs(rewards))
            worst_reward = np.max(np.abs(rewards))
            best_reward = np.min(np.abs(rewards))

            self.best_values_history.append(best_reward)

            pbar.update(1)
            pbar.set_description(
                f"Iteration {iteration}, Avg Return: {avg_reward:.2f}, "
                f"Worst Return: {worst_reward:.2f}, Best Return: {best_reward:.2f}"
            )

            if self.wdb_logging:
                wandb.log({
                    "Average Return": avg_reward,
                    "Worst Return": worst_reward,
                    "Best Return": best_reward,
                    "Total Sampling Time": total_sampling_time,
                    "Total Evaluation Time": total_evaluation_time,
                    "Total Update Time": total_update_time
                }, step=iteration)

            iteration += 1

        stop_conditions = self.es.stop()
        if stop_conditions:
            print(f"CMA-ES stopped after {iteration} iterations. Stop conditions: {stop_conditions}")
        else:
            print(f"CMA-ES completed all {self.n_iterations} iterations.")

        pbar.close()

        best_normalized = self.es.result.xbest
        best_traj_error_index = np.argmin(self.best_values_history)
        best_traj_error = self.best_values_history[best_traj_error_index]
        best_params = self.normalizer.denormalize(best_normalized)
        iteration_of_best = (self.es.result.evals_best - 1) // self.es.popsize
        return best_params, iteration_of_best, best_traj_error, best_traj_error_index
