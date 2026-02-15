import numpy as np
from tqdm import tqdm
import wandb
import time
import cma

from scripts.domain_adaptation.normalization import MinMaxNormalizer


class CMAPlanner:
    """
    CMA-ES based planner for parameter optimization.
    Supports both paddle (simple sampling) and puck (hits_array interval sampling) pipelines.
    """

    def __init__(self, eval_fn, trajectories, elite_frac=0.2, n_samples=100,
                 n_iterations=10, variance=0.2, n_starts=20, traj_length=50,
                 lower_bounds=None, upper_bounds=None, param_names=None,
                 wdb_logging=False, log_file=None):
        self.eval_fn = eval_fn
        # Normalize to list-of-episodes format.
        # Accepts: list of dicts, or a single dict (legacy single-trajectory).
        if isinstance(trajectories, list):
            self.episodes = trajectories
        elif isinstance(trajectories, dict):
            self.episodes = [trajectories]
        else:
            raise ValueError(f"Unexpected trajectories type: {type(trajectories)}")

        self.n_samples = n_samples  # Population size
        self.n_iterations = n_iterations
        self.variance = variance  # sigma0
        self.n_starts = n_starts
        self.traj_length = traj_length
        self.param_names = param_names
        self.wdb_logging = wdb_logging
        self.log_file = log_file

        # Parameter bounds
        self.lower_bounds = np.array(lower_bounds) if lower_bounds is not None else None
        self.upper_bounds = np.array(upper_bounds) if upper_bounds is not None else None
        self.normalizer = MinMaxNormalizer(min_val=self.lower_bounds, max_val=self.upper_bounds)

        # CMA-ES state
        self.es = None
        self.mean = None
        self.best_values_history = []

        # Use hits_array-based interval sampling if available, otherwise simple sampling
        has_hits = any('hits_array' in ep for ep in self.episodes)
        if has_hits:
            self.tuple_non_hit_array_start_end = self._divide_into_non_hit_intervals()
        else:
            self.tuple_non_hit_array_start_end = None

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

        sigma0 = self.variance

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

    def _divide_into_non_hit_intervals(self):
        """Find contiguous non-hit intervals across all episodes.

        Returns list of (episode_idx, start, end) tuples where
        hits_array[start:end] are all zeros.
        """
        print("Making the division across all episodes")
        tuple_array = []
        for ep_idx, ep in enumerate(self.episodes):
            hits = ep['hits_array']
            prev = -1
            start = 0
            for i in range(len(hits)):
                if prev == 0 and hits[i] == 0:
                    pass  # extend current run
                elif prev == 0 and hits[i] == 1:
                    tuple_array.append((ep_idx, start, i))
                elif prev == 1 and hits[i] == 0:
                    start = i
                # prev==1 and hits[i]==1: skip
                prev = hits[i]
            # Close out trailing non-hit run
            if prev == 0:
                tuple_array.append((ep_idx, start, len(hits)))

        print(f"  Found {len(tuple_array)} non-hit intervals across {len(self.episodes)} episodes")
        return tuple_array

    def sample_trajectories(self, iteration, num_samples=20, traj_length=200):
        if iteration == 0:
            total_obs = sum(len(ep['observations']) for ep in self.episodes)
            print("=" * 60)
            print(f'Total observations across {len(self.episodes)} episodes: {total_obs}')
            print(f'The length of the trajectory segment is {traj_length}')
            print("=" * 60)

        rng = np.random.RandomState(seed=iteration)

        if self.tuple_non_hit_array_start_end is not None:
            # Puck-style: sample from non-hit intervals
            valid_intervals = [(ep_idx, s, e) for ep_idx, s, e in self.tuple_non_hit_array_start_end
                               if e - s > traj_length]
            if len(valid_intervals) == 0:
                raise ValueError(f"No non-hit intervals are long enough for traj_length={traj_length}")

            sampled_obs_segments = []
            sampled_act_segments = []
            start_points = []
            for _ in range(num_samples):
                range_index = rng.randint(0, len(valid_intervals))
                ep_idx, start_episode, end_episode = valid_intervals[range_index]
                ep = self.episodes[ep_idx]

                start = np.random.randint(start_episode, end_episode - traj_length)
                start_points.append((ep_idx, start))
                sampled_obs_segments.append(ep['observations'][start: start + traj_length])
                sampled_act_segments.append(ep['actions'][start: start + traj_length])
        else:
            # Paddle-style: simple random sampling across episodes
            sampled_obs_segments = []
            sampled_act_segments = []
            start_points = []
            for _ in range(num_samples):
                ep_idx = rng.randint(0, len(self.episodes))
                ep = self.episodes[ep_idx]
                N = len(ep['observations'])
                start = rng.randint(0, N - traj_length)
                start_points.append((ep_idx, start))
                sampled_obs_segments.append(ep['observations'][start: start + traj_length])
                sampled_act_segments.append(ep['actions'][start: start + traj_length])

        sampled_states = np.stack(sampled_obs_segments, axis=0)
        sampled_actions = np.stack(sampled_act_segments, axis=0)
        return {'observations': sampled_states, 'actions': sampled_actions, 'start_points': start_points, 'traj_length': traj_length}

    def optimize(self, initial_guess):
        """Run CMA-ES optimization."""
        self.initialize(initial_guess)

        if self.log_file is not None:
            with open(self.log_file, "a") as lf:
                lf.write("iteration,min_reward,max_reward,std_reward\n")

        pbar = tqdm(range(self.n_iterations), desc='Initializing...')
        total_sampling_time = 0
        total_evaluation_time = 0
        total_update_time = 0

        iteration = 0

        #holdout samples
        holdout = self.sample_trajectories(1000000, num_samples = self.n_starts, traj_length = self.traj_length)

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

            trajs = self.sample_trajectories(iteration = iteration, num_samples=self.n_starts, traj_length=self.traj_length,)

            start_time = time.time()
            rewards = np.array([self.eval_fn(sample, trajs)[0] for sample in samples])

            #evaluating the sample parameters on holdout trajectories
            holdout_reward = np.array([self.eval_fn(holdout_sample, trajs)[0] for holdout_sample in holdout])
            print(f"Reward spread: min={min(rewards):.4f}, max={max(rewards):.4f}, std={np.std(rewards):.4f}")
            if self.log_file is not None:
                with open(self.log_file, "a") as lf:
                    lf.write(f"{iteration},{min(rewards):.6f},{max(rewards):.6f},{np.std(rewards):.6f}\n")
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
                    "Holdout Return": holdout_reward,
                    "Total Sampling Time": total_sampling_time,
                    "Total Evaluation Time": total_evaluation_time,
                    "Total Update Time": total_update_time
                }, step=iteration)

            iteration += 1

        pbar.close()

        stop_conditions = self.es.stop()
        if stop_conditions:
            print(f"CMA-ES stopped after {iteration} iterations. Stop conditions: {stop_conditions}")
        else:
            print(f"CMA-ES completed all {self.n_iterations} iterations.")

        best_normalized = self.es.result.xbest
        best_traj_error_index = np.argmin(self.best_values_history)
        best_traj_error = self.best_values_history[best_traj_error_index]
        best_params = self.normalizer.denormalize(best_normalized)
        iteration_of_best = (self.es.result.evals_best - 1) // self.es.popsize
        return best_params, iteration_of_best, best_traj_error, best_traj_error_index
