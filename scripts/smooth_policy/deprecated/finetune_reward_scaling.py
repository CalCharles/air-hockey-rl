import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import yaml

from airhockey import AirHockeyEnv
import gymnasium as gym

from dataclasses import dataclass
import tyro

import os
from datetime import datetime

from scripts.smooth_policy.evaluate import evaluate_iterative_smoothing
from scripts.smooth_policy.agent import Agent
from scripts.utils import save_tensorboard_plots


@dataclass
class FinetuneRewardScalingArgs:
    num_envs: int = 8
    num_steps: int = 512
    learning_rate: float = 1e-5  # Lower learning rate for finetuning
    num_iterations: int = 500
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    minibatch_size: int = 64
    update_epochs: int = 10
    clip_coef: float = 0.1  # Lower clip coefficient for finetuning
    clip_vloss: bool = True  # Use clipped value loss for stability
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    action_coef: float = 0.0
    max_grad_norm: float = 0.25  # Lower max grad norm for stability
    target_kl: float = None
    batch_size: int = 0  # computed in runtime

    config: str = "scripts/smooth_policy/configs/puck_touch/default_config.yaml"
    args_file: str = None
    model_path: str = None  # Path to pre-trained model state dict (REQUIRED for finetuning)
    run_name: str = "finetune_reward_scaling"

    # Finetuning specific parameters
    seed: int = 0
    norm_adv: bool = True
    reward_scaling: bool = True
    device: str = "cuda:0"
    log_parent_dir: str = None
    
    # Reward scaling specific parameters
    initial_base_reward_scaling: float = 1.0  # Starting reward scaling
    target_base_reward_scaling: float = 1.0/3  # Target minimum scaling (1/3)
    reward_scaling_factor: float = 0.9  # Factor to multiply by (9/10)
    performance_window_size: int = 40  # Size of performance tracking window
    stability_check_size: int = 20  # Size of recent vs previous comparison
    stability_threshold: float = 0.05  # 5% threshold for stability check
    use_target_base_reward_scaling: bool = False # if True, will stop training when the target reward scaling is reached
    
    # DEPRECATED/BACKWARD COMPATIBILITY parameters (set to 0 by default)
    performance_threshold: float = 0.0  # Not used in reward scaling approach
    performance_buffer_iterations: int = 5  # Not used
    caps_increment: float = 0.0  # Not used
    training_buffer_iterations: int = 5  # Not used
    initial_caps_coef_nearby: float = 0.0  # Not used
    initial_caps_coef_consecutive: float = 0.0  # Not used
    target_caps_coef: float = 0.0  # Not used
    use_target_caps_coef: bool = False  # Not used
    caps_coef_nearby: float = 0.0  # Not used
    caps_coef_consecutive: float = 0.0  # Not used
    dynamic_freeze_policy: bool = False  # Not used
    dynamic_reward_scaling: bool = False  # Not used
    finetune: bool = False  # Not used
    reward_normalization: bool = False


def make_env(env_id, initial_reward_scaling=1.0):
    def _thunk():
        curr_seed = random.randint(0, int(1e8))
        config["air_hockey"]["seed"] = curr_seed
        config["air_hockey"]["base_reward_scaling"] = initial_reward_scaling
        env = AirHockeyEnv(config["air_hockey"])
        return env
    return _thunk


class RewardScalingManager:
    def __init__(self, args):
        self.initial_scaling = args.initial_base_reward_scaling
        self.target_scaling = args.target_base_reward_scaling
        self.scaling_factor = args.reward_scaling_factor
        self.window_size = args.performance_window_size
        self.stability_size = args.stability_check_size
        self.threshold = args.stability_threshold
        
        # Current state
        self.current_scaling = self.initial_scaling
        self.episodic_returns = []  # Store last 40 iterations
        self.max_returns_per_iteration = []  # Store max return per iteration
        self.scaling_decreased = False
        
    def update_performance(self, avg_return, max_return, iteration):
        """Update performance tracking and check for stability"""
        # Add to tracking lists
        self.episodic_returns.append(avg_return)
        self.max_returns_per_iteration.append(max_return)
        
        # Keep only last window_size entries
        if len(self.episodic_returns) > self.window_size:
            self.episodic_returns.pop(0)
            self.max_returns_per_iteration.pop(0)
        
        # Check if we have enough data and should decrease scaling
        if len(self.episodic_returns) >= self.window_size:
            return self._check_stability()
        
        return False
    
    def _check_stability(self):
        """Check if performance has stabilized according to criteria"""
        if len(self.episodic_returns) < self.window_size:
            return False
        
        # Split into recent 20 and previous 20
        recent_20_returns = self.episodic_returns[-self.stability_size:]
        previous_20_returns = self.episodic_returns[-2*self.stability_size:-self.stability_size]
        
        recent_20_max = self.max_returns_per_iteration[-self.stability_size:]
        previous_20_max = self.max_returns_per_iteration[-2*self.stability_size:-self.stability_size]
        
        # Calculate averages
        avg_recent = np.mean(recent_20_returns)
        avg_previous = np.mean(previous_20_returns)
        avg_recent_max = np.mean(recent_20_max)
        avg_previous_max = np.mean(previous_20_max)
        
        # Check stability criteria (within 5%)
        return_stable = abs(avg_recent - avg_previous) <= self.threshold * abs(avg_previous)
        max_return_stable = abs(avg_recent_max - avg_previous_max) <= self.threshold * abs(avg_previous_max)
        
        print(f"Stability check - Recent avg: {avg_recent:.2f}, Previous avg: {avg_previous:.2f}")
        print(f"Stability check - Recent max avg: {avg_recent_max:.2f}, Previous max avg: {avg_previous_max:.2f}")
        print(f"Return stable: {return_stable}, Max return stable: {max_return_stable}")
        
        return return_stable and max_return_stable
    
    def decrease_scaling(self, envs):
        """Decrease the reward scaling and update environments"""
        if self.current_scaling <= self.target_scaling:
            return False  # Already at target
        
        # Calculate new scaling
        new_scaling = self.current_scaling * self.scaling_factor
        new_scaling = max(new_scaling, self.target_scaling)  # Don't go below target
        
        self.current_scaling = new_scaling
        
        # Update all environments
        self._update_env_reward_scaling(envs, new_scaling)
        
        # Reset performance tracking for next phase
        self.episodic_returns = []
        self.max_returns_per_iteration = []
        self.scaling_decreased = True
        
        return True
    
    def _update_env_reward_scaling(self, envs, new_scaling):
        """Update reward scaling for all environments"""
        try:
            envs.call('set_base_reward_scaling', new_scaling)
        except Exception as e:
            print(f"Warning: Could not update environment reward scaling: {e}")
            print("This may be due to vectorized environment limitations.")
    
    def should_continue(self):
        """Check if we should continue training (haven't reached target)"""
        return self.current_scaling > self.target_scaling


if __name__ == "__main__":
    temp_args = tyro.cli(FinetuneRewardScalingArgs)
    if temp_args.args_file is not None:
        with open(temp_args.args_file, "r") as f:
            file_args_dict = yaml.load(f, Loader=yaml.FullLoader)
        default_args = FinetuneRewardScalingArgs(**file_args_dict)
    else:
        default_args = FinetuneRewardScalingArgs()

    # Command line args override file args
    args = tyro.cli(FinetuneRewardScalingArgs, default=default_args)
    args.batch_size = args.num_envs * args.num_steps

    # Require model path for finetuning
    if args.model_path is None or not os.path.exists(args.model_path):
        raise ValueError(f"Model path is required for finetuning and must exist: {args.model_path}")

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create parallel envs with initial reward scaling
    envs = gym.vector.AsyncVectorEnv([make_env(i, args.initial_base_reward_scaling) for i in range(args.num_envs)])

    # Create folder with all results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    task_name = config["air_hockey"].get("task")
    run_name = args.run_name

    log_parent_dir = args.log_parent_dir
    if log_parent_dir is None:
        log_parent_dir = f"runs/finetune_reward_scaling/{task_name}/{run_name}_{timestamp}"
    if os.path.exists(log_parent_dir):
        raise FileExistsError(f"Log directory {log_parent_dir} already exists.")
    os.makedirs(log_parent_dir, exist_ok=True)

    writer = SummaryWriter(log_parent_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Save config and args
    with open(f"{log_parent_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(f"{log_parent_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)
    
    # Initialize agent and load pre-trained model
    agent = Agent(envs, init_reward_scaling=0.1).to(args.device)
    print(f"Loading pre-trained model from {args.model_path}")
    agent.load_state_dict(torch.load(args.model_path, map_location=args.device))
    print("Model loaded successfully")
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Initialize reward scaling manager
    scaling_manager = RewardScalingManager(args)

    # Training loop setup (same as iterative_smoothing)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(args.device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(args.device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(args.device)

    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(args.device)
    next_done = torch.zeros(args.num_envs).to(args.device)
    iteration = 1
    
    while True:
        # Reset episodic return tracking for this iteration
        episodic_returns = []
        success_rates = []
        
        # Annealing the rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Get current reward scaling
        current_base_reward_scaling = scaling_manager.current_scaling

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and log data
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            
            if args.reward_scaling:
                reward *= 0.1 # manually scale down

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(args.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(args.device), torch.Tensor(next_done).to(args.device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode_return" in info:
                        episodic_returns.append(info['episode_return'])
                        success_rates.append(1.0 if info['success'] else 0.0)
                        writer.add_scalar("charts/episodic_return", info['episode_return'], global_step)
                        writer.add_scalar("charts/episodic_length", info['episode_length'], global_step)

        # Bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(args.device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

            # Log statistics
            writer.add_scalar("charts/return_mean", returns.mean().item(), global_step)
            writer.add_scalar("charts/return_std", returns.std().item(), global_step)
            writer.add_scalar("charts/advantage_mean", advantages.mean().item(), global_step)
            writer.add_scalar("charts/advantage_std", advantages.std().item(), global_step)
            writer.add_scalar("charts/value_mean", values.mean().item(), global_step)
            writer.add_scalar("charts/value_std", values.std().item(), global_step)

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_dones = dones.reshape(-1).bool()

        # Evaluate smoothness metrics periodically for monitoring
        if iteration % 5 == 0: 
            with torch.no_grad():
                # Generate nearby observations for smoothness evaluation
                noise_std = 0.01
                nearby_obs = b_obs + torch.randn_like(b_obs) * noise_std
                
                # Get action means for current and nearby observations
                curr_actions_mean = agent.get_action_mean(b_obs)
                nearby_actions_mean = agent.get_action_mean(nearby_obs)
                
                # Calculate nearby action loss (smoothness w.r.t. observation perturbations)
                nearby_action_loss = ((curr_actions_mean - nearby_actions_mean) ** 2.0).mean()
                
                # Calculate consecutive action loss (smoothness w.r.t. temporal sequence)
                # Only consider non-terminal transitions
                non_done_mask = ~b_dones[:-1]  # Exclude last element to avoid index error
                if non_done_mask.sum() > 0:
                    consecutive_action_loss = ((curr_actions_mean[:-1] - curr_actions_mean[1:]) ** 2.0)[non_done_mask].mean()
                else:
                    consecutive_action_loss = torch.tensor(0.0, device=args.device)
                
                action_loss = (curr_actions_mean ** 2.0).mean()
                
                # Log evaluation metrics
                writer.add_scalar("eval/consecutive_action_loss", consecutive_action_loss.item(), global_step)
                writer.add_scalar("eval/nearby_action_loss", nearby_action_loss.item(), global_step)
                writer.add_scalar("eval/action_loss", action_loss.item(), global_step)



        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Action regularization (optional)
                action_loss = (b_actions[mb_inds] ** 2.0).mean()

                entropy_loss = entropy.mean()
                loss = (pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + 
                       args.action_coef * action_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log metrics
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/action_loss", action_loss.item(), global_step)
        
        # Log reward scaling
        writer.add_scalar("charts/base_reward_scaling", current_base_reward_scaling, global_step)
        
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Calculate and log episodic return statistics
        if episodic_returns:
            avg_return = np.mean(episodic_returns)
            min_return = np.min(episodic_returns)
            max_return = np.max(episodic_returns)
            
            print(f"Iteration {iteration}: Avg Return: {avg_return:.2f}, Min Return: {min_return:.2f}, Max Return: {max_return:.2f}")
            print(f"Iteration {iteration}: Avg Success Rate: {np.mean(success_rates):.2f}")
            print(f"Iteration {iteration}: Base Reward Scaling: {current_base_reward_scaling:.4f}")
            
            writer.add_scalar("charts/avg_episodic_return", avg_return, iteration)
            writer.add_scalar("charts/min_episodic_return", min_return, iteration)
            writer.add_scalar("charts/max_episodic_return", max_return, iteration)
            writer.add_scalar("charts/avg_success_rate", np.mean(success_rates), iteration)
            
            # Update scaling manager with performance
            should_decrease_scaling = scaling_manager.update_performance(avg_return, max_return, iteration)
            if should_decrease_scaling:

                # update the environments
                scaling_manager.decrease_scaling(envs)

                # Save model checkpoint
                checkpoint_dir = os.path.join(log_parent_dir, f"scaling_{scaling_manager.current_scaling:.4f}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model_path = f"{checkpoint_dir}/reward_scaling_model.pth"
                torch.save(agent.state_dict(), model_path)
                
                print(f"Performance stabilized! Reward scaling decreased to {scaling_manager.current_scaling:.4f}")
                print(f"Model saved to {model_path}")
                
                # Evaluate the model
                evaluate_iterative_smoothing(model_path, checkpoint_dir, config["air_hockey"], n_eps=6, n_gifs=1)
                
                writer.add_scalar("charts/reward_scaling_decreased", 1, iteration)
                
                # Check if we should stop
                if not scaling_manager.should_continue():
                    print("Reached target reward scaling. Training complete.")
                    break
            else:
                writer.add_scalar("charts/reward_scaling_decreased", 0, iteration)
        else:
            print(f"Iteration {iteration}: No episodes completed")
            print(f"Iteration {iteration}: Base Reward Scaling: {current_base_reward_scaling:.4f}")

        # Save checkpoints periodically
        if iteration % 50 == 0:
            checkpoint_dir = os.path.join(log_parent_dir, f"checkpoint_{iteration}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_path = f"{checkpoint_dir}/reward_scaling_model.pth"
            torch.save(agent.state_dict(), model_path)

            # Evaluate the model
            evaluate_iterative_smoothing(model_path, checkpoint_dir, config["air_hockey"], n_eps=5, n_gifs=1)
            print(f"Iteration {iteration} complete")

        iteration += 1
        if not args.use_target_base_reward_scaling and iteration > args.num_iterations:
            break

    # Save final model
    torch.save(agent.state_dict(), f"{log_parent_dir}/reward_scaling_model.pth")

    # Final evaluation
    evaluate_iterative_smoothing(f"{log_parent_dir}/reward_scaling_model.pth", log_parent_dir, config["air_hockey"])
    save_tensorboard_plots(log_parent_dir, config, 
        metrics=['charts/avg_episodic_return', 
                'charts/max_episodic_return', 
                'charts/min_episodic_return', 
                'charts/episodic_return', 
                'losses/approx_kl', 
                'losses/value_loss', 
                'losses/policy_loss', 
                'charts/avg_success_rate',
                'losses/action_loss',
                'charts/base_reward_scaling'])

    print("Reward scaling finetuning complete!")
    print(f"Final reward scaling: {scaling_manager.current_scaling:.4f}")
    print(f"Target was: {scaling_manager.target_scaling:.4f}")
