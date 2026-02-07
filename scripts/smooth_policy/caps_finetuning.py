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

from scripts.smooth_policy.evaluate import evaluate_agent   
from scripts.smooth_policy.agent import Agent
from scripts.utils import save_tensorboard_plots


@dataclass
class FinetuneArgs:
    num_envs: int = 8
    num_steps: int = 512
    learning_rate: float = 1e-4  # Match default_training
    num_iterations: int = 100  # Match default_training
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    minibatch_size: int = 64
    update_epochs: int = 10
    clip_coef: float = 0.2  # Match default_training
    clip_vloss: bool = False  # Match default_training
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5  # Match default_training
    target_kl: float = None
    minibatch_size: int = 64  # Add duplicate line to match default_training structure
    batch_size: int = 0  # computed at runtime
    norm_adv: bool = True  # Match default_training

    # CAPS hyperparameters (match default_training structure)
    caps_coef_nearby: float = 0.0
    caps_coef_consecutive: float = 0.0

    # Paths
    config: str = "scripts/smooth_policy/configs/puck_touch/default_config.yaml"
    args_file: str = None
    model_path: str = None  # Path to pre-trained model state dict (REQUIRED for finetuning)
    log_parent_dir: str = None
    run_name: str = "finetune_smoothing"

    # Others
    seed: int = 0
    device: str = "cuda:0"
    
    # Adaptive caps coefficient parameters
    performance_threshold: float = 3000.0  # Minimum average performance to maintain (this is very doable)
    performance_buffer_iterations: int = 5  # Iterations to check performance before increasing caps
    caps_increment: float = 50  # Amount to increase caps coefficients
    training_buffer_iterations: int = 5  # Iterations to train after caps increase
    
    # Initial caps coefficients (start at 0)
    initial_caps_coef_nearby: float = 0.0
    initial_caps_coef_consecutive: float = 0.0

    # target caps coefficient
    target_caps_coef: float = 1000
    use_target_caps_coef: bool = False


def make_env(env_id):
    def _thunk():
        curr_seed = random.randint(0, int(1e8))
        config["air_hockey"]["seed"] = curr_seed
        env = AirHockeyEnv(config["air_hockey"])
        return env
    return _thunk


class AdaptiveCapsManager:
    def __init__(self, args):
        self.performance_threshold = args.performance_threshold
        self.performance_buffer_iterations = args.performance_buffer_iterations
        self.caps_increment = args.caps_increment
        self.training_buffer_iterations = args.training_buffer_iterations
        
        # Current caps coefficients
        self.caps_coef_nearby = args.initial_caps_coef_nearby
        self.caps_coef_consecutive = args.initial_caps_coef_consecutive
        
        # Performance tracking
        self.performance_history = []
        self.iterations_since_caps_increase = 0
        self.in_training_buffer = False
        
    def update_performance(self, avg_return, iteration):
        """Update performance history and check if caps should be increased"""
        self.performance_history.append(avg_return)
        
        # Only keep recent performance history
        if len(self.performance_history) > self.performance_buffer_iterations:
            self.performance_history.pop(0)
        
        # If we're in training buffer, just count down
        if self.in_training_buffer:
            self.iterations_since_caps_increase += 1
            if self.iterations_since_caps_increase >= self.training_buffer_iterations:
                self.in_training_buffer = False
                self.iterations_since_caps_increase = 0
                print(f"Training buffer complete. Ready to evaluate performance again.")
            return False
        
        # Check if we have enough performance data
        if len(self.performance_history) < self.performance_buffer_iterations:
            return False
        
        # Check if performance is consistently above threshold
        avg_performance = np.mean(self.performance_history)
        min_performance = np.min(self.performance_history)
        if min_performance >= self.performance_threshold:
            # Increase caps coefficients
            self.caps_coef_nearby += self.caps_increment
            self.caps_coef_consecutive += self.caps_increment
            
            # Reset tracking and enter training buffer
            self.performance_history = []
            self.in_training_buffer = True
            self.iterations_since_caps_increase = 0
            
            print(f"Performance maintained above {self.performance_threshold:.1f} "
                  f"(avg: {avg_performance:.1f}) (min: {min_performance:.1f}). Increasing caps coefficients to "
                  f"nearby: {self.caps_coef_nearby:.4f}, consecutive: {self.caps_coef_consecutive:.4f}")
            print(f"Entering training buffer for {self.training_buffer_iterations} iterations.")
            return True
        
        return False


if __name__ == "__main__":
    temp_args = tyro.cli(FinetuneArgs)
    if temp_args.args_file is not None:
        with open(temp_args.args_file, "r") as f:
            file_args_dict = yaml.load(f, Loader=yaml.FullLoader)
        default_args = FinetuneArgs(**file_args_dict)
    else:
        default_args = FinetuneArgs()

    # Command line args override file args
    args = tyro.cli(FinetuneArgs, default=default_args)
    args.batch_size = args.num_envs * args.num_steps

    # Require model path for finetuning
    if args.model_path is None or not os.path.exists(args.model_path):
        raise ValueError(f"Model path is required for finetuning and must exist: {args.model_path}")

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create parallel envs
    envs = gym.vector.AsyncVectorEnv([make_env(i) for i in range(args.num_envs)])

    # Create folder with all results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    task_name = config["air_hockey"].get("task")
    run_name = args.run_name

    log_parent_dir = args.log_parent_dir
    if log_parent_dir is None:
        log_parent_dir = f"runs/finetune_smoothing/{task_name}/{run_name}_{timestamp}"
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
    agent = Agent(envs, action_scale=0.02, action_bias=0.0).to(args.device)
    print(f"Loading pre-trained model from {args.model_path}")
    agent.load_state_dict(torch.load(args.model_path, map_location=args.device))
    print("Model loaded successfully")
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-6)
    
    # Initialize adaptive caps manager
    caps_manager = AdaptiveCapsManager(args)

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
    
    # Tracking lists for motion metrics
    velocity_magnitudes = []
    acceleration_magnitudes = []
    jerk_magnitudes = []
    
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

        # Get current caps coefficients
        current_caps_nearby = caps_manager.caps_coef_nearby
        current_caps_consecutive = caps_manager.caps_coef_consecutive

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

            # REWARD SCALING is done on the environment level, not here

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
                        
                        # Extract motion data if available
                        if 'motion_data' in info:
                            velocity_magnitudes.extend(info['motion_data']['velocity_mags'])
                            acceleration_magnitudes.extend(info['motion_data']['acceleration_mags'])
                            jerk_magnitudes.extend(info['motion_data']['jerk_mags'])

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



        # Evaluate CAPS loss periodically for monitoring
        if iteration % 5 == 0:  # Log every 5000 steps instead of every 5 steps
            with torch.no_grad():
                # Generate nearby observations for smoothness evaluation
                noise_std = 0.01
                nearby_obs = b_obs + torch.randn_like(b_obs) * noise_std
                
                # Get actions for current and nearby observations (matching default_training)
                curr_actions, _, _, _ = agent.get_action_and_value(b_obs)
                nearby_actions, _, _, _ = agent.get_action_and_value(nearby_obs)
                next_actions = b_actions[np.clip(np.arange(args.batch_size) + 1, 0, args.batch_size - 1)] # ignore bias from the last action
                non_done_mask = ~b_dones

                # L2 losses
                nearby_action_loss_l2 = ((nearby_actions - b_actions) ** 2.0).mean()
                consecutive_action_loss_l2 = ((next_actions - b_actions) ** 2.0)[non_done_mask].sum() / non_done_mask.sum() # average over non-done steps
                action_loss_l2 = (b_actions ** 2.0).mean()
                caps_loss = nearby_action_loss_l2 * current_caps_nearby + consecutive_action_loss_l2 * current_caps_consecutive

                # L1 losses
                nearby_action_loss_l1 = ((nearby_actions - b_actions).abs()).mean()
                consecutive_action_loss_l1 = ((next_actions - b_actions).abs())[non_done_mask].sum() / non_done_mask.sum() # average over non-done steps
                action_loss_l1 = (b_actions.abs()).mean()

                writer.add_scalar("losses/consecutive_action_loss_l2", consecutive_action_loss_l2.item(), global_step)
                writer.add_scalar("losses/nearby_action_loss_l2", nearby_action_loss_l2.item(), global_step)
                writer.add_scalar("losses/caps_loss", caps_loss.item(), global_step)
                writer.add_scalar("losses/action_loss_l2", action_loss_l2.item(), global_step) # plot out, but not used in training
                writer.add_scalar("losses/consecutive_action_loss_l1", consecutive_action_loss_l1.item(), global_step)
                writer.add_scalar("losses/nearby_action_loss_l1", nearby_action_loss_l1.item(), global_step)
                writer.add_scalar("losses/action_loss_l1", action_loss_l1.item(), global_step) # plot out, but not used in training



        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            noise_std = 0.01
            nearby_obs = b_obs + torch.randn_like(b_obs) * noise_std
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, _, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
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

                # Action regularization
                action_loss = (b_actions[mb_inds] ** 2.0).mean()

                # CAPS loss (smoothness regularization) - matching default_training
                next_inds = np.clip(mb_inds + 1, 0, len(b_obs) - 1)
                curr_actions, _, _, _ = agent.get_action_and_value(b_obs[mb_inds])
                nearby_actions, _, _, _ = agent.get_action_and_value(nearby_obs[mb_inds]) # for now just use sample from gaussian
                next_actions, _, _, _ = agent.get_action_and_value(b_obs[next_inds])

                # Calculate CAPS losses (matching default_training)
                nearby_action_loss = ((nearby_actions - curr_actions) ** 2.0).mean()
                non_done_mask = ~b_dones[mb_inds]
                consecutive_action_loss = ((next_actions - curr_actions) ** 2.0)[non_done_mask].sum() / non_done_mask.sum() # average over non-done steps
                caps_loss = (nearby_action_loss * current_caps_nearby + 
                           consecutive_action_loss * current_caps_consecutive)

                # entropy_loss = entropy.mean()
                entropy_loss = (-newlogprob).mean() # unbiased estimate of entropy
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + caps_loss

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
        writer.add_scalar("losses/caps_loss", caps_loss.item(), global_step)
        
        # Log caps coefficients
        writer.add_scalar("charts/caps_coef_nearby", current_caps_nearby, global_step)
        writer.add_scalar("charts/caps_coef_consecutive", current_caps_consecutive, global_step)
        writer.add_scalar("charts/in_training_buffer", int(caps_manager.in_training_buffer), global_step)
        
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Calculate and log episodic return statistics
        if episodic_returns:
            avg_return = np.mean(episodic_returns)
            min_return = np.min(episodic_returns)
            max_return = np.max(episodic_returns)
            
            print(f"Iteration {iteration}: Avg Return: {avg_return:.2f}, Min Return: {min_return:.2f}, Max Return: {max_return:.2f}")
            print(f"Iteration {iteration}: Avg Success Rate: {np.mean(success_rates):.2f}")
            print(f"Iteration {iteration}: CAPS Coefficients - Nearby: {current_caps_nearby:.4f}, Consecutive: {current_caps_consecutive:.4f}")
            
            writer.add_scalar("charts/avg_episodic_return", avg_return, iteration)
            writer.add_scalar("charts/min_episodic_return", min_return, iteration)
            writer.add_scalar("charts/max_episodic_return", max_return, iteration)
            writer.add_scalar("charts/avg_success_rate", np.mean(success_rates), iteration)
        else:
            print(f"Iteration {iteration}: No episodes completed")
        
        # Calculate and log motion statistics
        if velocity_magnitudes:
            avg_vel_mag = np.mean(velocity_magnitudes)
            avg_acc_mag = np.mean(acceleration_magnitudes) 
            avg_jerk_mag = np.mean(jerk_magnitudes)
            
            print(f"Iteration {iteration}: Avg Velocity Mag: {avg_vel_mag:.4f}, Avg Acceleration Mag: {avg_acc_mag:.4f}, Avg Jerk Mag: {avg_jerk_mag:.4f}")
            
            writer.add_scalar("motion/avg_velocity_magnitude", avg_vel_mag, iteration)
            writer.add_scalar("motion/avg_acceleration_magnitude", avg_acc_mag, iteration)
            writer.add_scalar("motion/avg_jerk_magnitude", avg_jerk_mag, iteration)
            
            # Clear lists for next iteration
            velocity_magnitudes.clear()
            acceleration_magnitudes.clear()
            jerk_magnitudes.clear()
            
            # Update caps manager with performance
            caps_increased = caps_manager.update_performance(avg_return, iteration)
            if caps_increased:
                writer.add_scalar("charts/caps_coefficient_increased", 1, iteration)
                current_caps_nearby = caps_manager.caps_coef_nearby
                current_caps_consecutive = caps_manager.caps_coef_consecutive
                if args.use_target_caps_coef and current_caps_nearby >= args.target_caps_coef and current_caps_consecutive >= args.target_caps_coef:
                    break
            else:
                writer.add_scalar("charts/caps_coefficient_increased", 0, iteration)
        else:
            print(f"Iteration {iteration}: No episodes completed")
            print(f"Iteration {iteration}: CAPS Coefficients - Nearby: {current_caps_nearby:.4f}, Consecutive: {current_caps_consecutive:.4f}")

        # Save checkpoints periodically
        if iteration % 20 == 0:
            checkpoint_dir = os.path.join(log_parent_dir, f"checkpoint_{iteration}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_path = f"{checkpoint_dir}/model.pth"
            torch.save(agent.state_dict(), model_path)

            # Evaluate the model
            evaluate_agent(model_path, checkpoint_dir, config["air_hockey"], n_eps=4, n_gifs=1)
            print(f"Iteration {iteration} complete")

        iteration += 1
        if not args.use_target_caps_coef and iteration > args.num_iterations:
            break

    # Save final model
    torch.save(agent.state_dict(), f"{log_parent_dir}/model.pth")

    # Final evaluation
    evaluate_agent(f"{log_parent_dir}/model.pth", log_parent_dir, config["air_hockey"])
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
                'losses/caps_loss',
                'motion/avg_velocity_magnitude',
                'motion/avg_acceleration_magnitude',
                'motion/avg_jerk_magnitude',
                'charts/caps_coef_nearby',
                'charts/caps_coef_consecutive'])

    print("Finetuning complete!")
    print(f"Final CAPS coefficients - Nearby: {caps_manager.caps_coef_nearby:.4f}, Consecutive: {caps_manager.caps_coef_consecutive:.4f}")
