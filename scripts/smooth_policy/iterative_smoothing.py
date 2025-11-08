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



class RewardNormalizer:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.count = 0
        self.mean = 0.0
        self.var = 1.0
        
    def update(self, rewards):
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        
        batch_count = len(rewards)
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        
        # Update count
        new_count = self.count + batch_count
        
        # Update mean using Welford's online algorithm
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / new_count
        
        # Update variance using Welford's online algorithm
        if self.count > 0:
            delta2 = batch_mean - self.mean
            self.var = (self.count * self.var + batch_count * batch_var + 
                       delta * delta2 * self.count * batch_count / new_count) / new_count
        else:
            self.var = batch_var
            
        self.count = new_count
    
    def normalize(self, rewards):
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        return (rewards - self.mean) / (np.sqrt(self.var) + self.epsilon)


@dataclass
class Args:
    num_envs: int = 8
    num_steps: int = 512
    learning_rate: float = 1e-4
    num_iterations: int = 100
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    minibatch_size: int = 64
    update_epochs: int = 10
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    caps_coef_nearby: float = 0.0 # disable CAPS
    caps_coef_consecutive: float = 0.0
    action_coef: float = 0.0 # disable action regularization
    max_grad_norm: float = 0.5
    target_kl: float = None
    minibatch_size: int = 64
    batch_size: int = 0 # computed in runtime

    config: str = "scripts/smooth_policy/configs/puck_touch/default_config.yaml"
    args_file: str = None
    model_path: str = None  # Path to pre-trained model state dict
    run_name: str = "default"

    # Hyperparameters to vary
    seed: int = 0
    norm_adv: bool = True
    reward_scaling: bool = True
    reward_normalization: bool = False
    device: str = "cuda:0"
    log_parent_dir: str = None
    dynamic_reward_scaling: bool = False
    dynamic_freeze_policy: bool = False

    finetune: bool = False



def make_env(env_id):
    def _thunk():
        curr_seed = random.randint(0, int(1e8))
        config["air_hockey"]["seed"] = curr_seed
        env = AirHockeyEnv(config["air_hockey"])
        return env
    return _thunk

# Example usage:
if __name__ == "__main__":

    temp_args = tyro.cli(Args) # checks for a passed in args file
    if temp_args.args_file is not None:
        with open(temp_args.args_file, "r") as f:
            file_args_dict = yaml.load(f, Loader=yaml.FullLoader)
        default_args = Args(**file_args_dict)
    else:
        default_args = Args()  # Use class defaults

    # command line args override file args
    args = tyro.cli(Args, default=default_args)

    args.batch_size = args.num_envs * args.num_steps

    if args.finetune: # finetune mode has some preset hyperparameters
        args.learning_rate = 1e-5 # lower learning rate
        args.clip_coef = 0.1 # lower clip coef
        args.clip_vloss = True # use clipped value loss
        args.max_grad_norm = 0.25 # lower max grad norm
        args.action_coef = 0.5 # action regularization
        args.num_iterations = 500

        # args.caps_coef_nearby = 20
        # args.caps_coef_consecutive = 20
        # args.num_envs = 8
        # args.num_steps = 1024
        # args.device = "cuda:0"
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.dynamic_reward_scaling:
        args.vf_coef = 0.5

    # should just create parallel envs for future use (can just use sync, async as placeholders)
    envs = gym.vector.AsyncVectorEnv([make_env(i) for i in range(args.num_envs)])

    # Create folder with all results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    task_name = config["air_hockey"].get("task")
    run_name = args.run_name

    log_parent_dir = args.log_parent_dir
    if log_parent_dir is None:
        log_parent_dir = f"runs/iterative_smoothing/{task_name}/{run_name}_{timestamp}"
    if os.path.exists(log_parent_dir):
        raise FileExistsError(f"Log directory {log_parent_dir} already exists.")
    os.makedirs(log_parent_dir, exist_ok=True)

    writer = SummaryWriter(log_parent_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # save yaml config into log_parent_dir
    with open(f"{log_parent_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)
    # save args into log_parent_dir
    with open(f"{log_parent_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)
    
    agent = Agent(envs, init_reward_scaling=0.1).to(args.device) # prevent initial blowup
    
    # Load pre-trained model if path is provided
    if args.model_path is not None and os.path.exists(args.model_path):
        print(f"Loading pre-trained model from {args.model_path}")
        agent.load_state_dict(torch.load(args.model_path, map_location=args.device))
        print("Model loaded successfully")
    
    # TODO: save the optimizer state dict
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    reward_normalizer = RewardNormalizer()

    # TAKEN FROM CLEANRL
    # main training loop
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(args.device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(args.device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(args.device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(args.device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(args.device)
    next_done = torch.zeros(args.num_envs).to(args.device)

    freeze_policy_iterations = 0
    
    for iteration in range(1, args.num_iterations + 1):
        # Reset episodic return tracking for this iteration
        episodic_returns = []
        success_rates = []
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        reward_scaling = agent.reward_scaling.item()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            if args.dynamic_reward_scaling:
                reward = reward * reward_scaling
            elif args.reward_scaling:
                reward *= 0.1 # manually scale down

            if args.reward_normalization:
                reward_normalizer.update(reward)
                reward = reward_normalizer.normalize(reward)
            
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(args.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(args.device), torch.Tensor(next_done).to(args.device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode_return" in info:
                        episodic_returns.append(info['episode_return'])
                        success_rates.append(1.0 if info['success'] else 0.0)
                        # print(f"global_step={global_step}, episodic_return={info['episode_return']}")
                        writer.add_scalar("charts/episodic_return", info['episode_return'], global_step)
                        writer.add_scalar("charts/episodic_length", info['episode_length'], global_step)

        # bootstrap value if not done
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

            # log statistics of the returns, advantages, values
            writer.add_scalar("charts/return_mean", returns.mean().item(), global_step)
            writer.add_scalar("charts/return_std", returns.std().item(), global_step)
            writer.add_scalar("charts/advantage_mean", advantages.mean().item(), global_step)
            writer.add_scalar("charts/advantage_std", advantages.std().item(), global_step)
            writer.add_scalar("charts/value_mean", values.mean().item(), global_step)
            writer.add_scalar("charts/value_std", values.std().item(), global_step)
        
        # adjust reward scaling (if not already frozen)
        if freeze_policy_iterations == 0 and args.dynamic_reward_scaling and values.mean().item() >= 5:
            agent.reward_scaling *= 0.9
            # running only value iterations for a few iterations until value loss is more stable
            if args.dynamic_freeze_policy: # freeze on a parameter
                freeze_policy_iterations = 3
                print("Freezing policy iterations for 3 iterations")

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_dones = dones.reshape(-1).bool()

        
        # EVALUATE the loss before optimization
        noise_std = 0.01
        nearby_obs = b_obs + torch.randn_like(b_obs) * noise_std # new sample of noise
        nearby_actions_mean = agent.get_action_mean(nearby_obs)

        consecutive_action_loss = 0.0
        nearby_action_loss = 0.0
        cnt_consecutive = 0
        for i in range(args.batch_size):
            if i < args.batch_size - 1 and not b_dones[i].item():
                cnt_consecutive += 1
                consecutive_action_loss += (b_actions[i] - b_actions[i+1]) ** 2
            nearby_action_loss += (b_actions[i] - nearby_actions_mean[i]) ** 2
        consecutive_action_loss /= cnt_consecutive
        nearby_action_loss /= args.batch_size
        consecutive_action_loss = consecutive_action_loss.mean()
        nearby_action_loss = nearby_action_loss.mean()

        caps_loss = nearby_action_loss * args.caps_coef_nearby + consecutive_action_loss * args.caps_coef_consecutive
        action_loss = (b_actions ** 2.0).mean()
        writer.add_scalar("losses/consecutive_action_loss", consecutive_action_loss.item(), global_step)
        writer.add_scalar("losses/nearby_action_loss", nearby_action_loss.item(), global_step)
        writer.add_scalar("losses/caps_loss", caps_loss.item(), global_step)
        writer.add_scalar("losses/action_loss", action_loss.item(), global_step)
        

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            noise_std = 0.01
            nearby_obs = b_obs + torch.randn_like(b_obs) * noise_std # new sample of noise
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
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

                # ADDITION: penalize large actions (movements can still be jerky, but magnitude is controlled)
                action_loss = (b_actions[mb_inds] ** 2.0).mean() 

                # ADDITION: CAPS loss (regularize consecutive states and nearby states)
                curr_actions_mean = agent.get_action_mean(b_obs[mb_inds])
                nearby_actions_mean = agent.get_action_mean(nearby_obs[mb_inds]) # for now just use sample from gaussian
                # Ensure indices do not go out of bounds for next actions (shift by 1, but clip at batch length)
                next_inds = mb_inds + 1
                next_inds = np.clip(next_inds, 0, len(b_obs) - 1)
                next_actions_mean = agent.get_action_mean(b_obs[next_inds])
                # calculate losses

                # BUG: sampled actions are not differentiable, and don't have requires grad
                # solution: use the action mean
                nearby_action_loss = ((nearby_actions_mean - curr_actions_mean) ** 2.0).mean()
                non_done_mask = ~b_dones[mb_inds]
                consecutive_action_loss = ((next_actions_mean - curr_actions_mean) ** 2.0)[non_done_mask].mean()
                caps_loss = nearby_action_loss * args.caps_coef_nearby + consecutive_action_loss * args.caps_coef_consecutive

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + args.action_coef * action_loss + caps_loss


                if freeze_policy_iterations > 0:
                    loss = v_loss * args.vf_coef
    
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        
        
        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/reward_scaling", agent.reward_scaling.item(), global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        writer.add_scalar("charts/policy_frozen", int(freeze_policy_iterations > 0), global_step)
        if freeze_policy_iterations > 0:
            freeze_policy_iterations -= 1 # decrement the counter
        
        # Calculate and log episodic return statistics for this iteration
        if episodic_returns:
            avg_return = np.mean(episodic_returns)
            min_return = np.min(episodic_returns)
            max_return = np.max(episodic_returns)
            
            print(f"Iteration {iteration}: Avg Return: {avg_return:.2f}, Min Return: {min_return:.2f}, Max Return: {max_return:.2f}")
            print(f"Iteration {iteration}: Avg Success Rate: {np.mean(success_rates):.2f}, Max Success Rate: {np.max(success_rates):.2f}")
            writer.add_scalar("charts/avg_episodic_return", avg_return, iteration)
            writer.add_scalar("charts/min_episodic_return", min_return, iteration)
            writer.add_scalar("charts/max_episodic_return", max_return, iteration)
            writer.add_scalar("charts/avg_success_rate", np.mean(success_rates), iteration)
            writer.add_scalar("charts/max_success_rate", np.max(success_rates), iteration)
            episodic_returns = []
            success_rates = []
        else:
            print(f"Iteration {iteration}: No episodes completed")

        if iteration % 10 == 0 or min_return >= 3500: # start cherry-picking good policies
            # save a checkpoint of the model
            # create a subfolder for the checkpoint
            checkpoint_dir = os.path.join(log_parent_dir, f"checkpoint_{iteration}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_path = f"{checkpoint_dir}/iterative_smoothing_model.pth"
            torch.save(agent.state_dict(), model_path)

            # evaluate the model
            evaluate_iterative_smoothing(model_path, checkpoint_dir, config["air_hockey"], n_eps=6, n_gifs=1)
            
            print(f"Iteration {iteration} complete")

    # save model
    torch.save(agent.state_dict(), f"{log_parent_dir}/iterative_smoothing_model.pth")

    # evaluate the model and save results
    evaluate_iterative_smoothing(f"{log_parent_dir}/iterative_smoothing_model.pth", log_parent_dir, config["air_hockey"])
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
    'losses/caps_loss'])






