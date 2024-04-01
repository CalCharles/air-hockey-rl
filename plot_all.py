import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_rewards_and_steps(event_file_path):
    # Initialize an event accumulator with a large size guidance to load scalar events
    event_acc = EventAccumulator(event_file_path, size_guidance={'scalars': 0})
    event_acc.Reload()
    if 'rollout/ep_rew_mean' in event_acc.scalars.Keys():
        events = event_acc.scalars.Items('rollout/ep_rew_mean')
        steps, rewards = zip(*[(e.step, e.value) for e in events])
        return np.array(steps), np.array(rewards)
    return np.array([]), np.array([])

def average_rewards_across_seeds(task_path):
    steps_list, rewards_list = [], []
    for root, dirs, files in os.walk(task_path):
        for file in files:
            if file.startswith('events.out'):
                steps, rewards = extract_rewards_and_steps(os.path.join(root, file))
                if steps.size > 0 and rewards.size > 0:
                    steps_list.append(steps)
                    rewards_list.append(rewards)
    
    # Determine the common steps range for interpolation
    min_step, max_step = max(s[0] for s in steps_list), min(s[-1] for s in steps_list)
    common_steps = np.linspace(min_step, max_step, num=1000)  # 1000 points interpolation
    
    # Interpolate rewards for each seed to the common steps
    interpolated_rewards = []
    for steps, rewards in zip(steps_list, rewards_list):
        interp_func = interp1d(steps, rewards, kind='linear')
        interpolated_rewards.append(interp_func(common_steps))
    
    # Average the interpolated rewards across seeds
    avg_rewards = np.mean(np.vstack(interpolated_rewards), axis=0)
    return common_steps, avg_rewards

def plot_task_performances(base_dir='baseline_models'):
    plt.figure(figsize=(10, 6))
    
    all_rewards = []
    for task in os.listdir(base_dir):
        task_path = os.path.join(base_dir, task)
        if os.path.isdir(task_path):
            steps, avg_rewards = average_rewards_across_seeds(task_path)
            all_rewards.extend(avg_rewards)
            # normalize rewards here
            avg_rewards = (avg_rewards - avg_rewards.min()) / (avg_rewards.max() - avg_rewards.min())
            assert avg_rewards.min() == 0.0 and avg_rewards.max() == 1.0
            if steps[-1] < 1e6-1:
                continue # means it did not finish training
            plt.plot(steps, avg_rewards, label=task)
    
    # Normalize all rewards and re-plot
    all_rewards = np.array(all_rewards)
    # min_reward, max_reward = all_rewards.min(), all_rewards.max()
    plt.cla()  # Clear the current plot to redraw with normalized rewards
    
    for task in os.listdir(base_dir):
        task_path = os.path.join(base_dir, task)
        if os.path.isdir(task_path):
            steps, avg_rewards = average_rewards_across_seeds(task_path)
            avg_rewards = (avg_rewards - avg_rewards.min()) / (avg_rewards.max() - avg_rewards.min())
            assert avg_rewards.min() == 0.0 and avg_rewards.max() == 1.0
            # if steps[-1] < 1e6-1:
            #     continue # means it did not finish training
            # normalized_rewards = (avg_rewards - min_reward) / (max_reward - min_reward)
            plt.plot(steps, avg_rewards, label=task)
    
    plt.xlabel('Timesteps')
    plt.ylabel('Normalized Average Reward')
    plt.title('Box2D Task Performance Over Training Steps')
    plt.legend()
    plt.grid()
    # save for plot in research paper
    # cut off excess space
    plt.savefig('box2d_training_performance.pdf', bbox_inches='tight')

# Run the plotting function
plot_task_performances('baseline_models')
