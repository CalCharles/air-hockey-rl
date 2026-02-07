"""
Simple test script to verify dimensions and basic functionality
of the PID controller policy implementation.
"""
import torch
import numpy as np
import gymnasium as gym
from airhockey import AirHockeyEnv
import yaml
from scripts.smooth_policy.pid_controller_policy.agent import Agent

# Load config
with open("scripts/smooth_policy/configs/puck_touch/default_config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Create environment
env = AirHockeyEnv(config["air_hockey"])

# Get dimensions
obs_space = env.observation_space
action_space = env.action_space

obs_dim = int(np.prod(obs_space.shape))
act_dim = int(np.prod(action_space.shape))

print("=" * 60)
print("DIMENSION CHECK")
print("=" * 60)
print(f"Observation dimension: {obs_dim}")
print(f"Action dimension: {act_dim}")
print(f"Augmented observation dimension: {obs_dim + act_dim}")

# Create agent with augmented observation dimension
augmented_obs_dim = obs_dim + act_dim
agent = Agent(obs_dim=augmented_obs_dim, act_dim=act_dim, action_scale=0.02, action_bias=0.0)

print("\n" + "=" * 60)
print("AGENT ARCHITECTURE")
print("=" * 60)
print(f"Agent input dimension: {augmented_obs_dim}")
print(f"Agent output dimension: {act_dim}")

# Test forward pass
obs, _ = env.reset()
print("\n" + "=" * 60)
print("FORWARD PASS TEST")
print("=" * 60)
print(f"Initial observation shape: {obs.shape}")
print(f"Initial observation (first 10 elements): {obs[:10]}")

# Initialize target position to paddle position (first act_dim elements)
target_pos = obs[:act_dim].copy()
print(f"\nInitial target position: {target_pos}")

# Create augmented observation
augmented_obs = np.concatenate([obs, target_pos])
print(f"Augmented observation shape: {augmented_obs.shape}")
assert augmented_obs.shape[0] == augmented_obs_dim, f"Expected {augmented_obs_dim}, got {augmented_obs.shape[0]}"

# Test agent forward pass
with torch.no_grad():
    delta_target, logprob, mean, value = agent.get_action_and_value(
        torch.FloatTensor(augmented_obs).unsqueeze(0)
    )

print(f"\nDelta target shape: {delta_target.shape}")
print(f"Delta target value: {delta_target.squeeze().numpy()}")
print(f"Log probability: {logprob.item():.4f}")
print(f"Value estimate: {value.item():.4f}")

# Update target position
new_target_pos = target_pos + delta_target.squeeze().numpy()
print(f"New target position: {new_target_pos}")

# Test environment step
obs, reward, terminated, truncated, info = env.step(new_target_pos)
print(f"\nReward: {reward:.4f}")
print(f"Terminated: {terminated}")
print(f"Truncated: {truncated}")

print("\n" + "=" * 60)
print("MULTI-ENVIRONMENT TEST")
print("=" * 60)

# Test with vectorized environments
num_envs = 4
envs = gym.vector.SyncVectorEnv([lambda: AirHockeyEnv(config["air_hockey"]) for _ in range(num_envs)])

obs, _ = envs.reset()
print(f"Vectorized observations shape: {obs.shape}")

# Initialize target positions
target_positions = torch.FloatTensor(obs[:, :act_dim])
print(f"Target positions shape: {target_positions.shape}")

# Create augmented observations
augmented_obs = torch.cat([torch.FloatTensor(obs), target_positions], dim=-1)
print(f"Augmented observations shape: {augmented_obs.shape}")
assert augmented_obs.shape == (num_envs, augmented_obs_dim)

# Test agent with batch
with torch.no_grad():
    delta_targets, logprobs, means, values = agent.get_action_and_value(augmented_obs)

print(f"\nBatch delta targets shape: {delta_targets.shape}")
print(f"Batch logprobs shape: {logprobs.shape}")
print(f"Batch values shape: {values.shape}")

# Update target positions
target_positions = target_positions + delta_targets
print(f"Updated target positions shape: {target_positions.shape}")

# Test environment step
obs, rewards, terminations, truncations, infos = envs.step(target_positions.numpy())
print(f"\nRewards shape: {rewards.shape}")
print(f"Rewards: {rewards}")

print("\n" + "=" * 60)
print("RESET HANDLING TEST")
print("=" * 60)

# Simulate some environments terminating
done_mask = np.array([True, False, True, False])
print(f"Done mask: {done_mask}")

# Reset target positions for done environments
for env_idx in range(num_envs):
    if done_mask[env_idx]:
        old_target = target_positions[env_idx].numpy()
        target_positions[env_idx] = torch.FloatTensor(obs[env_idx, :act_dim])
        new_target = target_positions[env_idx].numpy()
        print(f"Env {env_idx}: Reset target from {old_target} to {new_target}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)

