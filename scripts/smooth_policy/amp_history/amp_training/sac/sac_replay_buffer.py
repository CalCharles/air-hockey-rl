"""
Replay buffer for SAC algorithm.

Stores transitions (obs, action, reward, next_obs, done) and provides
random sampling for off-policy learning.
"""

import torch
import numpy as np


class SACReplayBuffer:
    """
    Replay buffer for SAC that stores transitions and provides random sampling.
    
    Compatible with vectorized environments.
    """
    
    def __init__(self, buffer_size, obs_shape, action_shape, device='cuda', n_envs=1, disc_obs_dim=None):
        """
        Initialize replay buffer.
        
        Args:
            buffer_size: Maximum number of transitions to store
            obs_shape: Shape of observations (e.g., (obs_dim,))
            action_shape: Shape of actions (e.g., (action_dim,))
            device: Device to store tensors on
            n_envs: Number of parallel environments
            disc_obs_dim: Dimension of discriminator observations (for AMP), None to disable AMP storage
        """
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.n_envs = n_envs
        self.disc_obs_dim = disc_obs_dim
        self.use_amp = disc_obs_dim is not None
        
        # Storage buffers
        self.observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32, device=device)
        self.next_observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, *action_shape), dtype=torch.float32, device=device)
        self.prev_actions = torch.zeros((buffer_size, *action_shape), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size,), dtype=torch.float32, device=device)
        
        # AMP: Store discriminator observations and validity mask (only if AMP enabled)
        if self.use_amp:
            self.disc_obs = torch.zeros((buffer_size, disc_obs_dim), dtype=torch.float32, device=device)
            self.disc_valid = torch.zeros((buffer_size,), dtype=torch.bool, device=device)
        else:
            self.disc_obs = None
            self.disc_valid = None
        
        self.position = 0
        self.size = 0
        
    def add(self, obs, next_obs, actions, rewards, dones, prev_action=None, disc_obs=None, disc_valid=None):
        """
        Add transitions to the buffer.
        
        Args:
            obs: Observations [n_envs, *obs_shape] or [*obs_shape]
            next_obs: Next observations [n_envs, *obs_shape] or [*obs_shape]
            actions: Actions [n_envs, *action_shape] or [*action_shape]
            rewards: Rewards [n_envs,] or scalar (task rewards only, discriminator rewards computed later)
            dones: Done flags [n_envs,] or scalar
            prev_action: Previous action appended to policy state [n_envs, *action_shape] (optional)
            disc_obs: Discriminator observations [n_envs, disc_obs_dim] (optional, for AMP)
            disc_valid: Validity mask for discriminator observations [n_envs,] (optional, for AMP)
        """
        # Convert to tensors if needed
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if not isinstance(next_obs, torch.Tensor):
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        if prev_action is None:
            prev_action = torch.zeros_like(actions)
        if not isinstance(prev_action, torch.Tensor):
            prev_action = torch.tensor(prev_action, dtype=torch.float32, device=self.device)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        if not isinstance(dones, torch.Tensor):
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # Handle AMP discriminator observations
        if disc_obs is not None:
            if not isinstance(disc_obs, torch.Tensor):
                disc_obs = torch.tensor(disc_obs, dtype=torch.float32, device=self.device)
        if disc_valid is not None:
            if not isinstance(disc_valid, torch.Tensor):
                disc_valid = torch.tensor(disc_valid, dtype=torch.bool, device=self.device)
        
        # Handle single transitions (add batch dimension)
        if obs.dim() == len(self.obs_shape):
            obs = obs.unsqueeze(0)
            next_obs = next_obs.unsqueeze(0)
            actions = actions.unsqueeze(0)
            prev_action = prev_action.unsqueeze(0) if prev_action.dim() == len(self.action_shape) else prev_action
            rewards = rewards.unsqueeze(0) if rewards.dim() == 0 else rewards
            dones = dones.unsqueeze(0) if dones.dim() == 0 else dones
            if disc_obs is not None:
                disc_obs = disc_obs.unsqueeze(0) if disc_obs.dim() == 1 else disc_obs
            if disc_valid is not None:
                disc_valid = disc_valid.unsqueeze(0) if disc_valid.dim() == 0 else disc_valid
        
        batch_size = obs.shape[0]
        
        # Add transitions one by one (circular buffer)
        for i in range(batch_size):
            idx = self.position
            self.observations[idx] = obs[i]
            self.next_observations[idx] = next_obs[i]
            self.actions[idx] = actions[i]
            self.prev_actions[idx] = prev_action[i]
            self.rewards[idx] = rewards[i]
            self.dones[idx] = dones[i]
            
            # Store AMP discriminator observations (only if AMP enabled)
            if self.use_amp:
                if disc_obs is not None:
                    self.disc_obs[idx] = disc_obs[i]
                if disc_valid is not None:
                    self.disc_valid[idx] = disc_valid[i]
                else:
                    self.disc_valid[idx] = False  # Default to invalid
            
            self.position = (self.position + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size):
        """
        Sample random batch from buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary with keys: observations, next_observations, actions, prev_actions, rewards, dones, disc_obs, disc_valid
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Sample random indices
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        result = {
            'observations': self.observations[indices],
            'next_observations': self.next_observations[indices],
            'actions': self.actions[indices],
            'prev_actions': self.prev_actions[indices],
            'rewards': self.rewards[indices],
            'dones': self.dones[indices],
        }
        
        # Include AMP data if enabled
        if self.use_amp:
            result['disc_obs'] = self.disc_obs[indices]
            result['disc_valid'] = self.disc_valid[indices]
        
        return result
    
    def __len__(self):
        """Return current buffer size."""
        return self.size
    
    def clear(self):
        """Clear the buffer."""
        self.position = 0
        self.size = 0
        self.observations.zero_()
        self.next_observations.zero_()
        self.actions.zero_()
        self.prev_actions.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        if self.use_amp:
            self.disc_obs.zero_()
            self.disc_valid.zero_()