"""
Replay buffer for storing and sampling discriminator observations.

Used to store past agent-generated observations to stabilize discriminator training
and prevent catastrophic forgetting.
"""

import torch
import numpy as np


class ReplayBuffer:
    """
    Circular buffer for storing discriminator observations.
    
    Stores agent-generated observations and provides random sampling for
    discriminator training.
    """
    
    def __init__(self, capacity, obs_shape, device='cuda'):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of observations to store
            obs_shape: Shape of a single observation (e.g., (8,) for disc_obs)
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.device = device
        
        # Storage
        self.buffer = torch.zeros((capacity, *obs_shape), device=device, dtype=torch.float32)
        self.position = 0
        self.size = 0
        
    def push(self, observations):
        """
        Add observations to the buffer.
        
        Args:
            observations: Tensor of shape [batch_size, *obs_shape] or [*obs_shape]
        """
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, device=self.device, dtype=torch.float32)
        
        observations = observations.to(device=self.device, dtype=torch.float32)
        
        # Handle single observation
        if observations.dim() == len(self.obs_shape):
            observations = observations.unsqueeze(0)
        
        batch_size = observations.shape[0]
        
        # Add observations one by one (circular buffer)
        for i in range(batch_size):
            self.buffer[self.position] = observations[i]
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """
        Sample random batch from buffer.
        
        Args:
            batch_size: Number of observations to sample
            
        Returns:
            Tensor of shape [batch_size, *obs_shape]
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Sample with replacement if batch_size > size
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.buffer[indices]
    
    def is_full(self):
        """Check if buffer is at capacity."""
        return self.size >= self.capacity
    
    def __len__(self):
        """Return current buffer size."""
        return self.size
    
    def clear(self):
        """Clear the buffer."""
        self.position = 0
        self.size = 0
        self.buffer.zero_()
    
    def state_dict(self):
        """Return state dictionary for saving."""
        return {
            'buffer': self.buffer[:self.size].cpu(),
            'position': self.position,
            'size': self.size,
            'capacity': self.capacity,
            'obs_shape': self.obs_shape
        }
    
    def load_state_dict(self, state_dict):
        """Load state from dictionary."""
        self.capacity = state_dict['capacity']
        self.obs_shape = state_dict['obs_shape']
        self.position = state_dict['position']
        self.size = state_dict['size']
        
        # Recreate buffer with correct capacity
        self.buffer = torch.zeros((self.capacity, *self.obs_shape), 
                                  device=self.device, dtype=torch.float32)
        
        # Load saved data
        saved_buffer = state_dict['buffer'].to(device=self.device, dtype=torch.float32)
        self.buffer[:self.size] = saved_buffer
