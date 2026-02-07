"""
Normalizer for discriminator observations.

Tracks running mean and standard deviation for normalizing inputs to the discriminator.
This prevents the discriminator from overfitting to unnormalized scale differences.
"""

import torch
import torch.nn as nn


class Normalizer:
    """
    Running statistics normalizer for discriminator observations.
    
    Maintains running mean and standard deviation, and provides normalization
    with optional clipping.
    """
    
    def __init__(self, shape, clip=10.0, device='cuda', dtype=torch.float32):
        """
        Initialize normalizer.
        
        Args:
            shape: Shape of the observations to normalize (e.g., (8,) for disc_obs)
            clip: Maximum absolute value after normalization
            device: Device to store tensors on
            dtype: Data type for tensors
        """
        self.shape = shape
        self.clip = clip
        self.device = device
        self.dtype = dtype
        
        # Running statistics
        self.mean = torch.zeros(shape, device=device, dtype=dtype)
        self.var = torch.ones(shape, device=device, dtype=dtype)
        self.count = torch.zeros(1, device=device, dtype=dtype)
        
        # Accumulation buffers (reset after each update)
        self.sum = torch.zeros(shape, device=device, dtype=dtype)
        self.sum_sq = torch.zeros(shape, device=device, dtype=dtype)
        self.batch_count = 0
        
    def record(self, data):
        """
        Record new data for statistics computation.
        
        Args:
            data: Tensor of shape [batch_size, ...] or [...] matching self.shape
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, device=self.device, dtype=self.dtype)
        
        # Ensure data is on correct device
        data = data.to(device=self.device, dtype=self.dtype)
        
        # Flatten batch dimension if needed
        if data.dim() > len(self.shape):
            # Reshape to [batch_size, *self.shape]
            batch_size = data.shape[0]
            data = data.reshape(batch_size, -1)
        else:
            # Single sample, add batch dimension
            data = data.unsqueeze(0)
            batch_size = 1
        
        # Accumulate statistics
        self.sum += data.sum(dim=0)
        self.sum_sq += (data ** 2).sum(dim=0)
        self.batch_count += batch_size
        
    def update(self):
        """
        Update running mean and variance from accumulated data.
        Uses Welford's online algorithm for numerical stability.
        """
        if self.batch_count == 0:
            return
        
        # Compute batch statistics
        batch_mean = self.sum / self.batch_count
        batch_var = (self.sum_sq / self.batch_count) - (batch_mean ** 2)
        batch_var = torch.clamp(batch_var, min=1e-8)  # Prevent negative variance
        
        # Update running statistics using parallel algorithm
        total_count = self.count + self.batch_count
        
        # Update mean
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (self.batch_count / total_count)
        
        # Update variance
        m_a = self.var * self.count
        m_b = batch_var * self.batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * self.batch_count / total_count
        new_var = M2 / total_count
        
        # Apply updates
        self.mean = new_mean
        self.var = torch.clamp(new_var, min=1e-8)
        self.count = total_count
        
        # Reset accumulation buffers
        self.sum.zero_()
        self.sum_sq.zero_()
        self.batch_count = 0
        
    def normalize(self, data):
        """
        Normalize data using current statistics.
        
        Args:
            data: Tensor to normalize
            
        Returns:
            Normalized tensor with same shape as input
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, device=self.device, dtype=self.dtype)
        
        data = data.to(device=self.device, dtype=self.dtype)
        
        # Compute standard deviation
        std = torch.sqrt(self.var)
        
        # Normalize
        normalized = (data - self.mean) / (std + 1e-8)
        
        # Clip to prevent extreme values
        if self.clip is not None:
            normalized = torch.clamp(normalized, -self.clip, self.clip)
        
        return normalized
    
    def denormalize(self, data):
        """
        Denormalize data (inverse of normalize).
        
        Args:
            data: Normalized tensor
            
        Returns:
            Denormalized tensor
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, device=self.device, dtype=self.dtype)
        
        data = data.to(device=self.device, dtype=self.dtype)
        
        std = torch.sqrt(self.var)
        return data * (std + 1e-8) + self.mean
    
    def state_dict(self):
        """Return state dictionary for saving."""
        return {
            'mean': self.mean,
            'var': self.var,
            'count': self.count,
            'shape': self.shape,
            'clip': self.clip
        }
    
    def load_state_dict(self, state_dict):
        """Load state from dictionary."""
        self.mean = state_dict['mean'].to(device=self.device, dtype=self.dtype)
        self.var = state_dict['var'].to(device=self.device, dtype=self.dtype)
        self.count = state_dict['count'].to(device=self.device, dtype=self.dtype)
