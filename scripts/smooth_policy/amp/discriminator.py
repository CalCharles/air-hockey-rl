"""
Discriminator network for AMP (Adversarial Motion Priors).

The discriminator learns to distinguish between expert demonstrations and
agent-generated trajectories based on state transition pairs.
"""

import torch
import torch.nn as nn
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer weights using orthogonal initialization."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Discriminator(nn.Module):
    """
    Discriminator network for AMP.
    
    Takes concatenated state pairs [s_t, s_{t+1}] as input and outputs a logit
    indicating whether the transition is from expert demonstrations (positive)
    or agent-generated (negative).
    """
    
    def __init__(self, obs_dim, hidden_dims=[64, 64], activation='relu'):
        """
        Initialize discriminator.
        
        Args:
            obs_dim: Dimension of discriminator observations (e.g., 8 for [s_t, s_{t+1}])
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu' or 'tanh')
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.hidden_dims = hidden_dims
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(layer_init(nn.Linear(prev_dim, hidden_dim)))
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        # Output layer (single logit)
        self.logit_layer = layer_init(nn.Linear(prev_dim, 1), std=1.0)
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, disc_obs):
        """
        Forward pass.
        
        Args:
            disc_obs: Discriminator observations of shape [batch_size, obs_dim]
            
        Returns:
            Logits of shape [batch_size, 1]
        """
        features = self.network(disc_obs)
        logits = self.logit_layer(features)
        return logits
    
    def get_logit_weights(self):
        """
        Get weights of the final logit layer.
        
        Returns:
            Flattened tensor of logit layer weights
        """
        return self.logit_layer.weight.flatten()
    
    def get_all_weights(self):
        """
        Get all network weights for weight decay regularization.
        
        Returns:
            Concatenated tensor of all weights
        """
        weights = []
        for param in self.parameters():
            if param.requires_grad:
                weights.append(param.flatten())
        return torch.cat(weights)
    
    def compute_grad_penalty(self, disc_obs):
        """
        Compute gradient penalty for Lipschitz constraint.
        
        Args:
            disc_obs: Discriminator observations with requires_grad=True
            
        Returns:
            Gradient penalty scalar
        """
        disc_obs.requires_grad_(True)
        
        logits = self.forward(disc_obs).squeeze(-1)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=logits,
            inputs=disc_obs,
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty (L2 norm squared)
        grad_penalty = torch.mean(torch.sum(gradients ** 2, dim=-1))
        
        return grad_penalty
    
    def predict_probs(self, disc_obs):
        """
        Predict probabilities (expert vs agent).
        
        Args:
            disc_obs: Discriminator observations
            
        Returns:
            Probabilities of being expert data (0-1)
        """
        with torch.no_grad():
            logits = self.forward(disc_obs).squeeze(-1)
            probs = torch.sigmoid(logits)
        return probs
