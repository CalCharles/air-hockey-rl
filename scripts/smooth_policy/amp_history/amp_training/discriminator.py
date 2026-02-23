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
    
    def __init__(
        self,
        obs_dim,
        hidden_layer_size=64,
        num_hidden_layers=2,
        activation='leaky_relu',
        hidden_dims=None,
    ):
        """
        Initialize discriminator.
        
        Args:
            obs_dim: Dimension of discriminator observations (e.g., 8 for [s_t, s_{t+1}])
            hidden_layer_size: Width of each hidden layer (preferred spec)
            num_hidden_layers: Number of hidden layers (preferred spec)
            activation: Activation function ('leaky_relu', 'relu' or 'tanh')
            hidden_dims: Optional legacy list of hidden layer dimensions
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        if hidden_dims is not None:
            hidden_dims = [int(size) for size in hidden_dims]
            if len(hidden_dims) == 0:
                raise ValueError("hidden_dims must contain at least one layer size.")
            if any(size <= 0 for size in hidden_dims):
                raise ValueError(f"hidden_dims must be positive, got: {hidden_dims}")
            self.hidden_dims = hidden_dims
        else:
            hidden_layer_size = int(hidden_layer_size)
            num_hidden_layers = int(num_hidden_layers)
            if hidden_layer_size <= 0:
                raise ValueError(f"hidden_layer_size must be positive, got {hidden_layer_size}")
            if num_hidden_layers < 1:
                raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")
            self.hidden_dims = [hidden_layer_size] * num_hidden_layers
        self.hidden_layer_size = int(self.hidden_dims[0])
        self.num_hidden_layers = int(len(self.hidden_dims))
        
        # Choose activation function
        if activation == 'leaky_relu':
            activation_ctor = lambda: nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'relu':
            activation_ctor = nn.ReLU
        elif activation == 'tanh':
            activation_ctor = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(layer_init(nn.Linear(prev_dim, hidden_dim)))
            layers.append(activation_ctor())
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
