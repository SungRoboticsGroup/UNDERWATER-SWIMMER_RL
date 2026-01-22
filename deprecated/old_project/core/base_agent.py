"""
Base classes for RL agents in the SALP project.
Only includes utilities needed by custom components (like Discriminator).
For standard RL agents, use Stable Baselines3 implementations.
"""

from typing import Optional, List
import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    """
    Base neural network class with common functionality.
    Used by custom components like Discriminator.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int], 
                 activation: str = "relu", output_activation: Optional[str] = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        
        # Activation functions
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.activation = activations[activation]()
        self.output_activation = activations[output_activation]() if output_activation else None
        
        # Build network layers
        self.layers = self._build_layers()
    
    def _build_layers(self) -> nn.ModuleList:
        """Build network layers."""
        layers = nn.ModuleList()
        
        # Input layer
        prev_size = self.input_dim
        
        # Hidden layers
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, self.output_dim))
        
        return layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Hidden layers with activation
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        
        # Output layer
        x = self.layers[-1](x)
        
        # Output activation if specified
        if self.output_activation is not None:
            x = self.output_activation(x)
        
        return x


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float):
    """Soft update of target network parameters."""
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def hard_update(target_net: nn.Module, source_net: nn.Module):
    """Hard update of target network parameters."""
    target_net.load_state_dict(source_net.state_dict())
