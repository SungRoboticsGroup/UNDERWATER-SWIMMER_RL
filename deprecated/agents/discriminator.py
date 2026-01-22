"""
Discriminator network for GAIL (Generative Adversarial Imitation Learning).
Distinguishes between expert demonstrations and agent behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple

from salp.core.base_agent import BaseNetwork


class Discriminator(BaseNetwork):
    """
    Discriminator network for GAIL.
    
    Takes (state, action) pairs and outputs probability that the pair
    came from expert demonstrations (as opposed to the learning agent).
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list, 
                 activation: str = "relu", learning_rate: float = 3e-4, device: str = "cpu"):
        # Input is concatenated state and action
        input_dim = obs_dim + action_dim
        # Output is probability (single value)
        output_dim = 1
        
        # Initialize base network without output activation
        super().__init__(input_dim, output_dim, hidden_sizes, activation, output_activation=None)
        
        self.device = torch.device(device)
        self.to(self.device)
        
        # Optimizer for discriminator training
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Training metrics
        self.training_step = 0
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            action: Action tensor [batch_size, action_dim]
        
        Returns:
            Probability that (obs, action) is from expert [batch_size, 1]
        """
        # Concatenate state and action
        x = torch.cat([obs, action], dim=-1)
        
        # Pass through network
        x = super().forward(x)
        
        # Apply sigmoid to get probability
        prob = torch.sigmoid(x)
        
        return prob
    
    def predict_reward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute GAIL reward based on discriminator output.
        
        The GAIL reward is: -log(1 - D(s,a))
        This encourages the agent to fool the discriminator.
        
        Args:
            obs: Observation tensor
            action: Action tensor
        
        Returns:
            GAIL reward tensor
        """
        with torch.no_grad():
            prob = self.forward(obs, action)
            # GAIL reward: -log(1 - D(s,a))
            # Add small epsilon for numerical stability
            reward = -torch.log(1 - prob + 1e-8)
        
        return reward
    
    def update(self, expert_batch: Dict[str, np.ndarray], 
               agent_batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Update discriminator to distinguish expert from agent.
        
        Args:
            expert_batch: Batch of expert (state, action) pairs
            agent_batch: Batch of agent (state, action) pairs
        
        Returns:
            Dictionary with training metrics
        """
        # Convert to tensors
        expert_obs = torch.FloatTensor(expert_batch['observations']).to(self.device)
        expert_actions = torch.FloatTensor(expert_batch['actions']).to(self.device)
        
        agent_obs = torch.FloatTensor(agent_batch['observations']).to(self.device)
        agent_actions = torch.FloatTensor(agent_batch['actions']).to(self.device)
        
        # Get discriminator predictions
        expert_probs = self.forward(expert_obs, expert_actions)
        agent_probs = self.forward(agent_obs, agent_actions)
        
        # Binary cross-entropy loss
        # Expert samples should have label 1 (from expert)
        # Agent samples should have label 0 (not from expert)
        expert_loss = F.binary_cross_entropy(expert_probs, torch.ones_like(expert_probs))
        agent_loss = F.binary_cross_entropy(agent_probs, torch.zeros_like(agent_probs))
        
        # Total discriminator loss
        loss = expert_loss + agent_loss
        
        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.training_step += 1
        
        # Calculate accuracy
        with torch.no_grad():
            expert_correct = (expert_probs > 0.5).float().mean()
            agent_correct = (agent_probs <= 0.5).float().mean()
            accuracy = (expert_correct + agent_correct) / 2.0
        
        return {
            'discriminator_loss': loss.item(),
            'expert_loss': expert_loss.item(),
            'agent_loss': agent_loss.item(),
            'discriminator_accuracy': accuracy.item(),
            'expert_prob_mean': expert_probs.mean().item(),
            'agent_prob_mean': agent_probs.mean().item()
        }
    
    def save(self, filepath: str):
        """Save discriminator parameters."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step
        }, filepath)
    
    def load(self, filepath: str):
        """Load discriminator parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
