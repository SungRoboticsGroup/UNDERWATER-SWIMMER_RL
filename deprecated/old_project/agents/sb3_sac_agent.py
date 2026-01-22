"""
Stable Baselines3 SAC agent wrapper for SALP environments.
This provides a standard, well-tested SAC implementation using SB3.

You can easily test different network architectures by modifying the policy_kwargs
parameter without rewriting any core RL code. SB3 handles all the algorithm logic.

Example architectures:
    # Small network (faster, less capacity):
    policy_kwargs = dict(net_arch=[64, 64])
    
    # Medium network (balanced):
    policy_kwargs = dict(net_arch=[256, 256])
    
    # Large network (more capacity):
    policy_kwargs = dict(net_arch=[512, 512, 256])
    
    # Asymmetric (different for actor/critic):
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[512, 512]))
    
    # With different activation:
    policy_kwargs = dict(net_arch=[256, 256], activation_fn=torch.nn.ELU)
"""

import numpy as np
from typing import Dict, Any, Optional, List
import os
import torch.nn as nn

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from salp.config.base_config import AgentConfig


class SB3SACAgent:
    """
    Wrapper around Stable Baselines3's SAC implementation.
    
    This provides a clean, standardized SAC agent that's compatible with
    the existing SALP training infrastructure while using SB3's battle-tested
    implementation.
    """
    
    def __init__(self, config: AgentConfig, env, verbose: int = 1):
        """
        Initialize SB3 SAC agent.
        
        Args:
            config: Agent configuration (from base_config.py)
            env: Gymnasium environment
            verbose: Verbosity level for SB3 (0: no output, 1: info, 2: debug)
        """
        self.config = config
        self.env = env
        
        # Training state (for compatibility with existing code)
        self.training_step = 0
        self.episode_count = 0
        
        # Extract dimensions from environment
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_space = env.action_space
        
        # Create SB3 SAC agent with configuration
        self.model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.params.get('learning_starts', 1000),
            batch_size=config.batch_size,
            tau=config.tau,
            gamma=config.gamma,
            train_freq=config.params.get('train_freq', 1),
            gradient_steps=config.params.get('gradient_steps', 1),
            ent_coef=config.params.get('alpha', 'auto'),  # Entropy coefficient (alpha)
            target_update_interval=config.params.get('target_update_interval', 1),
            target_entropy=config.params.get('target_entropy', 'auto'),
            use_sde=config.params.get('use_sde', False),
            sde_sample_freq=config.params.get('sde_sample_freq', -1),
            use_sde_at_warmup=config.params.get('use_sde_at_warmup', False),
            policy_kwargs=dict(
                net_arch=config.hidden_sizes,
                # activation_fn can be: nn.ReLU, nn.Tanh, nn.ELU, nn.LeakyReLU
            ),
            verbose=verbose,
            seed=config.params.get('seed', None),
            device=config.params.get('device', 'auto')
        )
        
        print(f"✓ Initialized Stable Baselines3 SAC agent")
        print(f"  • Policy: MlpPolicy")
        print(f"  • Hidden layers: {config.hidden_sizes}")
        print(f"  • Learning rate: {config.learning_rate}")
        print(f"  • Gamma: {config.gamma}")
        print(f"  • Batch size: {config.batch_size}")
        print(f"  • Buffer size: {config.buffer_size}")
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action using the policy.
        
        Args:
            observation: Current observation
            deterministic: If True, use deterministic (mean) action
        
        Returns:
            Action array
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def learn(self, total_timesteps: int, callback: Optional[BaseCallback] = None):
        """
        Train the agent.
        
        Args:
            total_timesteps: Number of timesteps to train
            callback: Optional callback for logging/early stopping
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10,
            reset_num_timesteps=False  # Don't reset if continuing training
        )
        
        # Update training state
        self.training_step = self.model.num_timesteps
    
    def save(self, filepath: str):
        """
        Save agent to file.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load agent from file.
        
        Args:
            filepath: Path to load the model from
        """
        self.model = SAC.load(filepath, env=self.env)
        self.training_step = self.model.num_timesteps
        print(f"✓ Model loaded from {filepath}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information for logging."""
        return {
            "training_step": self.training_step,
            "episode_count": self.episode_count,
            "num_timesteps": self.model.num_timesteps,
            "algorithm": "SAC (Stable Baselines3)",
            "config": self.config.to_dict()
        }
    
    @property
    def replay_buffer(self):
        """Access to SB3's replay buffer."""
        return self.model.replay_buffer
    
    def __repr__(self) -> str:
        return f"SB3SACAgent(timesteps={self.model.num_timesteps})"
