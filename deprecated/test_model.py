#!/usr/bin/env python3
"""
Test a trained SALP model with visualization.

Usage:
    python test_model.py
    python test_model.py --episodes 10
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch

# Direct imports to avoid circular dependencies
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

# Import only what we need
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    """Actor network matching the saved checkpoint structure."""
    def __init__(self, obs_dim, action_dim, hidden_sizes=[256, 256]):
        super().__init__()
        
        # Build layers to match saved structure: layers.0, layers.1, layers.2
        all_layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            all_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        all_layers.append(nn.Linear(prev_size, action_dim * 2))  # mean + log_std
        
        self.layers = nn.ModuleList(all_layers)
        
    def forward(self, obs, deterministic=False):
        x = obs
        # Apply all but last layer with ReLU
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        
        # Last layer gives mean and log_std
        output = self.layers[-1](x)
        mean, log_std = output.chunk(2, dim=-1)
        
        if deterministic:
            return torch.tanh(mean)
        
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        dist = Normal(mean, std)
        action = dist.rsample()
        return torch.tanh(action)


def test_model(model_path: str, n_episodes: int = 5):
    """Test a trained model."""
    
    # Import environment
    from salp.environments.salp_snake_env import SalpSnakeEnv
    
    print("=" * 70)
    print("SALP MODEL TEST")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print("=" * 70 + "\n")
    
    # Create single food environment
    env = SalpSnakeEnv(
        render_mode="human",
        num_food_items=1,
        food_reward=1000.0,
        collision_penalty=-10.0,
        time_penalty=-0.1,
        proximity_reward_weight=5.0,
        respawn_food=True,
        forced_breathing=True,
        max_steps_without_food=1500
    )
    
    print(f"Environment created")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}\n")
    
    # Load model
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Debug: check what keys are in the checkpoint
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create actor
    actor = Actor(obs_dim, action_dim, hidden_sizes=[256, 256])
    
    # Try different possible keys for the policy
    if 'policy' in checkpoint:
        actor.load_state_dict(checkpoint['policy'])
    elif 'actor' in checkpoint:
        actor.load_state_dict(checkpoint['actor'])
    elif 'policy_state_dict' in checkpoint:
        actor.load_state_dict(checkpoint['policy_state_dict'])
    else:
        # Assume the checkpoint itself is the state dict
        actor.load_state_dict(checkpoint)
    
    actor.eval()
    
    print("✓ Model loaded\n")
    
    # Run episodes
    episode_rewards = []
    episode_lengths = []
    episode_food = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        print(f"🎮 Episode {episode + 1}/{n_episodes}")
        
        while not done and steps < 1500:
            # Get action from trained agent
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_tensor = actor(obs_tensor, deterministic=True)
                action = action_tensor.squeeze(0).numpy()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            # Render
            env.render()
            
            # Small delay
            import time
            time.sleep(0.02)
        
        # Episode summary
        food_collected = info.get('food_collected', 0)
        collision = info.get('collision', False)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        episode_food.append(food_collected)
        
        print(f"   Reward: {episode_reward:.1f}")
        print(f"   Steps: {steps}")
        print(f"   Food: {food_collected}")
        print(f"   Collision: {collision}\n")
    
    env.close()
    
    # Final statistics
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Episodes: {n_episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average Food: {np.mean(episode_food):.2f} ± {np.std(episode_food):.2f}")
    print(f"Success Rate: {100 * np.sum(np.array(episode_food) > 0) / n_episodes:.1f}%")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test a trained SALP model")
    
    parser.add_argument(
        '--model',
        default='data/models/single_food_optimal_navigation/best_model.pth',
        help='Path to model .pth file'
    )
    
    parser.add_argument(
        '--episodes', '-n',
        type=int,
        default=5,
        help='Number of test episodes'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        print("\nAvailable models:")
        models_dir = Path('data/models')
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    best_model = model_dir / 'best_model.pth'
                    if best_model.exists():
                        print(f"  {best_model}")
        sys.exit(1)
    
    test_model(args.model, args.episodes)


if __name__ == "__main__":
    main()
