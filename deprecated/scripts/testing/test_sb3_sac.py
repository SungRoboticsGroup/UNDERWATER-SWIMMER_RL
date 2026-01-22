"""
Test/demo script for trained Stable Baselines3 SAC models.

Usage:
    python scripts/test_sb3_sac.py <model_path> [--episodes 5] [--render]
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from environments.salp_snake_env import SalpSnakeEnv
from config.base_config import get_sac_snake_config
import numpy as np


def test_model(model_path: str, n_episodes: int = 5, render: bool = False):
    """
    Test a trained SB3 SAC model.
    
    Args:
        model_path: Path to saved model (.zip file or directory)
        n_episodes: Number of episodes to test
        render: Whether to render the environment
    """
    
    print("=" * 70)
    print("Testing Stable Baselines3 SAC Model")
    print("=" * 70)
    
    # Load configuration
    config = get_sac_snake_config()
    
    # Create environment
    render_mode = "human" if render else None
    env = SalpSnakeEnv(render_mode=render_mode, **config.environment.params)
    
    print(f"\n✓ Environment created")
    print(f"  • Render mode: {render_mode}")
    
    # Load model
    if not model_path.endswith('.zip'):
        # Try common model names
        for name in ['best_model.zip', 'final_model.zip']:
            full_path = os.path.join(model_path, name)
            if os.path.exists(full_path):
                model_path = full_path
                break
    
    print(f"\n✓ Loading model from: {model_path}")
    model = SAC.load(model_path, env=env)
    print(f"  • Training timesteps: {model.num_timesteps:,}")
    
    # Run test episodes
    print(f"\n{'=' * 70}")
    print(f"Running {n_episodes} test episodes...")
    print(f"{'=' * 70}\n")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Use deterministic actions for testing
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}/{n_episodes}:")
        print(f"  • Reward: {episode_reward:.2f}")
        print(f"  • Length: {episode_length} steps")
        if 'score' in info:
            print(f"  • Score: {info['score']}")
        print()
    
    # Print summary statistics
    print(f"{'=' * 70}")
    print("Test Summary")
    print(f"{'=' * 70}")
    print(f"Episodes: {n_episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"{'=' * 70}\n")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Test trained SB3 SAC model on SALP Snake environment"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to saved model (.zip file or directory containing model)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of test episodes (default: 5)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during testing"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        sys.exit(1)
    
    test_model(args.model_path, args.episodes, args.render)


if __name__ == "__main__":
    main()
