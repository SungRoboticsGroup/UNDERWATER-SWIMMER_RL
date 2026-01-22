#!/usr/bin/env python3
"""
Quick-start script to watch trained SALP agents in action.

Usage:
    # Auto-detect and use best available model
    python quickstart/watch_agent.py
    
    # Specify a model
    python quickstart/watch_agent.py --model data/models/single_food_long_horizon_20251130_223921/best_model.zip
    
    # Watch multiple episodes
    python quickstart/watch_agent.py --episodes 10
    
    # Use stochastic (exploratory) actions instead of deterministic
    python quickstart/watch_agent.py --stochastic
"""

import os
import sys
import argparse
import glob
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def find_best_model():
    """Find the most recent best_model in data/models/"""
    models_dir = project_root / "data" / "models"
    
    if not models_dir.exists():
        print("❌ No models directory found at data/models/")
        return None
    
    # Look for all best_model files
    zip_models = list(models_dir.glob("*/best_model.zip"))
    pth_models = list(models_dir.glob("*/best_model.pth"))
    
    all_models = zip_models + pth_models
    
    if not all_models:
        print("❌ No trained models found in data/models/")
        print("   Train a model first or specify a model path with --model")
        return None
    
    # Get the most recent model by modification time
    best_model = max(all_models, key=lambda p: p.stat().st_mtime)
    return best_model


def load_model_and_env(model_path: Path, render: bool = True):
    """Load model and create appropriate environment"""
    print(f"\n{'='*70}")
    print(f"Loading Model: {model_path.parent.name}/{model_path.name}")
    print(f"{'='*70}\n")
    
    model_path_str = str(model_path)
    
    # Determine model type and load appropriately
    if model_path.suffix == '.zip':
        # SB3 model
        print("📦 Model type: Stable Baselines3 (.zip)")
        from stable_baselines3 import SAC
        
        # Determine environment based on model directory name
        model_dir = model_path.parent.name.lower()
        
        # Create environment
        render_mode = "human" if render else None
        
        if 'robot' in model_dir or 'salp_robot' in model_dir:
            from salp.environments.salp_robot_env import SalpRobotEnv
            print("🌊 Environment: SalpRobotEnv")
            env = SalpRobotEnv(render_mode=render_mode)
        else:
            # Default to SalpSnakeEnv (most common)
            from salp.environments.salp_snake_env import SalpSnakeEnv
            print("🐍 Environment: SalpSnakeEnv")
            # Check for num_food_items parameter
            num_food = getattr(load_model_and_env, '_num_food_items', 5)
            env = SalpSnakeEnv(render_mode=render_mode, num_food_items=num_food)
        
        # Load model
        model = SAC.load(model_path_str, env=env)
        print(f"✓ Model loaded (trained for {model.num_timesteps:,} steps)")
        
        return model, env
        
    elif model_path.suffix == '.pth':
        # Custom PyTorch model
        print("🔥 Model type: Custom PyTorch (.pth)")
        
        # Import custom agent
        from salp.agents.sb3_sac_agent import SB3SACAgent
        from salp.config.base_config import get_sac_snake_config
        
        # Load config
        config = get_sac_snake_config()
        
        # Create environment
        render_mode = "human" if render else None
        from salp.environments.salp_snake_env import SalpSnakeEnv
        print("🐍 Environment: SalpSnakeEnv")
        env = SalpSnakeEnv(render_mode=render_mode, **config.environment.params)
        
        # Create and load agent
        agent = SB3SACAgent(config.agent, env, verbose=0)
        agent.load(model_path_str)
        print(f"✓ Model loaded")
        
        return agent, env
    
    else:
        raise ValueError(f"Unknown model format: {model_path.suffix}")


def watch_agent(model_path: Path, n_episodes: int = 5, deterministic: bool = True, render: bool = True):
    """Watch a trained agent perform in the environment"""
    
    # Load model and environment
    model, env = load_model_and_env(model_path, render=render)
    
    print(f"\n{'='*70}")
    print(f"Watching Agent for {n_episodes} episodes")
    print(f"Action Mode: {'Deterministic (best actions)' if deterministic else 'Stochastic (exploratory)'}")
    print(f"{'='*70}\n")
    
    # Track statistics
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    
    try:
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            print(f"\n▶ Episode {episode + 1}/{n_episodes}")
            
            while not done:
                # Get action from model
                action, _ = model.predict(obs, deterministic=deterministic)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                # Render if requested
                if render:
                    env.render()
            
            # Store statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Print episode summary
            print(f"  ├─ Reward: {episode_reward:.2f}")
            print(f"  ├─ Length: {episode_length} steps")
            if 'score' in info:
                episode_scores.append(info['score'])
                print(f"  └─ Score: {info['score']}")
            if 'food_collected' in info:
                print(f"     └─ Food collected: {info['food_collected']}")
    
    except KeyboardInterrupt:
        print("\n\n⏸ Stopped by user")
    
    finally:
        # Print summary
        if episode_rewards:
            import numpy as np
            print(f"\n{'='*70}")
            print("📊 Summary Statistics")
            print(f"{'='*70}")
            print(f"Episodes completed: {len(episode_rewards)}")
            print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
            print(f"Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
            print(f"Best reward: {np.max(episode_rewards):.2f}")
            print(f"Worst reward: {np.min(episode_rewards):.2f}")
            if episode_scores:
                print(f"Mean score: {np.mean(episode_scores):.1f} ± {np.std(episode_scores):.1f}")
            print(f"{'='*70}\n")
        
        # Clean up
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Watch trained SALP agents in action",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Watch best available model
  python quickstart/watch_agent.py
  
  # Watch specific model
  python quickstart/watch_agent.py --model data/models/single_food_long_horizon_20251130_223921/best_model.zip
  
  # Watch 10 episodes with stochastic actions
  python quickstart/watch_agent.py --episodes 10 --stochastic
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Path to model file (.zip or .pth). If not specified, uses most recent best_model.'
    )
    
    parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=5,
        help='Number of episodes to watch (default: 5)'
    )
    
    parser.add_argument(
        '--stochastic', '-s',
        action='store_true',
        help='Use stochastic (exploratory) actions instead of deterministic'
    )
    
    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Run without rendering (for testing/benchmarking)'
    )
    
    parser.add_argument(
        '--num-food',
        type=int,
        default=5,
        help='Number of food items in environment (default: 5, use 1 for single target)'
    )
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"❌ Model not found: {args.model}")
            sys.exit(1)
    else:
        print("🔍 Auto-detecting best available model...")
        model_path = find_best_model()
        if model_path is None:
            sys.exit(1)
        print(f"✓ Found: {model_path.parent.name}/{model_path.name}\n")
    
    # Set num_food_items parameter for environment creation
    load_model_and_env._num_food_items = args.num_food
    
    # Watch the agent
    watch_agent(
        model_path=model_path,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=not args.no_render
    )


if __name__ == "__main__":
    main()
