#!/usr/bin/env python3
"""
Create a video recording of a trained SALP model.
Records a single episode showing the SALP navigating and collecting food.
"""

import os
import sys
import time
import numpy as np
from typing import Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.base_config import get_sac_snake_config
from agents.sac_agent import SACAgent
from environments.salp_snake_env import SalpSnakeEnv
from gymnasium.wrappers import RecordVideo


def find_best_model() -> Optional[str]:
    """Find the best available trained model - prioritize the 1000ep model."""
    # Prioritize the 1000 episode model as it should be the best performing
    best_model = "models/salp_snake_1000ep_fixed12/best_model.pth"
    if os.path.exists(best_model):
        print(f"✅ Using 1000-episode trained model: {best_model}")
        return best_model
    
    # Fallback to other models if 1000ep not available
    potential_models = [
        "models/salp_snake_sac_optimized/best_model.pth",
        "models/salp_snake_sac_overnight/best_model.pth", 
        "models/salp_snake_sac_continued/best_model.pth",
        "models/salp_snake_50ep_visual/best_model.pth",
    ]
    
    for model_path in potential_models:
        if os.path.exists(model_path):
            print(f"✅ Found fallback model: {model_path}")
            return model_path
    
    # If no predefined models found, search directories
    models_dir = "models"
    if os.path.exists(models_dir):
        for exp_dir in os.listdir(models_dir):
            exp_path = os.path.join(models_dir, exp_dir)
            if os.path.isdir(exp_path):
                best_model = os.path.join(exp_path, "best_model.pth")
                if os.path.exists(best_model):
                    print(f"✅ Found model: {best_model}")
                    return best_model
    
    return None


def create_salp_video(model_path: str, video_name: str = "salp_demo"):
    """Create a video of the SALP agent in action."""
    print(f"🎬 Creating SALP video from model: {model_path}")
    
    # Load configuration
    config = get_sac_snake_config()
    
    # Create base environment (rgb_array mode for video recording)
    base_env = SalpSnakeEnv(
        render_mode="rgb_array",  # Required for video recording
        width=config.environment.width,
        height=config.environment.height,
        **config.environment.params
    )
    
    # Create videos directory
    video_dir = "videos"
    os.makedirs(video_dir, exist_ok=True)
    
    # Wrap environment with video recorder
    env = RecordVideo(
        env=base_env,
        video_folder=video_dir,
        episode_trigger=lambda episode_id: True,  # Record every episode
        name_prefix=video_name
    )
    
    # Create agent with same configuration as training
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SACAgent(
        config=config.agent,
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_space=env.action_space
    )
    
    # Load the trained model
    try:
        agent.load(model_path)
        print(f"✅ Successfully loaded trained model")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print(f"🎯 Starting video recording...")
    print(f"   Environment: {config.environment.name}")
    print(f"   Model: {os.path.basename(model_path)}")
    print(f"   Food items: {config.environment.params.get('num_food_items', 'unknown')}")
    print(f"   Video will be saved to: {video_dir}/")
    
    # Record one episode
    obs, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    food_collected = 0
    
    done = False
    truncated = False
    max_steps = 3000  # Limit episode length for reasonable video duration
    
    start_time = time.time()
    
    while not done and not truncated and episode_steps < max_steps:
        # Select action using trained policy (deterministic for consistent behavior)
        action = agent.select_action(obs, deterministic=True)
        
        # Take step in environment
        obs, reward, done, truncated, info = env.step(action)
        
        episode_reward += reward
        episode_steps += 1
        food_collected = info.get('food_collected', 0)
        
        # Print progress every 500 steps
        if episode_steps % 500 == 0:
            print(f"   Step {episode_steps}: Score={episode_reward:.1f}, Food={food_collected}")
    
    episode_time = time.time() - start_time
    
    # Close environment (this triggers video saving)
    env.close()
    
    # Print final results
    print(f"\n🎉 Video recording completed!")
    print(f"   Final Score: {episode_reward:.1f}")
    print(f"   Food Collected: {food_collected}")
    print(f"   Episode Steps: {episode_steps}")
    print(f"   Recording Time: {episode_time:.1f} seconds")
    print(f"   Episode Status: {'Success' if done else ('Timeout' if truncated else 'Stopped')}")
    
    # Find and report the video file
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if video_files:
        latest_video = max([os.path.join(video_dir, f) for f in video_files], key=os.path.getctime)
        file_size_mb = os.path.getsize(latest_video) / (1024 * 1024)
        print(f"\n📹 Video saved: {latest_video}")
        print(f"   File size: {file_size_mb:.1f} MB")
        print(f"   To view: open {latest_video}")
    else:
        print(f"\n⚠️  No video file found in {video_dir}/")
    
    return episode_reward, food_collected, episode_steps


def main():
    """Main function to create SALP video."""
    print("🎬 SALP Video Creator")
    print("=" * 50)
    
    # Find best available model
    model_path = find_best_model()
    
    if model_path is None:
        print("❌ No trained models found!")
        print("\nLooked for models in:")
        print("  - models/*/best_model.pth")
        print("\nPlease ensure you have a trained model available.")
        return
    
    # Get model name for video
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_dir)
    video_name = f"salp_{model_name}_demo"
    
    print(f"🤖 Using model: {model_name}")
    print(f"📹 Video name: {video_name}")
    print()
    
    try:
        # Create the video
        score, food, steps = create_salp_video(model_path, video_name)
        
        # Success message
        print("\n" + "="*50)
        print("🌟 SUCCESS! SALP video created successfully!")
        print(f"Your trained SALP agent achieved:")
        print(f"  • Score: {score:.1f}")
        print(f"  • Food collected: {food}")
        print(f"  • Episode length: {steps} steps")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\n⏹️  Video creation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error creating video: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
