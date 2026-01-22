#!/usr/bin/env python3
"""
Test a trained SALP model with a single food item.
Shows how well the agent can consistently find and collect one piece of food.
"""

import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.base_config import get_sac_snake_config
from training.trainer import EvaluationRunner
from environments.salp_snake_env import SalpSnakeEnv

def test_single_food_performance(model_path: str, num_episodes: int = 5):
    """Test trained model with single food item."""
    
    print(f"🧪 Testing trained model with single food item")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print()
    
    # Get configuration and modify for single food
    config = get_sac_snake_config()
    config.environment.params['num_food_items'] = 1  # Single food item
    config.environment.params['forced_breathing'] = True  # Keep forced breathing
    
    # Create evaluation runner
    runner = EvaluationRunner(config, model_path)
    
    print("🎮 Starting single food test episodes...")
    print("Watch the agent try to find and collect the single green food item!")
    print()
    
    # Track performance
    success_count = 0
    total_steps = 0
    total_time = 0
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        
        obs, _ = runner.env.reset()
        episode_reward = 0
        episode_steps = 0
        food_collected = 0
        
        done = False
        truncated = False
        start_time = time.time()
        
        while not done and not truncated:
            # Render environment
            runner.env.render()
            
            # Select action using trained model
            action = runner.agent.select_action(obs, deterministic=True)
            
            # Take step
            obs, reward, done, truncated, info = runner.env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            food_collected = info.get('food_collected', 0)
            
            # Slower frame rate for better viewing
            time.sleep(0.03)  # ~30 FPS
        
        episode_time = time.time() - start_time
        
        # Check if successful (collected food)
        if food_collected > 0:
            success_count += 1
            print(f"  ✅ SUCCESS! Food collected in {episode_steps} steps ({episode_time:.1f}s)")
        else:
            print(f"  ❌ Failed - No food collected in {episode_steps} steps ({episode_time:.1f}s)")
        
        print(f"  Score: {episode_reward:.1f}")
        print()
        
        total_steps += episode_steps
        total_time += episode_time
        
        # Brief pause between episodes
        time.sleep(2.0)
    
    # Summary statistics
    success_rate = (success_count / num_episodes) * 100
    avg_steps = total_steps / num_episodes
    avg_time = total_time / num_episodes
    
    print("📊 Test Results Summary:")
    print(f"  Success Rate: {success_rate:.1f}% ({success_count}/{num_episodes})")
    print(f"  Average Steps: {avg_steps:.1f}")
    print(f"  Average Time: {avg_time:.1f}s")
    print()
    
    if success_rate >= 80:
        print("🏆 EXCELLENT! Agent performs very well with single food items.")
    elif success_rate >= 60:
        print("👍 GOOD! Agent shows decent performance with single food items.")
    elif success_rate >= 40:
        print("🤔 OKAY! Agent sometimes finds food but needs improvement.")
    else:
        print("😞 POOR! Agent struggles to find single food items.")
    
    runner.env.close()
    return success_rate, avg_steps, avg_time

def main():
    """Main test function."""
    # Use the latest best model
    model_paths = [
        "models/salp_snake_50ep_visual/best_model.pth",
        "models/salp_snake_50ep_20250925_213317/best_model.pth"
    ]
    
    # Find the most recent model
    latest_model = None
    for path in model_paths:
        if os.path.exists(path):
            latest_model = path
            break
    
    if latest_model is None:
        print("❌ No trained models found!")
        print("Available models should be in:")
        for path in model_paths:
            print(f"  {path}")
        return
    
    print(f"Using model: {latest_model}")
    print()
    
    # Run the test
    test_single_food_performance(latest_model, num_episodes=5)

if __name__ == "__main__":
    main()
