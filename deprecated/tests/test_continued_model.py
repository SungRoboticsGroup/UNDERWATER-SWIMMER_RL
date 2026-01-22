#!/usr/bin/env python3
"""
Test the continued training model (425 episodes)
"""

import os
import time
import numpy as np
import torch

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.base_config import get_sac_snake_config
from agents.sac_agent import SACAgent
from environments.salp_snake_env import SalpSnakeEnv


def main():
    print("🧪 Testing the Continued Training Model (425 episodes)")
    print("=" * 60)
    
    # Use the continued training model
    model_path = "models/salp_snake_sac_continued/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Load config
    config = get_sac_snake_config()
    
    # Create environment with visual display
    env = SalpSnakeEnv(
        render_mode="human",
        width=config.environment.width,
        height=config.environment.height,
        **config.environment.params
    )
    
    # Create agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SACAgent(
        config=config.agent,
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_space=env.action_space
    )
    
    # Load the model
    try:
        agent.load(model_path)
        print(f"✅ Successfully loaded model: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print(f"🧪 Model Tester initialized")
    print(f"Model: {model_path}")
    print(f"Environment: {config.environment.name}")
    print(f"Ready to test trained SALP!")
    
    # Test for 3 episodes
    print(f"\n🎮 Testing trained SALP for 3 episodes...")
    print("Watch the SALP navigate and collect food!")
    print("Press Ctrl+C to stop early if needed.\n")
    
    episode_results = []
    
    try:
        for episode in range(3):
            print(f"🎯 Episode {episode + 1}/3")
            
            obs, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            food_collected = 0
            
            done = False
            truncated = False
            
            start_time = time.time()
            
            while not done and not truncated and episode_steps < 5000:
                # Render the environment
                env.render()
                
                # Select action using trained policy (deterministic)
                action = agent.select_action(obs, deterministic=True)
                
                # Take step
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                food_collected = info.get('food_collected', 0)
                
                # Small delay for better visualization
                time.sleep(0.02)  # 50 FPS
            
            episode_time = time.time() - start_time
            
            # Episode results
            result = {
                'episode': episode + 1,
                'score': episode_reward,
                'food_collected': food_collected,
                'steps': episode_steps,
                'time': episode_time,
                'reason': 'collision' if done else ('timeout' if truncated else 'completed')
            }
            
            episode_results.append(result)
            
            # Print episode summary
            print(f"   Score: {episode_reward:.1f}")
            print(f"   Food Collected: {food_collected}")
            print(f"   Steps: {episode_steps}")
            print(f"   Time: {episode_time:.1f}s")
            print(f"   Ended: {result['reason']}")
            print()
            
            # Brief pause between episodes
            time.sleep(2.0)
    
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
    
    finally:
        env.close()
    
    # Print summary
    if episode_results:
        print("=" * 50)
        print("🏆 TRAINING RESULTS SUMMARY")
        print("=" * 50)
        
        scores = [r['score'] for r in episode_results]
        food_counts = [r['food_collected'] for r in episode_results]
        steps = [r['steps'] for r in episode_results]
        
        print(f"Episodes Tested: {len(episode_results)}")
        print(f"Average Score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
        print(f"Best Score: {max(scores):.1f}")
        print(f"Average Food Collected: {np.mean(food_counts):.1f} ± {np.std(food_counts):.1f}")
        print(f"Best Food Collection: {max(food_counts)}")
        print(f"Average Episode Length: {np.mean(steps):.0f} steps")
        
        # Overall assessment
        avg_food = np.mean(food_counts)
        if avg_food >= 4:
            assessment = "🎉 EXCELLENT! The SALP learned to navigate and collect food very well!"
        elif avg_food >= 2:
            assessment = "👍 GOOD! The SALP learned basic navigation and food collection."
        elif avg_food >= 1:
            assessment = "👌 OKAY! The SALP learned some navigation skills."
        else:
            assessment = "😞 NEEDS MORE TRAINING! The SALP struggled to collect food."
        
        print(f"\nOverall Assessment: {assessment}")
        print("=" * 50)


if __name__ == "__main__":
    main()
