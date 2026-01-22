"""
Test script to evaluate the trained SALP model.
Run this in the morning to see how well the SALP learned overnight!
"""

import os
import time
import numpy as np
import torch
from typing import Dict, Any

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.base_config import get_sac_snake_config
from agents.sac_agent import SACAgent
from environments.salp_snake_env import SalpSnakeEnv


class ModelTester:
    """Test the trained SALP model."""
    
    def __init__(self, model_path: str = None):
        self.config = get_sac_snake_config()
        
        # Find the best model if no path specified
        if model_path is None:
            model_dir = os.path.join(self.config.training.model_dir, self.config.training.experiment_name)
            model_path = os.path.join(model_dir, "best_model.pth")
        
        self.model_path = model_path
        
        # Create environment with visual display
        self.env = SalpSnakeEnv(
            render_mode="human",
            width=self.config.environment.width,
            height=self.config.environment.height,
            **self.config.environment.params
        )
        
        # Create agent
        self.agent = self._create_agent()
        
        # Load trained model
        self._load_model()
        
        print(f"🧪 Model Tester initialized")
        print(f"Model: {model_path}")
        print(f"Environment: {self.config.environment.name}")
        print(f"Ready to test trained SALP!")
    
    def _create_agent(self):
        """Create agent with same architecture as training."""
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        return SACAgent(
            config=self.config.agent,
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_space=self.env.action_space
        )
    
    def _load_model(self):
        """Load the trained model."""
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found: {self.model_path}")
            print("Make sure overnight training completed successfully.")
            return False
        
        try:
            self.agent.load(self.model_path)
            print(f"✅ Successfully loaded model: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def test_episodes(self, num_episodes: int = 5, max_steps: int = 5000):
        """Test the trained model for multiple episodes."""
        print(f"\n🎮 Testing trained SALP for {num_episodes} episodes...")
        print("Watch the SALP navigate and collect food!")
        print("Press Ctrl+C to stop early if needed.\n")
        
        episode_results = []
        
        try:
            for episode in range(num_episodes):
                print(f"🎯 Episode {episode + 1}/{num_episodes}")
                
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_steps = 0
                food_collected = 0
                
                done = False
                truncated = False
                
                start_time = time.time()
                
                while not done and not truncated and episode_steps < max_steps:
                    # Render the environment
                    self.env.render()
                    
                    # Select action using trained policy (deterministic)
                    action = self.agent.select_action(obs, deterministic=True)
                    
                    # Take step
                    obs, reward, done, truncated, info = self.env.step(action)
                    
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
                    'collision': info.get('collision', False),
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
            self.env.close()
        
        # Print overall results
        self._print_summary(episode_results)
        
        return episode_results
    
    def _print_summary(self, results):
        """Print summary of test results."""
        if not results:
            return
        
        print("=" * 50)
        print("🏆 TRAINING RESULTS SUMMARY")
        print("=" * 50)
        
        scores = [r['score'] for r in results]
        food_counts = [r['food_collected'] for r in results]
        steps = [r['steps'] for r in results]
        
        print(f"Episodes Tested: {len(results)}")
        print(f"Average Score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
        print(f"Best Score: {max(scores):.1f}")
        print(f"Average Food Collected: {np.mean(food_counts):.1f} ± {np.std(food_counts):.1f}")
        print(f"Best Food Collection: {max(food_counts)}")
        print(f"Average Episode Length: {np.mean(steps):.0f} steps")
        
        # Performance categories
        excellent = sum(1 for f in food_counts if f >= 5)
        good = sum(1 for f in food_counts if 3 <= f < 5)
        okay = sum(1 for f in food_counts if 1 <= f < 3)
        poor = sum(1 for f in food_counts if f == 0)
        
        print(f"\nPerformance Breakdown:")
        print(f"  🌟 Excellent (5+ food): {excellent}/{len(results)} episodes")
        print(f"  👍 Good (3-4 food): {good}/{len(results)} episodes")
        print(f"  👌 Okay (1-2 food): {okay}/{len(results)} episodes")
        print(f"  😞 Poor (0 food): {poor}/{len(results)} episodes")
        
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
    
    def demo_single_episode(self, max_steps: int = 10000):
        """Run a single long demonstration episode."""
        print("\n🎬 Running single demonstration episode...")
        print("Watch the trained SALP navigate and collect food!")
        print("This episode can run up to 10,000 steps to show extended behavior.")
        print("Press Ctrl+C to stop early if needed.\n")
        
        try:
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            food_collected = 0
            
            done = False
            truncated = False
            
            start_time = time.time()
            
            while not done and not truncated and episode_steps < max_steps:
                # Render the environment
                self.env.render()
                
                # Select action using trained policy
                action = self.agent.select_action(obs, deterministic=True)
                
                # Take step
                obs, reward, done, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                food_collected = info.get('food_collected', 0)
                
                # Print progress every 1000 steps
                if episode_steps % 1000 == 0:
                    elapsed = time.time() - start_time
                    print(f"   Step {episode_steps}: Score={episode_reward:.1f}, Food={food_collected}, Time={elapsed:.1f}s")
                
                # Small delay for visualization
                time.sleep(0.02)
            
            episode_time = time.time() - start_time
            
            print(f"\n🏁 Demo Episode Complete!")
            print(f"   Final Score: {episode_reward:.1f}")
            print(f"   Food Collected: {food_collected}")
            print(f"   Steps: {episode_steps}")
            print(f"   Time: {episode_time:.1f}s")
            
        except KeyboardInterrupt:
            print("\n⏹️  Demo interrupted by user")
        
        finally:
            self.env.close()


def main():
    """Main testing function."""
    print("🌅 GOOD MORNING! Let's see how well the SALP learned overnight!")
    print("=" * 60)
    
    # Check if training completed
    config = get_sac_snake_config()
    model_dir = os.path.join(config.training.model_dir, config.training.experiment_name)
    summary_path = os.path.join(model_dir, "training_summary.txt")
    
    if os.path.exists(summary_path):
        print("📋 Training Summary:")
        with open(summary_path, 'r') as f:
            print(f.read())
    else:
        print("⚠️  Training summary not found. Training may still be running or failed.")
    
    # Test the model
    try:
        tester = ModelTester()
        
        print("\nChoose testing mode:")
        print("1. Quick test (5 episodes)")
        print("2. Single long demo (up to 10,000 steps)")
        print("3. Both")
        
        choice = input("Enter choice (1/2/3): ").strip()
        
        if choice in ['1', '3']:
            tester.test_episodes(num_episodes=5)
        
        if choice in ['2', '3']:
            if choice == '3':
                input("\nPress Enter to start long demo...")
            tester.demo_single_episode()
    
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
