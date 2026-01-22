#!/usr/bin/env python3
"""
Watch Navigation Live
Visualize the agent navigating from start to goal position in real-time.
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC
from src.salp.environments.salp_snake_env import SalpSnakeEnv


def watch_navigation(model_path: str, 
                     start_pos: tuple = (100, 300),
                     goal_pos: tuple = (700, 300),
                     max_steps: int = 3000,
                     env_width: int = 800,
                     env_height: int = 600):
    """
    Watch the agent navigate from start to goal with live visualization.
    """
    print("="*60)
    print("Loading model...")
    model = SAC.load(model_path)
    print("✓ Model loaded successfully")
    
    print("\nCreating environment with visualization...")
    env = SalpSnakeEnv(
        render_mode='human',  # Enable visualization!
        width=env_width,
        height=env_height,
        num_food_items=1,
        forced_breathing=True,
        respawn_food=False,
        max_steps_without_food=max_steps
    )
    
    print(f"\nStarting navigation task:")
    print(f"  Start: {start_pos}")
    print(f"  Goal:  {goal_pos}")
    print(f"  Max steps: {max_steps}")
    print("="*60)
    
    # Reset and setup
    obs, _ = env.reset()
    env.robot_pos = np.array(start_pos, dtype=float)
    env.robot_velocity = np.array([0.0, 0.0], dtype=float)
    env.robot_angle = np.random.uniform(-np.pi, np.pi)
    env.robot_angular_velocity = 0.0
    
    # Place food at goal
    env.food_positions = [np.array(goal_pos, dtype=float)]
    env.steps_since_food = 0
    
    # Update observation
    obs = env._get_extended_observation()
    
    print("\n🎥 Starting visualization... (Close window to stop)")
    print("   Green circle = Start")
    print("   Food (at goal) = Target")
    print("="*60)
    
    step = 0
    done = False
    
    while step < max_steps:
        # Explicitly render the environment
        env.render()
        
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Take step
        obs, reward, done, truncated, info = env.step(action)
        
        step += 1
        
        # Check if goal reached
        distance_to_goal = np.linalg.norm(env.robot_pos - np.array(goal_pos))
        
        # Print progress every 100 steps
        if step % 100 == 0:
            print(f"Step {step:4d} | Distance to goal: {distance_to_goal:.1f}px")
        
        # Check if goal reached
        if distance_to_goal < 50:
            print(f"\n✅ SUCCESS! Goal reached in {step} steps!")
            print(f"   Final distance: {distance_to_goal:.1f}px")
            # Render final state
            env.render()
            break
    
    if step >= max_steps:
        distance_to_goal = np.linalg.norm(env.robot_pos - np.array(goal_pos))
        print(f"\n⏱️  Reached max steps ({max_steps})")
        print(f"   Final distance to goal: {distance_to_goal:.1f}px")
    
    print("\n" + "="*60)
    print("Closing environment...")
    env.close()
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Watch Navigation Live')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.zip)')
    parser.add_argument('--start', type=str, default='100,300',
                       help='Start position as x,y')
    parser.add_argument('--goal', type=str, default='700,300',
                       help='Goal position as x,y')
    parser.add_argument('--max-steps', type=int, default=3000,
                       help='Maximum steps per trial')
    parser.add_argument('--width', type=int, default=800,
                       help='Environment width in pixels')
    parser.add_argument('--height', type=int, default=600,
                       help='Environment height in pixels')
    
    args = parser.parse_args()
    
    # Parse positions
    start_pos = tuple(map(float, args.start.split(',')))
    goal_pos = tuple(map(float, args.goal.split(',')))
    
    # Run visualization
    watch_navigation(args.model, start_pos, goal_pos, args.max_steps, args.width, args.height)


if __name__ == "__main__":
    main()
