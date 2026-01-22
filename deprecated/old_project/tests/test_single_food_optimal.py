"""
Test script for single food optimal navigation experiment.

This test demonstrates:
1. Single food item environment
2. Proximity-based reward shaping
3. Time penalty for brevity incentive
4. Episode ends when food is collected (no respawn)
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.salp_snake_env import SalpSnakeEnv
import pygame


def test_single_food_optimal_navigation():
    """Test single food environment with proximity rewards and time penalty."""
    
    print("=" * 70)
    print("SINGLE FOOD OPTIMAL NAVIGATION TEST")
    print("=" * 70)
    print("\nExperiment Configuration:")
    print("  - Single food item (num_food_items=1)")
    print("  - Proximity reward weight: 0.5")
    print("  - Time penalty: -0.1 per step")
    print("  - Episode ends when food collected (no respawn)")
    print("  - Forced breathing mode enabled")
    print("\nReward Structure:")
    print("  - Food collection: +100.0")
    print("  - Collision: -50.0")
    print("  - Proximity: 0.0 to +0.5 (higher when closer to food)")
    print("  - Time: -0.1 per step")
    print("\nControls:")
    print("  - HOLD SPACE: Inhale (in manual mode)")
    print("  - RELEASE SPACE: Exhale")
    print("  - ←/→ Arrow Keys: Steer nozzle")
    print("  - ESC: Quit")
    print("=" * 70)
    
    # Create environment with single food and optimal navigation incentives
    env = SalpSnakeEnv(
        render_mode="human",
        num_food_items=1,              # Single food item
        proximity_reward_weight=0.5,    # Proximity-based shaping
        time_penalty=-0.1,              # Brevity incentive
        respawn_food=False,             # Episode ends when food collected
        forced_breathing=True,          # Simplified action space
        max_steps_without_food=500      # Shorter timeout for single food
    )
    
    observation, info = env.reset()
    
    print("\nStarting test episodes...")
    print("Try to reach the food as quickly as possible!")
    print("Notice the reward feedback - you get rewarded for being closer to food.\n")
    
    episode = 1
    total_steps = 0
    episode_reward = 0.0
    running = True
    nozzle_direction = 0.0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Get keyboard input for manual control
        keys = pygame.key.get_pressed()
        
        # Nozzle steering
        if keys[pygame.K_LEFT]:
            nozzle_direction = min(1.0, nozzle_direction + 0.03)
        elif keys[pygame.K_RIGHT]:
            nozzle_direction = max(-1.0, nozzle_direction - 0.03)
        
        # In forced breathing mode, only nozzle control
        action = np.array([nozzle_direction])
        
        # Step environment
        observation, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        total_steps += 1
        
        # Display reward feedback every 30 steps
        if total_steps % 30 == 0:
            distance = env._get_nearest_food_distance()
            if distance is not None:
                print(f"Step {total_steps}: Distance to food = {distance:.1f} pixels, "
                      f"Episode reward = {episode_reward:.2f}")
        
        # Render
        env.render()
        
        # Episode ended
        if done or truncated:
            print(f"\n{'='*70}")
            print(f"EPISODE {episode} COMPLETE!")
            print(f"  Steps taken: {total_steps}")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Food collected: {info['food_collected']}")
            print(f"  Collision: {info['collision']}")
            
            if info['food_collected'] > 0:
                print(f"  ✓ SUCCESS! Reached food in {total_steps} steps")
                print(f"  Efficiency: {100.0 / total_steps:.2f} points per step")
            elif info['collision']:
                print(f"  ✗ FAILED: Collision with wall")
            else:
                print(f"  ✗ FAILED: Timeout")
            
            print(f"{'='*70}\n")
            
            # Reset for next episode
            observation, info = env.reset()
            episode += 1
            total_steps = 0
            episode_reward = 0.0
            nozzle_direction = 0.0
    
    env.close()
    print("\nTest completed!")


def test_reward_calculation():
    """Test reward calculation with different configurations."""
    
    print("\n" + "="*70)
    print("REWARD CALCULATION TEST")
    print("="*70)
    
    # Test different proximity reward weights
    configs = [
        {"proximity_reward_weight": 0.0, "name": "No proximity reward"},
        {"proximity_reward_weight": 0.3, "name": "Low proximity reward (0.3)"},
        {"proximity_reward_weight": 0.5, "name": "Medium proximity reward (0.5)"},
        {"proximity_reward_weight": 1.0, "name": "High proximity reward (1.0)"},
    ]
    
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"  Weight: {config['proximity_reward_weight']}")
        
        env = SalpSnakeEnv(
            render_mode=None,
            num_food_items=1,
            proximity_reward_weight=config['proximity_reward_weight'],
            time_penalty=-0.1,
            respawn_food=False
        )
        
        env.reset()
        
        # Simulate different distances
        test_distances = [0, 100, 300, 600, 1000]  # pixels
        max_distance = np.sqrt(env.width**2 + env.height**2)
        
        print(f"  Reward at different distances:")
        for dist in test_distances:
            if dist <= max_distance:
                normalized_dist = dist / max_distance
                proximity_reward = config['proximity_reward_weight'] * (1 - normalized_dist)
                total_reward = proximity_reward + env.time_penalty
                print(f"    {dist:4d} pixels: proximity={proximity_reward:+.3f}, "
                      f"time={env.time_penalty:.1f}, total={total_reward:+.3f}")
        
        env.close()
    
    print("\n" + "="*70)


def test_time_penalty_impact():
    """Test the impact of different time penalty values."""
    
    print("\n" + "="*70)
    print("TIME PENALTY IMPACT TEST")
    print("="*70)
    
    time_penalties = [0.0, -0.05, -0.1, -0.2, -0.5]
    
    for penalty in time_penalties:
        print(f"\nTime penalty: {penalty}")
        print(f"  100 steps cost: {100 * penalty:.1f}")
        print(f"  200 steps cost: {200 * penalty:.1f}")
        print(f"  500 steps cost: {500 * penalty:.1f}")
        
        # Calculate steps needed to offset food reward
        if penalty < 0:
            steps_to_offset = -100.0 / penalty
            print(f"  Steps to offset +100 food reward: {steps_to_offset:.0f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test single food optimal navigation")
    parser.add_argument('--test', choices=['interactive', 'rewards', 'time', 'all'],
                       default='interactive',
                       help='Which test to run')
    
    args = parser.parse_args()
    
    if args.test == 'interactive':
        test_single_food_optimal_navigation()
    elif args.test == 'rewards':
        test_reward_calculation()
    elif args.test == 'time':
        test_time_penalty_impact()
    elif args.test == 'all':
        test_reward_calculation()
        test_time_penalty_impact()
        input("\nPress Enter to start interactive test...")
        test_single_food_optimal_navigation()
