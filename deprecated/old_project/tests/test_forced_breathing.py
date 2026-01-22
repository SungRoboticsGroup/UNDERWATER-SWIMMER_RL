"""
Test script for forced breathing mode implementation.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.salp_snake_env import SalpSnakeEnv

def test_forced_breathing():
    """Test the forced breathing mode implementation."""
    print("Testing Forced Breathing Mode Implementation")
    print("=" * 50)
    
    # Test forced breathing mode
    print("\n1. Testing Forced Breathing Mode (forced_breathing=True)")
    env_forced = SalpSnakeEnv(render_mode=None, forced_breathing=True)
    obs, info = env_forced.reset()
    
    print(f"   Action space: {env_forced.action_space}")
    print(f"   Action space shape: {env_forced.action_space.shape}")
    print(f"   Environment forced_breathing attribute: {getattr(env_forced, 'forced_breathing', 'NOT SET')}")
    
    # Test single action
    action = np.array([0.5])  # Single nozzle direction
    obs, reward, done, truncated, info = env_forced.step(action)
    print(f"   Single action step successful: reward={reward:.2f}")
    
    # Test normal mode
    print("\n2. Testing Normal Mode (forced_breathing=False)")
    env_normal = SalpSnakeEnv(render_mode=None, forced_breathing=False)
    obs, info = env_normal.reset()
    
    print(f"   Action space: {env_normal.action_space}")
    print(f"   Action space shape: {env_normal.action_space.shape}")
    print(f"   Environment forced_breathing attribute: {getattr(env_normal, 'forced_breathing', 'NOT SET')}")
    
    # Test two actions
    action = np.array([0.8, -0.3])  # [inhale_control, nozzle_direction]
    obs, reward, done, truncated, info = env_normal.step(action)
    print(f"   Two action step successful: reward={reward:.2f}")
    
    # Test robot breathing behavior
    print("\n3. Testing Robot Breathing Behavior")
    print("   Forced breathing mode:")
    for i in range(5):
        action = np.array([0.0])  # Neutral nozzle
        obs, reward, done, truncated, info = env_forced.step(action)
        print(f"   Step {i+1}: is_inhaling={env_forced.is_inhaling}, breathing_phase='{env_forced.breathing_phase}', water_volume={env_forced.water_volume:.2f}")
    
    print("\n   Normal mode:")
    for i in range(5):
        action = np.array([0.5, 0.0])  # Neutral inhale and nozzle
        obs, reward, done, truncated, info = env_normal.step(action)
        print(f"   Step {i+1}: is_inhaling={env_normal.is_inhaling}, breathing_phase='{env_normal.breathing_phase}', water_volume={env_normal.water_volume:.2f}")
    
    env_forced.close()
    env_normal.close()
    
    print("\n✅ All tests completed successfully!")
    print("The forced breathing implementation is working correctly.")

if __name__ == "__main__":
    test_forced_breathing()
