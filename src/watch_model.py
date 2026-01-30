"""
Continuous Model Watcher - Watch the robot learn in real-time!

This script runs alongside training to continuously visualize
the latest best model's performance. It automatically reloads
the model when training saves a new best checkpoint.

Controls:
    SPACE - Pause/Resume
    Q     - Quit
    R     - Reload model now

Usage:
    # Watch training progress (auto-reloads best model):
    python watch_model.py
    
    # Watch a specific model file:
    python watch_model.py --model ../salp_robot_finalv2.zip
    python watch_model.py --model ./logs/best_model/best_model.zip
"""

from stable_baselines3 import SAC
from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle
import numpy as np
import time
import os
import sys
import pygame

def create_robot():
    """Create a robot instance."""
    nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.00016, mass=1.0)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, max_contraction=0.06, nozzle=nozzle)
    robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
    robot.set_environment(density=1000)
    return robot

def load_latest_model(model_path, env):
    """Load the latest model from disk."""
    try:
        if os.path.exists(model_path):
            model = SAC.load(model_path, env=env, device="cpu")
            return model, os.path.getmtime(model_path)
        else:
            print(f"⚠️  Model not found: {model_path}")
            return None, None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def run_episode(env, model, max_cycles=20):
    """Run a single episode and return stats."""
    obs, _ = env.reset()
    episode_reward = 0
    steps = 0
    done = False
    
    distances = []
    
    while not done and steps < max_cycles:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        steps += 1
        
        # Track distance to target
        distance = np.linalg.norm(env.robot.position[0:2] - env.target_point)
        distances.append(distance)
        
        # Wait for animation
        env.wait_for_animation()
        
    return {
        'reward': episode_reward,
        'steps': steps,
        'min_distance': min(distances) if distances else float('inf'),
        'final_distance': distances[-1] if distances else float('inf'),
        'success': done and terminated  # Reached target
    }

def main():
    # Parse command-line arguments
    MODEL_PATH = "./logs/best_model/best_model.zip"  # Default
    AUTO_RELOAD = True  # Auto-reload by default
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--model" and len(sys.argv) > 2:
            MODEL_PATH = sys.argv[2]
            AUTO_RELOAD = False  # Don't auto-reload for specific models
            print(f"📁 Using custom model: {MODEL_PATH}")
        elif sys.argv[1] in ["-h", "--help"]:
            print("Usage:")
            print("  python watch_model.py                              # Watch training (auto-reload)")
            print("  python watch_model.py --model <path>               # Watch specific model")
            print("\nExamples:")
            print("  python watch_model.py --model ../salp_robot_finalv2.zip")
            print("  python watch_model.py --model ./logs/best_model/best_model.zip")
            return
    
    CHECK_INTERVAL = 5  # Seconds between checking for model updates
    
    print("="*70)
    print("🎬 MODEL WATCHER - Real-time Visualization")
    print("="*70)
    print(f"Model: {MODEL_PATH}")
    if AUTO_RELOAD:
        print(f"Mode: Auto-reload (checks every {CHECK_INTERVAL}s)")
    else:
        print("Mode: Fixed model (no auto-reload)")
    print("\nControls:")
    print("  SPACE - Pause/Resume")
    print("  Q     - Quit")
    print("  R     - Reload model now")
    print("="*70 + "\n")
    
    # Initialize pygame first
    pygame.init()
    
    # Create environment
    robot = create_robot()
    env = SalpRobotEnv(render_mode="human", robot=robot)
    
    # Load initial model
    model, last_modified = load_latest_model(MODEL_PATH, env)
    
    if model is None:
        print("⏳ Waiting for training to create first model...")
        while model is None:
            time.sleep(5)
            model, last_modified = load_latest_model(MODEL_PATH, env)
    
    print(f"✅ Model loaded! Starting visualization...\n")
    
    # State
    paused = False
    episode_count = 0
    last_check_time = time.time()
    total_rewards = []
    
    try:
        while True:
            # Check for keyboard input
            keys = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\n👋 Quitting...")
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                        if paused:
                            print("\n⏸️  PAUSED - Press SPACE to resume")
                        else:
                            print("▶️  RESUMED\n")
                    elif event.key == pygame.K_q:
                        print("\n👋 Quitting...")
                        env.close()
                        return
                    elif event.key == pygame.K_r:
                        print("\n🔄 Reloading model...")
                        new_model, new_modified = load_latest_model(MODEL_PATH, env)
                        if new_model is not None:
                            model = new_model
                            last_modified = new_modified
                            print("✅ Model reloaded!\n")
            
            # Skip episode if paused
            if paused:
                time.sleep(0.1)
                continue
            
            # Check for model updates periodically (only if AUTO_RELOAD is enabled)
            if AUTO_RELOAD:
                current_time = time.time()
                if current_time - last_check_time > CHECK_INTERVAL:
                    last_check_time = current_time
                    current_modified = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
                    
                    if current_modified and current_modified > last_modified:
                        print("\n🆕 New model detected! Reloading...")
                        new_model, new_modified = load_latest_model(MODEL_PATH, env)
                        if new_model is not None:
                            model = new_model
                            last_modified = new_modified
                            print("✅ Updated to latest model!\n")
            
            # Run episode
            episode_count += 1
            print(f"{'='*70}")
            print(f"Episode {episode_count}")
            print(f"{'='*70}")
            
            stats = run_episode(env, model, max_cycles=20)
            total_rewards.append(stats['reward'])
            
            # Print stats
            status = "✅ SUCCESS" if stats['success'] else "❌ INCOMPLETE"
            print(f"{status}")
            print(f"  Reward:        {stats['reward']:8.2f}")
            print(f"  Steps:         {stats['steps']:8d}")
            print(f"  Min Distance:  {stats['min_distance']:8.3f}m")
            print(f"  Final Distance:{stats['final_distance']:8.3f}m")
            
            # Rolling average
            if len(total_rewards) >= 10:
                recent_avg = np.mean(total_rewards[-10:])
                print(f"  Avg Last 10:   {recent_avg:8.2f}")
            
            print(f"{'='*70}\n")
            
            # Small delay before next episode
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
    finally:
        env.close()
        print("✅ Watcher stopped")

if __name__ == "__main__":
    main()
