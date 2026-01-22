"""
Continue training from a previously saved SB3 SAC model WITH VISUALIZATION.

Usage:
    python scripts/continue_sb3_training_visual.py --timesteps 50000
"""

import os
import sys
import argparse
import threading
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from environments.salp_snake_env import SalpSnakeEnv
from config.base_config import get_single_food_optimal_config
import numpy as np


class VisualCallback(BaseCallback):
    """Callback that saves best model and signals visual updates."""
    
    def __init__(self, eval_env, eval_freq=5000, save_path=".", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.eval_count = 0
        self.n_eval_episodes = 10
        self.new_best_model = False
        
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            self.eval_count += 1
            
            # Evaluate
            episode_rewards = []
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                ep_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    done = terminated or truncated
                    ep_reward += reward
                episode_rewards.append(ep_reward)
            
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            
            elapsed = time.time() - self.training_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            
            print(f"\n📊 Evaluation #{self.eval_count} (Step {self.n_calls:,}, Time: {minutes}m {seconds}s)")
            print(f"   Mean Reward: {mean_reward:.1f} ± {std_reward:.1f}")
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.new_best_model = True
                print(f"   🏆 NEW BEST MODEL! (Previous best: {self.best_mean_reward:.1f})")
                print(f"   ✓ Model saved and visual display updated!")
                # Save best model
                self.model.save(os.path.join(self.save_path, "best_model"))
            else:
                print(f"   (Current best: {self.best_mean_reward:.1f})")
        
        return True
    
    def _on_training_start(self):
        self.training_start_time = time.time()


def visual_display_loop(model_path, config, stop_event):
    """Run continuous visual display in separate thread."""
    env = SalpSnakeEnv(render_mode="human", **config.environment.params)
    
    episode_num = 0
    
    print("\n🎬 Starting continuous visual display...")
    print("   The agent will play continuously while training in background")
    print("   Visual will update automatically when better models are found\n")
    
    try:
        while not stop_event.is_set():
            episode_num += 1
            
            # Try to load latest model
            try:
                model = SAC.load(model_path)
                print(f"🎬 Episode {episode_num} - Playing with current model")
            except:
                print(f"🎬 Episode {episode_num} - Waiting for model...")
                time.sleep(2)
                continue
            
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            food_collected = 0
            
            while not done and not stop_event.is_set():
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
                
                # Track food collected
                if 'food_collected' in info and info['food_collected']:
                    food_collected += 1
                
                env.render()
                time.sleep(0.01)  # Small delay for visibility
            
            print(f"   Reward: {episode_reward:.1f}, Food: {food_collected}, Steps: {steps}\n")
            
    finally:
        env.close()
        print("\n🎬 Visual display closed\n")


def main():
    parser = argparse.ArgumentParser(
        description="Continue training with visualization"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/Users/avielstein/Desktop/GRASP_LAB_SALP/models/sb3_sac_continuous_20251116_170828/best_model.zip",
        help="Path to saved model"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Additional timesteps to train"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="Evaluate every N steps"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎯 CONTINUE TRAINING - Continuous Food Collection")
    print("=" * 70)
    
    # Get configuration
    config = get_single_food_optimal_config()
    
    print(f"\n🎯 Using continuous food collection mode")
    print(f"   • Food respawns after collection")
    print(f"   • Max steps without food: {config.environment.params['max_steps_without_food']}")
    
    # Create training environment (no render)
    train_env = SalpSnakeEnv(render_mode=None, **config.environment.params)
    
    # Create evaluation environment (no render)
    eval_env = SalpSnakeEnv(render_mode=None, **config.environment.params)
    
    print(f"✓ Environments created")
    
    # Load model
    print(f"Loading model from: {args.model}")
    model = SAC.load(args.model, env=train_env)
    print(f"✓ Model loaded")
    print(f"  • Previous training: {model.num_timesteps:,} timesteps")
    
    # Setup save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/sb3_sac_continued_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save initial model for visual display
    model.save(os.path.join(save_dir, "best_model"))
    
    print(f"✓ Save directory: {save_dir}")
    
    # Create callback
    callback = VisualCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        save_path=save_dir
    )
    
    print("=" * 70)
    print(f"🚀 Continuing training for {args.timesteps:,} timesteps")
    print(f"   Evaluation frequency: every {args.eval_freq} steps")
    print("=" * 70)
    print()
    
    # Start visual display in separate thread
    stop_event = threading.Event()
    visual_thread = threading.Thread(
        target=visual_display_loop,
        args=(os.path.join(save_dir, "best_model"), config, stop_event)
    )
    visual_thread.daemon = True
    visual_thread.start()
    
    try:
        # Train (this blocks until complete)
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            reset_num_timesteps=False,
            log_interval=100
        )
        
        print("\n" + "=" * 70)
        print("✅ Training completed successfully!")
        print(f"   Best mean reward: {callback.best_mean_reward:.1f}")
        print(f"   Total time: ", end="")
        
        # Save final model
        model.save(os.path.join(save_dir, "continued_model"))
        print(f"\n✓ Final model saved to: {save_dir}/continued_model")
        print(f"✓ Total training: {model.num_timesteps:,} timesteps")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        model.save(os.path.join(save_dir, "interrupted_model"))
        print(f"✓ Model saved to: {save_dir}/interrupted_model")
    
    finally:
        stop_event.set()
        visual_thread.join(timeout=2)
        train_env.close()
        eval_env.close()
    
    print("\n" + "=" * 70)
    print("Training session complete!")
    print(f"Models saved in: {save_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
