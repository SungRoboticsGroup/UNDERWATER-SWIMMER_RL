"""
Continuous visual training with Stable Baselines3 SAC.

Shows the agent playing continuously while training in the background.
The visual display automatically updates when better models are found.

Now uses the reusable continuous_visual_trainer utility!
"""

import os
import sys
import time
import argparse
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utilities'))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from environments.salp_snake_env import SalpSnakeEnv
from config.base_config import get_sac_snake_config, get_single_food_optimal_config
from continuous_visual_trainer import ContinuousVisualTrainer


def train_sb3_sac_with_visual(
    visual_trainer: ContinuousVisualTrainer,
    config,
    total_timesteps: int = 100000,
    eval_freq: int = 5000,
    load_model: str = None
):
    """
    Training function that runs in background thread.
    
    Args:
        visual_trainer: The ContinuousVisualTrainer instance
        config: Training configuration
        total_timesteps: Total timesteps to train
        eval_freq: Evaluate every N steps
        load_model: Optional path to model to continue training from
    """
    # Create environments
    print("Creating training environments...")
    train_env = SalpSnakeEnv(render_mode=None, **config.environment.params)
    eval_env = SalpSnakeEnv(render_mode=None, **config.environment.params)
    print("✓ Environments created")
    
    # Create or load agent
    if load_model:
        print(f"Loading SB3 SAC agent from: {load_model}")
        agent = SAC.load(load_model, env=train_env)
        print(f"✓ Agent loaded (previously trained: {agent.num_timesteps:,} timesteps)")
    else:
        print("Creating SB3 SAC agent...")
        agent = SAC(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=config.agent.learning_rate,
            buffer_size=config.agent.buffer_size,
            batch_size=config.agent.batch_size,
            tau=config.agent.tau,
            gamma=config.agent.gamma,
            policy_kwargs=dict(net_arch=config.agent.hidden_sizes),
            verbose=0,
        )
        print("✓ Agent created")
    
    # Initialize visual with current agent
    visual_trainer.update_model(agent)
    
    # Training state
    best_mean_reward = -float('inf')
    
    # Save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/sb3_sac_continuous_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"✓ Save directory: {save_dir}")
    
    def evaluate_model(model, n_episodes=5):
        """Evaluate model performance."""
        episode_rewards = []
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
            episode_rewards.append(episode_reward)
        return np.mean(episode_rewards), np.std(episode_rewards)
    
    # Custom callback for evaluation
    class EvalAndUpdateCallback(BaseCallback):
        def __init__(self, eval_freq):
            super().__init__()
            self.eval_freq = eval_freq
            self.n_evals = 0
            self.start_time = time.time()
            
        def _on_step(self):
            nonlocal best_mean_reward
            
            # Evaluate periodically
            if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
                mean_reward, std_reward = evaluate_model(self.model)
                self.n_evals += 1
                
                elapsed = time.time() - self.start_time
                time_str = f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m" if elapsed >= 3600 else f"{int(elapsed//60)}m {int(elapsed%60)}s"
                
                print(f"\n📊 Evaluation #{self.n_evals} (Step {self.n_calls:,}, Time: {time_str})")
                print(f"   Mean Reward: {mean_reward:.1f} ± {std_reward:.1f}")
                
                # Update training info
                visual_trainer.update_training_info({
                    'step': f'{self.n_calls:,}',
                    'best': f'{best_mean_reward:.1f}'
                })
                
                # Check if this is a new best
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    
                    # Save best model
                    best_path = os.path.join(save_dir, "best_model")
                    self.model.save(best_path)
                    
                    # Update visual model
                    new_visual_model = SAC.load(best_path, env=train_env)
                    visual_trainer.update_model(new_visual_model)
                    
                    print(f"   🏆 NEW BEST MODEL!")
                    print(f"   ✓ Model saved and visual display updated!")
                else:
                    print(f"   (Current best: {best_mean_reward:.1f})")
            
            return True
    
    print(f"\n{'=' * 70}")
    print(f"🚀 Starting training for {total_timesteps:,} timesteps")
    print(f"   Evaluation frequency: every {eval_freq} steps")
    print(f"{'=' * 70}\n")
    
    try:
        callback = EvalAndUpdateCallback(eval_freq)
        agent.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=None
        )
        
        print(f"\n{'=' * 70}")
        print("✅ Training completed successfully!")
        print(f"   Best mean reward: {best_mean_reward:.1f}")
        print(f"{'=' * 70}\n")
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save final model
        final_path = os.path.join(save_dir, "final_model")
        agent.save(final_path)
        print(f"✓ Final model saved to: {final_path}")
        
        # Clean up environments
        train_env.close()
        eval_env.close()
        
        # Stop visual trainer
        visual_trainer.stop()
        
        print(f"\nModels saved in: {save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train SB3 SAC with continuous visualization"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total timesteps to train (default: 100000)"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="Evaluate every N steps (default: 5000)"
    )
    parser.add_argument(
        "--single-food",
        action="store_true",
        help="Use single food configuration (default: multi-food)"
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to model to continue training from (.zip file)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("SB3 SAC Training with Continuous Visualization")
    if args.load_model:
        print("🔄 CONTINUE TRAINING - Loading existing model")
    elif args.single_food:
        print("🎯 SINGLE FOOD LEARNING - Agent learns to reach one target")
    else:
        print("🍎 MULTI FOOD LEARNING - Agent learns to collect multiple targets")
    print("=" * 70 + "\n")
    
    # Get configuration
    if args.single_food:
        config = get_single_food_optimal_config()
        print("Using SINGLE FOOD configuration")
    else:
        config = get_sac_snake_config()
        print("Using MULTI FOOD configuration")
    
    # Create environment creator for visual trainer
    def create_visual_env():
        return SalpSnakeEnv(render_mode="human", **config.environment.params)
    
    # Create continuous visual trainer
    visual_trainer = ContinuousVisualTrainer(
        env_creator=create_visual_env,
        fps=30.0,  # 30 FPS for smooth viewing
        episode_delay=1.0,
        verbose=True
    )
    
    # Start training in background thread
    visual_trainer.start_training_thread(
        train_sb3_sac_with_visual,
        config=config,
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        load_model=args.load_model
    )
    
    # Give training a moment to start
    time.sleep(2)
    
    # Run visual loop on main thread (required for macOS pygame)
    try:
        visual_trainer.run_visual_loop()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        visual_trainer.stop()
    
    print(f"\n{'=' * 70}")
    print("Training session complete!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
