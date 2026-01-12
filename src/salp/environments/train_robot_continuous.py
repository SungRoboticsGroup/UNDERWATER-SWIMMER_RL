"""
Continuous visual training for SALP Robot with SAC.

Shows the robot navigating in real-time while training in the background.
The visual display automatically updates when better models are found.
"""

import os
import sys
import time
import argparse
from datetime import datetime
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle
from continuous_visual_trainer import ContinuousVisualTrainer


def create_robot_env(render_mode=None):
    """Create a SalpRobotEnv with robot."""
    nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.00016, mass=1.0)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                  max_contraction=0.06, nozzle=nozzle)
    robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
    robot.set_environment(density=1000)  # water density in kg/m^3
    
    return SalpRobotEnv(render_mode=render_mode, robot=robot)


def train_robot_with_visual(
    visual_trainer: ContinuousVisualTrainer,
    total_timesteps: int = 200000,
    eval_freq: int = 5000,
    num_train_envs: int = 8,
    load_model: str = None
):
    """
    Training function that runs in background thread.
    
    Args:
        visual_trainer: The ContinuousVisualTrainer instance
        total_timesteps: Total timesteps to train
        eval_freq: Evaluate every N steps
        num_train_envs: Number of parallel training environments
        load_model: Optional path to model to continue training from
    """
    # Create parallel training environments using DummyVecEnv
    print(f"Creating {num_train_envs} parallel training environments (DummyVecEnv)...")
    
    def make_env():
        def _init():
            return create_robot_env(render_mode=None)
        return _init
    
    train_env = DummyVecEnv([make_env() for _ in range(num_train_envs)])
    
    # Create single eval environment  
    eval_env = create_robot_env(render_mode=None)
    print(f"✓ {num_train_envs} training environments created with DummyVecEnv")
    
    # Create or load agent
    if load_model:
        print(f"Loading SAC agent from: {load_model}")
        agent = SAC.load(load_model, env=train_env)
        print(f"✓ Agent loaded (previously trained: {agent.num_timesteps:,} timesteps)")
    else:
        print("Creating SAC agent...")
        agent = SAC(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=64,
            learning_starts=100,  # Start training after filling buffer minimally
            tau=0.005,
            gamma=0.99,
            ent_coef='auto',
            device="cpu",
            verbose=1,  # Enable verbose logging to see what's happening
            tensorboard_log="./sac_robot_tensorboard/",
        )
        print("✓ Agent created with TensorBoard logging (starts training at 100 steps)")
    
    # Initialize visual with current agent
    visual_trainer.update_model(agent)
    
    # Training state
    best_mean_reward = -float('inf')
    
    # Save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models_robot/sac_robot_continuous_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"✓ Save directory: {save_dir}")
    
    def evaluate_model(model, n_episodes=3):
        """Evaluate model performance."""
        episode_rewards = []
        distances_to_target = []
        success_count = 0
        
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done and steps < 500:  # Max 500 cycles per episode
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
            
            episode_rewards.append(episode_reward)
            
            # Check if reached target
            if terminated:
                success_count += 1
            
            # Get final distance to target
            robot_pos = eval_env.robot.position[0:2]
            target_pos = eval_env.target_point
            final_dist = np.linalg.norm(robot_pos - target_pos)
            distances_to_target.append(final_dist)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'success_rate': success_count / n_episodes,
            'mean_distance': np.mean(distances_to_target)
        }
    
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
                eval_results = evaluate_model(self.model)
                self.n_evals += 1
                
                elapsed = time.time() - self.start_time
                time_str = f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m" if elapsed >= 3600 else f"{int(elapsed//60)}m {int(elapsed%60)}s"
                
                print(f"\n📊 Evaluation #{self.n_evals} (Step {self.n_calls:,}, Time: {time_str})")
                print(f"   Mean Reward: {eval_results['mean_reward']:.1f} ± {eval_results['std_reward']:.1f}")
                print(f"   Success Rate: {eval_results['success_rate']*100:.1f}%")
                print(f"   Mean Distance: {eval_results['mean_distance']:.4f}m")
                
                # Update training info
                visual_trainer.update_training_info({
                    'step': f"{self.n_calls:,}",
                    'best': f"{best_mean_reward:.1f}",
                    'success': f"{eval_results['success_rate']*100:.0f}%"
                })
                
                # Check if this is a new best
                if eval_results['mean_reward'] > best_mean_reward:
                    best_mean_reward = eval_results['mean_reward']
                    
                    # Save best model
                    best_path = os.path.join(save_dir, "best_model")
                    self.model.save(best_path)
                    
                    # Update visual model (create new instance for thread safety)
                    new_visual_model = SAC.load(best_path)
                    visual_trainer.update_model(new_visual_model)
                    
                    print(f"   🏆 NEW BEST MODEL!")
                    print(f"   ✓ Model saved and visual display updated!")
                else:
                    print(f"   (Current best: {best_mean_reward:.1f})")
            
            return True
    
    print(f"\n{'=' * 70}")
    print(f"🚀 Starting robot training for {total_timesteps:,} timesteps")
    print(f"   Parallel environments: {num_train_envs}")
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
        description="Train SALP Robot with continuous visualization"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200000,
        help="Total timesteps to train (default: 200000)"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="Evaluate every N steps (default: 5000)"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel training environments (default: 4)"
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to model to continue training from (.zip file)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("🤖 SALP Robot Training with Continuous Visualization")
    if args.load_model:
        print("🔄 CONTINUE TRAINING - Loading existing model")
    else:
        print("🆕 NEW TRAINING - Starting from scratch")
    print("=" * 70 + "\n")
    
    # Create environment creator for visual trainer
    def create_visual_env():
        return create_robot_env(render_mode="human")
    
    # Create custom action selector that handles robot animations
    def robot_action_selector(model, obs):
        """Custom action selector for robot environment."""
        # Reshape observation to add batch dimension if needed
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        action, _ = model.predict(obs, deterministic=True)
        # Remove batch dimension from action
        if len(action.shape) > 1:
            action = action[0]
        return action
    
    # Create continuous visual trainer
    visual_trainer = ContinuousVisualTrainer(
        env_creator=create_visual_env,
        fps=20.0,  # 20 FPS (robot animations are slower)
        episode_delay=0.5,  # Short delay between episodes
        verbose=True
    )
    
    # Start training in background thread
    visual_trainer.start_training_thread(
        train_robot_with_visual,
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        num_train_envs=args.num_envs,
        load_model=args.load_model
    )
    
    # Give training a moment to start
    print("Starting training in background...")
    time.sleep(3)
    
    # Run visual loop on main thread (required for macOS pygame)
    try:
        print("\n🎬 Starting visual display...")
        print("   Watch the robot learn to navigate to targets in real-time!")
        print("   Press Ctrl+C to stop\n")
        visual_trainer.run_visual_loop(action_selector=robot_action_selector)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        visual_trainer.stop()
    
    print(f"\n{'=' * 70}")
    print("Training session complete!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
