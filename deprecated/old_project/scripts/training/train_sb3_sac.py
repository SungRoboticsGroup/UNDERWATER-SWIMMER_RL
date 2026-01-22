"""
Simple training script for Stable Baselines3 SAC on SALP Snake environment.

This script demonstrates the standard SB3 SAC implementation, making it easy to:
1. Compare with custom implementations
2. Benchmark performance
3. Use well-tested, documented code
"""

import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.sb3_sac_agent import SB3SACAgent
from environments.salp_snake_env import SalpSnakeEnv
from config.base_config import get_sac_snake_config
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


def main():
    """Train SB3 SAC agent on SALP Snake environment."""
    
    print("=" * 70)
    print("SALP Snake Training with Stable Baselines3 SAC")
    print("=" * 70)
    
    # Get configuration
    config = get_sac_snake_config()
    
    # Create environment
    env = SalpSnakeEnv(render_mode=None, **config.environment.params)
    print(f"\n✓ Environment created: {env}")
    print(f"  • Observation space: {env.observation_space}")
    print(f"  • Action space: {env.action_space}")
    
    # Create eval environment
    eval_env = SalpSnakeEnv(render_mode=None, **config.environment.params)
    
    # Create agent
    agent = SB3SACAgent(config.agent, env, verbose=1)
    
    # Setup callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/sb3_sac_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10k steps
        save_path=save_dir,
        name_prefix="sac_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Evaluation callback - evaluate and save best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=5000,  # Evaluate every 5k steps
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    
    print(f"\n✓ Training setup complete")
    print(f"  • Save directory: {save_dir}")
    print(f"  • Checkpoint frequency: every 10k steps")
    print(f"  • Evaluation frequency: every 5k steps")
    
    # Train
    total_timesteps = 100000  # 100k timesteps for initial test
    print(f"\n{'=' * 70}")
    print(f"Starting training for {total_timesteps:,} timesteps...")
    print(f"{'=' * 70}\n")
    
    try:
        agent.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback]
        )
        
        print(f"\n{'=' * 70}")
        print("Training completed successfully! 🎉")
        print(f"{'=' * 70}")
        
        # Save final model
        final_model_path = os.path.join(save_dir, "final_model")
        agent.save(final_model_path)
        print(f"\n✓ Final model saved to: {final_model_path}")
        
        # Print summary
        print(f"\nTraining Summary:")
        print(f"  • Total timesteps: {agent.training_step:,}")
        print(f"  • Models saved in: {save_dir}")
        print(f"  • Best model: {os.path.join(save_dir, 'best_model.zip')}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        # Save current model
        interrupt_path = os.path.join(save_dir, "interrupted_model")
        agent.save(interrupt_path)
        print(f"✓ Model saved to: {interrupt_path}")
    
    finally:
        env.close()
        eval_env.close()
    
    print(f"\n{'=' * 70}")
    print("To test the trained model, run:")
    print(f"  python scripts/test_sb3_sac.py {save_dir}/best_model.zip")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
