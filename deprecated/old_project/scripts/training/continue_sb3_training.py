"""
Continue training from a previously saved SB3 SAC model.

Usage:
    python scripts/continue_sb3_training.py <model_path> --timesteps 100000
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from environments.salp_snake_env import SalpSnakeEnv
from config.base_config import get_single_food_optimal_config
from stable_baselines3.common.callbacks import CheckpointCallback


def main():
    parser = argparse.ArgumentParser(
        description="Continue training from saved SB3 SAC model"
    )
    parser.add_argument(
        "model_path",
        type=str,
        nargs='?',
        default="/Users/avielstein/Desktop/GRASP_LAB_SALP/models/sb3_sac_continuous_20251116_170828/best_model.zip",
        help="Path to saved model (.zip file)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Additional timesteps to train (default: 100000)"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="Evaluate every N steps (default: 5000)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Continue Training SB3 SAC - Continuous Food Collection")
    print("=" * 70)
    print(f"\nLoading model from: {args.model_path}")
    
    # Get configuration
    config = get_single_food_optimal_config()
    
    # Create environment with NEW settings (continuous food!)
    env = SalpSnakeEnv(render_mode=None, **config.environment.params)
    print(f"✓ Environment created with continuous food collection")
    print(f"  • respawn_food: {config.environment.params['respawn_food']}")
    print(f"  • max_steps_without_food: {config.environment.params['max_steps_without_food']}")
    
    # Load the model
    model = SAC.load(args.model_path, env=env)
    print(f"✓ Model loaded successfully")
    print(f"  • Previous training: {model.num_timesteps:,} timesteps")
    
    # Setup save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/sb3_sac_continued_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_dir,
        name_prefix="sac_checkpoint",
        save_replay_buffer=True,
    )
    
    print(f"✓ Save directory: {save_dir}")
    print(f"\n{'=' * 70}")
    print(f"Continuing training for {args.timesteps:,} more timesteps...")
    print(f"{'=' * 70}\n")
    
    try:
        # Continue training
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=False,  # Don't reset - continue from where we left off
            log_interval=10
        )
        
        print(f"\n{'=' * 70}")
        print("✅ Training completed successfully!")
        print(f"{'=' * 70}")
        
        # Save final model
        final_path = os.path.join(save_dir, "continued_model")
        model.save(final_path)
        print(f"\n✓ Final model saved to: {final_path}")
        print(f"✓ Total training: {model.num_timesteps:,} timesteps")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        interrupt_path = os.path.join(save_dir, "interrupted_model")
        model.save(interrupt_path)
        print(f"✓ Model saved to: {interrupt_path}")
    
    finally:
        env.close()
    
    print(f"\n{'=' * 70}")
    print("To test the model:")
    print(f"  python scripts/test_sb3_sac.py {save_dir}/continued_model.zip --render")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
