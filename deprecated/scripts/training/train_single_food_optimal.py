"""
Training script for single food optimal navigation experiment.
Uses continuous visual training with SAC+GAIL and proximity/time rewards.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.base_config import get_single_food_optimal_config
from training.continuous_trainer import ContinuousTrainer


def main():
    """Run single food optimal navigation training."""
    
    print("=" * 70)
    print("SINGLE FOOD OPTIMAL NAVIGATION TRAINING")
    print("=" * 70)
    print("\n🎯 Experiment Configuration:")
    print("  - Single food item per episode")
    print("  - Proximity reward (0.5 weight) guides agent to food")
    print("  - Time penalty (-0.1/step) encourages efficiency")
    print("  - Episode ends when food collected (no respawn)")
    print("  - SAC + GAIL with expert demonstrations")
    print("\n📊 Reward Structure:")
    print("  - Food collection: +100.0")
    print("  - Proximity: 0.0 to +0.5 (closer = higher)")
    print("  - Time: -0.1 per timestep")
    print("  - Collision: -50.0")
    print("\n🎮 Training Features:")
    print("  - Continuous visual display shows best agent")
    print("  - Background training runs in parallel")
    print("  - Models saved every 50 episodes")
    print("  - Evaluation every 25 episodes")
    print("\n" + "=" * 70 + "\n")
    
    # Get configuration
    config = get_single_food_optimal_config()
    
    # Display config details
    print("Configuration Details:")
    print(f"  Max Episodes: {config.training.max_episodes}")
    print(f"  Max Steps/Episode: {config.training.max_steps_per_episode}")
    print(f"  Network: {config.agent.hidden_sizes}")
    print(f"  Learning Rate: {config.agent.learning_rate}")
    print(f"  Batch Size: {config.agent.batch_size}")
    
    if config.gail:
        print(f"\n  GAIL Enabled:")
        print(f"    Environment Weight: {config.gail.reward_env_weight}")
        print(f"    GAIL Weight: {config.gail.reward_gail_weight}")
        print(f"    Expert Demos Path: {config.gail.expert_demos_path}")
        print(f"    Load Human Demos: {config.gail.load_human_demos}")
    
    print("\n" + "=" * 70)
    print("\n🚀 Starting training...\n")
    
    # Create and run trainer
    trainer = ContinuousTrainer(config)
    
    # Disable adaptive difficulty for fixed single food experiment
    trainer.recent_food_collection_rates = []  # Clear tracking
    trainer.adaptive_window = 999999  # Effectively disable adaptive difficulty
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print(f"Best score achieved: {trainer.best_eval_score:.2f}")
        print(f"Total episodes completed: {trainer.episode}")
        trainer.visual_running = False
        trainer._save_model("interrupted_model.pth")
        trainer.logger.save_metrics()
    except Exception as e:
        print(f"\n\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        trainer.visual_running = False
    
    print("\n" + "=" * 70)
    print("Training session complete!")
    print(f"Experiment: {config.training.experiment_name}")
    print(f"Models saved to: {trainer.model_dir}")
    print(f"Logs saved to: {trainer.logger.log_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
