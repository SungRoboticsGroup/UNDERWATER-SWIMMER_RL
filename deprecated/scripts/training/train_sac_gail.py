"""
Training script for SAC+GAIL agent.
Combines Soft Actor-Critic with Generative Adversarial Imitation Learning.
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.base_config import get_sac_gail_snake_config
from training.trainer import Trainer


def main():
    """Run SAC+GAIL training."""
    parser = argparse.ArgumentParser(description='Train SAC+GAIL agent')
    parser.add_argument('--env-weight', type=float, default=None,
                       help='Environment reward weight (overrides config)')
    parser.add_argument('--gail-weight', type=float, default=None,
                       help='GAIL reward weight (overrides config)')
    parser.add_argument('--max-episodes', type=int, default=None,
                       help='Maximum episodes (overrides config)')
    parser.add_argument('--expert-path', type=str, default=None,
                       help='Path to expert demonstrations (overrides config)')
    parser.add_argument('--no-human-demos', action='store_true',
                       help='Do not load human demonstrations')
    parser.add_argument('--no-agent-demos', action='store_true',
                       help='Do not load agent demonstrations')
    
    args = parser.parse_args()
    
    # Get default configuration
    config = get_sac_gail_snake_config()
    
    # Apply command line overrides
    if args.env_weight is not None:
        config.gail.reward_env_weight = args.env_weight
    if args.gail_weight is not None:
        config.gail.reward_gail_weight = args.gail_weight
    if args.max_episodes is not None:
        config.training.max_episodes = args.max_episodes
    if args.expert_path is not None:
        config.gail.expert_demos_path = args.expert_path
    if args.no_human_demos:
        config.gail.load_human_demos = False
    if args.no_agent_demos:
        config.gail.load_agent_demos = False
    
    # Print configuration
    print("\n" + "="*70)
    print("SAC+GAIL TRAINING CONFIGURATION")
    print("="*70)
    print(f"Environment: {config.environment.name}")
    print(f"Agent: {config.agent.name}")
    print(f"Max Episodes: {config.training.max_episodes}")
    print(f"\nGAIL Settings:")
    print(f"  Use GAIL: {config.gail.use_gail}")
    print(f"  Expert demos path: {config.gail.expert_demos_path}")
    print(f"  Load human demos: {config.gail.load_human_demos}")
    print(f"  Load agent demos: {config.gail.load_agent_demos}")
    print(f"  Reward weights: env={config.gail.reward_env_weight:.2f}, gail={config.gail.reward_gail_weight:.2f}")
    print(f"  Discriminator update freq: {config.gail.discriminator_update_freq}")
    print("="*70 + "\n")
    
    # Create trainer
    trainer = Trainer(config)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        print("Saving current model...")
        trainer._save_model("interrupted_model.pth")
        print("Model saved.")


if __name__ == "__main__":
    main()
