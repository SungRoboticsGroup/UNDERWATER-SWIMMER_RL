#!/usr/bin/env python3
"""
Continue training from the best existing model using continuous trainer.
This script loads the best model from previous training and continues with visual feedback.
"""

import os
import sys
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.base_config import get_sac_snake_config
from training.continuous_trainer import ContinuousTrainer


def find_best_model():
    """Find the best performing model from available checkpoints."""
    models_dir = "models"
    
    # Search for any available model directories
    if not os.path.exists(models_dir):
        return None, None
    
    # Look for directories with best_model.pth
    model_dirs = []
    for dir_name in os.listdir(models_dir):
        dir_path = os.path.join(models_dir, dir_name)
        if os.path.isdir(dir_path):
            best_model_path = os.path.join(dir_path, "best_model.pth")
            if os.path.exists(best_model_path):
                # Try to get score from training state
                training_state_path = best_model_path.replace('.pth', '_training_state.pth')
                score = -float('inf')
                if os.path.exists(training_state_path):
                    try:
                        import torch
                        training_state = torch.load(training_state_path, map_location='cpu')
                        score = training_state.get('best_eval_score', -float('inf'))
                    except Exception as e:
                        print(f"Warning: Could not load training state for {dir_name}: {e}")
                
                model_dirs.append((dir_name, best_model_path, score))
    
    if not model_dirs:
        return None, None
    
    # Sort by score (highest first)
    model_dirs.sort(key=lambda x: x[2], reverse=True)
    
    # Return the best one
    best_dir, best_path, best_score = model_dirs[0]
    print(f"Using model from: {best_dir}")
    if best_score > -float('inf'):
        print(f"Score: {best_score:.2f}")
    else:
        print(f"Score: unknown")
    
    return best_path, best_score


class ContinuousTrainerFromCheckpoint(ContinuousTrainer):
    """Extended continuous trainer that can load from existing checkpoints."""
    
    def __init__(self, config, checkpoint_path=None):
        super().__init__(config)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
            print(f"✅ Loaded checkpoint from: {checkpoint_path}")
        else:
            print("🆕 Starting fresh training (no checkpoint loaded)")
    
    def load_checkpoint(self, checkpoint_path):
        """Load agent and training state from checkpoint."""
        try:
            # Load agent state
            self.agent.load(checkpoint_path)
            print(f"Loaded agent from: {checkpoint_path}")
            
            # Try to load training state
            training_state_path = checkpoint_path.replace('.pth', '_training_state.pth')
            if os.path.exists(training_state_path):
                training_state = torch.load(training_state_path, map_location=self.agent.device)
                
                # Update training state
                self.episode = training_state.get('episode', 0)
                self.total_steps = training_state.get('total_steps', 0)
                self.best_eval_score = training_state.get('best_eval_score', -float('inf'))
                
                print(f"Loaded training state:")
                print(f"  - Episode: {self.episode}")
                print(f"  - Total steps: {self.total_steps}")
                print(f"  - Best eval score: {self.best_eval_score:.2f}")
            
            # Initialize visual agent with loaded model
            self._update_visual_agent()
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Continuing with fresh training...")


def main():
    """Main function to continue training from best checkpoint."""
    print("🚀 SALP Continuous Training - Resume from Best Model")
    print("=" * 60)
    
    # Find best existing model
    print("🔍 Searching for best existing model...")
    best_model_path, best_score = find_best_model()
    
    if best_model_path:
        print(f"✅ Found best model: {best_model_path}")
        if isinstance(best_score, (int, float)):
            print(f"📊 Best score: {best_score:.2f}")
        else:
            print(f"📊 Model type: {best_score}")
    else:
        print("⚠️  No existing models found, starting fresh training")
    
    # Get configuration
    config = get_sac_snake_config()
    
    # Update experiment name to indicate continuation
    if best_model_path:
        config.training.experiment_name = "salp_snake_sac_continued"
    else:
        config.training.experiment_name = "salp_snake_sac_fresh"
    
    # Set training parameters for continuation
    config.training.max_episodes = 1000  # Continue for more episodes
    config.training.eval_frequency = 10   # Evaluate more frequently
    config.training.save_frequency = 25   # Save more frequently
    
    print(f"🎯 Experiment: {config.training.experiment_name}")
    print(f"📈 Max episodes: {config.training.max_episodes}")
    print(f"🔄 Eval frequency: {config.training.eval_frequency}")
    print(f"💾 Save frequency: {config.training.save_frequency}")
    
    # Create continuous trainer with checkpoint
    print("\n🎬 Initializing continuous trainer...")
    trainer = ContinuousTrainerFromCheckpoint(config, best_model_path)
    
    print("\n🎮 Starting continuous training with visual feedback!")
    print("Press Ctrl+C to stop training gracefully.")
    print("=" * 60)
    
    try:
        # Start training
        trainer.train()
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        print("Models and logs have been saved.")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Training session completed!")


if __name__ == "__main__":
    main()
