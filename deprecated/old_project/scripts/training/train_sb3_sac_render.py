"""
Training script for Stable Baselines3 SAC with real-time visualization.

Watch the agent learn in real-time!

Usage:
    python scripts/train_sb3_sac_render.py [--timesteps 100000] [--eval-freq 5000]
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.sb3_sac_agent import SB3SACAgent
from environments.salp_snake_env import SalpSnakeEnv
from config.base_config import get_sac_snake_config
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import numpy as np


class RenderCallback(BaseCallback):
    """
    Callback for rendering episodes during training.
    Shows the agent's behavior periodically.
    """
    
    def __init__(self, render_freq: int = 1000, n_render_episodes: int = 1, verbose: int = 0):
        """
        Args:
            render_freq: Render every N timesteps
            n_render_episodes: Number of episodes to render each time
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.render_freq = render_freq
        self.n_render_episodes = n_render_episodes
        self.render_env = None
        
    def _on_training_start(self) -> None:
        """Create render environment."""
        # Get environment config from training environment
        from config.base_config import get_sac_snake_config
        config = get_sac_snake_config()
        
        # Create rendering environment
        self.render_env = SalpSnakeEnv(render_mode="human", **config.environment.params)
        print(f"\n🎬 Rendering enabled - will show agent every {self.render_freq} steps")
    
    def _on_step(self) -> bool:
        """Render episode if it's time."""
        if self.n_calls % self.render_freq == 0 and self.n_calls > 0:
            self._render_episodes()
        return True
    
    def _render_episodes(self):
        """Render a few episodes to visualize learning."""
        print(f"\n{'=' * 70}")
        print(f"🎬 VISUALIZATION (Step {self.n_calls:,})")
        print(f"{'=' * 70}")
        
        for episode in range(self.n_render_episodes):
            obs, _ = self.render_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                # Use trained model to predict
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = self.render_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
            
            print(f"  Episode {episode + 1}: Reward={episode_reward:.1f}, Length={episode_length}")
        
        print(f"{'=' * 70}\n")
    
    def _on_training_end(self) -> None:
        """Close render environment."""
        if self.render_env is not None:
            self.render_env.close()
            print("🎬 Rendering environment closed")


class ProgressCallback(BaseCallback):
    """Enhanced progress tracking with statistics."""
    
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        """Track episode statistics."""
        # Accumulate rewards
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Check if episode is done
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Log progress
            if len(self.episode_rewards) % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                recent_lengths = self.episode_lengths[-10:]
                print(f"📊 Episodes: {len(self.episode_rewards)} | "
                      f"Avg Reward (last 10): {np.mean(recent_rewards):.1f} | "
                      f"Avg Length: {np.mean(recent_lengths):.0f}")
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Train SB3 SAC with real-time visualization"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total timesteps to train (default: 100000)"
    )
    parser.add_argument(
        "--render-freq",
        type=int,
        default=5000,
        help="Render agent every N steps (default: 5000)"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10000,
        help="Save checkpoint every N steps (default: 10000)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SALP Snake Training with Stable Baselines3 SAC + Rendering")
    print("=" * 70)
    print("\n⚠️  This will show the agent's behavior during training!")
    print("    Close the visualization window to continue training.\n")
    
    # Get configuration
    config = get_sac_snake_config()
    
    # Create training environment (no rendering)
    env = SalpSnakeEnv(render_mode=None, **config.environment.params)
    print(f"✓ Training environment created")
    
    # Create agent
    agent = SB3SACAgent(config.agent, env, verbose=1)
    
    # Setup callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/sb3_sac_render_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=save_dir,
        name_prefix="sac_checkpoint",
        save_replay_buffer=True,
    )
    
    # Render callback - shows agent behavior
    render_callback = RenderCallback(
        render_freq=args.render_freq,
        n_render_episodes=1,
        verbose=1
    )
    
    # Progress callback
    progress_callback = ProgressCallback(
        log_freq=1000,
        verbose=1
    )
    
    print(f"\n✓ Training setup complete")
    print(f"  • Save directory: {save_dir}")
    print(f"  • Checkpoint frequency: every {args.checkpoint_freq} steps")
    print(f"  • Render frequency: every {args.render_freq} steps")
    
    # Train
    print(f"\n{'=' * 70}")
    print(f"Starting training for {args.timesteps:,} timesteps...")
    print(f"{'=' * 70}\n")
    
    try:
        agent.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, render_callback, progress_callback]
        )
        
        print(f"\n{'=' * 70}")
        print("Training completed successfully! 🎉")
        print(f"{'=' * 70}")
        
        # Save final model
        final_model_path = os.path.join(save_dir, "final_model")
        agent.save(final_model_path)
        print(f"\n✓ Final model saved to: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        interrupt_path = os.path.join(save_dir, "interrupted_model")
        agent.save(interrupt_path)
        print(f"✓ Model saved to: {interrupt_path}")
    
    finally:
        env.close()
    
    print(f"\n{'=' * 70}")
    print("To test the trained model:")
    print(f"  python scripts/test_sb3_sac.py {save_dir}/final_model.zip --render")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
