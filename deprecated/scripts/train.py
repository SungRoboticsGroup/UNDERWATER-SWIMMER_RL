#!/usr/bin/env python3
"""
Unified SALP Training Script

A single, modular training script that supports all training configurations.
Pass in what you need via command-line arguments instead of using separate scripts.

Examples:
    # Basic SB3 SAC training
    python scripts/train.py --algorithm sac --implementation sb3 --timesteps 100000
    
    # SB3 SAC with continuous visual feedback
    python scripts/train.py --algorithm sac --implementation sb3 --visual continuous
    
    # SAC + GAIL with expert demos
    python scripts/train.py --algorithm sac_gail --implementation custom --visual continuous
    
    # Continue from checkpoint
    python scripts/train.py --algorithm sac --implementation sb3 --continue-from models/best_model.zip
    
    # Single food experiment
    python scripts/train.py --config single_food --visual continuous
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Optional, Callable
import numpy as np

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utilities'))

# Imports will be conditional based on arguments
from config.base_config import (
    get_sac_snake_config,
    get_sac_gail_snake_config,
    get_single_food_optimal_config
)
from environments.salp_snake_env import SalpSnakeEnv


class UnifiedTrainer:
    """Unified trainer that handles all training configurations."""
    
    def __init__(self, args):
        self.args = args
        self.config = self._get_config()
        self.visual_trainer = None
        
        # Update config based on args
        if args.timesteps:
            self.config.training.max_episodes = args.timesteps // 1000  # Rough conversion
        if args.eval_freq:
            self.config.training.eval_frequency = args.eval_freq
        
        print("=" * 70)
        print("SALP UNIFIED TRAINING")
        print("=" * 70)
        print(f"Algorithm: {args.algorithm}")
        print(f"Implementation: {args.implementation}")
        print(f"Visualization: {args.visual}")
        print(f"Configuration: {args.config}")
        print("=" * 70 + "\n")
    
    def _get_config(self):
        """Get configuration based on algorithm and config preset."""
        if self.args.config == 'single_food':
            return get_single_food_optimal_config()
        elif self.args.algorithm == 'sac_gail':
            return get_sac_gail_snake_config()
        else:
            return get_sac_snake_config()
    
    def _create_env(self, render_mode=None):
        """Create environment."""
        return SalpSnakeEnv(
            render_mode=render_mode,
            width=self.config.environment.width,
            height=self.config.environment.height,
            **self.config.environment.params
        )
    
    def train(self):
        """Main training entry point."""
        if self.args.implementation == 'sb3':
            self._train_sb3()
        elif self.args.implementation == 'custom':
            self._train_custom()
        else:
            raise ValueError(f"Unknown implementation: {self.args.implementation}")
    
    def _train_sb3(self):
        """Train using Stable Baselines3."""
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
        
        # Create environments
        train_env = self._create_env(render_mode=None)
        eval_env = self._create_env(render_mode=None)
        
        # Create or load agent
        if self.args.continue_from:
            print(f"Loading model from: {self.args.continue_from}")
            agent = SAC.load(self.args.continue_from, env=train_env)
            print(f"✓ Model loaded (trained: {agent.num_timesteps:,} steps)")
        else:
            print("Creating new SB3 SAC agent...")
            agent = SAC(
                policy="MlpPolicy",
                env=train_env,
                learning_rate=self.config.agent.learning_rate,
                buffer_size=self.config.agent.buffer_size,
                batch_size=self.config.agent.batch_size,
                tau=self.config.agent.tau,
                gamma=self.config.agent.gamma,
                policy_kwargs=dict(net_arch=self.config.agent.hidden_sizes),
                verbose=1 if self.args.visual == 'none' else 0,
            )
            print("✓ Agent created")
        
        # Setup save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"models/{self.args.algorithm}_{self.args.implementation}_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"✓ Save directory: {save_dir}\n")
        
        # Choose visualization mode
        if self.args.visual == 'continuous':
            self._train_sb3_continuous(agent, train_env, eval_env, save_dir)
        elif self.args.visual == 'periodic':
            self._train_sb3_periodic(agent, train_env, save_dir)
        else:  # none
            self._train_sb3_basic(agent, train_env, save_dir)
        
        # Cleanup
        train_env.close()
        eval_env.close()
        
        print(f"\n✓ Training complete! Models saved to: {save_dir}")
    
    def _train_sb3_continuous(self, agent, train_env, eval_env, save_dir):
        """Train SB3 with continuous visual feedback."""
        from continuous_visual_trainer import ContinuousVisualTrainer
        
        # Create visual trainer
        visual_trainer = ContinuousVisualTrainer(
            env_creator=lambda: self._create_env(render_mode="human"),
            fps=30.0,
            episode_delay=1.0,
            verbose=True
        )
        
        # Define training function
        def train_with_visual(vt: ContinuousVisualTrainer):
            vt.update_model(agent)
            best_reward = -float('inf')
            
            def evaluate(model, n_episodes=5):
                rewards = []
                for _ in range(n_episodes):
                    obs, _ = eval_env.reset()
                    done = False
                    ep_reward = 0
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, _ = eval_env.step(action)
                        done = terminated or truncated
                        ep_reward += reward
                    rewards.append(ep_reward)
                return np.mean(rewards)
            
            class VisualCallback(BaseCallback):
                def __init__(self):
                    super().__init__()
                    self.n_evals = 0
                    
                def _on_step(self):
                    nonlocal best_reward
                    if self.n_calls % self.args.eval_freq == 0 and self.n_calls > 0:
                        mean_reward = evaluate(self.model)
                        self.n_evals += 1
                        
                        print(f"\n📊 Eval #{self.n_evals} (Step {self.n_calls:,}): {mean_reward:.1f}")
                        
                        vt.update_training_info({
                            'step': f'{self.n_calls:,}',
                            'best': f'{best_reward:.1f}'
                        })
                        
                        if mean_reward > best_reward:
                            best_reward = mean_reward
                            best_path = os.path.join(save_dir, "best_model")
                            self.model.save(best_path)
                            new_model = SAC.load(best_path, env=train_env)
                            vt.update_model(new_model)
                            print("🏆 NEW BEST MODEL!")
                    return True
            
            agent.learn(
                total_timesteps=self.args.timesteps,
                callback=VisualCallback(),
                log_interval=None
            )
            
            # Save final
            agent.save(os.path.join(save_dir, "final_model"))
            vt.stop()
        
        # Start training thread and run visual loop
        visual_trainer.start_training_thread(train_with_visual)
        time.sleep(2)
        
        try:
            visual_trainer.run_visual_loop()
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            visual_trainer.stop()
    
    def _train_sb3_periodic(self, agent, train_env, save_dir):
        """Train SB3 with periodic visualization."""
        from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
        
        class PeriodicRenderCallback(BaseCallback):
            def __init__(self, render_freq):
                super().__init__()
                self.render_freq = render_freq
                self.render_env = None
                
            def _on_training_start(self):
                self.render_env = SalpSnakeEnv(
                    render_mode="human",
                    **self.config.environment.params
                )
                
            def _on_step(self):
                if self.n_calls % self.render_freq == 0 and self.n_calls > 0:
                    print(f"\n🎬 Rendering at step {self.n_calls:,}")
                    obs, _ = self.render_env.reset()
                    done = False
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, _, terminated, truncated, _ = self.render_env.step(action)
                        done = terminated or truncated
                return True
                
            def _on_training_end(self):
                if self.render_env:
                    self.render_env.close()
        
        checkpoint_cb = CheckpointCallback(
            save_freq=10000,
            save_path=save_dir,
            name_prefix="checkpoint"
        )
        
        render_cb = PeriodicRenderCallback(render_freq=self.args.render_freq)
        
        agent.learn(
            total_timesteps=self.args.timesteps,
            callback=[checkpoint_cb, render_cb]
        )
        
        agent.save(os.path.join(save_dir, "final_model"))
    
    def _train_sb3_basic(self, agent, train_env, save_dir):
        """Basic SB3 training without visualization."""
        from stable_baselines3.common.callbacks import CheckpointCallback
        
        checkpoint_cb = CheckpointCallback(
            save_freq=10000,
            save_path=save_dir,
            name_prefix="checkpoint"
        )
        
        print(f"🚀 Starting training for {self.args.timesteps:,} timesteps...\n")
        
        agent.learn(
            total_timesteps=self.args.timesteps,
            callback=checkpoint_cb
        )
        
        agent.save(os.path.join(save_dir, "final_model"))
    
    def _train_custom(self):
        """Train using custom SAC implementation."""
        if self.args.algorithm == 'sac_gail':
            self._train_custom_gail()
        else:
            self._train_custom_sac()
    
    def _train_custom_sac(self):
        """Train custom SAC."""
        from training.continuous_trainer import ContinuousTrainer
        
        print("Using custom SAC implementation with continuous trainer...")
        trainer = ContinuousTrainer(self.config)
        trainer.train()
    
    def _train_custom_gail(self):
        """Train custom SAC+GAIL."""
        from training.continuous_trainer import ContinuousTrainer
        from training.expert_buffer import ExpertBuffer
        
        print("Using custom SAC+GAIL implementation...")
        
        # Load expert demos
        obs_dim = self._create_env().observation_space.shape[0]
        action_dim = self._create_env().action_space.shape[0]
        expert_buffer = ExpertBuffer(obs_dim, action_dim)
        
        if self.config.gail.load_human_demos:
            human_path = os.path.join(self.config.gail.expert_demos_path, "human")
            if os.path.exists(human_path):
                expert_buffer.load_directory(human_path, source_filter='human')
        
        print(f"Loaded {len(expert_buffer)} expert transitions")
        
        trainer = ContinuousTrainer(self.config)
        trainer.train()


def main():
    parser = argparse.ArgumentParser(
        description="Unified SALP training script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Algorithm selection
    parser.add_argument(
        '--algorithm', '-a',
        choices=['sac', 'sac_gail'],
        default='sac',
        help='Training algorithm (default: sac)'
    )
    
    # Implementation selection
    parser.add_argument(
        '--implementation', '-i',
        choices=['sb3', 'custom'],
        default='sb3',
        help='Implementation to use (default: sb3)'
    )
    
    # Visualization mode
    parser.add_argument(
        '--visual', '-v',
        choices=['none', 'periodic', 'continuous'],
        default='none',
        help='Visualization mode (default: none)'
    )
    
    # Configuration preset
    parser.add_argument(
        '--config', '-c',
        choices=['multi_food', 'single_food', 'navigation'],
        default='multi_food',
        help='Environment configuration preset (default: multi_food)'
    )
    
    # Training parameters
    parser.add_argument(
        '--timesteps', '-t',
        type=int,
        default=100000,
        help='Total training timesteps (default: 100000)'
    )
    
    parser.add_argument(
        '--eval-freq',
        type=int,
        default=5000,
        help='Evaluate every N steps (default: 5000)'
    )
    
    parser.add_argument(
        '--render-freq',
        type=int,
        default=10000,
        help='For periodic visual mode, render every N steps (default: 10000)'
    )
    
    # Checkpoint management
    parser.add_argument(
        '--continue-from',
        type=str,
        default=None,
        help='Path to checkpoint to continue training from'
    )
    
    args = parser.parse_args()
    
    # Create and run trainer
    trainer = UnifiedTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
