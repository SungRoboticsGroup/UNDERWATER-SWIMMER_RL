#!/usr/bin/env python3
"""
SALP Unified Training Script

Simple, config-driven training. All parameters come from YAML files.

Usage:
    # Use default config
    python train.py
    
    # Use a preset config
    python train.py --config single_food
    
    # Use custom config file
    python train.py --config path/to/custom.yaml
    
    # With visual feedback
    python train.py --visual
    
    # Use SB3 implementation (default is custom)
    python train.py --sb3
    
    # Continue from checkpoint
    python train.py --checkpoint data/models/best_model.zip
    
    # Override config values
    python train.py --config single_food --timesteps 50000 --eval-freq 1000
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from salp.config.config_loader import load_config
from salp.environments.salp_snake_env import SalpSnakeEnv


def train_sb3(config, args):
    """Train using Stable Baselines3 SAC."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
    import numpy as np
    from datetime import datetime
    
    # Create environments
    env_params = config.environment.get('params', {})
    train_env = SalpSnakeEnv(render_mode=None, **env_params)
    eval_env = SalpSnakeEnv(render_mode=None, **env_params)
    
    # Create or load agent
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        agent = SAC.load(args.checkpoint, env=train_env)
    else:
        agent_config = config.agent
        agent = SAC(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=agent_config.get('learning_rate', 3e-4),
            buffer_size=agent_config.get('buffer_size', 500000),
            batch_size=agent_config.get('batch_size', 128),
            tau=agent_config.get('tau', 0.005),
            gamma=agent_config.get('gamma', 0.99),
            policy_kwargs=dict(net_arch=agent_config.get('hidden_sizes', [256, 256])),
            verbose=0 if args.visual else 1,
        )
    
    # Setup save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config.training.get('experiment_name', 'salp_training')
    save_dir = Path(config.training.get('model_dir', 'data/models')) / f"{exp_name}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if args.visual:
        _train_sb3_visual(agent, train_env, eval_env, save_dir, config, args)
    else:
        _train_sb3_basic(agent, train_env, save_dir, config, args)
    
    train_env.close()
    eval_env.close()
    print(f"\n✓ Training complete! Models saved to: {save_dir}")


def _train_sb3_visual(agent, train_env, eval_env, save_dir, config, args):
    """Train SB3 with continuous visual feedback."""
    import time
    import numpy as np
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    
    # Import visual trainer
    sys.path.append(str(Path(__file__).parent / 'scripts' / 'utilities'))
    from continuous_visual_trainer import ContinuousVisualTrainer
    
    env_params = config.environment.get('params', {})
    visual_trainer = ContinuousVisualTrainer(
        env_creator=lambda: SalpSnakeEnv(render_mode="human", **env_params),
        fps=30.0,
        verbose=True
    )
    
    def train_with_visual(vt):
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
                self.eval_freq = config.training.get('eval_frequency', 5000)
                if args.eval_freq:
                    self.eval_freq = args.eval_freq
                    
            def _on_step(self):
                nonlocal best_reward
                if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
                    mean_reward = evaluate(self.model)
                    print(f"\n📊 Step {self.n_calls:,}: Reward {mean_reward:.1f}")
                    
                    vt.update_training_info({'step': f'{self.n_calls:,}', 'best': f'{best_reward:.1f}'})
                    
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                        best_path = save_dir / "best_model"
                        self.model.save(str(best_path))
                        new_model = SAC.load(str(best_path), env=train_env)
                        vt.update_model(new_model)
                        print("🏆 NEW BEST!")
                return True
        
        timesteps = args.timesteps if args.timesteps else config.training.get('max_episodes', 1000) * 1000
        agent.learn(total_timesteps=timesteps, callback=VisualCallback(), log_interval=None)
        agent.save(str(save_dir / "final_model"))
        vt.stop()
    
    visual_trainer.start_training_thread(train_with_visual)
    time.sleep(2)
    
    try:
        visual_trainer.run_visual_loop()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
        visual_trainer.stop()


def _train_sb3_basic(agent, train_env, save_dir, config, args):
    """Basic SB3 training without visualization."""
    from stable_baselines3.common.callbacks import CheckpointCallback
    
    checkpoint_cb = CheckpointCallback(
        save_freq=10000,
        save_path=str(save_dir),
        name_prefix="checkpoint"
    )
    
    timesteps = args.timesteps if args.timesteps else config.training.get('max_episodes', 1000) * 1000
    print(f"🚀 Training for {timesteps:,} timesteps...\n")
    
    agent.learn(total_timesteps=timesteps, callback=checkpoint_cb)
    agent.save(str(save_dir / "final_model"))


def train_custom(config, args):
    """Train using custom SAC/GAIL implementation."""
    from salp.training.continuous_trainer import ContinuousTrainer
    from salp.config.base_config import ExperimentConfig, EnvironmentConfig, AgentConfig, TrainingConfig, GAILConfig
    
    # Convert new config format to old format (temporary bridge)
    env_cfg = EnvironmentConfig(
        name=config.environment.get('name'),
        type=config.environment.get('type'),
        width=config.environment.get('width', 800),
        height=config.environment.get('height', 600),
        params=config.environment.get('params', {})
    )
    
    agent_cfg = AgentConfig(
        name=config.agent.get('name'),
        type=config.agent.get('type'),
        hidden_sizes=config.agent.get('hidden_sizes', [256, 256]),
        learning_rate=config.agent.get('learning_rate', 3e-4),
        batch_size=config.agent.get('batch_size', 128),
        buffer_size=config.agent.get('buffer_size', 500000),
        gamma=config.agent.get('gamma', 0.99),
        tau=config.agent.get('tau', 0.005),
        params=config.agent.get('params', {})
    )
    
    training_cfg = TrainingConfig(
        max_episodes=config.training.get('max_episodes', 1000),
        max_steps_per_episode=config.training.get('max_steps_per_episode', 5000),
        eval_frequency=config.training.get('eval_frequency', 25),
        save_frequency=config.training.get('save_frequency', 50),
        start_training_after=config.training.get('start_training_after', 500),
        log_dir=config.training.get('log_dir', 'data/logs'),
        model_dir=config.training.get('model_dir', 'data/models'),
        experiment_name=config.training.get('experiment_name', 'salp_training')
    )
    
    gail_cfg = None
    if config.gail and config.gail.get('use_gail', False):
        gail_cfg = GAILConfig(**config.gail)
    
    exp_config = ExperimentConfig(env_cfg, agent_cfg, training_cfg, gail_cfg)
    
    trainer = ContinuousTrainer(exp_config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--config', '-c', default='defaults', help='Config file or preset name')
    parser.add_argument('--sb3', action='store_true', help='Use Stable Baselines3 (default: custom)')
    parser.add_argument('--visual', '-v', action='store_true', help='Enable continuous visual feedback')
    parser.add_argument('--checkpoint', help='Path to checkpoint to continue from')
    parser.add_argument('--timesteps', '-t', type=int, help='Override total timesteps')
    parser.add_argument('--eval-freq', type=int, help='Override evaluation frequency')
    
    args = parser.parse_args()
    
    # Load configuration
    overrides = {}
    if args.timesteps:
        overrides['training'] = {'max_episodes': args.timesteps // 1000}
    if args.eval_freq:
        overrides.setdefault('training', {})['eval_frequency'] = args.eval_freq
    
    config = load_config(args.config, **overrides)
    
    print("=" * 70)
    print("SALP TRAINING")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Implementation: {'SB3' if args.sb3 else 'Custom'}")
    print(f"Visual: {'Yes' if args.visual else 'No'}")
    print("=" * 70 + "\n")
    
    # Train
    if args.sb3:
        train_sb3(config, args)
    else:
        train_custom(config, args)


if __name__ == "__main__":
    main()
