"""
Continuous visual GAIL training system.
Shows the agent learning in real-time with GAIL bootstrapping from expert demos.
"""

import sys
import os
import time
import threading

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.base_config import get_sac_gail_snake_config
from training.expert_buffer import ExpertBuffer
from agents.sac_gail_agent import SACGAILAgent
from environments.salp_snake_env import SalpSnakeEnv
from core.base_agent import ReplayBuffer, Logger
import numpy as np
import torch


class ContinuousGAILTrainer:
    """GAIL trainer with continuous visual feedback."""
    
    def __init__(self, config):
        self.config = config
        
        # Create environments
        self.env = self._create_environment(render_mode=None)
        self.eval_env = self._create_environment(render_mode=None)
        self.visual_env = self._create_environment(render_mode="human")
        
        # Load expert demonstrations
        self.expert_buffer = self._load_expert_demonstrations()
        
        # Create GAIL agent
        self.agent = self._create_agent()
        
        # Enable adaptive weights
        self.agent.enable_adaptive_weights(True)
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.agent.buffer_size,
            obs_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0]
        )
        
        # Create logger
        log_dir = os.path.join(config.training.log_dir, config.training.experiment_name)
        self.logger = Logger(log_dir)
        
        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_eval_score = -float('inf')
        
        # Visual state
        self.current_best_agent = None
        self.visual_running = True
        
        # Create directories
        self.model_dir = os.path.join(config.training.model_dir, config.training.experiment_name)
        os.makedirs(self.model_dir, exist_ok=True)
        
        print(f"🎬 Continuous GAIL Trainer initialized!")
        print(f"Device: {self.agent.device}")
        print("Real-time visual display with GAIL learning!")
    
    def _create_environment(self, render_mode=None):
        """Create environment."""
        env_params = self.config.environment.params.copy()
        return SalpSnakeEnv(
            render_mode=render_mode,
            width=self.config.environment.width,
            height=self.config.environment.height,
            **env_params
        )
    
    def _load_expert_demonstrations(self):
        """Load expert demonstrations."""
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        expert_buffer = ExpertBuffer(obs_dim, action_dim)
        
        if self.config.gail.load_human_demos:
            human_path = os.path.join(self.config.gail.expert_demos_path, "human")
            if os.path.exists(human_path):
                print(f"Loading human demonstrations from {human_path}...")
                expert_buffer.load_directory(human_path, source_filter='human')
        
        stats = expert_buffer.get_statistics()
        print(f"\n📊 Expert Buffer Statistics:")
        print(f"  Episodes: {stats['num_episodes']}")
        print(f"  Transitions: {stats['num_transitions']}")
        print(f"  Avg reward: {stats['avg_episode_reward']:.2f}")
        print(f"  Sources: {stats['sources']}\n")
        
        return expert_buffer
    
    def _create_agent(self):
        """Create SAC+GAIL agent."""
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        return SACGAILAgent(
            config=self.config.agent,
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_space=self.env.action_space,
            expert_buffer=self.expert_buffer,
            use_gail=self.config.gail.use_gail,
            reward_env_weight=self.config.gail.reward_env_weight,
            reward_gail_weight=self.config.gail.reward_gail_weight,
            discriminator_update_freq=self.config.gail.discriminator_update_freq
        )
    
    def train(self):
        """Main training loop with visualization."""
        print(f"\n🚀 Starting GAIL training for {self.config.training.max_episodes} episodes...")
        print("🎮 Visual display will update with best models!")
        print("Press Ctrl+C to stop.\n")
        
        # Initialize current best agent
        self.current_best_agent = self._create_agent()
        
        # Start background training
        training_thread = threading.Thread(target=self._background_training, daemon=True)
        training_thread.start()
        
        # Run visual loop (main thread for macOS)
        self._visual_loop()
    
    def _background_training(self):
        """Background training loop."""
        start_time = time.time()
        
        try:
            for episode in range(self.config.training.max_episodes):
                if not self.visual_running:
                    break
                
                self.episode = episode
                episode_metrics = self._run_episode()
                
                # Update performance for adaptive weights
                self.agent.update_performance(episode_metrics['score'])
                
                self.logger.log_episode(episode, episode_metrics)
                self.agent.episode_count = episode
                
                # Evaluation
                if episode % self.config.training.eval_frequency == 0:
                    eval_metrics = self._evaluate()
                    
                    for key, value in eval_metrics.items():
                        self.logger.log_scalar(f"eval_{key}", value, episode)
                    
                    if eval_metrics['mean_score'] > self.best_eval_score:
                        self.best_eval_score = eval_metrics['mean_score']
                        self._save_model("best_model.pth")
                        self._update_visual_agent()
                        print(f"🏆 NEW BEST! Score: {self.best_eval_score:.2f}")
                
                # Progress with adaptive status
                elapsed = time.time() - start_time
                disc_acc = episode_metrics.get('avg_discriminator_accuracy', 0)
                adaptive_status = self.agent.get_adaptive_status()
                
                if adaptive_status['enabled']:
                    level = adaptive_status.get('level', 'warming_up')
                    env_w = adaptive_status['env_weight']
                    gail_w = adaptive_status['gail_weight']
                    print(f"Ep {episode}/{self.config.training.max_episodes} | "
                          f"Score: {episode_metrics['score']:.1f} | "
                          f"Food: {episode_metrics['food_collected']} | "
                          f"Level: {level} | "
                          f"Weights: E{env_w:.0%}/G{gail_w:.0%} | "
                          f"Disc: {disc_acc:.2%} | "
                          f"{elapsed:.1f}s")
                else:
                    print(f"Ep {episode}/{self.config.training.max_episodes} | "
                          f"Score: {episode_metrics['score']:.1f} | "
                          f"Food: {episode_metrics['food_collected']} | "
                          f"Disc Acc: {disc_acc:.2%} | "
                          f"Time: {elapsed:.1f}s")
        
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.visual_running = False
            self._save_model("final_model.pth")
            self.logger.save_metrics()
            print(f"\n✅ Training complete! Best score: {self.best_eval_score:.2f}")
    
    def _visual_loop(self):
        """Continuous visual display."""
        print("🎬 Starting visual display...\n")
        
        try:
            import pygame
            os.environ['SDL_VIDEODRIVER'] = 'cocoa'
            pygame.init()
        except:
            pass
        
        while self.visual_running:
            try:
                obs, _ = self.visual_env.reset()
                episode_reward = 0
                episode_steps = 0
                done = False
                truncated = False
                
                print(f"🎮 Visual episode (Training Ep {self.episode})...")
                
                while not done and not truncated and self.visual_running:
                    self.visual_env.render()
                    
                    if self.current_best_agent:
                        action = self.current_best_agent.select_action(obs, deterministic=True)
                    else:
                        action = self.visual_env.action_space.sample()
                    
                    obs, reward, done, truncated, info = self.visual_env.step(action)
                    episode_reward += reward
                    episode_steps += 1
                    time.sleep(0.05)
                
                food = info.get('food_collected', 0)
                print(f"   Score: {episode_reward:.1f}, Food: {food}, Steps: {episode_steps}")
                time.sleep(1.0)
                
            except Exception as e:
                print(f"Visual error: {e}")
                break
    
    def _update_visual_agent(self):
        """Update visual agent with current best."""
        try:
            new_agent = self._create_agent()
            new_agent.policy.load_state_dict(self.agent.policy.state_dict())
            new_agent.q1.load_state_dict(self.agent.q1.state_dict())
            new_agent.q2.load_state_dict(self.agent.q2.state_dict())
            if self.agent.discriminator:
                new_agent.discriminator.load_state_dict(self.agent.discriminator.state_dict())
            self.current_best_agent = new_agent
        except Exception as e:
            print(f"Update visual agent error: {e}")
    
    def _run_episode(self):
        """Run training episode."""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_losses = []
        done = False
        truncated = False
        
        while not done and not truncated and episode_steps < self.config.training.max_steps_per_episode:
            action = self.agent.select_action(obs, deterministic=False)
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # Use hybrid reward for GAIL
            if hasattr(self.agent, 'compute_hybrid_reward'):
                hybrid_reward = self.agent.compute_hybrid_reward(reward, obs, action)
            else:
                hybrid_reward = reward
            
            self.replay_buffer.add(obs, action, reward, next_obs, done or truncated)
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            
            if (len(self.replay_buffer) >= self.config.training.start_training_after and 
                self.total_steps % self.config.training.train_frequency == 0):
                batch = self.replay_buffer.sample(self.config.agent.batch_size)
                loss_info = self.agent.update(batch)
                episode_losses.append(loss_info)
                
                for key, value in loss_info.items():
                    self.logger.log_scalar(f"train_{key}", value, self.total_steps)
            
            obs = next_obs
        
        metrics = {
            'score': episode_reward,
            'steps': episode_steps,
            'food_collected': info.get('food_collected', 0),
            'collision': info.get('collision', False)
        }
        
        if episode_losses:
            for key in episode_losses[0].keys():
                metrics[f'avg_{key}'] = np.mean([loss[key] for loss in episode_losses])
        
        return metrics
    
    def _evaluate(self, num_episodes=3):
        """Evaluate agent."""
        scores = []
        food_list = []
        
        for _ in range(num_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            done = False
            truncated = False
            steps = 0
            
            while not done and not truncated and steps < self.config.training.max_steps_per_episode:
                action = self.agent.select_action(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                steps += 1
            
            scores.append(episode_reward)
            food_list.append(info.get('food_collected', 0))
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'mean_food_collected': np.mean(food_list)
        }
    
    def _save_model(self, filename):
        """Save model."""
        filepath = os.path.join(self.model_dir, filename)
        self.agent.save(filepath)


def main():
    """Run continuous GAIL training."""
    config = get_sac_gail_snake_config()
    trainer = ContinuousGAILTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
