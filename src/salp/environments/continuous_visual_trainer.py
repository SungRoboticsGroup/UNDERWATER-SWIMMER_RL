"""
Reusable continuous visual training utility for SALP RL experiments.

This utility provides:
- Live visual feedback during training
- Training runs in background thread  
- Visual display updates automatically when better models are found

Can be used with any RL training setup (SB3, custom SAC, etc.)
"""

import os
import time
import threading
from typing import Callable, Optional, Any
import numpy as np


class ContinuousVisualTrainer:
    """
    Generic continuous visual training wrapper.
    
    Handles the threading and visual display logic so training scripts
    can focus on the actual training implementation.
    """
    
    def __init__(
        self,
        env_creator: Callable,
        model_loader: Optional[Callable] = None,
        fps: float = 20.0,
        episode_delay: float = 1.0,
        verbose: bool = True
    ):
        """
        Initialize continuous visual trainer.
        
        Args:
            env_creator: Function that creates a new environment with render_mode="human"
            model_loader: Optional function that returns the current best model
            fps: Frames per second for visual display (default: 20)
            episode_delay: Delay in seconds between visual episodes (default: 1.0)
            verbose: Whether to print episode summaries (default: True)
        """
        self.env_creator = env_creator
        self.model_loader = model_loader
        self.fps = fps
        self.frame_delay = 1.0 / fps
        self.episode_delay = episode_delay
        self.verbose = verbose
        
        # State
        self.training_active = True
        self.visual_env = None
        self.current_model = None
        self.model_updated = False
        self.training_info = {}
        
        # Episode tracking
        self.episode_count = 0
        
    def update_model(self, model: Any):
        """
        Update the model used for visual display.
        Call this when you have a new best model.
        
        Args:
            model: The new model to use for visualization
        """
        self.current_model = model
        self.model_updated = True
        
    def update_training_info(self, info: dict):
        """
        Update training information to display.
        
        Args:
            info: Dictionary with training metrics (e.g., {'step': 1000, 'best_reward': 100})
        """
        self.training_info = info.copy()
    
    def start_training_thread(self, training_function: Callable, **kwargs):
        """
        Start training in a background thread.
        
        Args:
            training_function: The training function to run (should accept trainer as first arg)
            **kwargs: Additional arguments to pass to training function
        """
        training_thread = threading.Thread(
            target=training_function,
            args=(self,),
            kwargs=kwargs,
            daemon=True
        )
        training_thread.start()
        return training_thread
    
    def run_visual_loop(self, action_selector: Optional[Callable] = None):
        """
        Run the continuous visual display loop (blocks until training completes).
        This should be called on the main thread for macOS compatibility.
        
        Args:
            action_selector: Optional function(model, obs) -> action.
                           If None, uses model.predict(obs, deterministic=True)[0]
        """
        # Initialize pygame for macOS
        try:
            import pygame
            os.environ['SDL_VIDEODRIVER'] = 'cocoa'
            os.environ['SDL_VIDEO_WINDOW_POS'] = '100,100'
            pygame.init()
            pygame.display.init()
        except:
            pass
        
        # Create visual environment
        self.visual_env = self.env_creator()
        
        if self.verbose:
            print("🎬 Starting continuous visual display...")
            print(f"   FPS: {self.fps}, Episode delay: {self.episode_delay}s")
            print()
        
        try:
            while self.training_active:
                self._run_visual_episode(action_selector)
                time.sleep(self.episode_delay)
                
        except KeyboardInterrupt:
            if self.verbose:
                print("\n⚠️  Visual display interrupted by user")
            self.training_active = False
            
        finally:
            if self.visual_env is not None:
                self.visual_env.close()
                if self.verbose:
                    print("🎬 Visual display closed")
    
    def _run_visual_episode(self, action_selector: Optional[Callable] = None):
        """Run a single visual episode."""
        try:
            obs, _ = self.visual_env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            
            # Show status
            if self.verbose:
                status = f"🎬 Episode {self.episode_count + 1}"
                if self.training_info:
                    info_str = ", ".join([f"{k}={v}" for k, v in self.training_info.items()])
                    status += f" ({info_str})"
                if self.model_updated:
                    status += " - ✨ UPDATED MODEL"
                    self.model_updated = False
                print(status)
            
            while not done and self.training_active:
                # Render
                self.visual_env.render()
                
                # Get action
                if self.current_model is not None:
                    if action_selector is not None:
                        action = action_selector(self.current_model, obs)
                    else:
                        # Default: assume model has predict method (SB3 style)
                        try:
                            action, _ = self.current_model.predict(obs, deterministic=True)
                        except AttributeError:
                            # Fallback: try select_action method
                            action = self.current_model.select_action(obs, deterministic=True)
                else:
                    # No model yet, use random actions
                    action = self.visual_env.action_space.sample()
                
                # Step
                obs, reward, terminated, truncated, info = self.visual_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_steps += 1
                
                # Frame rate control
                time.sleep(self.frame_delay)
            
            # Episode summary
            if self.verbose:
                food = info.get('food_collected', 0)
                print(f"   Reward: {episode_reward:.1f}, Food: {food}, Steps: {episode_steps}\n")
            
            self.episode_count += 1
            
        except Exception as e:
            if self.verbose:
                print(f"Visual episode error: {e}")
            import traceback
            traceback.print_exc()
    
    def stop(self):
        """Stop the visual training loop."""
        self.training_active = False


def create_visual_trainer_sb3(
    env_creator: Callable,
    initial_model: Optional[Any] = None,
    **kwargs
) -> ContinuousVisualTrainer:
    """
    Convenience function for creating a visual trainer for Stable Baselines3 models.
    
    Args:
        env_creator: Function that creates environment with render_mode="human"
        initial_model: Optional initial SB3 model
        **kwargs: Additional arguments for ContinuousVisualTrainer
    
    Returns:
        ContinuousVisualTrainer instance
    """
    trainer = ContinuousVisualTrainer(env_creator, **kwargs)
    if initial_model is not None:
        trainer.update_model(initial_model)
    return trainer


# Example usage functions
def example_sb3_training():
    """Example of how to use continuous visual trainer with SB3."""
    from stable_baselines3 import SAC
    
    def create_env():
        from salp.environments.salp_snake_env import SalpSnakeEnv
        return SalpSnakeEnv(render_mode="human", num_food_items=12)
    
    def create_training_env():
        from salp.environments.salp_snake_env import SalpSnakeEnv
        return SalpSnakeEnv(render_mode=None, num_food_items=12)
    
    def train_agent(visual_trainer: ContinuousVisualTrainer, total_timesteps: int = 100000):
        """Training function that runs in background."""
        # Create training environment
        env = create_training_env()
        
        # Create or load agent
        agent = SAC("MlpPolicy", env, verbose=0)
        visual_trainer.update_model(agent)
        
        # Training loop with periodic updates
        eval_freq = 5000
        best_reward = -float('inf')
        
        for step in range(0, total_timesteps, eval_freq):
            # Train
            agent.learn(total_timesteps=eval_freq, reset_num_timesteps=False)
            
            # Evaluate
            eval_reward = evaluate_model(agent, env, n_episodes=3)
            
            # Update visual if better
            if eval_reward > best_reward:
                best_reward = eval_reward
                visual_trainer.update_model(agent)
            
            # Update info
            visual_trainer.update_training_info({
                'step': agent.num_timesteps,
                'best_reward': f'{best_reward:.1f}'
            })
        
        env.close()
        visual_trainer.stop()
    
    def evaluate_model(model, env, n_episodes=3):
        """Simple evaluation function."""
        rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            rewards.append(episode_reward)
        return np.mean(rewards)
    
    # Create visual trainer
    visual_trainer = create_visual_trainer_sb3(create_env)
    
    # Start training in background
    visual_trainer.start_training_thread(train_agent, total_timesteps=100000)
    
    # Run visual loop on main thread (blocking)
    visual_trainer.run_visual_loop()


if __name__ == "__main__":
    print("This is a utility module. Import and use ContinuousVisualTrainer in your training scripts.")
    print("\nFor an example, see the example_sb3_training() function in this file.")
