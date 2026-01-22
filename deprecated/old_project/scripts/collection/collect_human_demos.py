"""
Script for collecting human expert demonstrations.
Allows manual gameplay with keyboard controls while recording trajectories.
"""

import pygame
import numpy as np
import os
import sys
import pickle
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.salp_snake_env import SalpSnakeEnv
from config.base_config import get_sac_snake_config


class HumanDemoCollector:
    """Collects human demonstrations for GAIL training."""
    
    def __init__(self, env, save_dir: str = "expert_demos/human"):
        self.env = env
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Current episode storage
        self.current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': []
        }
        
        # Session statistics
        self.episodes_collected = 0
        self.total_episodes_attempted = 0
        self.session_scores = []
    
    def reset_episode(self):
        """Reset current episode storage."""
        self.current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': []
        }
    
    def record_transition(self, obs, action, reward, next_obs, done):
        """Record a single transition."""
        self.current_episode['observations'].append(obs)
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['next_observations'].append(next_obs)
        self.current_episode['dones'].append(done)
    
    def save_episode(self, score: float, accept: bool = True):
        """
        Save current episode to disk.
        
        Args:
            score: Total score for the episode
            accept: Whether to save this episode (user decision)
        """
        if not accept:
            print("Episode discarded.")
            return
        
        # Convert lists to numpy arrays
        episode_data = {
            'observations': np.array(self.current_episode['observations']),
            'actions': np.array(self.current_episode['actions']),
            'rewards': np.array(self.current_episode['rewards']),
            'next_observations': np.array(self.current_episode['next_observations']),
            'dones': np.array(self.current_episode['dones']),
            'metadata': {
                'source': 'human',
                'score': score,
                'timestamp': datetime.now().isoformat(),
                'episode_length': len(self.current_episode['observations'])
            }
        }
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"human_demo_{timestamp}_score{int(score)}.pkl"
        filepath = os.path.join(self.save_dir, filename)
        
        # Save to disk
        with open(filepath, 'wb') as f:
            pickle.dump(episode_data, f)
        
        self.episodes_collected += 1
        self.session_scores.append(score)
        
        print(f"✓ Episode saved: {filename}")
        print(f"  Score: {score:.2f}")
        print(f"  Length: {len(self.current_episode['observations'])} steps")
        print(f"  Total collected this session: {self.episodes_collected}")
    
    def run_collection_session(self, target_episodes: int = 10, min_score: float = None):
        """
        Run human demonstration collection session.
        
        Args:
            target_episodes: Number of episodes to collect
            min_score: Minimum score threshold (None = collect all)
        """
        print("\n" + "="*70)
        print("HUMAN DEMONSTRATION COLLECTION")
        print("="*70)
        print("\nControls:")
        if self.env.forced_breathing:
            print("  ←/→ Arrow Keys: Steer nozzle left/right")
            print("  (Breathing is automatic)")
        else:
            print("  SPACE: Inhale/Exhale (hold to control)")
            print("  ←/→ Arrow Keys: Steer nozzle left/right")
        print("\n  After episode ends:")
        print("  Y: Save this episode")
        print("  N: Discard this episode")
        print("  Q: Quit collection session")
        print("\nObjective: Collect food efficiently!")
        print(f"Target: {target_episodes} good episodes")
        if min_score:
            print(f"Minimum score: {min_score}")
        print("="*70 + "\n")
        
        # Initialize pygame (in case environment didn't)
        if not pygame.get_init():
            pygame.init()
        
        running = True
        nozzle_direction = 0.0
        
        while running and self.episodes_collected < target_episodes:
            # Start new episode
            self.total_episodes_attempted += 1
            self.reset_episode()
            
            obs, info = self.env.reset()
            episode_score = 0
            episode_steps = 0
            done = False
            truncated = False
            
            print(f"\nEpisode {self.total_episodes_attempted} started...")
            
            quit_requested = False
            
            while not done and not truncated and running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_requested = True
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            quit_requested = True
                            running = False
                
                if not running:
                    break
                
                # Get keyboard input
                keys = pygame.key.get_pressed()
                
                # Nozzle steering
                if keys[pygame.K_LEFT]:
                    nozzle_direction = min(1.0, nozzle_direction + 0.03)
                elif keys[pygame.K_RIGHT]:
                    nozzle_direction = max(-1.0, nozzle_direction - 0.03)
                
                # Create action based on mode
                if self.env.forced_breathing:
                    action = np.array([nozzle_direction])
                else:
                    inhale_control = 1.0 if keys[pygame.K_SPACE] else 0.0
                    action = np.array([inhale_control, nozzle_direction])
                
                # Step environment
                next_obs, reward, done, truncated, info = self.env.step(action)
                
                # Record transition
                self.record_transition(obs, action, reward, next_obs, done or truncated)
                
                episode_score += reward
                episode_steps += 1
                obs = next_obs
                
                # Render
                self.env.render()
            
            # Handle quit during episode - use terminal input for decisions
            if quit_requested and len(self.current_episode['observations']) > 0:
                print(f"\n\n{'='*50}")
                print("QUIT DURING EPISODE")
                print(f"{'='*50}")
                print(f"  Partial Score: {episode_score:.2f}")
                print(f"  Steps recorded: {episode_steps}")
                
                # Use terminal input for save decision
                save_response = input("\nSave partial episode? (y/n): ").lower().strip()
                
                if save_response == 'y':
                    self.save_episode(episode_score, accept=True)
                else:
                    print("Partial episode discarded.")
                
                # Ask if want to continue collecting
                if self.episodes_collected < target_episodes:
                    continue_response = input("\nContinue collecting more episodes? (y/n): ").lower().strip()
                    if continue_response == 'y':
                        running = True  # Reset running flag to continue
                        quit_requested = False  # Reset quit flag for next episode
                    else:
                        running = False
                else:
                    running = False
            
            if not running:
                print("\nCollection session terminated by user.")
                break
            
            # Skip normal save prompt if we already handled quit
            if quit_requested:
                quit_requested = False  # Reset for next episode
                continue
            
            # Episode ended normally - ask user if they want to save it
            print(f"\nEpisode {self.total_episodes_attempted} completed!")
            print(f"  Score: {episode_score:.2f}")
            print(f"  Steps: {episode_steps}")
            print(f"  Food collected: {info.get('food_collected', 0)}")
            
            # Check if meets minimum score requirement
            meets_threshold = min_score is None or episode_score >= min_score
            
            # Use terminal input for save decision
            if meets_threshold:
                save_response = input("\nSave this episode? (y/n/q): ").lower().strip()
            else:
                save_response = input(f"\nScore below threshold ({min_score}). Save anyway? (y/n/q): ").lower().strip()
            
            if save_response == 'y':
                self.save_episode(episode_score, accept=True)
            elif save_response == 'n':
                print("Episode discarded.")
            elif save_response == 'q':
                print("Episode discarded. Quitting collection.")
                running = False
        
        # Session summary
        print("\n" + "="*70)
        print("COLLECTION SESSION COMPLETE")
        print("="*70)
        print(f"Episodes attempted: {self.total_episodes_attempted}")
        print(f"Episodes collected: {self.episodes_collected}")
        print(f"Save directory: {self.save_dir}")
        
        if self.session_scores:
            print(f"\nScore statistics:")
            print(f"  Average: {np.mean(self.session_scores):.2f}")
            print(f"  Min: {np.min(self.session_scores):.2f}")
            print(f"  Max: {np.max(self.session_scores):.2f}")
        
        print("="*70 + "\n")


def main():
    """Run human demonstration collection."""
    # Get configuration
    config = get_sac_snake_config()
    
    # Create environment with rendering
    env_params = config.environment.params.copy()
    env = SalpSnakeEnv(
        render_mode="human",
        width=config.environment.width,
        height=config.environment.height,
        **env_params
    )
    
    # Create collector
    collector = HumanDemoCollector(env)
    
    # Run collection session
    try:
        # Collect 10 demonstrations with minimum score of 50
        collector.run_collection_session(
            target_episodes=10,
            min_score=50.0  # Adjust based on your environment
        )
    except KeyboardInterrupt:
        print("\nCollection interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
