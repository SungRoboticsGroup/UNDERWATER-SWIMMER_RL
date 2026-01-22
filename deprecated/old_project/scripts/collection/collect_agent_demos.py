"""
Script for collecting agent expert demonstrations.
Runs a trained SAC agent and records high-performing episodes.
"""

import numpy as np
import os
import sys
import pickle
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.salp_snake_env import SalpSnakeEnv
from agents.sac_agent import SACAgent
from config.base_config import get_sac_snake_config


class AgentDemoCollector:
    """Collects demonstrations from trained agent."""
    
    def __init__(self, agent, env, save_dir: str = "expert_demos/agent"):
        self.agent = agent
        self.env = env
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Session statistics
        self.episodes_collected = 0
        self.episodes_evaluated = 0
        self.session_scores = []
    
    def collect_episode(self, deterministic: bool = True, render: bool = False) -> dict:
        """
        Run one episode and collect trajectory.
        
        Args:
            deterministic: Use deterministic policy
            render: Whether to render episode
        
        Returns:
            Dictionary with episode data and metadata
        """
        obs, info = self.env.reset()
        
        # Storage for episode
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        
        episode_score = 0
        episode_steps = 0
        done = False
        truncated = False
        
        while not done and not truncated:
            # Select action
            action = self.agent.select_action(obs, deterministic=deterministic)
            
            # Step environment
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # Record transition
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            dones.append(done or truncated)
            
            episode_score += reward
            episode_steps += 1
            obs = next_obs
            
            if render:
                self.env.render()
        
        # Convert to arrays
        episode_data = {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_observations': np.array(next_observations),
            'dones': np.array(dones),
            'metadata': {
                'source': 'agent',
                'score': episode_score,
                'timestamp': datetime.now().isoformat(),
                'episode_length': episode_steps,
                'food_collected': info.get('food_collected', 0),
                'deterministic': deterministic
            }
        }
        
        self.episodes_evaluated += 1
        
        return episode_data
    
    def save_episode(self, episode_data: dict):
        """Save episode to disk."""
        score = episode_data['metadata']['score']
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_demo_{timestamp}_score{int(score)}.pkl"
        filepath = os.path.join(self.save_dir, filename)
        
        # Save to disk
        with open(filepath, 'wb') as f:
            pickle.dump(episode_data, f)
        
        self.episodes_collected += 1
        self.session_scores.append(score)
        
        print(f"✓ Episode saved: {filename}")
        print(f"  Score: {score:.2f}")
        print(f"  Length: {episode_data['metadata']['episode_length']} steps")
        print(f"  Food collected: {episode_data['metadata']['food_collected']}")
    
    def collect_demonstrations(self, num_episodes: int, 
                               min_score: Optional[float] = None,
                               top_k: Optional[int] = None,
                               deterministic: bool = True,
                               render: bool = False):
        """
        Collect multiple demonstrations from agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            min_score: Minimum score threshold (save episodes >= this score)
            top_k: Only save top K episodes by score
            deterministic: Use deterministic policy
            render: Whether to render episodes
        """
        print("\n" + "="*70)
        print("AGENT DEMONSTRATION COLLECTION")
        print("="*70)
        print(f"Episodes to evaluate: {num_episodes}")
        print(f"Policy mode: {'Deterministic' if deterministic else 'Stochastic'}")
        if min_score:
            print(f"Minimum score threshold: {min_score}")
        if top_k:
            print(f"Save top {top_k} episodes only")
        print("="*70 + "\n")
        
        # Collect all episodes
        all_episodes = []
        
        for i in range(num_episodes):
            print(f"Collecting episode {i+1}/{num_episodes}...")
            episode_data = self.collect_episode(deterministic, render)
            all_episodes.append(episode_data)
            
            score = episode_data['metadata']['score']
            print(f"  Score: {score:.2f}")
        
        print("\n" + "-"*70)
        print("FILTERING EPISODES")
        print("-"*70)
        
        # Filter by score threshold
        if min_score is not None:
            filtered_episodes = [ep for ep in all_episodes 
                               if ep['metadata']['score'] >= min_score]
            print(f"Episodes meeting score threshold ({min_score}): {len(filtered_episodes)}/{len(all_episodes)}")
        else:
            filtered_episodes = all_episodes
        
        # Select top K episodes
        if top_k is not None and len(filtered_episodes) > top_k:
            # Sort by score descending
            filtered_episodes.sort(key=lambda x: x['metadata']['score'], reverse=True)
            filtered_episodes = filtered_episodes[:top_k]
            print(f"Selected top {top_k} episodes")
        
        # Save filtered episodes
        print(f"\nSaving {len(filtered_episodes)} episodes...")
        for episode_data in filtered_episodes:
            self.save_episode(episode_data)
        
        # Session summary
        print("\n" + "="*70)
        print("COLLECTION SESSION COMPLETE")
        print("="*70)
        print(f"Episodes evaluated: {self.episodes_evaluated}")
        print(f"Episodes saved: {self.episodes_collected}")
        print(f"Save directory: {self.save_dir}")
        
        if self.session_scores:
            print(f"\nScore statistics (saved episodes):")
            print(f"  Average: {np.mean(self.session_scores):.2f}")
            print(f"  Min: {np.min(self.session_scores):.2f}")
            print(f"  Max: {np.max(self.session_scores):.2f}")
        
        all_scores = [ep['metadata']['score'] for ep in all_episodes]
        print(f"\nScore statistics (all evaluated):")
        print(f"  Average: {np.mean(all_scores):.2f}")
        print(f"  Min: {np.min(all_scores):.2f}")
        print(f"  Max: {np.max(all_scores):.2f}")
        
        print("="*70 + "\n")


def main():
    """Run agent demonstration collection."""
    parser = argparse.ArgumentParser(description='Collect agent demonstrations')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--num-episodes', type=int, default=50,
                       help='Number of episodes to evaluate')
    parser.add_argument('--min-score', type=float, default=None,
                       help='Minimum score threshold')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Save only top K episodes')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy (default: deterministic)')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes')
    parser.add_argument('--save-dir', type=str, default='expert_demos/agent',
                       help='Directory to save demonstrations')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_sac_snake_config()
    
    # Create environment
    env_params = config.environment.params.copy()
    render_mode = "human" if args.render else None
    env = SalpSnakeEnv(
        render_mode=render_mode,
        width=config.environment.width,
        height=config.environment.height,
        **env_params
    )
    
    # Create agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SACAgent(
        config=config.agent,
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_space=env.action_space
    )
    
    # Load trained model
    print(f"Loading model from {args.model}...")
    agent.load(args.model)
    print("Model loaded successfully!")
    
    # Create collector
    collector = AgentDemoCollector(agent, env, save_dir=args.save_dir)
    
    # Collect demonstrations
    try:
        collector.collect_demonstrations(
            num_episodes=args.num_episodes,
            min_score=args.min_score,
            top_k=args.top_k,
            deterministic=not args.stochastic,
            render=args.render
        )
    except KeyboardInterrupt:
        print("\nCollection interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
