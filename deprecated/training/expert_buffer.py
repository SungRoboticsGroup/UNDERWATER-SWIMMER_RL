"""
Expert demonstration buffer for GAIL.
Manages loading, storing, and sampling expert trajectories.
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Tuple
import glob


class ExpertBuffer:
    """
    Buffer for storing and sampling expert demonstrations.
    Supports both human and agent demonstrations.
    """
    
    def __init__(self, obs_dim: int, action_dim: int):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Storage for expert data
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []
        
        # Episode metadata
        self.episodes = []  # List of episode info dicts
        self.num_transitions = 0
    
    def add_episode(self, episode_data: Dict[str, np.ndarray], metadata: Optional[Dict] = None):
        """
        Add a complete episode to the buffer.
        
        Args:
            episode_data: Dictionary with keys:
                - 'observations': [T, obs_dim]
                - 'actions': [T, action_dim]
                - 'rewards': [T]
                - 'next_observations': [T, obs_dim]
                - 'dones': [T]
            metadata: Optional episode info (score, source, etc.)
        """
        # Validate shapes
        T = len(episode_data['observations'])
        assert episode_data['actions'].shape == (T, self.action_dim), \
            f"Action shape mismatch: expected ({T}, {self.action_dim}), got {episode_data['actions'].shape}"
        assert episode_data['observations'].shape == (T, self.obs_dim), \
            f"Observation shape mismatch: expected ({T}, {self.obs_dim}), got {episode_data['observations'].shape}"
        
        # Add to storage
        self.observations.append(episode_data['observations'])
        self.actions.append(episode_data['actions'])
        self.rewards.append(episode_data['rewards'])
        self.next_observations.append(episode_data['next_observations'])
        self.dones.append(episode_data['dones'])
        
        # Store metadata
        episode_info = {
            'num_transitions': T,
            'total_reward': np.sum(episode_data['rewards']),
            'episode_length': T
        }
        if metadata:
            episode_info.update(metadata)
        self.episodes.append(episode_info)
        
        self.num_transitions += T
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample a batch of transitions from expert demonstrations.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Dictionary with sampled transitions
        """
        if self.num_transitions == 0:
            raise ValueError("Expert buffer is empty. Add demonstrations first.")
        
        # Flatten all episodes into single arrays
        all_observations = np.concatenate(self.observations, axis=0)
        all_actions = np.concatenate(self.actions, axis=0)
        all_rewards = np.concatenate(self.rewards, axis=0)
        all_next_observations = np.concatenate(self.next_observations, axis=0)
        all_dones = np.concatenate(self.dones, axis=0)
        
        # Random sample
        indices = np.random.randint(0, self.num_transitions, size=batch_size)
        
        return {
            'observations': all_observations[indices],
            'actions': all_actions[indices],
            'rewards': all_rewards[indices],
            'next_observations': all_next_observations[indices],
            'dones': all_dones[indices]
        }
    
    def save(self, filepath: str):
        """Save expert buffer to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'observations': self.observations,
            'actions': self.actions,
            'rewards': self.rewards,
            'next_observations': self.next_observations,
            'dones': self.dones,
            'episodes': self.episodes,
            'num_transitions': self.num_transitions,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Expert buffer saved to {filepath}")
        print(f"  Episodes: {len(self.episodes)}")
        print(f"  Transitions: {self.num_transitions}")
    
    def load(self, filepath: str):
        """Load expert buffer from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.observations = data['observations']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.next_observations = data['next_observations']
        self.dones = data['dones']
        self.episodes = data['episodes']
        self.num_transitions = data['num_transitions']
        
        # Verify dimensions match
        assert self.obs_dim == data['obs_dim'], "Observation dimension mismatch"
        assert self.action_dim == data['action_dim'], "Action dimension mismatch"
        
        print(f"Expert buffer loaded from {filepath}")
        print(f"  Episodes: {len(self.episodes)}")
        print(f"  Transitions: {self.num_transitions}")
    
    def load_directory(self, directory: str, source_filter: Optional[str] = None):
        """
        Load all expert demonstrations from a directory.
        
        Args:
            directory: Path to directory containing .pkl files
            source_filter: Optional filter for episode source ('human', 'agent', or None for all)
        """
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist")
            return
        
        # Find all pickle files
        pattern = os.path.join(directory, "**", "*.pkl")
        files = glob.glob(pattern, recursive=True)
        
        if not files:
            print(f"Warning: No .pkl files found in {directory}")
            return
        
        loaded_count = 0
        for filepath in files:
            try:
                with open(filepath, 'rb') as f:
                    episode_data = pickle.load(f)
                
                # Check source filter
                metadata = episode_data.get('metadata', {})
                source = metadata.get('source', 'unknown')
                
                if source_filter is None or source == source_filter:
                    # Add episode to buffer
                    self.add_episode(episode_data, metadata)
                    loaded_count += 1
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
        
        print(f"Loaded {loaded_count} episodes from {directory}")
        if source_filter:
            print(f"  Filtered by source: {source_filter}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about the expert demonstrations."""
        if not self.episodes:
            return {
                'num_episodes': 0,
                'num_transitions': 0,
                'avg_episode_length': 0,
                'avg_episode_reward': 0,
                'sources': {}
            }
        
        # Calculate statistics
        episode_lengths = [ep['episode_length'] for ep in self.episodes]
        episode_rewards = [ep['total_reward'] for ep in self.episodes]
        
        # Count by source
        sources = {}
        for ep in self.episodes:
            source = ep.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        return {
            'num_episodes': len(self.episodes),
            'num_transitions': self.num_transitions,
            'avg_episode_length': np.mean(episode_lengths),
            'min_episode_length': np.min(episode_lengths),
            'max_episode_length': np.max(episode_lengths),
            'avg_episode_reward': np.mean(episode_rewards),
            'min_episode_reward': np.min(episode_rewards),
            'max_episode_reward': np.max(episode_rewards),
            'sources': sources
        }
    
    def filter_by_reward(self, min_reward: float):
        """
        Keep only episodes with total reward >= min_reward.
        
        Args:
            min_reward: Minimum total reward threshold
        """
        # Find indices of episodes to keep
        keep_indices = [i for i, ep in enumerate(self.episodes) 
                       if ep['total_reward'] >= min_reward]
        
        if not keep_indices:
            print("Warning: No episodes meet the reward threshold")
            return
        
        # Filter episodes
        self.observations = [self.observations[i] for i in keep_indices]
        self.actions = [self.actions[i] for i in keep_indices]
        self.rewards = [self.rewards[i] for i in keep_indices]
        self.next_observations = [self.next_observations[i] for i in keep_indices]
        self.dones = [self.dones[i] for i in keep_indices]
        self.episodes = [self.episodes[i] for i in keep_indices]
        
        # Recalculate transition count
        self.num_transitions = sum(ep['num_transitions'] for ep in self.episodes)
        
        print(f"Filtered to {len(keep_indices)} episodes with reward >= {min_reward}")
    
    def __len__(self) -> int:
        """Return number of transitions in buffer."""
        return self.num_transitions
    
    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (f"ExpertBuffer(episodes={stats['num_episodes']}, "
                f"transitions={stats['num_transitions']}, "
                f"avg_reward={stats['avg_episode_reward']:.2f})")


def merge_expert_buffers(buffers: List[ExpertBuffer], obs_dim: int, action_dim: int) -> ExpertBuffer:
    """
    Merge multiple expert buffers into one.
    
    Args:
        buffers: List of ExpertBuffer objects
        obs_dim: Observation dimension
        action_dim: Action dimension
    
    Returns:
        Merged ExpertBuffer
    """
    merged = ExpertBuffer(obs_dim, action_dim)
    
    for buffer in buffers:
        for i in range(len(buffer.episodes)):
            episode_data = {
                'observations': buffer.observations[i],
                'actions': buffer.actions[i],
                'rewards': buffer.rewards[i],
                'next_observations': buffer.next_observations[i],
                'dones': buffer.dones[i]
            }
            merged.add_episode(episode_data, buffer.episodes[i])
    
    return merged
