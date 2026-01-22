"""
Base configuration system for SALP RL experiments.
Supports different agents, environments, and training configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
import os


@dataclass
class EnvironmentConfig:
    """Configuration for environments."""
    name: str
    type: str  # "salp_basic", "salp_snake", "salp_navigation", etc.
    width: int = 800
    height: int = 600
    render_mode: Optional[str] = "human"
    
    # Environment-specific parameters
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "width": self.width,
            "height": self.height,
            "render_mode": self.render_mode,
            "params": self.params
        }


@dataclass
class AgentConfig:
    """Configuration for RL agents."""
    name: str
    type: str  # "sac", "td3", "ppo", etc.
    
    # Network architecture
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1000000
    gamma: float = 0.99
    tau: float = 0.005
    
    # Agent-specific parameters
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "gamma": self.gamma,
            "tau": self.tau,
            "params": self.params
        }


@dataclass
class GAILConfig:
    """Configuration for GAIL (Generative Adversarial Imitation Learning)."""
    use_gail: bool = False
    expert_demos_path: str = "expert_demos/"
    discriminator_lr: float = 3e-4
    discriminator_update_freq: int = 1
    reward_env_weight: float = 0.3
    reward_gail_weight: float = 0.7
    min_expert_episodes: int = 5
    load_human_demos: bool = True
    load_agent_demos: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_gail": self.use_gail,
            "expert_demos_path": self.expert_demos_path,
            "discriminator_lr": self.discriminator_lr,
            "discriminator_update_freq": self.discriminator_update_freq,
            "reward_env_weight": self.reward_env_weight,
            "reward_gail_weight": self.reward_gail_weight,
            "min_expert_episodes": self.min_expert_episodes,
            "load_human_demos": self.load_human_demos,
            "load_agent_demos": self.load_agent_demos
        }


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    max_episodes: int = 1000
    max_steps_per_episode: int = 1000
    eval_frequency: int = 100
    save_frequency: int = 500
    
    # Training parameters
    start_training_after: int = 1000
    train_frequency: int = 1
    target_update_frequency: int = 1
    
    # Logging and saving
    log_dir: str = "logs"
    model_dir: str = "models"
    experiment_name: str = "salp_experiment"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_episodes": self.max_episodes,
            "max_steps_per_episode": self.max_steps_per_episode,
            "eval_frequency": self.eval_frequency,
            "save_frequency": self.save_frequency,
            "start_training_after": self.start_training_after,
            "train_frequency": self.train_frequency,
            "target_update_frequency": self.target_update_frequency,
            "log_dir": self.log_dir,
            "model_dir": self.model_dir,
            "experiment_name": self.experiment_name
        }


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    environment: EnvironmentConfig
    agent: AgentConfig
    training: TrainingConfig
    gail: Optional[GAILConfig] = None
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        config_dict = {
            "environment": self.environment.to_dict(),
            "agent": self.agent.to_dict(),
            "training": self.training.to_dict()
        }
        
        if self.gail is not None:
            config_dict["gail"] = self.gail.to_dict()
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        gail_config = None
        if "gail" in config_dict:
            gail_config = GAILConfig(**config_dict["gail"])
        
        return cls(
            environment=EnvironmentConfig(**config_dict["environment"]),
            agent=AgentConfig(**config_dict["agent"]),
            training=TrainingConfig(**config_dict["training"]),
            gail=gail_config
        )


# Predefined configurations
def get_sac_snake_config() -> ExperimentConfig:
    """Get configuration for SAC agent in Snake-like environment."""
    env_config = EnvironmentConfig(
        name="salp_snake",
        type="salp_snake",
        width=800,
        height=600,
        params={
            "num_food_items": 12,  # Increased from 3 to 12 (4x) for more challenging environment
            "food_reward": 15.0,  # Increased reward for food collection
            "collision_penalty": -30.0,  # Reduced penalty to be less harsh
            "time_penalty": -0.05,  # Reduced time penalty
            "efficiency_bonus": 2.0,  # Increased efficiency bonus
            "forced_breathing": True  # Enable forced breathing mode for nozzle steering focus
        }
    )
    
    agent_config = AgentConfig(
        name="sac_salp",
        type="sac",
        hidden_sizes=[256, 256],
        learning_rate=3e-4,
        batch_size=128,  # Smaller batch size for faster learning
        buffer_size=500000,  # Reduced buffer size
        params={
            "alpha": 0.1,  # Lower entropy coefficient for more focused learning
            "target_entropy": -1.0,  # Adjusted target entropy
            "alpha_lr": 3e-4  # Learning rate for alpha
        }
    )
    
    training_config = TrainingConfig(
        max_episodes=2000,  # Increased for overnight training
        max_steps_per_episode=5000,  # Long episodes for extended learning
        eval_frequency=25,  # More frequent evaluation for better tracking
        save_frequency=50,  # Save models more frequently
        start_training_after=500,  # Start training sooner
        experiment_name="salp_snake_sac_overnight"
    )
    
    return ExperimentConfig(env_config, agent_config, training_config)


def get_sac_gail_snake_config() -> ExperimentConfig:
    """Get configuration for SAC+GAIL agent in Snake-like environment."""
    env_config = EnvironmentConfig(
        name="salp_snake_gail",
        type="salp_snake",
        width=800,
        height=600,
        params={
            "num_food_items": 12,
            "food_reward": 15.0,
            "collision_penalty": -30.0,
            "time_penalty": -0.05,
            "efficiency_bonus": 2.0,
            "forced_breathing": True
        }
    )
    
    agent_config = AgentConfig(
        name="sac_gail_salp",
        type="sac_gail",
        hidden_sizes=[256, 256],
        learning_rate=3e-4,
        batch_size=128,
        buffer_size=500000,
        params={
            "alpha": 0.1,
            "target_entropy": -1.0,
            "alpha_lr": 3e-4,
            "discriminator_lr": 3e-4
        }
    )
    
    training_config = TrainingConfig(
        max_episodes=1500,
        max_steps_per_episode=5000,
        eval_frequency=25,
        save_frequency=50,
        start_training_after=500,
        experiment_name="salp_snake_sac_gail"
    )
    
    gail_config = GAILConfig(
        use_gail=True,
        expert_demos_path="expert_demos/",
        discriminator_lr=3e-4,
        discriminator_update_freq=1,
        reward_env_weight=0.3,
        reward_gail_weight=0.7,
        min_expert_episodes=5,
        load_human_demos=True,
        load_agent_demos=True
    )
    
    return ExperimentConfig(env_config, agent_config, training_config, gail_config)


def get_sac_navigation_config() -> ExperimentConfig:
    """Get configuration for SAC agent in navigation environment."""
    env_config = EnvironmentConfig(
        name="salp_navigation",
        type="salp_navigation",
        width=800,
        height=600,
        params={
            "num_obstacles": 10,
            "goal_reward": 100.0,
            "collision_penalty": -100.0,
            "distance_reward_scale": 1.0
        }
    )
    
    agent_config = AgentConfig(
        name="sac_navigation",
        type="sac",
        hidden_sizes=[512, 512, 256],
        learning_rate=1e-4,
        params={
            "alpha": 0.1,
            "target_entropy": -2.0,
            "alpha_lr": 1e-4
        }
    )
    
    training_config = TrainingConfig(
        max_episodes=5000,
        max_steps_per_episode=2000,
        experiment_name="salp_navigation_sac"
    )
    
    return ExperimentConfig(env_config, agent_config, training_config)


def get_single_food_optimal_config() -> ExperimentConfig:
    """Get configuration for single food optimal navigation experiment with SAC+GAIL."""
    env_config = EnvironmentConfig(
        name="salp_single_food_optimal",
        type="salp_snake",
        width=800,
        height=600,
        params={
            "num_food_items": 1,              # Single food target at a time
            "proximity_reward_weight": 5.0,   # STRONG guidance - clear gradient to food!
            "time_penalty": -0.1,             # Tiny penalty - don't punish exploration too much
            "respawn_food": True,             # Food respawns after collection - continuous!
            "forced_breathing": True,         # Automatic breathing
            "max_steps_without_food": 1500,   # Timer resets on each collection
            "collision_penalty": -10.0,       # Small penalty - don't be too harsh
            "food_reward": 1000.0,            # HUGE reward for each collection!
            "efficiency_bonus": 0.0           # Keep it simple - just reach food
        }
    )
    
    agent_config = AgentConfig(
        name="sac_single_food",
        type="sac",                         # Use pure SAC agent
        hidden_sizes=[256, 256],
        learning_rate=3e-4,
        batch_size=128,
        buffer_size=500000,
        params={
            "alpha": 0.5,                   # Higher entropy = more exploration
            "target_entropy": -0.5,         # Less negative = more randomness
            "alpha_lr": 3e-4
        }
    )
    
    training_config = TrainingConfig(
        max_episodes=1000,
        max_steps_per_episode=1500,          # Matches environment timeout
        eval_frequency=25,
        save_frequency=50,
        start_training_after=500,
        experiment_name="single_food_optimal_navigation"
    )
    
    gail_config = GAILConfig(
        use_gail=False,                     # Disabled - using pure SAC with shaped rewards
        expert_demos_path="expert_demos/",
        discriminator_lr=3e-4,
        discriminator_update_freq=1,
        reward_env_weight=1.0,              # 100% environment rewards (proximity + time)
        reward_gail_weight=0.0,             # No GAIL
        min_expert_episodes=5,
        load_human_demos=False,             # Not needed without GAIL
        load_agent_demos=False
    )
    
    return ExperimentConfig(env_config, agent_config, training_config, gail_config)
