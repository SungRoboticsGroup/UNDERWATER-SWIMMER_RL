"""
SALP agent implementations.

Uses well-tested libraries for core RL algorithms:
- SB3SACAgent: SAC implementation using Stable Baselines3
- Discriminator: For GAIL/imitation learning (custom, but only for rewards)

For testing different architectures, see examples/test_architectures.py
"""

from salp.agents.sb3_sac_agent import SB3SACAgent
from salp.agents.discriminator import Discriminator

__all__ = ["SB3SACAgent", "Discriminator"]
