"""SALP core base classes."""

from salp.core.base_agent import BaseNetwork, soft_update, hard_update

__all__ = ["BaseNetwork", "soft_update", "hard_update"]
