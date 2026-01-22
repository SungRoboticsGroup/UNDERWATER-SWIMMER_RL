"""
Configuration loader for SALP experiments.
Loads configs from YAML files with sensible defaults.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """Complete experiment configuration loaded from YAML."""
    
    environment: Dict[str, Any]
    agent: Dict[str, Any]
    training: Dict[str, Any]
    gail: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_yaml(cls, config_path: str, overrides: Optional[Dict] = None) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file
            overrides: Optional dict of values to override
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Apply overrides if provided
        if overrides:
            config_dict = _deep_update(config_dict, overrides)
        
        return cls(
            environment=config_dict.get('environment', {}),
            agent=config_dict.get('agent', {}),
            training=config_dict.get('training', {}),
            gail=config_dict.get('gail')
        )
    
    @classmethod
    def from_preset(cls, preset: str = 'defaults', overrides: Optional[Dict] = None) -> 'Config':
        """
        Load configuration from preset name.
        
        Args:
            preset: Preset name (defaults, single_food, sac_gail)
            overrides: Optional dict of values to override
        """
        # Look in configs/ directory
        config_path = Path('configs') / f'{preset}.yaml'
        
        if not config_path.exists():
            # Fallback to defaults
            config_path = Path('configs') / 'defaults.yaml'
        
        return cls.from_yaml(str(config_path), overrides)
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a config value with dot notation support."""
        section_dict = getattr(self, section, {})
        return section_dict.get(key, default)


def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """Recursively update nested dictionaries."""
    result = base_dict.copy()
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_file: Optional[str] = None, **overrides) -> Config:
    """
    Convenience function to load configuration.
    
    Args:
        config_file: Path to config file or preset name (defaults, single_food, etc.)
        **overrides: Key-value pairs to override config values
    
    Returns:
        Config object
    
    Examples:
        # Load default config
        config = load_config()
        
        # Load from preset
        config = load_config('single_food')
        
        # Load with overrides
        config = load_config('defaults', training={'max_episodes': 2000})
        
        # Load specific file
        config = load_config('path/to/custom.yaml')
    """
    if config_file is None:
        config_file = 'defaults'
    
    # Check if it's a file path or preset name
    path = Path(config_file)
    if path.suffix == '.yaml' and path.exists():
        return Config.from_yaml(str(path), overrides if overrides else None)
    else:
        return Config.from_preset(config_file, overrides if overrides else None)
