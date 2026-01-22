"""
Generate trajectory visualization from trained SALP model.
Uses the best model from a training run to create a visual trajectory plot.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import torch

from config.base_config import get_sac_snake_config
from agents.sac_agent import SACAgent
from environments.salp_snake_env import SalpSnakeEnv


def load_best_model(experiment_name: str):
    """Load the best model from a training experiment."""
    config = get_sac_snake_config()
    config.training.experiment_name = experiment_name
    
    # Create environment to get dimensions
    env = SalpSnakeEnv(
        render_mode=None,
        width=config.environment.width,
        height=config.environment.height,
        **config.environment.params
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
    
    # Load best model
    model_dir = os.path.join(config.training.model_dir, experiment_name)
    model_path = os.path.join(model_dir, "best_model.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found at {model_path}")
    
    agent.load(model_path)
    print(f"✓ Loaded best model from {model_path}")
    
    return agent, env, config


def generate_trajectory(agent, env, max_steps=5000):
    """Generate a single trajectory using the trained agent."""
    obs, _ = env.reset()
    
    trajectory = {
        'positions': [],
        'actions': [],
        'rewards': [],
        'food_positions': [],
        'food_collected': []
    }
    
    episode_reward = 0
    steps = 0
    food_collected = 0
    
    done = False
    truncated = False
    
    while not done and not truncated and steps < max_steps:
        # Record current position
        trajectory['positions'].append(env.robot_pos.copy())
        trajectory['food_positions'].append([f.copy() for f in env.food_positions])
        
        # Select action
        action = agent.select_action(obs, deterministic=True)
        trajectory['actions'].append(action.copy())
        
        # Take step
        obs, reward, done, truncated, info = env.step(action)
        
        trajectory['rewards'].append(reward)
        episode_reward += reward
        steps += 1
        
        # Track food collection
        new_food_collected = info.get('food_collected', 0)
        if new_food_collected > food_collected:
            trajectory['food_collected'].append(len(trajectory['positions']) - 1)
            food_collected = new_food_collected
    
    print(f"Trajectory: {steps} steps, {food_collected} food collected, reward: {episode_reward:.1f}")
    
    return trajectory, episode_reward, food_collected, steps


def plot_trajectory_with_env(agent, env, trajectory, save_path='salp_trajectory.png'):
    """Create a visualization using actual environment rendering with trajectory overlay."""
    import pygame
    
    # Set up environment for rendering
    env_vis = SalpSnakeEnv(
        render_mode="rgb_array",  # Get RGB array instead of displaying
        width=800,
        height=600,
        num_food_items=12,
        random_food_count=False,
        respawn_food=False,
        forced_breathing=True
    )
    
    # Reset to final state
    obs, _ = env_vis.reset()
    
    # Run through the trajectory to get to final state
    positions = trajectory['positions']
    for i, action in enumerate(trajectory['actions']):
        obs, _, done, truncated, _ = env_vis.step(action)
        if done or truncated:
            break
    
    # Get the rendered frame
    frame = env_vis.render()
    
    # Convert to matplotlib figure
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(frame, origin='upper')
    ax.set_xlim(0, 800)
    ax.set_ylim(600, 0)  # Flip y-axis to match pygame coords
    ax.axis('off')
    
    # Overlay trajectory path
    positions_array = np.array(positions)
    ax.plot(positions_array[:, 0], positions_array[:, 1], 
            'c-', linewidth=2, alpha=0.7, label='Path')
    
    # Mark food collection points with bright stars
    for idx in trajectory['food_collected']:
        pos = positions[idx]
        ax.plot(pos[0], pos[1], '*', color='gold', markersize=20, 
                markeredgecolor='orange', markeredgewidth=2, zorder=25,
                label='Food Collected' if idx == trajectory['food_collected'][0] else '')
    
    # Mark start position
    ax.plot(positions[0][0], positions[0][1], 'go', markersize=10, 
            markeredgecolor='darkgreen', markeredgewidth=2, label='Start', zorder=20)
    
    # Add info
    info_text = f'Food: {len(trajectory["food_collected"])}/{env_vis.base_num_food_items} | Steps: {len(positions)} | Reward: {sum(trajectory["rewards"]):.1f}'
    ax.text(400, 30, info_text, ha='center', fontsize=16, color='white',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, pad=1.0),
           transform=ax.transData)
    
    ax.legend(loc='upper right', fontsize=12, framealpha=0.8)
    
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved trajectory visualization to {save_path}")
    plt.close()
    
    env_vis.close()


def main():
    """Generate trajectory visualization from the latest training run."""
    import sys
    
    # Get experiment name from command line or use latest
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
    else:
        # Find the latest training experiment
        model_dir = "models"
        experiments = [d for d in os.listdir(model_dir) if d.startswith("salp_training_")]
        if not experiments:
            print("❌ No training experiments found!")
            print("Please specify experiment name: python -m scripts.generate_trajectory_visualization <experiment_name>")
            return
        
        # Sort by timestamp (experiment names are like salp_training_YYYYMMDD_HHMMSS)
        experiments.sort(reverse=True)
        experiment_name = experiments[0]
        print(f"Using latest experiment: {experiment_name}")
    
    try:
        # Load best model
        agent, env, config = load_best_model(experiment_name)
        
        # Generate trajectory
        print("\nGenerating trajectory...")
        trajectory, reward, food_collected, steps = generate_trajectory(agent, env)
        
        # Create visualization with environment rendering
        output_path = f"salp_trajectory_{experiment_name}.png"
        plot_trajectory_with_env(agent, env, trajectory, output_path)
        
        print(f"\n✅ Trajectory visualization complete!")
        print(f"   Experiment: {experiment_name}")
        print(f"   Steps: {steps}")
        print(f"   Food collected: {food_collected}")
        print(f"   Total reward: {reward:.1f}")
        print(f"   Output: {output_path}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"   Make sure the training has created a best_model.pth file")
    except Exception as e:
        print(f"❌ Error generating trajectory: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
