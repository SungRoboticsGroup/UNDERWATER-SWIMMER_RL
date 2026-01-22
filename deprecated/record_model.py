#!/usr/bin/env python3
"""
Record a trained SALP model with synchronized live viewing and video.

Records EXACTLY what you see - single continuous video for full timesteps.

Usage:
    python record_model.py --watch --timesteps 5000
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from datetime import datetime
import subprocess
import cv2

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch.nn as nn
from torch.distributions import Normal
from salp.environments.salp_snake_env import SalpSnakeEnv


class CustomActor(nn.Module):
    """Actor network for custom .pth models."""
    def __init__(self, obs_dim, action_dim, hidden_sizes=[256, 256]):
        super().__init__()
        all_layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            all_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        all_layers.append(nn.Linear(prev_size, action_dim * 2))
        self.layers = nn.ModuleList(all_layers)
        
    def forward(self, obs, deterministic=False):
        x = obs
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        output = self.layers[-1](x)
        mean, log_std = output.chunk(2, dim=-1)
        
        if deterministic:
            return torch.tanh(mean)
        
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        action = dist.rsample()
        return torch.tanh(action)


def find_latest_model():
    """Find the most recently trained model."""
    models_dir = Path('data/models')
    if not models_dir.exists():
        return None
    
    model_dirs = sorted(models_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
    
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            continue
        sb3_model = model_dir / 'best_model.zip'
        if sb3_model.exists():
            return str(sb3_model)
        custom_model = model_dir / 'best_model.pth'
        if custom_model.exists():
            return str(custom_model)
    return None


def load_model(model_path, env):
    """Load either SB3 or custom model."""
    model_path = Path(model_path)
    
    if model_path.suffix == '.zip':
        from stable_baselines3 import SAC
        model = SAC.load(str(model_path))
        return model, 'sb3'
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        model = CustomActor(obs_dim, action_dim, hidden_sizes=[256, 256])
        model.load_state_dict(checkpoint['policy_state_dict'])
        model.eval()
        return model, 'custom'


def get_action(model, model_type, obs):
    """Get action from model."""
    if model_type == 'sb3':
        action, _ = model.predict(obs, deterministic=True)
        return action
    else:
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_tensor = model(obs_tensor, deterministic=True)
            return action_tensor.squeeze(0).numpy()


def record_model(model_path: str, timesteps: int = 5000, watch: bool = False):
    """Record model - what you see is what gets saved."""
    print("=" * 70)
    print("SALP MODEL RECORDING")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Timesteps: {timesteps}")
    print(f"Watch: {watch}")
    print("=" * 70 + "\n")
    
    # Create ONE environment
    render_mode = 'human' if watch else 'rgb_array'
    env = SalpSnakeEnv(
        render_mode=render_mode,
        num_food_items=1,
        respawn_food=True,
        forced_breathing=True,
        max_steps_without_food=10000
    )
    
    # Load model
    model, model_type = load_model(model_path, env)
    print(f"✓ {model_type.upper()} model loaded\n")
    
    # Setup video writer
    output_dir = Path("assets/videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = output_dir / f"salp_{timestamp}.mp4"
    
    # Video writer setup (will be initialized on first frame)
    video_writer = None
    
    # Run
    obs, _ = env.reset()
    total_reward = 0
    total_food = 0
    collisions = 0
    
    print(f"🎬 Recording {timesteps} steps...")
    if watch:
        print("   (Watch the pygame window - that's what's being recorded!)")
    print(f"   (Continues through collisions)\n")
    
    for step in range(timesteps):
        # Get action
        action = get_action(model, model_type, obs)
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_food = info.get('food_collected', 0)
        
        # Get frame for recording
        frame = env.render()
        if frame is not None and not watch:  # rgb_array mode returns frame
            # Initialize video writer on first frame
            if video_writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))
            
            # Write frame (convert RGB to BGR for opencv)
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Handle collision - reset but keep recording!
        if terminated:
            collisions += 1
            obs, _ = env.reset()
        
        # Progress
        if (step + 1) % 500 == 0:
            print(f"   Step {step + 1}/{timesteps} - Food: {total_food}, Collisions: {collisions}")
    
    # Cleanup
    env.close()
    if video_writer:
        video_writer.release()
    
    print(f"\n✓ Recording complete!")
    print(f"   Total steps: {timesteps}")
    print(f"   Food collected: {total_food}")
    print(f"   Collisions: {collisions}")
    print(f"   Total reward: {total_reward:.1f}\n")
    
    # Convert to GIF if video was saved
    if video_path.exists() and not watch:
        file_size = video_path.stat().st_size / 1024 / 1024
        print(f"📹 Video saved: {video_path}")
        print(f"   Size: {file_size:.1f} MB")
        
        print(f"\n🎨 Converting to GIF...")
        gif_path = video_path.with_suffix('.gif')
        
        try:
            subprocess.run([
                'ffmpeg', '-i', str(video_path),
                '-vf', 'fps=10,scale=400:-1:flags=lanczos',
                '-y', str(gif_path)
            ], check=True, capture_output=True)
            
            gif_size = gif_path.stat().st_size / 1024 / 1024
            print(f"✓ GIF created: {gif_path}")
            print(f"   Size: {gif_size:.1f} MB")
            return video_path, gif_path
        except:
            print(f"⚠️  GIF conversion skipped (ffmpeg not available)")
            return video_path, None
    elif watch:
        print("⚠️  Watch mode - no video saved (only live display)")
        return None, None
    else:
        return video_path, None


def main():
    parser = argparse.ArgumentParser(description="Record SALP model")
    parser.add_argument('--model', '-m', help='Model path (auto-detects if not specified)')
    parser.add_argument('--timesteps', '-t', type=int, default=5000, help='Timesteps to record')
    parser.add_argument('--watch', '-w', action='store_true', help='Show live view (no video saved)')
    args = parser.parse_args()
    
    # Find model
    if args.model:
        model_path = args.model
    else:
        print("Searching for latest model...")
        model_path = find_latest_model()
        if not model_path:
            print("Error: No models found")
            sys.exit(1)
        print(f"Found: {model_path}\n")
    
    if not Path(model_path).exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    
    # Record
    video, gif = record_model(model_path, args.timesteps, watch=args.watch)
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    if video:
        print(f"MP4: {video}")
    if gif:
        print(f"GIF: {gif}")
    print("=" * 70)


if __name__ == "__main__":
    main()
