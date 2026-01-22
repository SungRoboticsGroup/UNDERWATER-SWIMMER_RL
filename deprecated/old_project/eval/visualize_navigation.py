#!/usr/bin/env python3
"""
Visualize Navigation Results
Generates three visualizations from saved trajectory data:
1. Raw trajectories
2. Spline-fitted trajectories  
3. Heatmap of spline trajectories
"""

import os
import sys
import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_trajectory_data(input_path: str) -> Tuple[List[Dict], Dict]:
    """Load trajectory data from pickle file."""
    print(f"\nLoading trajectory data from: {input_path}")
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    results = data['results']
    metadata = {
        'start_pos': tuple(data['start_pos']),
        'goal_pos': tuple(data['goal_pos']),
        'env_width': data['env_width'],
        'env_height': data['env_height'],
        'goal_radius': data.get('goal_radius', 50),
        'optimal_distance': data['optimal_distance'],
        'timestamp': data.get('timestamp', 'unknown')
    }
    
    print(f"✓ Loaded {len(results)} trials")
    print(f"  Start: {metadata['start_pos']}, Goal: {metadata['goal_pos']}")
    print(f"  Success rate: {sum(r['success'] for r in results) / len(results) * 100:.1f}%\n")
    
    return results, metadata


def plot_raw_trajectories(results: List[Dict], metadata: Dict, output_path: str):
    """1. Plot raw trajectories (no splines)."""
    print("Generating raw trajectories visualization...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    start_pos = metadata['start_pos']
    goal_pos = metadata['goal_pos']
    env_width = metadata['env_width']
    env_height = metadata['env_height']
    goal_radius = metadata['goal_radius']
    
    # Background
    ax.set_facecolor('#f5f5f5')
    
    # Tank boundaries
    tank_margin = 50
    boundary = patches.Rectangle((tank_margin, tank_margin), 
                                env_width - 2*tank_margin, 
                                env_height - 2*tank_margin,
                                linewidth=2, edgecolor='gray', 
                                facecolor='white', alpha=0.5)
    ax.add_patch(boundary)
    
    # Optimal path
    ax.plot([start_pos[0], goal_pos[0]], [start_pos[1], goal_pos[1]], 
           'k--', linewidth=2, alpha=0.4, label='Optimal Path', zorder=1)
    
    # Color by path efficiency (path_ratio)
    path_ratios = np.array([r['path_ratio'] for r in results])
    
    # Normalize path ratios for colormap (RdYlGn_r: green=good/low, red=bad/high)
    vmin, vmax = path_ratios.min(), path_ratios.max()
    norm = plt.cm.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn_r  # Green for efficient, red for inefficient
    
    for i, result in enumerate(results):
        positions = result['positions']
        color = cmap(norm(result['path_ratio']))
        ax.plot(positions[:, 0], positions[:, 1], 
               color=color, linewidth=1.0, alpha=0.6, zorder=2)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Path Efficiency Ratio', rotation=270, labelpad=20, fontsize=11)
    
    # Start and goal
    ax.plot(start_pos[0], start_pos[1], 'go', 
           markersize=15, markeredgecolor='darkgreen', 
           markeredgewidth=2, label='Start', zorder=10)
    ax.plot(goal_pos[0], goal_pos[1], 'r*', 
           markersize=20, markeredgecolor='darkred', 
           markeredgewidth=2, label='Goal', zorder=10)
    
    # Goal radius
    goal_circle = plt.Circle(goal_pos, goal_radius, 
                            color='red', fill=False, 
                            linestyle='--', linewidth=2, alpha=0.3)
    ax.add_patch(goal_circle)
    
    # Stats
    success_rate = sum(r['success'] for r in results) / len(results)
    avg_path_ratio = np.mean([r['path_ratio'] for r in results])
    
    ax.set_xlim(0, env_width)
    ax.set_ylim(0, env_height)
    ax.set_xlabel('X Position (pixels)', fontsize=12)
    ax.set_ylabel('Y Position (pixels)', fontsize=12)
    ax.set_title(f'Raw Trajectories (N={len(results)} trials)\n'
                f'Success Rate: {success_rate*100:.1f}% | '
                f'Path Efficiency: {avg_path_ratio:.2f}x optimal', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved to: {output_path}\n")
    plt.close()


def plot_spline_trajectories(results: List[Dict], metadata: Dict, output_path: str):
    """2. Plot spline-fitted trajectories."""
    print("Generating spline trajectories visualization...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    start_pos = metadata['start_pos']
    goal_pos = metadata['goal_pos']
    env_width = metadata['env_width']
    env_height = metadata['env_height']
    goal_radius = metadata['goal_radius']
    
    # Background
    ax.set_facecolor('white')
    
    # Tank boundaries
    tank_margin = 50
    boundary = patches.Rectangle((tank_margin, tank_margin), 
                                env_width - 2*tank_margin, 
                                env_height - 2*tank_margin,
                                linewidth=2, edgecolor='gray', 
                                facecolor='none', alpha=0.3)
    ax.add_patch(boundary)
    
    # Optimal path
    ax.plot([start_pos[0], goal_pos[0]], [start_pos[1], goal_pos[1]], 
           'k--', linewidth=2, alpha=0.3, label='Optimal Path', zorder=1)
    
    # Calculate spline path ratios
    spline_path_lengths = []
    spline_path_ratios = []
    optimal_distance = metadata['optimal_distance']
    
    # First pass: calculate spline path lengths
    for result in results:
        positions = result['positions']
        
        # Remove consecutive duplicates
        unique_positions = [positions[0]]
        for j in range(1, len(positions)):
            if not np.array_equal(positions[j], positions[j-1]):
                unique_positions.append(positions[j])
        unique_array = np.array(unique_positions)
        
        # Fit spline with high smoothing
        if len(unique_array) >= 4:
            try:
                # Downsample to ~20 control points
                downsample_factor = max(1, len(unique_array) // 20)
                downsampled = unique_array[::downsample_factor]
                
                # Always include last point
                if not np.array_equal(downsampled[-1], unique_array[-1]):
                    downsampled = np.vstack([downsampled, unique_array[-1]])
                
                if len(downsampled) >= 4:
                    # Smooth spline
                    tck, u = splprep([downsampled[:, 0], downsampled[:, 1]], s=500.0, k=3)
                    u_fine = np.linspace(0, 1, 500)
                    spline_points = splev(u_fine, tck)
                    spline_positions = np.column_stack(spline_points)
                    
                    # Calculate spline path length
                    spline_length = np.sum(np.linalg.norm(np.diff(spline_positions, axis=0), axis=1))
                    spline_path_lengths.append(spline_length)
                    spline_path_ratios.append(spline_length / optimal_distance)
                else:
                    spline_path_lengths.append(None)
                    spline_path_ratios.append(None)
            except:
                spline_path_lengths.append(None)
                spline_path_ratios.append(None)
        else:
            spline_path_lengths.append(None)
            spline_path_ratios.append(None)
    
    # Use raw path ratios for coloring (since they're already computed)
    path_ratios = np.array([r['path_ratio'] for r in results])
    
    # Normalize for colormap
    vmin, vmax = path_ratios.min(), path_ratios.max()
    norm = plt.cm.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_efficiency = plt.cm.RdYlGn_r
    
    # Second pass: plot splines
    spline_count = 0
    for i, result in enumerate(results):
        if spline_path_ratios[i] is None:
            continue
            
        positions = result['positions']
        color = cmap_efficiency(norm(result['path_ratio']))
        
        # Remove consecutive duplicates
        unique_positions = [positions[0]]
        for j in range(1, len(positions)):
            if not np.array_equal(positions[j], positions[j-1]):
                unique_positions.append(positions[j])
        unique_array = np.array(unique_positions)
        
        # Fit spline with high smoothing
        if len(unique_array) >= 4:
            try:
                # Downsample to ~20 control points
                downsample_factor = max(1, len(unique_array) // 20)
                downsampled = unique_array[::downsample_factor]
                
                # Always include last point
                if not np.array_equal(downsampled[-1], unique_array[-1]):
                    downsampled = np.vstack([downsampled, unique_array[-1]])
                
                if len(downsampled) >= 4:
                    # Smooth spline
                    tck, u = splprep([downsampled[:, 0], downsampled[:, 1]], s=500.0, k=3)
                    u_fine = np.linspace(0, 1, 500)
                    spline_points = splev(u_fine, tck)
                    spline_x, spline_y = spline_points
                    
                    ax.plot(spline_x, spline_y, 
                           color=color, linewidth=2.0, alpha=0.75, zorder=2)
                    spline_count += 1
            except:
                pass
    
    # Print spline stats
    valid_spline_ratios = [r for r in spline_path_ratios if r is not None]
    if valid_spline_ratios:
        avg_spline_ratio = np.mean(valid_spline_ratios)
        avg_raw_ratio = np.mean(path_ratios)
        print(f"  Spline path ratio: {avg_spline_ratio:.3f}x optimal")
        print(f"  Raw path ratio: {avg_raw_ratio:.3f}x optimal")
        print(f"  Difference: {abs(avg_spline_ratio - avg_raw_ratio):.3f}x\n")
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_efficiency, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Path Efficiency Ratio', rotation=270, labelpad=20, fontsize=11)
    
    # Start and goal
    ax.plot(start_pos[0], start_pos[1], 'go', 
           markersize=15, markeredgecolor='darkgreen', 
           markeredgewidth=2, label='Start', zorder=10)
    ax.plot(goal_pos[0], goal_pos[1], 'r*', 
           markersize=20, markeredgecolor='darkred', 
           markeredgewidth=2, label='Goal', zorder=10)
    
    # Goal radius
    goal_circle = plt.Circle(goal_pos, goal_radius, 
                            color='red', fill=False, 
                            linestyle='--', linewidth=2, alpha=0.3)
    ax.add_patch(goal_circle)
    
    # Stats
    success_rate = sum(r['success'] for r in results) / len(results)
    avg_path_ratio = np.mean([r['path_ratio'] for r in results])
    
    ax.set_xlim(0, env_width)
    ax.set_ylim(0, env_height)
    ax.set_xlabel('X Position (pixels)', fontsize=12)
    ax.set_ylabel('Y Position (pixels)', fontsize=12)
    ax.set_title(f'Spline Trajectories (N={spline_count} trials)\n'
                f'Success Rate: {success_rate*100:.1f}% | '
                f'Path Efficiency: {avg_path_ratio:.2f}x optimal', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved to: {output_path}\n")
    plt.close()


def plot_spline_heatmap(results: List[Dict], metadata: Dict, output_path: str):
    """3. Plot heatmap of spline trajectories."""
    print("Generating spline heatmap visualization...")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    start_pos = metadata['start_pos']
    goal_pos = metadata['goal_pos']
    env_width = metadata['env_width']
    env_height = metadata['env_height']
    goal_radius = metadata['goal_radius']
    
    # Collect spline points
    positions = []
    for result in results:
        positions_array = result['positions']
        
        # Remove consecutive duplicates
        unique_positions = [positions_array[0]]
        for i in range(1, len(positions_array)):
            if not np.array_equal(positions_array[i], positions_array[i-1]):
                unique_positions.append(positions_array[i])
        unique_array = np.array(unique_positions)
        
        # Fit spline
        if len(unique_array) >= 4:
            try:
                tck, u = splprep([unique_array[:, 0], unique_array[:, 1]], s=5.0, k=3)
                u_fine = np.linspace(0, 1, len(unique_array) * 10)
                spline_points = splev(u_fine, tck)
                spline_array = np.column_stack(spline_points)
                positions.extend(spline_array)
            except:
                positions.extend(unique_array)
        else:
            positions.extend(unique_array)
    
    positions = np.array(positions)
    
    # Create 2D histogram
    bins_x = 200
    bins_y = int(bins_x * env_height / env_width)
    
    heatmap, xedges, yedges = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=[bins_x, bins_y],
        range=[[0, env_width], [0, env_height]]
    )
    
    # Smooth
    heatmap = gaussian_filter(heatmap, sigma=2.0)
    
    # Colormap
    colors = ['#000033', '#000080', '#0000FF', '#00FFFF', '#00FF00', 
             '#FFFF00', '#FF8800', '#FF0000', '#FF0080']
    cmap = LinearSegmentedColormap.from_list('traffic_flow', colors, N=256)
    
    # Plot heatmap
    extent = [0, env_width, 0, env_height]
    im = ax.imshow(heatmap.T, extent=extent, origin='lower', 
                  cmap=cmap, aspect='auto', alpha=1.0, interpolation='bilinear')
    
    # Optimal path
    ax.plot([start_pos[0], goal_pos[0]], [start_pos[1], goal_pos[1]], 
           'w--', linewidth=3, alpha=0.8, label='Optimal Path')
    
    # Start and goal
    ax.plot(start_pos[0], start_pos[1], 'go', 
           markersize=20, markeredgecolor='darkgreen', 
           markeredgewidth=3, label='Start', zorder=10)
    ax.plot(goal_pos[0], goal_pos[1], 'r*', 
           markersize=25, markeredgecolor='darkred', 
           markeredgewidth=2, label='Goal', zorder=10)
    
    # Goal radius
    goal_circle = plt.Circle(goal_pos, goal_radius, 
                            color='red', fill=False, 
                            linestyle='--', linewidth=2, alpha=0.5)
    ax.add_patch(goal_circle)
    
    # Tank boundaries
    tank_margin = 50
    boundary = patches.Rectangle((tank_margin, tank_margin), 
                                env_width - 2*tank_margin, 
                                env_height - 2*tank_margin,
                                linewidth=2, edgecolor='cyan', 
                                facecolor='none', alpha=0.3)
    ax.add_patch(boundary)
    
    # Stats
    success_rate = sum(r['success'] for r in results) / len(results)
    avg_path_length = np.mean([r['path_length'] for r in results])
    std_path_length = np.std([r['path_length'] for r in results])
    avg_straightness = np.mean([r['straightness'] for r in results])
    std_straightness = np.std([r['straightness'] for r in results])
    optimal_distance = metadata['optimal_distance']
    
    stats_text = (
        f"Trials: {len(results)}\n"
        f"Success Rate: {success_rate*100:.1f}%\n"
        f"Avg Path Length: {avg_path_length:.0f} ± {std_path_length:.0f} px\n"
        f"Straightness: {avg_straightness:.3f} ± {std_straightness:.3f}\n"
        f"Optimal Distance: {optimal_distance:.0f} px"
    )
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
           color='white', family='monospace')
    
    ax.set_xlim(0, env_width)
    ax.set_ylim(0, env_height)
    ax.set_xlabel('X Position (pixels)', fontsize=12)
    ax.set_ylabel('Y Position (pixels)', fontsize=12)
    ax.set_title(f'Spline Heatmap (N={len(results)} trials)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Visit Density', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {output_path}\n")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Navigation Results')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to trajectory data file (.pkl)')
    parser.add_argument('--output-dir', type=str, default='eval/results',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Load data
    results, metadata = load_trajectory_data(args.data)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate all three visualizations
    print("="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60 + "\n")
    
    # 1. Raw trajectories
    raw_path = f"{args.output_dir}/trajectories_{timestamp}.png"
    plot_raw_trajectories(results, metadata, raw_path)
    
    # 2. Spline trajectories
    spline_path = f"{args.output_dir}/splines_{timestamp}.png"
    plot_spline_trajectories(results, metadata, spline_path)
    
    # 3. Spline heatmap
    heatmap_path = f"{args.output_dir}/heatmap_{timestamp}.png"
    plot_spline_heatmap(results, metadata, heatmap_path)
    
    print("="*60)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*60)
    print(f"\n1. Raw trajectories: {raw_path}")
    print(f"2. Spline trajectories: {spline_path}")
    print(f"3. Spline heatmap: {heatmap_path}\n")


if __name__ == "__main__":
    main()
