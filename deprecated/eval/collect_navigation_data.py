#!/usr/bin/env python3
"""
Collect Navigation Trajectory Data
Runs navigation trials and saves trajectory data for later visualization.
Separates data collection from visualization.
"""

import os
import sys
import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from scipy.interpolate import splprep, splev

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC
from src.salp.environments.salp_snake_env import SalpSnakeEnv


class NavigationDataCollector:
    """Collects navigation trajectory data from multiple trials."""
    
    def __init__(self, model_path: str,
                 start_pos: Tuple[float, float] = (100, 300),
                 goal_pos: Tuple[float, float] = (700, 300),
                 env_width: int = 800,
                 env_height: int = 600):
        """
        Initialize the data collector.
        
        Args:
            model_path: Path to trained model (.zip for SB3)
            start_pos: Starting position (x, y)
            goal_pos: Goal position (x, y)
            env_width: Environment width in pixels
            env_height: Environment height in pixels
        """
        self.model_path = model_path
        self.start_pos = np.array(start_pos, dtype=float)
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.env_width = env_width
        self.env_height = env_height
        self.goal_radius = 50
        
        # Calculate optimal distance
        self.optimal_distance = np.linalg.norm(self.goal_pos - self.start_pos)
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = SAC.load(model_path)
        print("✓ Model loaded successfully")
    
    def create_environment(self) -> SalpSnakeEnv:
        """Create a clean navigation environment."""
        env = SalpSnakeEnv(
            render_mode=None,
            width=self.env_width,
            height=self.env_height,
            num_food_items=1,
            forced_breathing=True,
            respawn_food=False,
            max_steps_without_food=3000
        )
        return env
    
    def run_single_trial(self, env: SalpSnakeEnv, max_steps: int = 3000) -> Dict:
        """Run a single navigation trial and collect trajectory data."""
        # Reset environment and set custom start position
        obs, _ = env.reset()
        env.robot_pos = self.start_pos.copy()
        env.robot_velocity = np.array([0.0, 0.0], dtype=float)
        
        # Randomize initial orientation
        env.robot_angle = np.random.uniform(-np.pi, np.pi)
        env.robot_angular_velocity = 0.0
        
        # Place food at goal position
        env.food_positions = [self.goal_pos.copy()]
        env.steps_since_food = 0
        
        # Update observation
        obs = env._get_extended_observation()
        
        # Record trajectory
        positions = [self.start_pos.copy()]
        velocities = []
        actions_taken = []
        
        steps = 0
        while steps < max_steps:
            # Select action
            action, _ = self.model.predict(obs, deterministic=True)
            actions_taken.append(action.copy())
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            
            # Record state
            positions.append(env.robot_pos.copy())
            velocities.append(env.robot_velocity.copy())
            
            steps += 1
            
            # Check if goal reached
            distance_to_goal = np.linalg.norm(env.robot_pos - self.goal_pos)
            if distance_to_goal < self.goal_radius:
                break
        
        # Calculate metrics
        positions_array = np.array(positions)
        
        # Remove consecutive duplicates
        unique_positions = [positions_array[0]]
        for i in range(1, len(positions_array)):
            if not np.array_equal(positions_array[i], positions_array[i-1]):
                unique_positions.append(positions_array[i])
        unique_positions_array = np.array(unique_positions)
        
        # Path length
        path_length_to_stop = np.sum(np.linalg.norm(np.diff(unique_positions_array, axis=0), axis=1))
        final_distance = np.linalg.norm(positions_array[-1] - self.goal_pos)
        path_length = path_length_to_stop + final_distance
        
        # Success
        success = final_distance < self.goal_radius
        
        # Metrics
        path_ratio = path_length / self.optimal_distance if self.optimal_distance > 0 else float('inf')
        straightness = self.optimal_distance / path_length if path_length > 0 else 0
        
        # Calculate spline path ratio
        spline_path_length = None
        spline_path_ratio = None
        if len(unique_positions_array) >= 4:
            try:
                # Downsample to ~20 control points
                downsample_factor = max(1, len(unique_positions_array) // 20)
                downsampled = unique_positions_array[::downsample_factor]
                
                # Always include last point
                if not np.array_equal(downsampled[-1], unique_positions_array[-1]):
                    downsampled = np.vstack([downsampled, unique_positions_array[-1]])
                
                if len(downsampled) >= 4:
                    # Smooth spline
                    tck, u = splprep([downsampled[:, 0], downsampled[:, 1]], s=500.0, k=3)
                    u_fine = np.linspace(0, 1, 500)
                    spline_points = splev(u_fine, tck)
                    spline_positions = np.column_stack(spline_points)
                    
                    # Calculate spline path length (to last spline point)
                    spline_path_to_stop = np.sum(np.linalg.norm(np.diff(spline_positions, axis=0), axis=1))
                    
                    # Add distance from last spline point to goal (like raw path)
                    spline_final_pos = spline_positions[-1]
                    spline_final_distance = np.linalg.norm(spline_final_pos - self.goal_pos)
                    spline_path_length = spline_path_to_stop + spline_final_distance
                    spline_path_ratio = spline_path_length / self.optimal_distance if self.optimal_distance > 0 else float('inf')
            except:
                pass
        
        # Lateral deviation
        lateral_deviation = self._calculate_lateral_deviation(positions_array)
        
        # Area covered
        x_min, y_min = positions_array.min(axis=0)
        x_max, y_max = positions_array.max(axis=0)
        area_covered = (x_max - x_min) * (y_max - y_min)
        optimal_area = self.optimal_distance * (self.goal_radius * 2)
        area_ratio = area_covered / optimal_area if optimal_area > 0 else float('inf')
        
        return {
            'positions': positions_array,
            'velocities': velocities,
            'actions': actions_taken,
            'steps': steps,
            'path_length': path_length,
            'path_ratio': path_ratio,
            'spline_path_length': spline_path_length,
            'spline_path_ratio': spline_path_ratio,
            'straightness': straightness,
            'final_distance': final_distance,
            'success': success,
            'lateral_deviation': lateral_deviation,
            'area_covered': area_covered,
            'area_ratio': area_ratio,
            'x_range': x_max - x_min,
            'y_range': y_max - y_min
        }
    
    def _calculate_lateral_deviation(self, positions: np.ndarray) -> float:
        """Calculate mean perpendicular distance from optimal straight line."""
        if len(positions) < 2:
            return 0.0
        
        optimal_vector = self.goal_pos - self.start_pos
        optimal_length = np.linalg.norm(optimal_vector)
        
        if optimal_length == 0:
            return 0.0
        
        optimal_unit = optimal_vector / optimal_length
        
        deviations = []
        for pos in positions:
            pos_vector = pos - self.start_pos
            projection_length = np.dot(pos_vector, optimal_unit)
            projection = self.start_pos + projection_length * optimal_unit
            deviation = np.linalg.norm(pos - projection)
            deviations.append(deviation)
        
        return np.mean(deviations)
    
    def collect_data(self, num_trials: int = 100, max_steps: int = 3000) -> List[Dict]:
        """Collect trajectory data from multiple trials."""
        print(f"\n{'='*60}")
        print(f"Collecting navigation data from {num_trials} trials...")
        print(f"Start: {self.start_pos}, Goal: {self.goal_pos}")
        print(f"Optimal distance: {self.optimal_distance:.1f} px")
        print(f"{'='*60}\n")
        
        env = self.create_environment()
        results = []
        
        for i in range(num_trials):
            print(f"Trial {i+1}/{num_trials}...", end=' ')
            
            trial_result = self.run_single_trial(env, max_steps=max_steps)
            results.append(trial_result)
            
            status = "✓ SUCCESS" if trial_result['success'] else "✗ FAILED"
            spline_ratio_str = f"{trial_result['spline_path_ratio']:.2f}" if trial_result['spline_path_ratio'] is not None else "N/A"
            print(f"{status} | Steps: {trial_result['steps']} | "
                  f"Path ratio: {trial_result['path_ratio']:.2f} | "
                  f"Spline ratio: {spline_ratio_str} | "
                  f"Final dist: {trial_result['final_distance']:.1f}px")
        
        env.close()
        
        # Print summary
        success_rate = sum(r['success'] for r in results) / len(results)
        avg_path_length = np.mean([r['path_length'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        
        print(f"\n{'='*60}")
        print(f"DATA COLLECTION COMPLETE")
        print(f"{'='*60}")
        print(f"Success Rate: {success_rate*100:.1f}% ({sum(r['success'] for r in results)}/{len(results)})")
        print(f"Avg Path Length: {avg_path_length:.1f} px")
        print(f"Avg Steps: {avg_steps:.1f}")
        print(f"{'='*60}\n")
        
        return results
    
    def save_data(self, results: List[Dict], output_path: str):
        """Save trajectory data to pickle file."""
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Package data with metadata
        data = {
            'results': results,
            'start_pos': self.start_pos.tolist(),
            'goal_pos': self.goal_pos.tolist(),
            'env_width': self.env_width,
            'env_height': self.env_height,
            'goal_radius': self.goal_radius,
            'optimal_distance': self.optimal_distance,
            'model_path': self.model_path,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Trajectory data saved to: {output_path}")
        print(f"  ({len(results)} trials)")
        
        return output_path


def load_trajectory_data(input_path: str) -> Tuple[List[Dict], Dict]:
    """
    Load trajectory data from pickle file.
    
    Args:
        input_path: Path to pickle file
        
    Returns:
        Tuple of (results list, metadata dict)
    """
    print(f"Loading trajectory data from: {input_path}")
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    results = data['results']
    metadata = {
        'start_pos': tuple(data['start_pos']),
        'goal_pos': tuple(data['goal_pos']),
        'env_width': data['env_width'],
        'env_height': data['env_height'],
        'goal_radius': data['goal_radius'],
        'optimal_distance': data['optimal_distance'],
        'model_path': data.get('model_path', 'unknown'),
        'timestamp': data.get('timestamp', 'unknown')
    }
    
    print(f"✓ Loaded {len(results)} trials")
    print(f"  Start: {metadata['start_pos']}, Goal: {metadata['goal_pos']}")
    print(f"  Collected: {metadata['timestamp']}")
    
    return results, metadata


def main():
    parser = argparse.ArgumentParser(description='Collect Navigation Trajectory Data')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.zip)')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of trials to run (default: 100)')
    parser.add_argument('--max-steps', type=int, default=3000,
                       help='Max steps per trial')
    parser.add_argument('--start', type=str, default='150,300',
                       help='Start position as x,y')
    parser.add_argument('--goal', type=str, default='650,300',
                       help='Goal position as x,y')
    parser.add_argument('--output', type=str,
                       help='Output path for data file (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Parse positions
    start_pos = tuple(map(float, args.start.split(',')))
    goal_pos = tuple(map(float, args.goal.split(',')))
    
    # Create collector
    collector = NavigationDataCollector(
        model_path=args.model,
        start_pos=start_pos,
        goal_pos=goal_pos
    )
    
    # Collect data
    results = collector.collect_data(
        num_trials=args.trials,
        max_steps=args.max_steps
    )
    
    # Save data
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"eval/results/trajectory_data_{timestamp}.pkl"
    
    collector.save_data(results, output_path)
    
    print(f"\n✅ Data collection complete!")
    print(f"📁 Saved to: {output_path}")
    print(f"\n💡 Use this data for visualization:")
    print(f"   python eval/navigation_heatmap.py --load-data {output_path}")


if __name__ == "__main__":
    main()
