# test_robot.py
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle


def generate_circle_trajectory(center, radius, num_points=20):
    """
    Generate a circular trajectory.
    
    Args:
        center: Center point [x, y] in meters
        radius: Radius of circle in meters
        num_points: Number of waypoints along the circle
        
    Returns:
        List of [x, y] target points
    """
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    trajectory = []
    for angle in angles:
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        trajectory.append(np.array([x, y]))
    return trajectory


def generate_square_trajectory(center, side_length, num_points=20):
    """
    Generate a square trajectory.
    
    Args:
        center: Center point [x, y] in meters
        side_length: Length of each side in meters
        num_points: Number of waypoints along the square (distributed evenly)
        
    Returns:
        List of [x, y] target points
    """
    half_side = side_length / 2
    points_per_side = num_points // 4
    trajectory = []
    
    # Top side (left to right)
    for i in range(points_per_side):
        t = i / points_per_side
        x = center[0] - half_side + side_length * t
        y = center[1] + half_side
        trajectory.append(np.array([x, y]))
    
    # Right side (top to bottom)
    for i in range(points_per_side):
        t = i / points_per_side
        x = center[0] + half_side
        y = center[1] + half_side - side_length * t
        trajectory.append(np.array([x, y]))
    
    # Bottom side (right to left)
    for i in range(points_per_side):
        t = i / points_per_side
        x = center[0] + half_side - side_length * t
        y = center[1] - half_side
        trajectory.append(np.array([x, y]))
    
    # Left side (bottom to top)
    for i in range(points_per_side):
        t = i / points_per_side
        x = center[0] - half_side
        y = center[1] - half_side + side_length * t
        trajectory.append(np.array([x, y]))
    
    return trajectory


def generate_figure_eight_trajectory(center, width, height, num_points=40):
    """
    Generate a figure-eight (infinity symbol) trajectory.
    
    Args:
        center: Center point [x, y] in meters
        width: Width of the figure-eight in meters
        height: Height of the figure-eight in meters
        num_points: Number of waypoints
        
    Returns:
        List of [x, y] target points
    """
    t = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    trajectory = []
    for angle in t:
        # Parametric equations for figure-eight
        x = center[0] + (width/2) * np.sin(angle)
        y = center[1] + (height/2) * np.sin(angle) * np.cos(angle)
        trajectory.append(np.array([x, y]))
    return trajectory


def generate_spiral_trajectory(center, max_radius, num_loops=3, num_points=60):
    """
    Generate an outward spiral trajectory.
    
    Args:
        center: Center point [x, y] in meters
        max_radius: Maximum radius of spiral in meters
        num_loops: Number of complete loops
        num_points: Total number of waypoints
        
    Returns:
        List of [x, y] target points
    """
    trajectory = []
    for i in range(num_points):
        t = i / num_points
        angle = t * num_loops * 2 * np.pi
        radius = max_radius * t
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        trajectory.append(np.array([x, y]))
    return trajectory


def generate_zigzag_trajectory(start, end, amplitude, num_points=20):
    """
    Generate a zigzag trajectory from start to end point.
    
    Args:
        start: Starting point [x, y] in meters
        end: Ending point [x, y] in meters
        amplitude: Amplitude of zigzag perpendicular to main direction
        num_points: Number of waypoints
        
    Returns:
        List of [x, y] target points
    """
    trajectory = []
    direction = np.array(end) - np.array(start)
    perpendicular = np.array([-direction[1], direction[0]])
    perpendicular = perpendicular / np.linalg.norm(perpendicular) if np.linalg.norm(perpendicular) > 0 else perpendicular
    
    for i in range(num_points):
        t = i / (num_points - 1)
        # Main direction progress
        base_point = np.array(start) + t * direction
        # Zigzag offset
        offset = amplitude * np.sin(t * np.pi * 4) * perpendicular
        trajectory.append(base_point + offset)
    return trajectory


def generate_star_trajectory(center, outer_radius, inner_radius, num_points=10):
    """
    Generate a star-shaped trajectory.
    
    Args:
        center: Center point [x, y] in meters
        outer_radius: Radius of outer points in meters
        inner_radius: Radius of inner points in meters
        num_points: Number of points (must be even, half for outer, half for inner)
        
    Returns:
        List of [x, y] target points
    """
    trajectory = []
    for i in range(num_points):
        angle = i / num_points * 2 * np.pi
        # Alternate between outer and inner radius
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        trajectory.append(np.array([x, y]))
    return trajectory


def generate_sine_wave_trajectory(start, end, amplitude, frequency=2, num_points=30):
    """
    Generate a sine wave trajectory.
    
    Args:
        start: Starting point [x, y] in meters
        end: Ending point [x, y] in meters
        amplitude: Amplitude of the wave
        frequency: Number of complete waves
        num_points: Number of waypoints
        
    Returns:
        List of [x, y] target points
    """
    trajectory = []
    direction = np.array(end) - np.array(start)
    perpendicular = np.array([-direction[1], direction[0]])
    perpendicular = perpendicular / np.linalg.norm(perpendicular) if np.linalg.norm(perpendicular) > 0 else perpendicular
    
    for i in range(num_points):
        t = i / (num_points - 1)
        base_point = np.array(start) + t * direction
        offset = amplitude * np.sin(t * 2 * np.pi * frequency) * perpendicular
        trajectory.append(base_point + offset)
    return trajectory


def test_trajectory_tracking(env, model, trajectory, steps_per_target=50, render=True):
    """
    Test the robot's ability to track a trajectory.
    
    Args:
        env: The environment
        model: The trained model
        trajectory: List of target points
        steps_per_target: Number of steps to attempt reaching each target
        render: Whether to render the environment
        
    Returns:
        Dictionary with tracking statistics including actual trajectory
    """
    obs, _ = env.reset()
    
    # Set the full trajectory for visualization
    # env.set_trajectory(trajectory)
    
    # Position robot at the first waypoint with orientation toward the second
    if len(trajectory) >= 2:
        start_pos = trajectory[0]
        next_pos = trajectory[1]
        
        # Set robot position to first waypoint
        env.robot.position[0] = start_pos[0]
        env.robot.position[1] = start_pos[1]
        
        # Calculate orientation angle from first to second waypoint
        direction = next_pos - start_pos
        yaw_angle = np.arctan2(direction[1], direction[0])
        env.robot.euler_angle[2] = yaw_angle  # Set yaw
        
        print(f"Robot initialized at waypoint 0: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
        print(f"Orientation toward waypoint 1: {np.degrees(yaw_angle):.1f}°")
    elif len(trajectory) == 1:
        # Only one waypoint, just position at it
        start_pos = trajectory[0]
        env.robot.position[0] = start_pos[0]
        env.robot.position[1] = start_pos[1]
        print(f"Robot initialized at waypoint 0: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
    
    total_steps = 0
    targets_reached = 0
    distances_to_targets = []
    
    # Record actual robot trajectory
    actual_trajectory = [np.array([env.robot.position[0], env.robot.position[1]])]
    
    trajectory.append(trajectory[0])  # Loop back to start for continuous tracking
    trajectory = trajectory[1:]
    env.set_trajectory(trajectory) 
    for target_idx, target in enumerate(trajectory):
        # Update current waypoint index for visualization
        env.current_waypoint_index = target_idx
        # Set the new target in the environment
        env.target_point = target
        print(f"\nTarget {target_idx+1}/{len(trajectory)}: ({target[0]:.2f}, {target[1]:.2f})")
        
        min_distance = float('inf')
        
        for step in range(steps_per_target):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record robot position after each step
            actual_trajectory.append(np.array([env.robot.position[0], env.robot.position[1]]))
            
            if render:
                env.wait_for_animation()
            
            # Calculate distance to current target
            distance = np.linalg.norm(env.robot.position[0:-1] - target)
            min_distance = min(min_distance, distance)
            
            total_steps += 1
            
            if distance < 0.05:  # Threshold for "reaching" the target
                targets_reached += 1
                print(f"  ✓ Reached in {step+1} steps (distance: {distance:.3f}m)")
                break
            
            if truncated or terminated:
                obs, _ = env.reset()
                # Re-set the trajectory after reset
                env.set_trajectory(trajectory)
                env.current_waypoint_index = target_idx
                env.target_point = target
                print(f"  Reset environment (truncated={truncated}, terminated={terminated})")
        
        distances_to_targets.append(min_distance)
        if min_distance >= 0.05:
            print(f"  ✗ Closest approach: {min_distance:.3f}m")
    
    stats = {
        'total_targets': len(trajectory),
        'targets_reached': targets_reached,
        'success_rate': targets_reached / len(trajectory),
        'avg_min_distance': np.mean(distances_to_targets),
        'total_steps': total_steps,
        'actual_trajectory': actual_trajectory,
        'desired_trajectory': trajectory
    }
    
    return stats


def plot_trajectory_comparison(desired_trajectory, actual_trajectory, title="Trajectory Comparison", save_path=None):
    """
    Plot comparison between desired and actual robot trajectories.
    
    Args:
        desired_trajectory: List of desired waypoints [x, y]
        actual_trajectory: List of actual robot positions [x, y]
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Convert to numpy arrays
    desired = np.array(desired_trajectory)
    actual = np.array(actual_trajectory)
    
    # Plot desired trajectory
    ax.plot(desired[:, 0], desired[:, 1], 'b--', linewidth=2, label='Desired Trajectory', marker='o', markersize=8)
    
    # Plot actual trajectory
    ax.plot(actual[:, 0], actual[:, 1], 'r-', linewidth=1.5, label='Actual Trajectory', alpha=0.7)
    
    # Mark start and end points
    ax.plot(desired[0, 0], desired[0, 1], 'go', markersize=12, label='Start')
    ax.plot(actual[0, 0], actual[0, 1], 'g^', markersize=10)
    ax.plot(desired[-1, 0], desired[-1, 1], 'rs', markersize=12, label='End (Desired)')
    ax.plot(actual[-1, 0], actual[-1, 1], 'r^', markersize=10)
    
    # Calculate tracking error
    # For each actual point, find closest desired point
    errors = []
    for actual_pt in actual:
        distances = np.linalg.norm(desired - actual_pt, axis=1)
        min_dist = np.min(distances)
        errors.append(min_dist)
    
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    
    # Add statistics text box
    stats_text = f'Avg Error: {avg_error:.3f}m\n'
    stats_text += f'Max Error: {max_error:.3f}m\n'
    stats_text += f'Actual Points: {len(actual)}\n'
    stats_text += f'Desired Points: {len(desired)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='top', fontsize=10, family='monospace')
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory comparison plot saved to: {save_path}")
    else:
        plt.show()
    
    return fig


def plot_tracking_error_over_time(desired_trajectory, actual_trajectory, title="Tracking Error Over Time", save_path=None):
    """
    Plot tracking error as a function of time/steps.
    
    Args:
        desired_trajectory: List of desired waypoints
        actual_trajectory: List of actual robot positions
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    desired = np.array(desired_trajectory)
    actual = np.array(actual_trajectory)
    
    # Calculate error at each step (distance to closest desired point)
    errors = []
    for actual_pt in actual:
        distances = np.linalg.norm(desired - actual_pt, axis=1)
        min_dist = np.min(distances)
        errors.append(min_dist)
    
    steps = np.arange(len(errors))
    
    # Plot 1: Tracking error over time
    ax1.plot(steps, errors, 'r-', linewidth=1.5, label='Tracking Error')
    ax1.axhline(y=np.mean(errors), color='b', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.3f}m')
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Error (m)', fontsize=11)
    ax1.set_title('Tracking Error vs Step', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: X and Y position comparison
    # Interpolate desired trajectory to match actual trajectory length
    if len(desired) > 1:
        from scipy import interpolate
        # Create interpolation functions for desired trajectory
        t_desired = np.linspace(0, 1, len(desired))
        t_actual = np.linspace(0, 1, len(actual))
        
        fx = interpolate.interp1d(t_desired, desired[:, 0], kind='linear', fill_value='extrapolate')
        fy = interpolate.interp1d(t_desired, desired[:, 1], kind='linear', fill_value='extrapolate')
        
        desired_interp_x = fx(t_actual)
        desired_interp_y = fy(t_actual)
        
        ax2.plot(steps, desired_interp_x, 'b--', linewidth=2, label='Desired X', alpha=0.7)
        ax2.plot(steps, actual[:, 0], 'b-', linewidth=1.5, label='Actual X')
        ax2.plot(steps, desired_interp_y, 'r--', linewidth=2, label='Desired Y', alpha=0.7)
        ax2.plot(steps, actual[:, 1], 'r-', linewidth=1.5, label='Actual Y')
    else:
        ax2.plot(steps, actual[:, 0], 'b-', linewidth=1.5, label='Actual X')
        ax2.plot(steps, actual[:, 1], 'r-', linewidth=1.5, label='Actual Y')
    
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Position (m)', fontsize=11)
    ax2.set_title('X and Y Positions vs Step', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Tracking error plot saved to: {save_path}")
    else:
        plt.show()
    
    return fig


# Example usage
if __name__ == "__main__":
    nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.00016, mass=1.0)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                      max_contraction=0.06, nozzle=nozzle)
    robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
    env = SalpRobotEnv(render_mode="human", robot=robot)
    
    # Load the trained model
    model = SAC.load("./salp_robot_final_yaw_continuity", env=env)   
    
    # Choose a trajectory type
    center = np.array([0.0, 0.0])
    
    # Test different trajectories
    trajectories = {
        'circle': generate_circle_trajectory(center, radius=1.0, num_points=16),
        'square': generate_square_trajectory(center, side_length=2.0, num_points=20),
        'figure_eight': generate_figure_eight_trajectory(center, width=2.0, height=1.0, num_points=30),
        'spiral': generate_spiral_trajectory(center, max_radius=1.0, num_loops=2, num_points=40),
        'star': generate_star_trajectory(center, outer_radius=1.0, inner_radius=0.5, num_points=10),
        'sine_wave': generate_sine_wave_trajectory(
            start=np.array([-1.0, 0.0]), 
            end=np.array([1.0, 0.0]), 
            amplitude=0.5, 
            frequency=3,
            num_points=25
        )
    }
    
    # Select which trajectory to test (change this to test different shapes)
    trajectory_name = 'circle'  # Options: circle, square, figure_eight, spiral, star, sine_wave
    trajectory = trajectories[trajectory_name]
    
    print(f"\n{'='*60}")
    print(f"Testing {trajectory_name.upper()} trajectory")
    print(f"{'='*60}")
    
    # env.start_recording()
    
    # Test the trajectory
    stats = test_trajectory_tracking(env, model, trajectory, steps_per_target=100, render=True)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"TRAJECTORY TRACKING RESULTS - {trajectory_name.upper()}")
    print(f"{'='*60}")
    print(f"Total targets: {stats['total_targets']}")
    print(f"Targets reached: {stats['targets_reached']}")
    print(f"Success rate: {stats['success_rate']*100:.1f}%")
    print(f"Average minimum distance: {stats['avg_min_distance']:.3f}m")
    print(f"Total steps: {stats['total_steps']}")

    # gif_path = env.stop_recording(f"trajectory_{trajectory_name}_test.gif")
    env.close()
    
    # Generate trajectory comparison plots
    print(f"\n{'='*60}")
    print("Generating trajectory comparison plots...")
    print(f"{'='*60}")
    
    # Plot trajectory comparison
    plot_trajectory_comparison(
        stats['desired_trajectory'],
        stats['actual_trajectory'],
        title=f"Trajectory Comparison - {trajectory_name.upper()}",
        save_path=f"recordings/trajectory_comparison_{trajectory_name}.png"
    )
    
    # Plot tracking error over time
    plot_tracking_error_over_time(
        stats['desired_trajectory'],
        stats['actual_trajectory'],
        title=f"Tracking Error - {trajectory_name.upper()}",
        save_path=f"recordings/tracking_error_{trajectory_name}.png"
    )
    
    print(f"✓ All plots saved to recordings/ directory")