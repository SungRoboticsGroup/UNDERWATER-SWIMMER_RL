import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from salp_robot_env import SalpRobotEnv
from robot import Nozzle, Robot


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


def test_single_target(env, model, target, max_steps=200, render=True, threshold=0.05):
    """
    Test the robot's ability to reach a single target point.
    
    Args:
        env: The environment
        model: The trained model
        target: Target point [x, y] in meters
        max_steps: Maximum number of steps to attempt reaching the target
        render: Whether to render the environment
        threshold: Distance threshold for considering target reached (meters)
        
    Returns:
        Dictionary with test results and statistics
    """
    obs, _ = env.reset()

    env.set_trajectory([target])
    env.target_point = target
    env.current_waypoint_index = 0

    initial_pos = np.array([env.robot.position_world[0], env.robot.position_world[1]])
    initial_distance = np.linalg.norm(initial_pos - target)
    
    print(f"\n{'='*60}")
    print(f"SINGLE TARGET REACHING TEST")
    print(f"{'='*60}")
    print(f"Initial position: ({initial_pos[0]:.3f}, {initial_pos[1]:.3f})")
    print(f"Target position:  ({target[0]:.3f}, {target[1]:.3f})")
    print(f"Initial distance: {initial_distance:.3f}m")
    print(f"Success threshold: {threshold}m")
    print(f"{'='*60}\n")
    
    actual_trajectory = [initial_pos.copy()]
    distances = [initial_distance]
    min_distance = initial_distance
    min_distance_step = 0
    reached = False
    reached_step = None

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        current_pos = np.array([env.robot.position_world[0], env.robot.position_world[1]])
        actual_trajectory.append(current_pos.copy())

        distance = np.linalg.norm(current_pos - target)
        distances.append(distance)

        if distance < min_distance:
            min_distance = distance
            min_distance_step = step + 1

        if (step + 1) % 20 == 0:
            print(f"Step {step+1:3d}: Distance = {distance:.3f}m, Min = {min_distance:.3f}m")

        if distance < threshold and not reached:
            reached = True
            reached_step = step + 1
            print(f"\n✓ TARGET REACHED at step {reached_step}!")
            print(f"  Final distance: {distance:.3f}m")
            break

        if terminated or truncated:
            print(f"\nEnvironment terminated/truncated at step {step+1}")
            break

        if render:
            env.wait_for_animation()

    final_pos = actual_trajectory[-1]
    final_distance = distances[-1]
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"Success: {'YES' if reached else 'NO'}")
    if reached:
        print(f"Reached at step: {reached_step}/{max_steps}")
    print(f"Minimum distance: {min_distance:.3f}m at step {min_distance_step}")
    print(f"Final distance: {final_distance:.3f}m")
    print(f"Final position: ({final_pos[0]:.3f}, {final_pos[1]:.3f})")
    print(f"Total steps: {len(actual_trajectory)-1}")
    print(f"{'='*60}\n")
    
    return {
        'success': reached,
        'reached_step': reached_step,
        'min_distance': min_distance,
        'min_distance_step': min_distance_step,
        'final_distance': final_distance,
        'initial_distance': initial_distance,
        'initial_position': initial_pos,
        'final_position': final_pos,
        'target': target,
        'actual_trajectory': actual_trajectory,
        'distances': distances,
        'total_steps': len(actual_trajectory) - 1
    }


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

    if len(trajectory) >= 2:
        start_pos = trajectory[0]
        next_pos = trajectory[1]
        env.robot.position_world[0] = start_pos[0]
        env.robot.position_world[1] = start_pos[1]
        direction = next_pos - start_pos
        yaw_angle = np.arctan2(direction[1], direction[0])
        env.robot.euler_angle[2] = yaw_angle
        print(f"Robot initialized at waypoint 0: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
        print(f"Orientation toward waypoint 1: {np.degrees(yaw_angle):.1f}°")
    elif len(trajectory) == 1:
        start_pos = trajectory[0]
        env.robot.position_world[0] = start_pos[0]
        env.robot.position_world[1] = start_pos[1]
        print(f"Robot initialized at waypoint 0: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
    
    total_steps = 0
    targets_reached = 0
    distances_to_targets = []
    actual_trajectory = [np.array([env.robot.position_world[0], env.robot.position_world[1]])]

    trajectory.append(trajectory[0])  # loop back to start
    trajectory = trajectory[1:]
    env.set_trajectory(trajectory)
    for target_idx, target in enumerate(trajectory):
        env.current_waypoint_index = target_idx
        env.target_point = target
        env.prev_target_point = trajectory[target_idx - 1] if target_idx > 0 else trajectory[-1]

        print(f"\nTarget {target_idx+1}/{len(trajectory)}: ({target[0]:.2f}, {target[1]:.2f})")

        min_distance = float('inf')

        for step in range(steps_per_target):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            actual_trajectory.append(np.array([env.robot.position_world[0], env.robot.position_world[1]]))

            if render:
                env.wait_for_animation()

            distance = np.linalg.norm(env.robot.position_world[0:-1] - target)
            min_distance = min(min_distance, distance)
            total_steps += 1

            if distance < 0.10:
                targets_reached += 1
                print(f"  ✓ Reached in {step+1} steps (distance: {distance:.3f}m)")
                break

            if terminated:
                obs, _ = env.reset()
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

    desired = np.array(desired_trajectory)
    actual = np.array(actual_trajectory)

    ax.plot(desired[:, 0], desired[:, 1], 'b--', linewidth=2, label='Desired Trajectory', marker='o', markersize=8)
    ax.plot(actual[:, 0], actual[:, 1], 'r-', linewidth=1.5, label='Actual Trajectory', alpha=0.7)

    ax.plot(desired[0, 0], desired[0, 1], 'go', markersize=12, label='Start')
    ax.plot(actual[0, 0], actual[0, 1], 'g^', markersize=10)
    ax.plot(desired[-1, 0], desired[-1, 1], 'rs', markersize=12, label='End (Desired)')
    ax.plot(actual[-1, 0], actual[-1, 1], 'r^', markersize=10)

    errors = [np.min(np.linalg.norm(desired - pt, axis=1)) for pt in actual]
    avg_error = np.mean(errors)
    max_error = np.max(errors)

    stats_text = (f'Avg Error: {avg_error:.3f}m\n'
                  f'Max Error: {max_error:.3f}m\n'
                  f'Actual Points: {len(actual)}\n'
                  f'Desired Points: {len(desired)}')
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

    errors = [np.min(np.linalg.norm(desired - pt, axis=1)) for pt in actual]
    steps = np.arange(len(errors))

    ax1.plot(steps, errors, 'r-', linewidth=1.5, label='Tracking Error')
    ax1.axhline(y=np.mean(errors), color='b', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.3f}m')
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Error (m)', fontsize=11)
    ax1.set_title('Tracking Error vs Step', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    if len(desired) > 1:
        from scipy import interpolate
        t_desired = np.linspace(0, 1, len(desired))
        t_actual = np.linspace(0, 1, len(actual))
        fx = interpolate.interp1d(t_desired, desired[:, 0], kind='linear', fill_value='extrapolate')
        fy = interpolate.interp1d(t_desired, desired[:, 1], kind='linear', fill_value='extrapolate')
        ax2.plot(steps, fx(t_actual), 'b--', linewidth=2, label='Desired X', alpha=0.7)
        ax2.plot(steps, actual[:, 0], 'b-', linewidth=1.5, label='Actual X')
        ax2.plot(steps, fy(t_actual), 'r--', linewidth=2, label='Desired Y', alpha=0.7)
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

def test_single_target_tracking(env, model, target, max_steps=300, render=True, threshold=0.05):
    """
    Test the robot's ability to navigate to and hold a single fixed target point.

    The environment is reset once, the target is overridden with the supplied
    point, and the model is run for up to *max_steps*.  Every step the robot
    position, distance to target, and scalar reward are recorded.  Summary
    statistics are printed and a 2-panel figure is produced showing:
      - 2-D trajectory with the target marked
      - Distance to target and cumulative reward over time

    Args:
        env: The SalpRobotEnv instance (unwrapped, render_mode already set).
        model: Trained SB3 model with a ``predict`` method.
        target: 1-D array-like [x, y] in metres.
        max_steps: Maximum number of environment steps to run.
        render: If True, call ``env.wait_for_animation()`` each step.
        threshold: Distance (m) at which the target is considered reached.

    Returns:
        dict with keys:
            success, reached_step, min_distance, min_distance_step,
            final_distance, initial_distance, positions (N×2 ndarray),
            distances (list), rewards (list), total_steps.
    """
    target = np.asarray(target, dtype=float)

    obs, _ = env.reset()
    env.target_point = target.copy()
    tracking_point_pos = env.robot.get_tracking_point_position_world(env.tracking_point)
    env.prev_target_point = tracking_point_pos.copy()[0:2]
    env.prev_dist = np.linalg.norm(tracking_point_pos[0:2] - target)

    initial_pos = tracking_point_pos.copy()[0:2]
    initial_dist = np.linalg.norm(initial_pos - target)

    print(f"\n{'='*60}")
    print(f"SINGLE TARGET TRACKING TEST")
    print(f"{'='*60}")
    print(f"Initial position : ({initial_pos[0]:.3f}, {initial_pos[1]:.3f})")
    print(f"Target position  : ({target[0]:.3f}, {target[1]:.3f})")
    print(f"Initial distance : {initial_dist:.3f} m")
    print(f"Success threshold: {threshold} m")
    print(f"{'='*60}")

    positions = [initial_pos.copy()]
    distances = [initial_dist]
    rewards = []
    min_dist = initial_dist
    min_dist_step = 0
    reached = False
    reached_step = None

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        tracking_point_pos = env.robot.get_tracking_point_position_world(env.tracking_point)
        pos = tracking_point_pos.copy()[0:2]
        dist = np.linalg.norm(pos - target)

        positions.append(pos)
        distances.append(dist)
        rewards.append(float(reward))

        if dist < min_dist:
            min_dist = dist
            min_dist_step = step + 1

        if (step + 1) % 50 == 0:
            print(f"  step {step+1:4d} | dist = {dist:.4f} m | min = {min_dist:.4f} m | reward = {reward:.3f}")

        if dist < threshold and not reached:
            reached = True
            reached_step = step + 1
            print(f"\n  Target reached at step {reached_step}  (dist = {dist:.4f} m)")

        if terminated or truncated:
            print(f"\n  Episode ended at step {step+1} (terminated={terminated}, truncated={truncated})")
            break

        if render:
            env.wait_for_animation()

    positions = np.array(positions)
    final_dist = distances[-1]
    final_pos = positions[-1]
    total_steps = len(positions) - 1

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Success        : {'YES' if reached else 'NO'}")
    if reached:
        print(f"  Reached at step: {reached_step} / {total_steps}")
    print(f"  Min distance   : {min_dist:.4f} m  (step {min_dist_step})")
    print(f"  Final distance : {final_dist:.4f} m")
    print(f"  Final position : ({final_pos[0]:.3f}, {final_pos[1]:.3f})")
    print(f"  Total steps    : {total_steps}")
    print(f"{'='*60}\n")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 2-D trajectory
    ax1.plot(positions[:, 0], positions[:, 1], 'r-', linewidth=1.5, label='Robot path')
    ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(positions[-1, 0], positions[-1, 1], 'rs', markersize=10, label='End')
    ax1.plot(target[0], target[1], 'b*', markersize=14, label='Target')
    circle = plt.Circle(target, threshold, color='blue', fill=False, linestyle='--', linewidth=1, label=f'Threshold ({threshold} m)')
    ax1.add_patch(circle)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('2-D Trajectory')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Distance and reward over time
    steps_axis = np.arange(len(distances))
    ax2.plot(steps_axis, distances, 'r-', linewidth=1.5, label='Distance to target')
    ax2.axhline(threshold, color='blue', linestyle='--', linewidth=1, label=f'Threshold ({threshold} m)')
    ax2.axhline(min_dist, color='green', linestyle=':', linewidth=1, label=f'Min dist ({min_dist:.3f} m)')
    if rewards:
        ax2_twin = ax2.twinx()
        cumulative_reward = np.cumsum(rewards)
        ax2_twin.plot(np.arange(1, len(rewards) + 1), cumulative_reward, 'k--', linewidth=1, alpha=0.6, label='Cumulative reward')
        ax2_twin.set_ylabel('Cumulative reward')
        ax2_twin.legend(fontsize=9, loc='lower right')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Distance (m)')
    ax2.set_title('Distance to Target Over Time')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Single-target tracking  |  target=({target[0]:.2f}, {target[1]:.2f})  |  "
        f"{'Reached' if reached else 'Not reached'}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()

    return {
        'success': reached,
        'reached_step': reached_step,
        'min_distance': min_dist,
        'min_distance_step': min_dist_step,
        'final_distance': final_dist,
        'initial_distance': initial_dist,
        'positions': positions,
        'distances': distances,
        'rewards': rewards,
        'total_steps': total_steps,
    }

if __name__ == "__main__":
    # Robot physical parameters — DO NOT CHANGE
    nozzle = Nozzle(
        length1=0.052, length2=0.038, length3=0.050,
        area=np.pi * 0.01**2, mass=0.428,
        radius=0.1, inner_radius=0.022,
    )
    nozzle.set_angles(angle1=0.0, angle2=0.0)
    robot = Robot(
        dry_mass=0.738, init_length=0.26, init_width=0.135,
        max_contraction=0.04, nozzle=nozzle,
    )
    robot.set_environment(density=1000)
    robot.enable_history_recording()

    env = SalpRobotEnv(render_mode="human", robot=robot)
    model = SAC.load("./salp_robot_final_front_pos", env=env)  
    result = test_single_target_tracking(env, model, target=np.array([1.5, 0.9]), max_steps=300, threshold=0.05)
    
    # Choose a trajectory type
    center = np.array([0.0, 0.0])
    
    # Test different trajectories
    trajectories = {
        'circle': generate_circle_trajectory(center, radius=0.75, num_points=16),
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


