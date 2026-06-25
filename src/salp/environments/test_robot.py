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
    Test the robot's ability to track a trajectory with dual target setting.
    
    For each waypoint, the robot is given:
    - target_point (primary): Current waypoint to track
    - target_point_2 (secondary): Next waypoint for reward function
    
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
        env.robot.position_front_world[0] = start_pos[0]
        env.robot.position_front_world[1] = start_pos[1]

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
    actual_trajectory = [np.array([env.robot.position_front_world[0], env.robot.position_front_world[1]])]

    trajectory.append(trajectory[0])  # loop back to start
    trajectory = trajectory[1:]
    env.set_trajectory(trajectory)
    for target_idx, target in enumerate(trajectory):
        env.current_waypoint_index = target_idx
        # Set primary target (current waypoint for tracking)
        env.target_point = target
        # Set secondary target (next waypoint for reward function)
        next_target_idx = (target_idx + 1) % len(trajectory)
        env.target_point_2 = trajectory[next_target_idx]
        env.prev_target_point = trajectory[target_idx - 1] if target_idx > 0 else trajectory[-1]

        print(f"\nTarget {target_idx+1}/{len(trajectory)}: T1=({target[0]:.2f}, {target[1]:.2f}) T2=({trajectory[next_target_idx][0]:.2f}, {trajectory[next_target_idx][1]:.2f})")

        min_distance = float('inf')

        for step in range(steps_per_target):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            actual_trajectory.append(np.array([env.robot.position_front_world[0], env.robot.position_front_world[1]]))

            if render:
                env.wait_for_animation()

            distance = np.linalg.norm(env.robot.position_front_world[0:-1] - target)
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
                # Reset both targets
                env.target_point = target
                next_target_idx = (target_idx + 1) % len(trajectory)
                env.target_point_2 = trajectory[next_target_idx]
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


def _point_to_polyline_distance(point, polyline):
    """
    Compute the minimum distance from *point* to the closest point on any
    segment of *polyline* (a (K, 2) array of vertices).  The polyline is
    treated as open (no extra closing segment is added here).
    """
    point = np.asarray(point, dtype=float)
    polyline = np.asarray(polyline, dtype=float)
    min_dist = np.inf
    for i in range(len(polyline) - 1):
        p1, p2 = polyline[i], polyline[i + 1]
        seg = p2 - p1
        seg_len_sq = np.dot(seg, seg)
        if seg_len_sq < 1e-12:
            d = np.linalg.norm(point - p1)
        else:
            t = np.clip(np.dot(point - p1, seg) / seg_len_sq, 0.0, 1.0)
            closest = p1 + t * seg
            d = np.linalg.norm(point - closest)
        if d < min_dist:
            min_dist = d
    return min_dist


def _cross_track_errors(desired_trajectory, actual_trajectory):
    """
    For each point in *actual_trajectory*, compute the perpendicular (cross-track)
    distance to the nearest segment of the closed *desired_trajectory* polyline.
    """
    desired = np.asarray(desired_trajectory, dtype=float)
    # Close the loop for a closed trajectory
    closed = np.vstack([desired, desired[0]])
    return np.array([_point_to_polyline_distance(pt, closed) for pt in actual_trajectory])


def plot_trajectory_comparison(desired_trajectory, actual_trajectory, title="Trajectory Comparison", save_path=None):
    """
    Plot comparison between desired and actual robot trajectories.

    Args:
        desired_trajectory: List of desired waypoints [x, y]
        actual_trajectory: List of actual robot positions [x, y]
        title: Plot title
        save_path: Optional path to save the figure
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.linewidth": 0.8, "grid.linewidth": 0.5,
    })

    C_DESIRED = "#3C5488"
    C_ACTUAL  = "#E64B35"
    C_START   = "#00A087"
    C_END     = "#F39B7F"

    desired = np.asarray(desired_trajectory, dtype=float)
    actual  = np.asarray(actual_trajectory,  dtype=float)

    errors = _cross_track_errors(desired, actual)
    avg_error = float(np.mean(errors))
    max_error = float(np.max(errors))

    fig, ax = plt.subplots(figsize=(8, 8))

    # Close the desired loop for plotting
    desired_closed = np.vstack([desired, desired[0]])
    ax.plot(desired_closed[:, 0], desired_closed[:, 1],
            color=C_DESIRED, linestyle="--", linewidth=1.8,
            marker="o", markersize=6,
            markeredgecolor="white", markeredgewidth=0.5,
            label="Desired trajectory", zorder=3)
    ax.plot(actual[:, 0], actual[:, 1],
            color=C_ACTUAL, linewidth=1.6, alpha=0.85,
            label="Robot path", zorder=4)

    ax.plot(*actual[0],  marker="o", color=C_START, markersize=10, linestyle="None",
            markeredgecolor="white", markeredgewidth=0.6, label="Start", zorder=5)
    ax.plot(*actual[-1], marker="s", color=C_END,   markersize=9,  linestyle="None",
            markeredgecolor="white", markeredgewidth=0.6, label="End",   zorder=5)

    for i, wp in enumerate(desired):
        ax.annotate(str(i + 1), xy=wp, fontsize=8, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points", color=C_DESIRED)

    stats_text = (f"Avg cross-track error: {avg_error:.3f} m\n"
                  f"Max cross-track error: {max_error:.3f} m")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.7", alpha=0.9),
            verticalalignment="top", fontsize=10, family="monospace")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.invert_yaxis()  # Invert y-axis to match typical robot coordinate system
    # ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="0.7")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.35, color="grey")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Trajectory comparison plot saved to: {save_path}")
    else:
        plt.show()

    return fig


def plot_tracking_error_over_time(desired_trajectory, actual_trajectory, title="Tracking Error Over Time", save_path=None):
    """
    Plot cross-track tracking error and x/y positions as a function of actuation cycle.

    Args:
        desired_trajectory: List of desired waypoints [x, y]
        actual_trajectory: List of actual robot positions [x, y]
        title: Plot title
        save_path: Optional path to save the figure
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.linewidth": 0.8, "grid.linewidth": 0.5,
    })

    C_ERROR  = "#E64B35"
    C_MEAN   = "#3C5488"
    C_X      = "#4DBBD5"
    C_Y      = "#00A087"

    desired = np.asarray(desired_trajectory, dtype=float)
    actual  = np.asarray(actual_trajectory,  dtype=float)

    # Cross-track error (perpendicular distance to nearest path segment)
    errors = _cross_track_errors(desired, actual)
    steps  = np.arange(len(errors))
    mean_err = float(np.mean(errors))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                    gridspec_kw={"hspace": 0.42})

    # ── Top: error over time ─────────────────────────────────────────────
    ax1.plot(steps, errors, color=C_ERROR, linewidth=1.6, label="Cross-track error")
    ax1.fill_between(steps, errors, alpha=0.12, color=C_ERROR)
    ax1.axhline(mean_err, color=C_MEAN, linestyle="--", linewidth=1.4,
                label=f"Mean: {mean_err:.3f} m")
    ax1.set_xlabel("Actuation cycle")
    ax1.set_ylabel("Error (m)")
    # ax1.set_title("Cross-Track Error vs Cycle", fontweight="bold")
    ax1.set_xlim(0, len(steps) - 1)
    ax1.set_ylim(bottom=0)
    ax1.legend(framealpha=0.9, edgecolor="0.7")
    ax1.grid(True, linestyle="--", alpha=0.35, color="grey")
    for spine in ax1.spines.values():
        spine.set_linewidth(0.8)

    # ── Bottom: x and y positions ────────────────────────────────────────
    if len(desired) > 1:
        from scipy import interpolate
        t_d = np.linspace(0, 1, len(desired))
        t_a = np.linspace(0, 1, len(actual))
        fx = interpolate.interp1d(t_d, desired[:, 0], kind="linear", fill_value="extrapolate")
        fy = interpolate.interp1d(t_d, desired[:, 1], kind="linear", fill_value="extrapolate")
        ax2.plot(steps, fx(t_a), color=C_X, linestyle="--", linewidth=1.6,
                 alpha=0.7, label="Desired x")
        ax2.plot(steps, actual[:, 0], color=C_X, linewidth=1.4,
                 label="Actual x")
        ax2.plot(steps, fy(t_a), color=C_Y, linestyle="--", linewidth=1.6,
                 alpha=0.7, label="Desired y")
        ax2.plot(steps, actual[:, 1], color=C_Y, linewidth=1.4,
                 label="Actual y")
    else:
        ax2.plot(steps, actual[:, 0], color=C_X, linewidth=1.4, label="Actual x")
        ax2.plot(steps, actual[:, 1], color=C_Y, linewidth=1.4, label="Actual y")

    ax2.set_xlabel("Actuation cycle")
    ax2.set_ylabel("Position (m)")
    # ax2.set_title("x / y Positions vs Cycle", fontweight="bold")
    ax2.set_xlim(0, len(steps) - 1)
    ax2.legend(framealpha=0.9, edgecolor="0.7", ncol=2)
    ax2.grid(True, linestyle="--", alpha=0.35, color="grey")
    for spine in ax2.spines.values():
        spine.set_linewidth(0.8)

    # fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Tracking error plot saved to: {save_path}")
    else:
        plt.show()

    return fig

def test_single_target_tracking(env, model, target, target2=None, max_steps=300, render=True, threshold=0.05,
                                snapshot_every_n_cycles=2, snapshot_dir="snapshots"):
    """
    Test the robot's ability to navigate to and hold fixed target points.

    The environment is reset once, the targets are overridden with the supplied
    points, and the model is run for up to *max_steps*.  Every step the robot
    position, distance to targets, and scalar reward are recorded.  Summary
    statistics are printed and a 2-panel figure is produced showing:
      - 2-D trajectory with both targets marked
      - Distance to primary target and cumulative reward over time

    A pygame screenshot of the simulation window is saved to *snapshot_dir* after
    every *snapshot_every_n_cycles* actuation cycles (env steps), and also when the
    episode ends.  Set *snapshot_every_n_cycles* to 0 to disable saving.

    Args:
        env: The SalpRobotEnv instance (unwrapped, render_mode already set).
        model: Trained SB3 model with a ``predict`` method.
        target: 1-D array-like [x, y] in metres for the primary target.
        target2: Optional 1-D array-like [x, y] in metres for the secondary target.
                 If None, generates target2 as 0.1-0.2m further from target.
        max_steps: Maximum number of environment steps to run.
        render: If True, call ``env.wait_for_animation()`` each step.
        threshold: Distance (m) at which the primary target is considered reached.
        snapshot_every_n_cycles: Save a pygame screenshot after every N actuation
                                  cycles.  0 disables snapshots.
        snapshot_dir: Directory in which snapshots are saved.

    Returns:
        dict with keys:
            success, reached_step, min_distance, min_distance_step,
            final_distance, initial_distance, positions (N×2 ndarray),
            distances (list), rewards (list), total_steps.
    """
    target = np.asarray(target, dtype=float)
    
    # Generate second target if not provided
    if target2 is None:
        # Generate target2 at 0.1-0.2m further along direction from origin
        direction = target / np.linalg.norm(target) if np.linalg.norm(target) > 0 else np.array([1.0, 0.0])
        additional_distance = np.random.uniform(0.1, 0.2)
        target2 = target + additional_distance * direction
    else:
        target2 = np.asarray(target2, dtype=float)

    # Prepare snapshot directory
    import os
    from datetime import datetime
    if snapshot_every_n_cycles and snapshot_every_n_cycles > 0:
        os.makedirs(snapshot_dir, exist_ok=True)
        _snapshot_prefix = os.path.join(
            snapshot_dir,
            f"stt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    else:
        _snapshot_prefix = None

    def _save_snapshot(step_label):
        """Save a screenshot of the pygame simulation window."""
        import pygame
        if env.screen is None:
            return
        _path = f"{_snapshot_prefix}_step{step_label:05d}.png"
        pygame.image.save(env.screen, _path)
        print(f"  [snapshot] saved → {_path}")

    obs, _ = env.reset()
    env.target_point = target.copy()
    env.target_point_2 = target2.copy()
    tracking_point_pos = env.robot.get_tracking_point_position_world(env.tracking_point)
    env.prev_target_point = tracking_point_pos.copy()[0:2]

    initial_pos = tracking_point_pos.copy()[0:2]
    initial_dist = np.linalg.norm(initial_pos - target)
    initial_dist2 = np.linalg.norm(initial_pos - target2)

    print(f"\n{'='*60}")
    print(f"DUAL TARGET TRACKING TEST")
    print(f"{'='*60}")
    print(f"Initial position : ({initial_pos[0]:.3f}, {initial_pos[1]:.3f})")
    print(f"Target 1 (primary): ({target[0]:.3f}, {target[1]:.3f})")
    print(f"Target 2 (secondary): ({target2[0]:.3f}, {target2[1]:.3f})")
    print(f"Initial distance to T1: {initial_dist:.3f} m")
    print(f"Initial distance to T2: {initial_dist2:.3f} m")
    print(f"Distance between targets: {np.linalg.norm(target2 - target):.3f} m")
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

        # Render / animate the completed cycle
        needs_render = render or (_snapshot_prefix is not None)
        if needs_render:
            env.wait_for_animation()

        # Save snapshot after every N cycles
        if _snapshot_prefix and (step + 1) % snapshot_every_n_cycles == 0:
            _save_snapshot(step + 1)

        if terminated or truncated:
            print(f"\n  Episode ended at step {step+1} (terminated={terminated}, truncated={truncated})")
            if _snapshot_prefix and (step + 1) % snapshot_every_n_cycles != 0:
                # Save a final snapshot if we didn't just take one
                _save_snapshot(step + 1)
            break

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

    # ── Publication-quality plot ──────────────────────────────────────────
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
    })

    # Colour palette — colourblind-friendly
    C_PATH   = "#E64B35"   # red
    C_TARGET = "#4DBBD5"   # teal  (primary target)
    C_TARGET2= "#00A087"   # green (secondary target)
    C_START  = "#3C5488"   # navy
    C_END    = "#F39B7F"   # salmon
    C_THRESH = "#7E6148"   # brown

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                   gridspec_kw={"wspace": 0.32})

    # ── Left panel: 2-D trajectory ───────────────────────────────────────
    ax1.plot(positions[:, 0], positions[:, 1],
             color=C_PATH, linewidth=1.8, zorder=3, label="Robot path")

    # Threshold circle
    circle = mpl.patches.Circle(target, threshold,
                                 color=C_THRESH, fill=False,
                                 linestyle=(0, (4, 2)), linewidth=1.2,
                                 zorder=4, label=f"Success radius ({threshold} m)")
    ax1.add_patch(circle)

    # Targets
    ax1.plot(*target,  marker="*", color=C_TARGET,  markersize=14,
             zorder=5, label="Primary target", linestyle="None",
             markeredgecolor="white", markeredgewidth=0.5)
    ax1.plot(*target2, marker="*", color=C_TARGET2, markersize=12,
             zorder=5, label="Secondary target", linestyle="None",
             markeredgecolor="white", markeredgewidth=0.5)

    # Start / end markers
    ax1.plot(*positions[0],  marker="o", color=C_START, markersize=8,
             zorder=6, label="Start", linestyle="None",
             markeredgecolor="white", markeredgewidth=0.6)
    ax1.plot(*positions[-1], marker="s", color=C_END,   markersize=8,
             zorder=6, label="End",   linestyle="None",
             markeredgecolor="white", markeredgewidth=0.6)

    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.invert_yaxis()
    # ax1.set_title("Robot Trajectory")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, linestyle="--", alpha=0.35, color="grey")
    ax1.legend(loc="best", framealpha=0.9,
               edgecolor="0.7", handlelength=1.6)
    for spine in ax1.spines.values():
        spine.set_linewidth(0.8)

    # ── Right panel: distance over time ──────────────────────────────────
    steps_axis = np.arange(len(distances))
    ax2.plot(steps_axis, distances, color=C_PATH, linewidth=1.8,
             label="Distance to primary target", zorder=3)
    ax2.axhline(threshold, color=C_THRESH, linestyle=(0, (4, 2)),
                linewidth=1.2, label=f"Success threshold ({threshold} m)", zorder=2)
    # ax2.axhline(min_dist, color=C_START, linestyle=":",
    #             linewidth=1.2, label=f"Min. distance ({min_dist:.3f} m)", zorder=2)

    if reached:
        ax2.axvline(reached_step, color=C_TARGET2, linestyle="--",
                    linewidth=1.1, alpha=0.8,
                    label=f"Target reached (cycle {reached_step})", zorder=2)

    # Shade the region below threshold
    # ax2.fill_between(steps_axis, distances, threshold,
    #                  where=np.array(distances) < threshold,
    #                  color=C_TARGET2, alpha=0.15, zorder=1)

    ax2.set_xlabel("Actuation cycle")
    ax2.set_ylabel("Distance (m)")
    # ax2.set_title("Convergence to Target")
    ax2.set_xlim(0, len(distances) - 1)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, linestyle="--", alpha=0.35, color="grey")
    ax2.legend(loc="upper right", framealpha=0.9,
               edgecolor="0.7", handlelength=1.6)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.8)

    status_str = "Reached" if reached else "Not reached"
    # fig.suptitle(
    #     f"Single-target tracking  |  "
    #     f"T1=({target[0]:.2f}, {target[1]:.2f})  "
    #     f"T2=({target2[0]:.2f}, {target2[1]:.2f})  |  {status_str}",
    #     fontsize=12, fontweight="bold", y=1.01
    # )
    fig.tight_layout()
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


def _circle_segment_intersect(pos, radius, p1, p2):
    """
    Find intersections of a circle (centre=pos, radius=radius) with segment p1→p2.

    Returns a list of parameter values t ∈ [0, 1] where the intersection lies
    on the segment (0 = p1, 1 = p2).  The corresponding world point is
    p1 + t * (p2 - p1).
    """
    d = p2 - p1
    f = p1 - pos
    a = float(np.dot(d, d))
    b = 2.0 * float(np.dot(f, d))
    c = float(np.dot(f, f)) - radius ** 2
    disc = b * b - 4 * a * c
    if disc < 0 or a < 1e-12:
        return []
    sqrt_disc = np.sqrt(disc)
    ts = [(-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a)]
    return [t for t in ts if 0.0 <= t <= 1.0]


def pure_pursuit_lookahead(env, lookahead_distance: float = 0.3):
    """
    Compute the pure pursuit lookahead point by finding the intersection of the
    lookahead circle (centred at the robot's tracking point, radius=lookahead_distance)
    with the path segments starting from ``env.current_waypoint_index``.

    For each segment (waypoints[i] → waypoints[i+1]) the circle-segment intersection
    is computed geometrically.  The intersection furthest along the segment (highest t)
    is preferred.  The first segment (starting from current_waypoint_index) that
    produces a valid intersection gives the lookahead point.

    If no intersection is found (robot is far off-path or inside all segments),
    the function falls back to the next waypoint.

    Args:
        env: SalpRobotEnv instance with trajectory_waypoints set.
        lookahead_distance: Lookahead circle radius in metres.

    Returns:
        Tuple of (lookahead_point, next_point): both np.ndarray [x, y] in metres.
    """
    waypoints = env.trajectory_waypoints
    if not waypoints:
        return env.target_point.copy(), env.target_point.copy()

    tp = env.robot.get_tracking_point_position_world(env.tracking_point)
    pos = tp[:2].copy()
    n = len(waypoints)

    # current_waypoint_index is the NEXT waypoint to reach, so the robot is
    # currently on segment (current_waypoint_index - 1) → current_waypoint_index.
    # Start searching from that segment (offset = -1 relative to current index).
    start_idx = (env.current_waypoint_index - 1) % n

    # Search segments forward from the segment the robot is currently on.
    lookahead_point = None
    lookahead_seg_offset = None
    for offset in range(n):
        i = (start_idx + offset) % n
        j = (i + 1) % n
        p1 = np.asarray(waypoints[i], dtype=float)
        p2 = np.asarray(waypoints[j], dtype=float)

        ts = _circle_segment_intersect(pos, lookahead_distance, p1, p2)
        if ts:
            t_best = max(ts)
            lookahead_point = p1 + t_best * (p2 - p1)
            lookahead_seg_offset = offset
            break

    # Fallback: no circle-segment intersection found — head toward next waypoint.
    if lookahead_point is None:
        fallback_idx = env.current_waypoint_index
        lookahead_point = np.asarray(waypoints[fallback_idx], dtype=float).copy()
        lookahead_seg_offset = 1

    # Secondary target: intersection at 2× lookahead_distance, searched forward
    # from the segment where the primary lookahead was found.
    next_point = None
    for offset in range(lookahead_seg_offset, n):
        i = (start_idx + offset) % n
        j = (i + 1) % n
        p1 = np.asarray(waypoints[i], dtype=float)
        p2 = np.asarray(waypoints[j], dtype=float)

        ts = _circle_segment_intersect(pos, 2.0 * lookahead_distance, p1, p2)
        if ts:
            t_best = max(ts)
            candidate = p1 + t_best * (p2 - p1)
            # Must be further from the robot than the primary lookahead
            if np.linalg.norm(candidate - pos) > np.linalg.norm(lookahead_point - pos):
                next_point = candidate
                break

    if next_point is None:
        # Fallback: waypoint after the primary lookahead segment
        fallback_idx = (start_idx + lookahead_seg_offset + 1) % n
        next_point = np.asarray(waypoints[fallback_idx], dtype=float).copy()

    return lookahead_point, next_point


def test_pure_pursuit_tracking(env, model, trajectory, lookahead_distance: float = 0.3,
                               steps_per_waypoint: int = 60, waypoint_threshold: float = 0.15,
                               render: bool = True):
    """
    Track *trajectory* using pure pursuit geometry + RL model control with dual targets.

    At every step:
      1. Pure pursuit selects a lookahead point on the trajectory.
      2. ``env.target_point`` (primary) is set to the lookahead point.
      3. ``env.target_point_2`` (secondary) is set to the next point after lookahead.
      4. The RL model predicts an action given the updated observation.
      5. The environment steps forward.
      6. When the robot comes within *waypoint_threshold* of the current
         waypoint the waypoint index advances.

    Args:
        env: SalpRobotEnv instance (render_mode already set).
        model: Trained SB3 model with a ``predict`` method.
        trajectory: List of [x, y] waypoints in metres.
        lookahead_distance: Pure pursuit look-ahead radius (metres).
        steps_per_waypoint: Max env steps before forcing advance to next waypoint.
        waypoint_threshold: Distance (m) to consider a waypoint reached.
        render: If True call ``env.wait_for_animation()`` each step.

    Returns:
        dict with keys:
            targets_reached, total_targets, success_rate,
            avg_min_distance, total_steps,
            actual_trajectory (N×2 ndarray), desired_trajectory (list).
    """
    waypoints = [np.asarray(w, dtype=float) for w in trajectory]
    n = len(waypoints)

    obs, _ = env.reset()
    env.set_trajectory(waypoints)
    # Robot starts AT waypoints[0] heading toward waypoints[1], so the first
    # segment to track is 0→1 and current_waypoint_index should be 1.
    env.current_waypoint_index = 1 % n
    # Set initial targets (primary and secondary)
    env.target_point = waypoints[1 % n].astype(np.float32)
    env.target_point_2 = waypoints[2 % n].astype(np.float32) if n > 2 else waypoints[0].astype(np.float32)

    # Initialise robot at first waypoint, oriented toward second
    env.robot.position_world[0] = waypoints[0][0]
    env.robot.position_world[1] = waypoints[0][1]
    if n >= 2:
        d = waypoints[1] - waypoints[0]
        env.robot.euler_angle[2] = float(np.arctan2(d[1], d[0]))
    # Sync position_front_world so the tracking point reflects the new position
    env.robot.position_front_world = env.robot.get_front_position_world_frame()

    obs = env._get_observation()

    print(f"\n{'='*60}")
    print(f"PURE PURSUIT + RL MODEL TRAJECTORY TRACKING")
    print(f"{'='*60}")
    print(f"Waypoints       : {n}")
    print(f"Look-ahead dist : {lookahead_distance} m")
    print(f"WP threshold    : {waypoint_threshold} m")
    print(f"Steps / waypoint: {steps_per_waypoint}")
    print(f"{'='*60}")

    actual_positions = [waypoints[0].copy()]
    targets_reached = 0
    min_distances = [float('inf')] * n
    total_steps = 0
    waypoints_reached_set = set()
    total_max_steps = n * steps_per_waypoint

    # Single flat loop — pure pursuit runs continuously.
    for step in range(total_max_steps):
        tp_now = env.robot.get_tracking_point_position_world(env.tracking_point)[:2].copy()

        # Advance waypoint index when the robot comes within threshold of the current
        # waypoint — do this BEFORE computing the lookahead so the lookahead always
        # targets the next unvisited point.
        cur_idx = env.current_waypoint_index
        dist_to_cur = float(np.linalg.norm(tp_now - waypoints[cur_idx]))
        if dist_to_cur < waypoint_threshold:
            if cur_idx not in waypoints_reached_set:
                waypoints_reached_set.add(cur_idx)
                targets_reached += 1
                print(f"    \u2713 Waypoint {cur_idx+1}/{n} reached at step {step+1}"
                      f"  (dist={dist_to_cur:.3f} m)")
            env.current_waypoint_index = (cur_idx + 1) % n
            cur_idx = env.current_waypoint_index

        # Track minimum distance to the current (next) waypoint.
        dist_to_cur = float(np.linalg.norm(tp_now - waypoints[cur_idx]))
        if dist_to_cur < min_distances[cur_idx]:
            min_distances[cur_idx] = dist_to_cur

        if (step + 1) % 50 == 0:
            print(f"  step {step+1:4d} | wp {cur_idx+1}/{n} | dist={dist_to_cur:.3f} m")

        # Pure pursuit: update lookahead targets, recompute obs, then step.
        lookahead, next_point = pure_pursuit_lookahead(env, lookahead_distance)
        env.target_point = lookahead.astype(np.float32)
        env.target_point_2 = next_point.astype(np.float32)
        tp = env.robot.get_tracking_point_position_world(env.tracking_point)
        env.prev_dist = float(np.linalg.norm(tp[:2] - lookahead))
        obs = env._get_observation()

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1

        tp = env.robot.get_tracking_point_position_world(env.tracking_point)[:2].copy()
        actual_positions.append(tp)

        if render:
            env.wait_for_animation()

        if terminated:
            obs, _ = env.reset()
            env.set_trajectory(waypoints)
            obs = env._get_observation()
            print(f"    Episode reset at step {step+1}")

        if targets_reached == n:
            print(f"\n  All {n} waypoints reached at step {step+1}!")
            break

    actual_positions = np.array(actual_positions)
    # Replace any waypoints never visited with 0 for the average
    avg_min_dist = float(np.mean([d if d < float('inf') else 0.0 for d in min_distances]))
    stats = {
        'targets_reached': targets_reached,
        'total_targets': n,
        'success_rate': targets_reached / n,
        'avg_min_distance': avg_min_dist,
        'total_steps': total_steps,
        'actual_trajectory': actual_positions,
        'desired_trajectory': waypoints,
    }

    print(f"\n{'='*60}")
    print(f"PURE PURSUIT + RL RESULTS")
    print(f"{'='*60}")
    print(f"  Targets reached : {targets_reached} / {n}  ({stats['success_rate']*100:.1f}%)")
    print(f"  Avg min dist    : {avg_min_dist:.4f} m")
    print(f"  Total steps     : {total_steps}")
    print(f"{'='*60}\n")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))
    desired = np.array(waypoints)
    desired_closed = np.vstack([desired, desired[0]])  # close the loop
    ax.plot(desired_closed[:, 0], desired_closed[:, 1], 'b--o', linewidth=1.5, markersize=6, label='Desired trajectory')
    ax.plot(actual_positions[:, 0], actual_positions[:, 1], 'r-', linewidth=1.2, alpha=0.8, label='Robot path (PP + RL)')
    ax.plot(actual_positions[0, 0], actual_positions[0, 1], 'go', markersize=10, label='Start')
    ax.plot(actual_positions[-1, 0], actual_positions[-1, 1], 'rs', markersize=10, label='End')
    for i, wp in enumerate(waypoints):
        ax.annotate(str(i+1), xy=wp, fontsize=8, ha='center', va='bottom',
                    xytext=(0, 6), textcoords='offset points')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Pure Pursuit + RL  |  {targets_reached}/{n} waypoints reached', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    plt.show()

    return stats


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


    # ------------------------------------------------------------------ #
    # #  Pure Pursuit + RL model tracking                                   #
    # # ------------------------------------------------------------------ #
    model = SAC.load("./logs/salp_robot_two_targets_1000000_steps", env=env)
    # env.start_recording()
    # test_single_target_tracking(
    #     env, model, target=[-0.5, -0.5], target2=[-0.6, -0.6],
    #     max_steps=300, render=False, threshold=0.05
    # )  
    # gif_path = env.stop_recording(filename="sim_demo_track1.gif")

    # center = np.array([0.0, 0.0])
    # pp_trajectory = generate_circle_trajectory(center, radius=0.75, num_points=12)
    # pp_stats = test_pure_pursuit_tracking(
    #     env, model, pp_trajectory,
    #     lookahead_distance=0.35,
    #     steps_per_waypoint=80,
    #     waypoint_threshold=0.15,
    #     render=True,
    # )
    # env.close()


    # Choose a trajectory type
    center = np.array([0.0, 0.0])
    
    # Test different trajectories
    trajectories = {
        'circle': generate_circle_trajectory(center, radius=0.75, num_points=15),
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
    
    env.start_recording()
    
    # --- Choose: 'waypoint' for discrete waypoint tracking, 'pure_pursuit' for Pure Pursuit ---
    tracking_mode = 'pure_pursuit'   # change to 'waypoint' to use the old method

    if tracking_mode == 'pure_pursuit':
        stats = test_pure_pursuit_tracking(
            env, model, trajectory,
            lookahead_distance=0.30,
            steps_per_waypoint=100,
            waypoint_threshold=0.10,
            render=True,
        )
        print(f"\n{'='*60}")
        print(f"PURE PURSUIT RESULTS - {trajectory_name.upper()}")
        print(f"{'='*60}")
        print(f"Targets reached      : {stats['targets_reached']} / {stats['total_targets']} ({stats['success_rate']*100:.1f}%)")
        print(f"Avg min distance     : {stats['avg_min_distance']:.4f} m")
        print(f"Total steps          : {stats['total_steps']}")
        print(f"{'='*60}")
    else:
        stats = test_trajectory_tracking(env, model, trajectory, steps_per_target=100, render=True)
        print(f"\n{'='*60}")
        print(f"TRAJECTORY TRACKING RESULTS - {trajectory_name.upper()}")
        print(f"{'='*60}")
        print(f"Total targets: {stats['total_targets']}")
        print(f"Targets reached: {stats['targets_reached']}")
        print(f"Success rate: {stats['success_rate']*100:.1f}%")
        print(f"Average minimum distance: {stats['avg_min_distance']:.3f}m")
        print(f"Total steps: {stats['total_steps']}")

    gif_path = env.stop_recording(f"trajectory_{trajectory_name}_test.gif")
    env.close()

    # Generate trajectory comparison plots
    print(f"\n{'='*60}")
    print("Generating trajectory comparison plots...")
    print(f"{'='*60}")

    plot_trajectory_comparison(
        stats['desired_trajectory'],
        stats['actual_trajectory'],
        title=f"Trajectory Comparison ({tracking_mode}) - {trajectory_name.upper()}",
        save_path=f"recordings/trajectory_comparison_{trajectory_name}_{tracking_mode}.pdf"
    )

    plot_tracking_error_over_time(
        stats['desired_trajectory'],
        stats['actual_trajectory'],
        title=f"Tracking Error ({tracking_mode}) - {trajectory_name.upper()}",
        save_path=f"recordings/tracking_error_{trajectory_name}_{tracking_mode}.pdf"
    )

    print(f"✓ All plots saved to recordings/ directory")


