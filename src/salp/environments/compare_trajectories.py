"""
Compare robot trajectories with different action inputs.

This script simulates the robot with various combinations of:
- Contraction levels
- Coast times
- Nozzle yaw angles

and visualizes the resulting trajectories for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

from robot import Robot, Nozzle
from experiment import read_file as read_experiment_file


def compare_actions_with_states(actions, expected_states, robot=None, verbose=True):
    """Compare simulated trajectory from actions with expected states.
    
    Args:
        actions: Array of shape (n_steps, 3) containing [contraction, coast_time, yaw_angle]
        expected_states: Array of shape (n_steps, 6) containing expected [pos_x, pos_y, vel_x, vel_y, yaw, angular_vel]
        robot: Robot instance to use (if None, creates default robot)
        verbose: Whether to print detailed comparison results
        
    Returns:
        Dictionary containing:
            - actual_states: Simulated states from actions
            - expected_states: Input expected states
            - errors: State-by-state errors
            - position_error: Mean position error (m)
            - velocity_error: Mean velocity error (m/s)
            - angle_error: Mean angular error (rad)
            - max_position_error: Maximum position error (m)
    """
    if robot is None:
        nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.00016, mass=1.0)
        robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                      max_contraction=0.06, nozzle=nozzle)
        robot.set_environment(density=1000)
        robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
    
    robot.reset()
    
    n_steps = len(actions)
    actual_states = []
    
    # Simulate robot with given actions
    for i, action in enumerate(actions):
        contraction, coast_time, yaw_angle = action
        
        robot.nozzle.set_yaw_angle(yaw_angle=yaw_angle)
        robot.nozzle.solve_angles()
        robot.set_control(
            contraction=contraction,
            coast_time=coast_time,
            nozzle_angles=np.array([robot.nozzle.angle1, robot.nozzle.angle2])
        )
        robot.step_through_cycle()
        
        # Get final state after this cycle
        state = np.array([
            robot.position[0],
            robot.position[1],
            robot.velocity[0],
            robot.velocity[1],
            robot.euler_angle[2],
            robot.angular_velocity[2]
        ])
        actual_states.append(state)
    
    actual_states = np.array(actual_states)
    
    # Calculate errors
    errors = actual_states - expected_states
    position_errors = np.linalg.norm(errors[:, 0:2], axis=1)
    velocity_errors = np.linalg.norm(errors[:, 2:4], axis=1)
    angle_errors = np.abs(errors[:, 4])
    angular_vel_errors = np.abs(errors[:, 5])
    
    mean_pos_error = np.mean(position_errors)
    mean_vel_error = np.mean(velocity_errors)
    mean_angle_error = np.mean(angle_errors)
    max_pos_error = np.max(position_errors)
    
    if verbose:
        print("\n" + "=" * 70)
        print("TRAJECTORY COMPARISON: ACTIONS vs EXPECTED STATES")
        print("=" * 70)
        print(f"Number of steps: {n_steps}")
        print(f"\nSummary Statistics:")
        print(f"  Mean position error:     {mean_pos_error:.6f} m")
        print(f"  Mean velocity error:     {mean_vel_error:.6f} m/s")
        print(f"  Mean angle error:        {np.degrees(mean_angle_error):.6f}°")
        print(f"  Max position error:      {max_pos_error:.6f} m")
        print(f"\nStep-by-step comparison:")
        print("-" * 70)
        print(f"{'Step':<6} {'Pos Error (m)':<15} {'Vel Error (m/s)':<18} {'Angle Error (°)':<15}")
        print("-" * 70)
        for i in range(n_steps):
            print(f"{i:<6} {position_errors[i]:<15.6f} {velocity_errors[i]:<18.6f} {np.degrees(angle_errors[i]):<15.6f}")
        print("=" * 70)
    
    return {
        'actual_states': actual_states,
        'expected_states': expected_states,
        'errors': errors,
        'position_errors': position_errors,
        'velocity_errors': velocity_errors,
        'angle_errors': angle_errors,
        'position_error': mean_pos_error,
        'velocity_error': mean_vel_error,
        'angle_error': mean_angle_error,
        'max_position_error': max_pos_error
    }


def simulate_trajectory(robot, n_cycles, contraction, coast_time, yaw_angle):
    """Simulate robot trajectory with given action inputs.
    
    Args:
        robot: Robot instance
        n_cycles: Number of breathing cycles to simulate
        contraction: Contraction distance (m)
        coast_time: Coast phase duration (s)
        yaw_angle: Nozzle yaw angle (radians)
        
    Returns:
        Dictionary containing trajectory data
    """
    robot.reset()
    
    positions = []
    velocities = []
    euler_angles = []
    states = []
    times = []
    
    for i in range(n_cycles):
        robot.nozzle.set_yaw_angle(yaw_angle=yaw_angle)
        robot.nozzle.solve_angles()
        robot.set_control(
            contraction=contraction, 
            coast_time=coast_time, 
            nozzle_angles=np.array([robot.nozzle.angle1, robot.nozzle.angle2])
        )
        robot.step_through_cycle()
        
        # Create time array for this cycle
        cycle_start_time = robot.time - robot.cycle_time
        time_array = np.arange(cycle_start_time, robot.time, robot.dt)[:len(robot.length_history)-1]
        
        # Accumulate data
        times.extend(time_array)
        positions.extend(robot.position_front_world_history)
        velocities.extend(robot.velocity_world_history)
        euler_angles.extend(robot.euler_angle_history)
        states.extend(robot.state_history)

    positions = np.array(positions)
    positions -= positions[0, :]  # Normalize to start at origin
    return {
        'times': np.array(times),
        'positions': np.array(positions),
        'velocities': np.array(velocities),
        'euler_angles': np.array(euler_angles),
        'states': np.array(states)
    }


def plot_trajectory_comparison(trajectories, labels, title="Trajectory Comparison"):
    """Plot multiple trajectories for comparison.
    
    Args:
        trajectories: List of trajectory dictionaries
        labels: List of labels for each trajectory
        title: Plot title
    """
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    # Plot 1: X-Y trajectory
    ax = plt.subplot(1, 1, 1)
    for i, (traj, label) in enumerate(zip(trajectories, labels)):
        positions = traj['positions']
        ax.plot(positions[:, 0], positions[:, 1], '-', color=colors[i], 
                label=label, linewidth=2, alpha=0.7)
        # Mark start and end
        ax.plot(positions[0, 0], positions[0, 1], 'o', color=colors[i], 
                markersize=10, markeredgecolor='black')
        ax.plot(positions[-1, 0], positions[-1, 1], 's', color=colors[i], 
                markersize=10, markeredgecolor='black')
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('X-Y Trajectory', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.axis('equal')
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_trajectory_comparison_with_experiment(sim_trajectories, sim_labels, 
                                                exp_trajectories, exp_labels,
                                                title="Simulation vs Experiment"):
    """Plot simulation and experimental trajectories together.
    
    Args:
        sim_trajectories: List of simulation trajectory dictionaries
        sim_labels: List of labels for simulation trajectories
        exp_trajectories: List of experimental trajectory dictionaries
        exp_labels: List of labels for experimental trajectories
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Use distinct colors from tab10 - enough contrast between different angles
    distinct_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
    ]
    
    # Plot simulation trajectories (solid thick lines)
    for i, (traj, label) in enumerate(zip(sim_trajectories, sim_labels)):
        positions = traj['positions']
        color = distinct_colors[i % len(distinct_colors)]
        ax.plot(positions[:, 0], positions[:, 1], '-', color=color, 
                label=label, linewidth=3, alpha=0.9)
        # Mark start and end
        ax.plot(positions[0, 0], positions[0, 1], 'o', color=color, 
                markersize=12, markeredgecolor='black', markeredgewidth=1.5)
        ax.plot(positions[-1, 0], positions[-1, 1], 's', color=color, 
                markersize=12, markeredgecolor='black', markeredgewidth=1.5)
    
    # Plot experimental trajectories (dashed lines, same color as matching simulation)
    for i, (traj, label) in enumerate(zip(exp_trajectories, exp_labels)):
        positions = traj['positions']
        # Use same color as simulation for matching test cases
        color = distinct_colors[i % len(distinct_colors)]
        ax.plot(positions[:, 0], positions[:, 1], '--', color=color, 
                label=label, linewidth=2.5, alpha=0.85)
        # Mark start and end
        ax.plot(positions[0, 0], positions[0, 1], 'o', color=color, 
                markersize=9, markeredgecolor='black', markeredgewidth=1)
        ax.plot(positions[-1, 0], positions[-1, 1], 's', color=color, 
                markersize=9, markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_trajectory_mean_std_with_experiment(sim_stats, sim_labels, 
                                              exp_trajectories, exp_labels,
                                              title="Simulation (Mean±Std) vs Experiment"):
    """Plot simulation mean trajectories with std bands and experimental trajectories.
    
    Args:
        sim_stats: List of dicts with 'mean' and 'std' arrays of shape (n_points, 3)
        sim_labels: List of labels for simulation trajectories
        exp_trajectories: List of experimental trajectory dictionaries
        exp_labels: List of labels for experimental trajectories
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Use distinct colors
    distinct_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
    ]
    
    # Plot simulation mean trajectories with std bands
    for i, (stats, label) in enumerate(zip(sim_stats, sim_labels)):
        mean_pos = stats['mean']
        std_pos = stats['std']
        color = distinct_colors[i % len(distinct_colors)]
        
        # Plot mean trajectory
        ax.plot(mean_pos[:, 0], mean_pos[:, 1], '-', color=color, 
                label=label, linewidth=3, alpha=0.9)
        
        # Plot std band using fill_between (approximate as ellipses along trajectory)
        # We'll use a simplified approach: plot error bars at intervals
        n_points = len(mean_pos)
        interval = max(1, n_points // 10)  # Show ~10 error indicators
        for j in range(0, n_points, interval):
            ax.errorbar(mean_pos[j, 0], mean_pos[j, 1], 
                       xerr=std_pos[j, 0], yerr=std_pos[j, 1],
                       color=color, alpha=0.3, capsize=3, capthick=1)
        
        # Mark start and end
        ax.plot(mean_pos[0, 0], mean_pos[0, 1], 'o', color=color, 
                markersize=12, markeredgecolor='black', markeredgewidth=1.5)
        ax.plot(mean_pos[-1, 0], mean_pos[-1, 1], 's', color=color, 
                markersize=12, markeredgecolor='black', markeredgewidth=1.5)
    
    # Plot experimental trajectories (dashed lines, same color as matching simulation)
    for i, (traj, label) in enumerate(zip(exp_trajectories, exp_labels)):
        positions = traj['positions']
        color = distinct_colors[i % len(distinct_colors)]
        ax.plot(positions[:, 0], positions[:, 1], '--', color=color, 
                label=label, linewidth=2.5, alpha=0.85)
        # Mark start and end
        ax.plot(positions[0, 0], positions[0, 1], 'o', color=color, 
                markersize=9, markeredgecolor='black', markeredgewidth=1)
        ax.plot(positions[-1, 0], positions[-1, 1], 's', color=color, 
                markersize=9, markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.axis('equal')
    plt.tight_layout()
    plt.show()


def compare_contraction_levels():
    """Compare trajectories with different contraction levels."""
    print("Comparing different contraction levels...")


    # robot parameters (DO NOT CHANGE THESE)
    nozzle_length1 = 0.052
    nozzle_length2 = 0.038
    nozzle_length3 = 0.050
    nozzle_area = np.pi * 0.01**2
    nozzle_mass = 0.428
    nozzle_radius = 0.1
    nozzle_inner_radius = 0.022

    robot_mass = 0.738
    robot_init_length = 0.26
    robot_init_width = 0.135
    # robot parameters (DO NOT CHANGE THESE)

    nozzle = Nozzle(length1=nozzle_length1, length2=nozzle_length2, length3=nozzle_length3, area=nozzle_area, mass=nozzle_mass, radius=nozzle_radius, inner_radius=nozzle_inner_radius)
    nozzle.set_angles(angle1=0.0, angle2=0.0)
    robot = Robot(dry_mass=robot_mass, init_length=robot_init_length, init_width=robot_init_width, 
                  max_contraction=0.04, nozzle=nozzle)
    robot.set_environment(density=1000)
    robot.enable_history_recording()
    
    contractions = [0.02, 0.03, 0.04]  # Different contraction levels
    # contractions = [0.01]  # Different contraction levels
    n_cycles = 6
    coast_time = 2.0
    yaw_angle = 0.0
    
    trajectories = []
    labels = []
    
    for contraction in contractions:
        traj = simulate_trajectory(robot, n_cycles, contraction, coast_time, yaw_angle)
        trajectories.append(traj)
        labels.append(f'Contraction = {contraction:.2f} m')
        final_dist = np.linalg.norm(traj['positions'][-1])
        print(f"  Contraction {contraction:.2f}m: Final distance = {final_dist:.3f} m")
    
    plot_trajectory_comparison(trajectories, labels, "Comparison: Different Contraction Levels")

def compare_coast_times():
    """Compare trajectories with different coast times."""
    print("\nComparing different coast times...")
    
    # robot parameters (DO NOT CHANGE THESE)
    nozzle_length1 = 0.052
    nozzle_length2 = 0.038
    nozzle_length3 = 0.050
    nozzle_area = np.pi * 0.01**2
    nozzle_mass = 0.428
    nozzle_radius = 0.1
    nozzle_inner_radius = 0.022

    robot_mass = 0.738
    robot_init_length = 0.26
    robot_init_width = 0.135
    # robot parameters (DO NOT CHANGE THESE)

    nozzle = Nozzle(length1=nozzle_length1, length2=nozzle_length2, length3=nozzle_length3, area=nozzle_area, mass=nozzle_mass, radius=nozzle_radius, inner_radius=nozzle_inner_radius)
    nozzle.set_angles(angle1=0.0, angle2=0.0)
    robot = Robot(dry_mass=robot_mass, init_length=robot_init_length, init_width=robot_init_width, 
                  max_contraction=0.04, nozzle=nozzle)
    robot.set_environment(density=1000)
    robot.enable_history_recording()
   
    coast_times = [1.0, 2.0, 3.0]  # Different coast times
    n_cycles = 6
    contraction = 0.03
    yaw_angle = 0.0
    
    trajectories = []
    labels = []
    
    for coast_time in coast_times:
        traj = simulate_trajectory(robot, n_cycles, contraction, coast_time, yaw_angle)
        trajectories.append(traj)
        labels.append(f'Coast Time = {coast_time:.1f} s')
        final_dist = np.linalg.norm(traj['positions'][-1])
        print(f"  Coast time {coast_time:.1f}s: Final distance = {final_dist:.3f} m")

    plot_trajectory_comparison(trajectories, labels, "Comparison: Different Coast Times")

def compare_yaw_angles(n_trials=1):
    """Compare trajectories with different yaw angles (simulation + experiment).
    
    Runs n_trials randomized simulations per yaw angle and computes mean/std.
    
    Args:
        n_trials: Number of randomized simulation trials per yaw angle (default: 1000)
    """
    print(f"\nComparing different yaw angles with {n_trials} randomized trials each...")
    
    yaw_angles = [np.pi/2, np.pi/6, -np.pi/2, -np.pi/6, 0]  # Different yaw angles
    n_cycles = 6
    contraction = 0.03
    coast_time = 2.0
    
    sim_stats = []  # Will contain {'mean': array, 'std': array} for each yaw angle
    sim_labels = []
    
    for yaw_angle in yaw_angles:
        print(f"  Running {n_trials} trials for Yaw = {np.degrees(yaw_angle):.0f}°...")
        
        # Collect all trajectories for this yaw angle
        all_positions = []
        
        for trial in range(n_trials):
            # Create fresh robot for each trial to ensure independent randomization
            nozzle = Nozzle(length1=0.052, length2=0.039, length3=0.031, area=np.pi*0.01**2, mass=0.440)
            nozzle.set_angles(angle1=0.0, angle2=0.0)
            robot = Robot(dry_mass=0.756, init_length=0.26, init_width=0.14, 
                          max_contraction=0.04, nozzle=nozzle)
            robot.set_environment(density=1000)
            robot.enable_history_recording()
            # robot.enable_dynamic_randomization()
            
            traj = simulate_trajectory(robot, n_cycles, contraction, coast_time, yaw_angle)
            all_positions.append(traj['positions'])
            
            if (trial + 1) % 100 == 0:
                print(f"    Completed {trial + 1}/{n_trials} trials")
        
        # Resample all trajectories to same length for averaging
        # Use the median trajectory length as reference
        lengths = [len(pos) for pos in all_positions]
        target_len = int(np.median(lengths))
        
        resampled_positions = []
        for pos in all_positions:
            if len(pos) == target_len:
                resampled_positions.append(pos)
            else:
                # Linear interpolation to resample
                old_indices = np.linspace(0, 1, len(pos))
                new_indices = np.linspace(0, 1, target_len)
                resampled = np.zeros((target_len, 3))
                for dim in range(3):
                    resampled[:, dim] = np.interp(new_indices, old_indices, pos[:, dim])
                resampled_positions.append(resampled)
        
        resampled_positions = np.array(resampled_positions)  # Shape: (n_trials, target_len, 3)
        
        # Compute mean and std
        mean_pos = np.mean(resampled_positions, axis=0)
        std_pos = np.std(resampled_positions, axis=0)
        
        sim_stats.append({'mean': mean_pos, 'std': std_pos, 'all_positions': resampled_positions})
        sim_labels.append(f'Sim: Yaw = {np.degrees(yaw_angle):.0f}° (n={n_trials})')
        
        final_mean = mean_pos[-1]
        final_std = std_pos[-1]
        print(f"    Final position: mean=({final_mean[0]:.3f}, {final_mean[1]:.3f}, {final_mean[2]:.3f}) m")
        print(f"                    std=({final_std[0]:.3f}, {final_std[1]:.3f}, {final_std[2]:.3f}) m")
    
    # Load experimental data (ordered to match simulation yaw_angles: 90°, 30°, -90°, -30°, 0°)
    exp_start_times = [22, 118, 55, 165, 22]
    exp_file_names = [
        'compression_3cm_coast_2s_nozzle_90deg.csv',
        'compression_3cm_coast_2s_nozzle_30deg.csv',
        'compression_3cm_coast_2s_nozzle_-90deg.csv',
        'compression_3cm_coast_2s_nozzle_-30deg.csv',
        'compression_3cm_coast_2s_nozzle_0deg.csv',
    ]
    
    exp_trajectories = []
    exp_labels = []
    
    # Compute rotation angle from 0deg experiment trajectory to align with x-axis
    # 0deg file is at index 4 after reordering
    time_ref, positions_ref, _, _ = read_experiment_file(exp_file_names[4], start_time=exp_start_times[4])
    x_ref, z_ref = positions_ref[:, 0], positions_ref[:, 2]
    dx = x_ref[-1] - x_ref[0]
    dz = z_ref[-1] - z_ref[0]
    rotation_angle = -np.arctan2(dz, dx)  # Rotate to align with x-axis
    
    for file_name, start in zip(exp_file_names, exp_start_times):
        time, positions, velocities, euler_angles = read_experiment_file(file_name, start_time=start)
        x, z = positions[:, 0], positions[:, 2]
        
        # Apply rotation to align first trajectory with x-axis
        c, s = np.cos(rotation_angle), np.sin(rotation_angle)
        x_rot = c * x - s * z
        z_rot = s * x + c * z
        
        # Create trajectory dict compatible with simulation format
        # Note: experimental data is in X-Z plane, simulation is in X-Y plane
        exp_positions = np.column_stack([x_rot, z_rot, np.zeros_like(x_rot)])
        exp_trajectories.append({'positions': exp_positions})
        
        # Extract nozzle angle from filename
        import os
        label = os.path.splitext(file_name)[0]
        exp_labels.append(f'Exp: {label}')
    
    plot_trajectory_mean_std_with_experiment(
        sim_stats, sim_labels,
        exp_trajectories, exp_labels,
        f"Comparison: Simulation (n={n_trials}) vs Experiment (Different Yaw Angles)"
    )

def compare_action_combinations():
    """Compare trajectories with different combinations of actions."""
    print("\nComparing different action combinations...")
    
    nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.0036, mass=1.0)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                  max_contraction=0.06, nozzle=nozzle)
    robot.set_environment(density=1000)
    
    # Define different action combinations
    actions = [
        {'contraction': 0.06, 'coast_time': 1.0, 'yaw': 0.0, 'label': 'Max thrust, straight'},
        {'contraction': 0.03, 'coast_time': 1.0, 'yaw': 0.0, 'label': 'Half thrust, straight'},
        {'contraction': 0.06, 'coast_time': 0.5, 'yaw': 0.0, 'label': 'Max thrust, short coast'},
        {'contraction': 0.06, 'coast_time': 1.0, 'yaw': np.pi/6, 'label': 'Max thrust, turn right'},
        {'contraction': 0.06, 'coast_time': 1.0, 'yaw': -np.pi/6, 'label': 'Max thrust, turn left'},
    ]
    
    n_cycles = 5
    trajectories = []
    labels = []
    
    for action in actions:
        traj = simulate_trajectory(
            robot, n_cycles, 
            action['contraction'], 
            action['coast_time'], 
            action['yaw']
        )
        trajectories.append(traj)
        labels.append(action['label'])
        final_pos = traj['positions'][-1]
        print(f"  {action['label']}: Final position = ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}) m")
    
    plot_trajectory_comparison(trajectories, labels, "Comparison: Different Action Combinations")

def main():
    """Run all trajectory comparisons."""
    print("=" * 60)
    print("Robot Trajectory Comparison")
    print("=" * 60)
    
    # Compare individual action parameters
    # compare_contraction_levels()
    # compare_coast_times()
    compare_yaw_angles()
    
    # Compare action combinations
    # compare_action_combinations()

    
    print("\n" + "=" * 60)
    print("All comparisons complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
