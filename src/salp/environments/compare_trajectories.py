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
        time_array = np.arange(cycle_start_time, robot.time + robot.dt, robot.dt)[:len(robot.position_history)]
        
        # Accumulate data
        times.extend(time_array)
        positions.extend(robot.position_history)
        velocities.extend(robot.velocity_history)
        euler_angles.extend(robot.euler_angle_history)
        states.extend(robot.state_history)
    
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

def compare_contraction_levels():
    """Compare trajectories with different contraction levels."""
    print("Comparing different contraction levels...")
    
    nozzle = Nozzle(length1=0.01, length2=0.01, length3=0.01, area=0.00016, mass=1.0)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                  max_contraction=0.06, nozzle=nozzle)
    robot.set_environment(density=1000)
    robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
    
    contractions = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]  # Different contraction levels
    # contractions = [0.01]  # Different contraction levels
    n_cycles = 1
    coast_time = 1.0
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
    
    nozzle = Nozzle(length1=0.01, length2=0.01, length3=0.01, area=0.00016, mass=1.0)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                  max_contraction=0.06, nozzle=nozzle)
    robot.set_environment(density=1000)
    robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
    
    coast_times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # Different coast times
    n_cycles = 1
    contraction = 0.06
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

def compare_yaw_angles():
    """Compare trajectories with different yaw angles."""
    print("\nComparing different yaw angles...")
    
    nozzle = Nozzle(length1=0.01, length2=0.01, length3=0.01, area=0.00016, mass=1.0)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                  max_contraction=0.06, nozzle=nozzle)
    robot.set_environment(density=1000)
    robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
    
    yaw_angles = [-np.pi/2, -np.pi/4, -np.pi/8, np.pi/16, np.pi/32, 0.0, np.pi/32, np.pi/16, np.pi/8, np.pi/4, np.pi/2]  # Different yaw angles
    n_cycles = 1
    contraction = 0.06
    coast_time = 10.0
    
    trajectories = []
    labels = []
    
    for yaw_angle in yaw_angles:
        traj = simulate_trajectory(robot, n_cycles, contraction, coast_time, yaw_angle)
        trajectories.append(traj)
        labels.append(f'Yaw = {np.degrees(yaw_angle):.0f}°')  
        final_pos = traj['positions'][-1]
        final_dist = np.linalg.norm(final_pos)
        print(f"  Yaw {np.degrees(yaw_angle):.0f}°: Final position = ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}) m, Distance = {final_dist:.3f} m")
    
    plot_trajectory_comparison(trajectories, labels, "Comparison: Different Yaw Angles")

def compare_action_combinations():
    """Compare trajectories with different combinations of actions."""
    print("\nComparing different action combinations...")
    
    nozzle = Nozzle(length1=0.01, length2=0.01, length3=0.01, area=0.00009, mass=1.0)
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
