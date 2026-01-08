"""
Plotting utilities for SALP robot visualization.

This module contains all plotting functions for visualizing robot behavior,
including geometry, forces, velocities, torques, and other physical properties.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from enum import Enum


def _add_phase_backgrounds(ax, time_data, state_data):
    """
    Add colored background regions to show robot phases.
    
    Args:
        ax: Matplotlib axis object
        time_data: Array of time values
        state_data: Array of state values (Phase enum values)
    """
    # Phase order: REFILL=0, JET=1, COAST=2, REST=3
    phase_names = ["Refill", "Jet", "Coast", "Rest"]
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightgray']
    alphas = [0.2, 0.2, 0.2, 0.2]
    
    # Convert Phase enums to their integer values
    state_values = np.array([s.value if isinstance(s, Enum) else s for s in state_data])
    
    # Find phase boundaries
    current_phase = state_values[0]
    start_idx = 0
    
    for i in range(1, len(state_values)):
        if state_values[i] != current_phase:
            # Draw the region for the previous phase
            ax.axvspan(time_data[start_idx-1], time_data[i-1], 
                      color=colors[current_phase], alpha=alphas[current_phase],
                      label=phase_names[current_phase] if start_idx == 0 or current_phase not in state_values[:start_idx] else "")
            start_idx = i
            current_phase = state_values[i]
    
    # Draw the last region
    ax.axvspan(time_data[start_idx], time_data[-1], 
              color=colors[current_phase], alpha=alphas[current_phase],
              label=phase_names[current_phase] if current_phase not in state_values[:start_idx] else "")


def plot_robot_geometry(time_data, length_data, width_data, state_data=None, title="Robot Geometry Over Time"):
    """
    Plot robot length and width over time.
    
    Args:
        time_data: Array of time values
        length_data: Array of length values
        width_data: Array of width values
        state_data: Optional array of state values (0: refill, 1: jet, 2: coast, 3: rest)
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Add phase backgrounds if state_data provided
    if state_data is not None:
        _add_phase_backgrounds(ax1, time_data, state_data)
        _add_phase_backgrounds(ax2, time_data, state_data)
    
    # Length plot
    ax1.plot(time_data, length_data, 'b-', linewidth=2, label='Length', zorder=3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Length (m)')
    ax1.set_title('Robot Length')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Width plot
    ax2.plot(time_data, width_data, 'r-', linewidth=2, label='Width', zorder=3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Width (m)')
    ax2.set_title('Robot Width')
    ax2.grid(True, alpha=0.3, zorder=1)
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_robot_mass(time_data, mass_data, state_data=None, title="Robot Total Mass Over Time"):
    """
    Plot robot total mass over time.
    
    Args:
        time_data: Array of time values
        mass_data: Array of total mass values
        state_data: Optional array of state values (0: refill, 1: jet, 2: coast, 3: rest)
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add phase backgrounds if state_data provided
    if state_data is not None:
        _add_phase_backgrounds(ax, time_data, state_data)
    
    ax.plot(time_data, mass_data, 'k-', linewidth=2, label='Total Mass')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mass (kg)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_mass_rate(time_data, mass_data, state_data=None, title="Rate of Change of Mass Over Time"):
    """
    Plot the rate of change of mass over time.
    
    Args:
        time_data: Array of time values
        mass_data: Array of total mass values
        state_data: Optional array of state values (Phase enum values)
        title: Plot title
    """
    # Calculate rate of change of mass (dm/dt)
    mass_rate = np.gradient(mass_data, time_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add phase backgrounds if state_data provided
    if state_data is not None:
        _add_phase_backgrounds(ax, time_data, state_data)
    
    ax.plot(time_data, mass_rate, 'purple', linewidth=2, label='Mass Rate (dm/dt)', zorder=3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mass Rate (kg/s)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_volume_rate(time_data, volume_data, state_data=None, title="Rate of Change of Volume Over Time"):
    """
    Plot the rate of change of volume over time.
    
    Args:
        time_data: Array of time values
        volume_data: Array of volume values
        state_data: Optional array of state values (Phase enum values)
        title: Plot title
    """
    # Calculate rate of change of volume (dV/dt)
    volume_rate = np.gradient(volume_data, time_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add phase backgrounds if state_data provided
    if state_data is not None:
        _add_phase_backgrounds(ax, time_data, state_data)
    
    ax.plot(time_data, volume_rate, 'orange', linewidth=2, label='Volume Rate (dV/dt)', zorder=3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Volume Rate (m³/s)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_robot_forces(time_data, jet_force_data, drag_force_data, coriolis_force_data=None, 
                     state_data=None, title="Robot Forces Over Time"):
    """
    Plot all forces acting on the robot.
    
    Args:
        time_data: Array of time values
        jet_force_data: Array of jet force values (3D vectors)
        drag_force_data: Array of drag force values (3D vectors)
        coriolis_force_data: Optional array of coriolis force values (3D vectors)
        state_data: Optional array of state values (0: refill, 1: jet, 2: coast, 3: rest)
        title: Plot title
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    directions = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']
    
    for i, (ax, direction, color) in enumerate(zip(axes, directions, colors)):
        # Add phase backgrounds if state_data provided
        if state_data is not None:
            _add_phase_backgrounds(ax, time_data, state_data)
        
        ax.plot(time_data, jet_force_data[:, i], color=color, linestyle='-', 
                linewidth=2, label=f'Jet Force {direction}')
        ax.plot(time_data, drag_force_data[:, i], color=color, linestyle='--', 
                linewidth=2, label=f'Drag Force {direction}')
        
        if coriolis_force_data is not None:
            ax.plot(time_data, coriolis_force_data[:, i], color=color, linestyle=':', 
                    linewidth=2, label=f'Coriolis Force {direction}')
        
        total_force = jet_force_data[:, i] + drag_force_data[:, i]
        if coriolis_force_data is not None:
            total_force += coriolis_force_data[:, i]
        ax.plot(time_data, total_force, 'k-', linewidth=2.5, label=f'Total Force {direction}')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Force {direction} (N)')
        ax.set_title(f'Forces in {direction} Direction')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_jet_properties(time_data, jet_force_data, 
                       state_data=None, title="Jet Properties Over Time"):
    """
    Plot jet forces in X, Y, Z dimensions.
    
    Args:
        time_data: Array of time values
        jet_force_data: Array of jet force values (3D vectors)
        state_data: Optional array of state values (Phase enum values)
        title: Plot title
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    directions = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']
    
    for i, (ax, direction, color) in enumerate(zip(axes, directions, colors)):
        # Add phase backgrounds if state_data provided
        if state_data is not None:
            _add_phase_backgrounds(ax, time_data, state_data)
        
        # Plot jet force in each direction
        ax.plot(time_data, jet_force_data[:, i], color=color, linewidth=2, 
                label=f'Jet Force {direction}', zorder=3)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Jet Force {direction} (N)')
        ax.set_title(f'Jet Force in {direction} Direction')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_jet_velocity(time_data, jet_velocity_data, 
                     state_data=None, title="Jet Velocity Over Time"):
    """
    Plot jet velocity in X, Y, Z dimensions.
    
    Args:
        time_data: Array of time values
        jet_velocity_data: Array of jet velocity values (3D vectors)
        state_data: Optional array of state values (Phase enum values)
        title: Plot title
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    directions = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']
    
    for i, (ax, direction, color) in enumerate(zip(axes, directions, colors)):
        # Add phase backgrounds if state_data provided
        if state_data is not None:
            _add_phase_backgrounds(ax, time_data, state_data)
        
        # Plot jet velocity in each direction
        ax.plot(time_data, jet_velocity_data[:, i], color=color, linewidth=2, 
                label=f'Jet Velocity {direction}', zorder=3)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Jet Velocity {direction} (m/s)')
        ax.set_title(f'Jet Velocity in {direction} Direction')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_drag_properties(time_data, drag_force_data, 
                        state_data=None, title="Drag Properties Over Time"):
    """
    Plot drag forces in X, Y, and Z dimensions.
    
    Args:
        time_data: Array of time values
        drag_force_data: Array of drag force values (3D vectors)
        state_data: Optional array of state values (0: refill, 1: jet, 2: coast, 3: rest)
        title: Plot title
    """
    # Create subplots for X, Y, Z components
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Labels and colors for each dimension
    dimensions = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']
    
    for i, (ax, dim, color) in enumerate(zip(axes, dimensions, colors)):
        # Add phase backgrounds if state_data provided
        if state_data is not None:
            _add_phase_backgrounds(ax, time_data, state_data)
        
        ax.plot(time_data, drag_force_data[:, i], color=color, linewidth=2, zorder=3, label=f'Drag {dim}')
        ax.set_ylabel(f'Drag Force {dim} (N)')
        ax.set_title(f'Drag Force - {dim} Dimension')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_drag_coefficient(time_data, drag_coefficient_data, 
                         state_data=None, title="Drag Coefficient Over Time"):
    """
    Plot drag coefficient over time.
    
    Args:
        time_data: Array of time values
        drag_coefficient_data: Array of drag coefficient values
        state_data: Optional array of state values (0: refill, 1: jet, 2: coast, 3: rest)
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Add phase backgrounds if state_data provided
    if state_data is not None:
        _add_phase_backgrounds(ax, time_data, state_data)
    
    ax.plot(time_data, drag_coefficient_data, 'g-', linewidth=2, zorder=3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Drag Coefficient')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_robot_position(time_data, position_data, 
                       state_data=None, title="Robot Position Over Time"):
    """
    Plot robot position in X, Y, Z dimensions.
    
    Args:
        time_data: Array of time values
        position_data: Array of position values (3D vectors)
        state_data: Optional array of state values (0: refill, 1: jet, 2: coast, 3: rest)
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    directions = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']
    
    for i, (ax, direction, color) in enumerate(zip(axes, directions, colors)):
        # Add phase backgrounds if state_data provided
        if state_data is not None:
            _add_phase_backgrounds(ax, time_data, state_data)
        
        ax.plot(time_data, position_data[:, i], color=color, linewidth=2, 
                label=f'Position {direction}', zorder=3)
        ax.set_ylabel(f'Position {direction} (m)')
        ax.set_title(f'Position - {direction} Dimension')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_robot_velocity(time_data, velocity_data, 
                       state_data=None, title="Robot Velocity Over Time"):
    """
    Plot robot velocity in X, Y, Z dimensions.
    
    Args:
        time_data: Array of time values
        velocity_data: Array of velocity values (3D vectors)
        state_data: Optional array of state values (0: refill, 1: jet, 2: coast, 3: rest)
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    directions = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']
    
    for i, (ax, direction, color) in enumerate(zip(axes, directions, colors)):
        # Add phase backgrounds if state_data provided
        if state_data is not None:
            _add_phase_backgrounds(ax, time_data, state_data)
        
        ax.plot(time_data, velocity_data[:, i], color=color, linewidth=2, 
                label=f'Velocity {direction}', zorder=3)
        ax.set_ylabel(f'Velocity {direction} (m/s)')
        ax.set_title(f'Velocity - {direction} Dimension')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_robot_acceleration(time_data, acceleration_data, 
                           state_data=None, title="Robot Acceleration Over Time"):
    """
    Plot robot acceleration in X, Y, Z dimensions.
    
    Args:
        time_data: Array of time values
        acceleration_data: Array of acceleration values (3D vectors)
        state_data: Optional array of state values (0: refill, 1: jet, 2: coast, 3: rest)
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    directions = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']
    
    for i, (ax, direction, color) in enumerate(zip(axes, directions, colors)):
        # Add phase backgrounds if state_data provided
        if state_data is not None:
            _add_phase_backgrounds(ax, time_data, state_data)
        
        ax.plot(time_data, acceleration_data[:, i], color=color, linewidth=2, 
                label=f'Acceleration {direction}', zorder=3)
        ax.set_ylabel(f'Acceleration {direction} (m/s²)')
        ax.set_title(f'Acceleration - {direction} Dimension')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_cross_sectional_area(time_data, area_data, 
                             state_data=None, title="Robot Cross-Sectional Area Over Time"):
    """
    Plot cross-sectional area over time.
    
    Args:
        time_data: Array of time values
        area_data: Array of cross-sectional area values
        state_data: Optional array of state values (0: refill, 1: jet, 2: coast, 3: rest)
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add phase backgrounds if state_data provided
    if state_data is not None:
        _add_phase_backgrounds(ax, time_data, state_data)
    
    # Cross-sectional area
    ax.plot(time_data, area_data, 'teal', linewidth=2, label='Cross-Sectional Area')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Area (m²)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_nozzle_configuration(time_data, angle1_data, angle2_data, 
                              state_data=None, title="Nozzle Angle Configuration"):
    """
    Plot nozzle angles over time.
    
    Args:
        time_data: Array of time values
        angle1_data: Array of angle1 values (around y-axis)
        angle2_data: Array of angle2 values (around z-axis)
        state_data: Optional array of state values (0: refill, 1: jet, 2: coast, 3: rest)
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Add phase backgrounds if state_data provided
    if state_data is not None:
        _add_phase_backgrounds(ax1, time_data, state_data)
        _add_phase_backgrounds(ax2, time_data, state_data)
    
    # Angle 1
    ax1.plot(time_data, np.degrees(angle1_data), 'b-', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle 1 (degrees)')
    ax1.set_title('Nozzle Angle Around Y-axis')
    ax1.grid(True, alpha=0.3)
    
    # Angle 2
    ax2.plot(time_data, np.degrees(angle2_data), 'r-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle 2 (degrees)')
    ax2.set_title('Nozzle Angle Around Z-axis')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_angular_acceleration(time_data, angular_acceleration_data, 
                              state_data=None, title="Angular Acceleration Over Time"):
    """
    Plot angular acceleration components over time.
    
    Args:
        time_data: Array of time values
        angular_acceleration_data: Array of angular acceleration vectors (N x 3)
        state_data: Optional array of state values (0: refill, 1: jet, 2: coast, 3: rest)
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
    colors = ['r', 'g', 'b']
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        # Add phase backgrounds if state_data provided
        if state_data is not None:
            _add_phase_backgrounds(ax, time_data, state_data)
        
        ax.plot(time_data, angular_acceleration_data[:, i], color=color, linewidth=2, zorder=3)
        ax.set_ylabel(f'Angular Accel. (rad/s²)')
        ax.set_title(f'Angular Acceleration - {label}')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_angular_velocity(time_data, angular_velocity_data, 
                          state_data=None, title="Angular Velocity Over Time"):
    """
    Plot angular velocity components over time.
    
    Args:
        time_data: Array of time values
        angular_velocity_data: Array of angular velocity vectors (N x 3)
        state_data: Optional array of state values (0: refill, 1: jet, 2: coast, 3: rest)
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    labels = ['Roll Rate (X)', 'Pitch Rate (Y)', 'Yaw Rate (Z)']
    colors = ['r', 'g', 'b']
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        # Add phase backgrounds if state_data provided
        if state_data is not None:
            _add_phase_backgrounds(ax, time_data, state_data)
        
        ax.plot(time_data, angular_velocity_data[:, i], color=color, linewidth=2, zorder=3)
        ax.set_ylabel(f'Angular Vel. (rad/s)')
        ax.set_title(f'Angular Velocity - {label}')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_euler_angles(time_data, euler_angles_data, 
                      state_data=None, title="Euler Angles Over Time"):
    """
    Plot Euler angle components over time.
    
    Args:
        time_data: Array of time values
        euler_angles_data: Array of Euler angle vectors (N x 3)
        state_data: Optional array of state values (0: refill, 1: jet, 2: coast, 3: rest)
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
    colors = ['r', 'g', 'b']
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        # Add phase backgrounds if state_data provided
        if state_data is not None:
            _add_phase_backgrounds(ax, time_data, state_data)
        
        # Convert to degrees for better readability
        ax.plot(time_data, np.degrees(euler_angles_data[:, i]), color=color, linewidth=2, zorder=3)
        ax.set_ylabel(f'Angle (degrees)')
        ax.set_title(f'Euler Angle - {label}')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_drag_torque(time_data, drag_torque_data, 
                     state_data=None, title="Drag Torque Over Time"):
    """
    Plot drag torque components over time.
    
    Args:
        time_data: Array of time values
        drag_torque_data: Array of drag torque vectors (N x 3)
        state_data: Optional array of state values (0: refill, 1: jet, 2: coast, 3: rest)
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    labels = ['Roll Torque (X)', 'Pitch Torque (Y)', 'Yaw Torque (Z)']
    colors = ['r', 'g', 'b']
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        # Add phase backgrounds if state_data provided
        if state_data is not None:
            _add_phase_backgrounds(ax, time_data, state_data)
        
        ax.plot(time_data, drag_torque_data[:, i], color=color, linewidth=2, zorder=3)
        ax.set_ylabel(f'Torque (N·m)')
        ax.set_title(f'Drag Torque - {label}')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_jet_torque(time_data, jet_torque_data, 
                    state_data=None, title="Jet Torque Over Time"):
    """
    Plot jet torque components over time.
    
    Args:
        time_data: Array of time values
        jet_torque_data: Array of jet torque vectors (N x 3)
        state_data: Optional array of state values (0: refill, 1: jet, 2: coast, 3: rest)
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    labels = ['Roll Torque (X)', 'Pitch Torque (Y)', 'Yaw Torque (Z)']
    colors = ['r', 'g', 'b']
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        # Add phase backgrounds if state_data provided
        if state_data is not None:
            _add_phase_backgrounds(ax, time_data, state_data)
        
        ax.plot(time_data, jet_torque_data[:, i], color=color, linewidth=2, zorder=3)
        ax.set_ylabel(f'Torque (N·m)')
        ax.set_title(f'Jet Torque - {label}')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_trajectory_xy(position_data: np.ndarray, state_data: np.ndarray = None, euler_angle_data: np.ndarray = None):
    """Plot the robot's trajectory in the x-y plane.
    
    Args:
        position_data: Nx3 array of position data [x, y, z]
        state_data: Optional array of phase states for color coding
        euler_angle_data: Optional Nx3 array of euler angles [roll, pitch, yaw] for orientation visualization
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x_positions = position_data[:, 0]
    y_positions = position_data[:, 1]
    
    if state_data is not None:
        # Color code by phase
        from matplotlib.collections import LineCollection
        
        # Create segments for line collection
        points = np.array([x_positions, y_positions]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Map phases to colors
        phase_colors = {
            0: 'blue',    # REFILL
            1: 'red',     # JET
            2: 'green',   # COAST
            3: 'gray'     # REST
        }
        
        # Get phase values as integers
        phase_values = np.array([s.value if hasattr(s, 'value') else s for s in state_data])
        colors = [phase_colors.get(phase_values[i], 'black') for i in range(len(segments))]
        
        lc = LineCollection(segments, colors=colors, linewidths=2)
        ax.add_collection(lc)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Refill'),
            Patch(facecolor='red', label='Jet'),
            Patch(facecolor='green', label='Coast'),
            Patch(facecolor='gray', label='Rest')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    else:
        # Simple line plot without color coding
        ax.plot(x_positions, y_positions, 'b-', linewidth=2)
    
    # Mark start and end points
    ax.plot(x_positions[0], y_positions[0], 'go', markersize=10, label='Start', zorder=5)
    ax.plot(x_positions[-1], y_positions[-1], 'ro', markersize=10, label='End', zorder=5)
    
    # Add orientation arrows along trajectory
    arrow_interval = max(1, len(x_positions) // 20)
    
    # Calculate triangle size based on position data range
    x_range = np.max(x_positions) - np.min(x_positions)
    y_range = np.max(y_positions) - np.min(y_positions)
    data_range = max(x_range, y_range)
    
    # Scale triangle size to be approximately 2% of the data range
    triangle_size = data_range * 0.02 if data_range > 0 else 0.015
    
    if euler_angle_data is not None:
        # Use yaw angle to determine arrow direction
        yaw_angles = euler_angle_data[:, 2]  # Extract yaw (psi) angles
        
        from matplotlib.patches import Polygon
        
        for i in range(0, len(x_positions), arrow_interval):
            # Calculate triangle vertices based on yaw angle
            yaw = yaw_angles[i]
            
            # Triangle pointing in the direction of yaw
            # Tip of triangle
            tip_x = x_positions[i] + triangle_size * np.cos(yaw)
            tip_y = y_positions[i] + triangle_size * np.sin(yaw)
            
            # Base corners (perpendicular to yaw direction)
            base_offset = triangle_size * 0.4
            left_x = x_positions[i] + base_offset * np.cos(yaw + np.pi/2)
            left_y = y_positions[i] + base_offset * np.sin(yaw + np.pi/2)
            right_x = x_positions[i] + base_offset * np.cos(yaw - np.pi/2)
            right_y = y_positions[i] + base_offset * np.sin(yaw - np.pi/2)
            
            # Create triangle vertices
            triangle = np.array([[tip_x, tip_y], [left_x, left_y], [right_x, right_y]])
            
            # Draw hollow triangle with black outline
            poly = Polygon(triangle, facecolor='none', edgecolor='black', 
                          linewidth=1.5, alpha=0.9, zorder=4)
            ax.add_patch(poly)
    else:
        # Fallback: use movement direction if no yaw data
        from matplotlib.patches import Polygon
        
        for i in range(0, len(x_positions) - 1, arrow_interval):
            dx = x_positions[i+1] - x_positions[i]
            dy = y_positions[i+1] - y_positions[i]
            
            # Calculate angle from movement direction
            angle = np.arctan2(dy, dx)
            
            # Triangle pointing in movement direction
            tip_x = x_positions[i] + triangle_size * np.cos(angle)
            tip_y = y_positions[i] + triangle_size * np.sin(angle)
            
            base_offset = triangle_size * 0.4
            left_x = x_positions[i] + base_offset * np.cos(angle + np.pi/2)
            left_y = y_positions[i] + base_offset * np.sin(angle + np.pi/2)
            right_x = x_positions[i] + base_offset * np.cos(angle - np.pi/2)
            right_y = y_positions[i] + base_offset * np.sin(angle - np.pi/2)
            
            triangle = np.array([[tip_x, tip_y], [left_x, left_y], [right_x, right_y]])
            poly = Polygon(triangle, facecolor='none', edgecolor='black', 
                          linewidth=1.5, alpha=0.9, zorder=4)
            ax.add_patch(poly)
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    title = 'Robot Trajectory in X-Y Plane'
    if euler_angle_data is not None:
        title += ' (Arrows show yaw orientation)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    if state_data is None:
        ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_nozzle_direction(nozzle, euler_angles=None, title="Nozzle Direction Visualization"):
    """
    Visualize the nozzle direction in 3D space.
    
    Creates a 3D plot showing:
    - Robot body axes (X, Y, Z)
    - Nozzle position
    - Nozzle direction vector
    - Nozzle's rotation angles
    
    Args:
        nozzle: Nozzle object with position and direction methods
        euler_angles: Optional tuple of (roll, pitch, yaw) angles for robot orientation.
                     If provided, the visualization will show the nozzle direction
                     in the world frame. Otherwise shows body frame.
        title: Plot title
    
    Returns:
        Figure object
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get nozzle position and direction
    nozzle_position = nozzle.get_nozzle_position()
    nozzle_direction = nozzle.get_nozzle_direction()
    middle_position = nozzle.get_middle_position()
    
    # Normalize direction for visualization
    direction_normalized = nozzle_direction / (np.linalg.norm(nozzle_direction) + 1e-8)
    arrow_length = 0.1  # length of direction arrow in visualization
    
    # Plot robot body reference frame (origin at center of mass)
    origin = np.array([0, 0, 0])
    axis_length = 0.15
    
    ax.quiver(origin[0], origin[1], origin[2], axis_length, 0, 0, 
              color='r', arrow_length_ratio=0.2, linewidth=2, label='X-axis (forward)')
    ax.quiver(origin[0], origin[1], origin[2], 0, axis_length, 0, 
              color='g', arrow_length_ratio=0.2, linewidth=2, label='Y-axis (lateral)')
    ax.quiver(origin[0], origin[1], origin[2], 0, 0, axis_length, 
              color='b', arrow_length_ratio=0.2, linewidth=2, label='Z-axis (vertical)')
    
    # Plot nozzle middle position
    ax.scatter(*middle_position, color='orange', s=100, marker='o', 
               label='Nozzle joint', zorder=5)
    
    # Plot nozzle tip position
    ax.scatter(*nozzle_position, color='purple', s=100, marker='s', 
               label='Nozzle tip', zorder=5)
    
    # Plot nozzle direction vector
    ax.quiver(nozzle_position[0], nozzle_position[1], nozzle_position[2],
              direction_normalized[0] * arrow_length, 
              direction_normalized[1] * arrow_length, 
              direction_normalized[2] * arrow_length,
              color='darkred', arrow_length_ratio=0.3, linewidth=2.5, 
              label='Nozzle direction', zorder=4)
    
    # Plot nozzle structure (line from joint to tip)
    ax.plot([middle_position[0], nozzle_position[0]], 
            [middle_position[1], nozzle_position[1]], 
            [middle_position[2], nozzle_position[2]], 
            'k--', linewidth=1.5, alpha=0.7, label='Nozzle structure')
    
    # Set labels and limits
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)
    
    # Set equal aspect ratio and limits
    limit = 0.2
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text annotation with nozzle angles
    angle1_deg = np.degrees(nozzle.angle1)
    angle2_deg = np.degrees(nozzle.angle2)
    yaw_deg = np.degrees(nozzle.yaw)
    
    textstr = f'Angle1 (Y-axis): {angle1_deg:.1f}°\nAngle2 (Z-axis): {angle2_deg:.1f}°\nYaw: {yaw_deg:.1f}°'
    ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes, 
              fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_nozzle_direction_sequence(nozzle_directions, nozzle_positions=None, 
                                   title="Nozzle Direction Sequence"):
    """
    Visualize multiple nozzle directions to show steering capability.
    
    Useful for understanding the nozzle's reachable workspace and steering range.
    
    Args:
        nozzle_directions: List or array of direction vectors (Nx3)
        nozzle_positions: Optional list of corresponding nozzle tip positions (Nx3).
                         If not provided, all arrows start from origin.
        title: Plot title
    
    Returns:
        Figure object
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert to numpy array if needed
    nozzle_directions = np.array(nozzle_directions)
    if nozzle_positions is not None:
        nozzle_positions = np.array(nozzle_positions)
    
    # Color map based on index
    colors = plt.cm.rainbow(np.linspace(0, 1, len(nozzle_directions)))
    
    # Plot robot body reference frame
    origin = np.array([0, 0, 0])
    axis_length = 0.15
    
    ax.quiver(origin[0], origin[1], origin[2], axis_length, 0, 0, 
              color='r', arrow_length_ratio=0.2, linewidth=2, alpha=0.5)
    ax.quiver(origin[0], origin[1], origin[2], 0, axis_length, 0, 
              color='g', arrow_length_ratio=0.2, linewidth=2, alpha=0.5)
    ax.quiver(origin[0], origin[1], origin[2], 0, 0, axis_length, 
              color='b', arrow_length_ratio=0.2, linewidth=2, alpha=0.5)
    
    arrow_length = 0.08
    
    # Plot each nozzle direction
    for i, direction in enumerate(nozzle_directions):
        direction_normalized = direction / (np.linalg.norm(direction) + 1e-8)
        
        if nozzle_positions is not None:
            start_pos = nozzle_positions[i]
        else:
            start_pos = origin
        
        ax.quiver(start_pos[0], start_pos[1], start_pos[2],
                  direction_normalized[0] * arrow_length,
                  direction_normalized[1] * arrow_length,
                  direction_normalized[2] * arrow_length,
                  color=colors[i], arrow_length_ratio=0.3, linewidth=1.5, alpha=0.8)
        
        # Plot starting point
        ax.scatter(*start_pos, color=colors[i], s=50, alpha=0.6)
    
    # Set labels and limits
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)
    
    limit = 0.15
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig
