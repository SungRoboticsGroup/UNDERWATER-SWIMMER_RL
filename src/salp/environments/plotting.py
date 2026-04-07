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
    ax.axvspan(time_data[start_idx-1], time_data[-1], 
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
    plt.show(block=False)
    plt.pause(0.001)
    
    return fig


def plot_nozzle_yaw_angle(time_data, yaw_data, state_data=None, title="Nozzle Yaw Angle Over Time"):
    """
    Plot nozzle yaw angle over time.
    
    Args:
        time_data: Array of time values
        yaw_data: Array of nozzle yaw angle values (in radians)
        state_data: Optional array of state values (0: refill, 1: jet, 2: coast, 3: rest)
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add phase backgrounds if state_data provided
    if state_data is not None:
        _add_phase_backgrounds(ax, time_data, state_data)
    
    # Convert to degrees for better readability
    yaw_degrees = np.degrees(yaw_data)
    
    # Plot nozzle yaw angle
    ax.plot(time_data, yaw_degrees, 'purple', linewidth=2, label='Nozzle Yaw', zorder=3)
    
    # Add reference lines at 0 and typical limits
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=90, color='red', linestyle='--', linewidth=0.8, alpha=0.3, label='±90°')
    ax.axhline(y=-90, color='red', linestyle='--', linewidth=0.8, alpha=0.3)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Yaw Angle (degrees)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    
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
    plt.show(block=False)
    plt.pause(0.001)
    
    return fig


def plot_mass_rate(time_data, mass_rate_data, state_data=None, title="Rate of Change of Mass Over Time"):
    """
    Plot the rate of change of mass over time.
    
    Args:
        time_data: Array of time values
        mass_rate_data: Array of mass rate values
        state_data: Optional array of state values (Phase enum values)
        title: Plot title
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add phase backgrounds if state_data provided
    if state_data is not None:
        _add_phase_backgrounds(ax, time_data, state_data)
    
    ax.plot(time_data, mass_rate_data, 'purple', linewidth=2, label='Mass Rate (dm/dt)', zorder=3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mass Rate (kg/s)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    
    return fig


def plot_volume_rate(time_data, volume_rate_data, state_data=None, title="Rate of Change of Volume Over Time"):
    """
    Plot the rate of change of volume over time.
    
    Args:
        time_data: Array of time values
        volume_rate_data: Array of volume rate values
        state_data: Optional array of state values (Phase enum values)
        title: Plot title
    """
    # Calculate rate of change of volume (dV/dt)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add phase backgrounds if state_data provided
    if state_data is not None:
        _add_phase_backgrounds(ax, time_data, state_data)
    
    ax.plot(time_data, volume_rate_data, 'orange', linewidth=2, label='Volume Rate (dV/dt)', zorder=3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Volume Rate (m³/s)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    
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
    plt.show(block=False)
    plt.pause(0.001)
    
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
    plt.show(block=False)
    plt.pause(0.001)
    
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
    plt.show(block=False)
    plt.pause(0.001)
    
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
    plt.show(block=False)
    plt.pause(0.001)
    
    return fig


def plot_drag_coefficient(time_data, drag_coefficient_data, 
                         state_data=None, title="Drag Coefficient Over Time"):
    """
    Plot drag coefficient over time for all three axes.
    
    Args:
        time_data: Array of time values
        drag_coefficient_data: Array of drag coefficient values (Nx3 for X, Y, Z)
        state_data: Optional array of state values (0: refill, 1: jet, 2: coast, 3: rest)
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axis_labels = ['X', 'Y', 'Z']
    colors = ['blue', 'green', 'red']
    
    # Ensure drag_coefficient_data is 2D
    if len(drag_coefficient_data.shape) == 1:
        drag_coefficient_data = drag_coefficient_data.reshape(-1, 1)
    
    for idx, (ax, label, color) in enumerate(zip(axes, axis_labels, colors)):
        # Add phase backgrounds if state_data provided
        if state_data is not None:
            _add_phase_backgrounds(ax, time_data, state_data)
        
        # Plot the data for this axis
        if drag_coefficient_data.shape[1] > idx:
            ax.plot(time_data, drag_coefficient_data[:, idx], color=color, linewidth=2, zorder=3, label=f'Drag Coeff {label}')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Drag Coefficient {label}')
        ax.set_title(f'{title} - {label} Direction')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    
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
    plt.show(block=False)
    plt.pause(0.001)
    
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
    plt.show(block=False)
    plt.pause(0.001)
    
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
    plt.show(block=False)
    plt.pause(0.001)
    
    return fig


def plot_cross_sectional_area(time_data, area_data, 
                             state_data=None, title="Robot Cross-Sectional Areas Over Time"):
    """
    Plot all three cross-sectional areas in separate subplots.
    
    Args:
        time_data: Array of time values
        area_data: Array of cross-sectional area values (3D: [A_yz, A_xz, A_xy])
        state_data: Optional array of state values (0: refill, 1: jet, 2: coast, 3: rest)
        title: Plot title
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # Extract individual area components
    area_data = np.array(area_data)
    if area_data.ndim == 2:
        A_yz = area_data[:, 0]
        A_xz = area_data[:, 1]
        A_xy = area_data[:, 2]
        
        # Plot A_yz
        if state_data is not None:
            _add_phase_backgrounds(axes[0], time_data, state_data)
        axes[0].plot(time_data, A_yz, 'teal', linewidth=2)
        axes[0].set_ylabel('Area (m²)')
        axes[0].set_title('A_yz (y-z plane)')
        axes[0].grid(True, alpha=0.3)
        
        # Plot A_xz
        if state_data is not None:
            _add_phase_backgrounds(axes[1], time_data, state_data)
        axes[1].plot(time_data, A_xz, 'coral', linewidth=2)
        axes[1].set_ylabel('Area (m²)')
        axes[1].set_title('A_xz (x-z plane)')
        axes[1].grid(True, alpha=0.3)
        
        # Plot A_xy
        if state_data is not None:
            _add_phase_backgrounds(axes[2], time_data, state_data)
        axes[2].plot(time_data, A_xy, 'steelblue', linewidth=2)
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Area (m²)')
        axes[2].set_title('A_xy (x-y plane)')
        axes[2].grid(True, alpha=0.3)
    else:
        # Fallback for 1D array
        if state_data is not None:
            _add_phase_backgrounds(axes[0], time_data, state_data)
        axes[0].plot(time_data, area_data, 'teal', linewidth=2)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Area (m²)')
        axes[0].set_title('Cross-Sectional Area')
        axes[0].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    
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
    plt.show(block=False)
    plt.pause(0.001)
    
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
    plt.show(block=False)
    plt.pause(0.001)
    
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
    plt.show(block=False)
    plt.pause(0.001)
    
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
    plt.show(block=False)
    plt.pause(0.001)
    
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
    plt.show(block=False)
    plt.pause(0.001)
    
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
    plt.show(block=False)
    plt.pause(0.001)
    
    return fig

def plot_trajectory_xy(position_data: np.ndarray, state_data: np.ndarray = None,
                       euler_angle_data: np.ndarray = None, nozzle_yaw_data: np.ndarray = None):
    """Plot the robot's trajectory in the x-y plane.
    
    Args:
        position_data: Nx3 array of position data [x, y, z]
        state_data: Optional array of phase states for color coding
        euler_angle_data: Optional Nx3 array of euler angles [roll, pitch, yaw] for body orientation visualization
        nozzle_yaw_data: Optional array of nozzle yaw values (radians). When provided,
                nozzle orientation is shown along the trajectory.
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
    
    yaw_angles = None
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

    # Add nozzle orientation vectors along trajectory
    if nozzle_yaw_data is not None:
        if len(nozzle_yaw_data) != len(x_positions):
            raise ValueError(
                f"Length mismatch: nozzle_yaw_data has {len(nozzle_yaw_data)} samples, "
                f"but position_data has {len(x_positions)} samples"
            )

        # Nozzle is at the rear of the robot, so we add π to get the rear-facing direction
        # then add the nozzle yaw angle (relative to body)
        if yaw_angles is not None:
            nozzle_world_angles = yaw_angles + np.pi + nozzle_yaw_data
        else:
            nozzle_world_angles = np.pi + nozzle_yaw_data

        nozzle_vector_length = triangle_size * 0.9
        for i in range(0, len(x_positions), arrow_interval):
            nozzle_end_x = x_positions[i] + nozzle_vector_length * np.cos(nozzle_world_angles[i])
            nozzle_end_y = y_positions[i] + nozzle_vector_length * np.sin(nozzle_world_angles[i])
            ax.plot(
                [x_positions[i], nozzle_end_x],
                [y_positions[i], nozzle_end_y],
                color='purple', linewidth=1.4, alpha=0.85, zorder=4
            )

        ax.plot([], [], color='purple', linewidth=1.8, label='Nozzle orientation')
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    title = 'Robot Trajectory in X-Y Plane'
    if euler_angle_data is not None:
        title += ' (Arrows show yaw orientation)'
    if nozzle_yaw_data is not None:
        title += ' + nozzle orientation'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    if state_data is None:
        ax.legend()
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)


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
    plt.show(block=False)
    plt.pause(0.001)
    
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
    plt.show(block=False)
    plt.pause(0.001)
    
    return fig


def plot_coriolis_force(time_data, coriolis_force_data, 
                        state_data=None, title="Coriolis Force Over Time"):
    """
    Plot Coriolis force in X, Y, Z dimensions.
    
    Args:
        time_data: Array of time values
        coriolis_force_data: Array of Coriolis force values (3D vectors)
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
        
        # Plot Coriolis force in each direction
        ax.plot(time_data, coriolis_force_data[:, i], color=color, linewidth=2, 
                label=f'Coriolis Force {direction}', zorder=3)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Coriolis Force {direction} (N)')
        ax.set_title(f'Coriolis Force in {direction} Direction')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    
    return fig


def plot_acceleration_force(time_data, acceleration_force_data, 
                            state_data=None, title="Acceleration Force Over Time"):
    """
    Plot acceleration force (fictitious forces due to moving center of mass) in X, Y, Z dimensions.
    
    The acceleration force includes contributions from:
    - Centripetal acceleration
    - Coriolis acceleration (from rotating frame)
    - Tangential acceleration
    - Recoil acceleration (from center of mass movement)
    
    Args:
        time_data: Array of time values
        acceleration_force_data: Array of acceleration force values (3D vectors in Newtons)
        state_data: Optional array of state values (Phase enum values)
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
        
        # Plot acceleration force in each direction
        ax.plot(time_data, acceleration_force_data[:, i], color=color, linewidth=2, 
                label=f'Acceleration Force {direction}', zorder=3)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Acceleration Force {direction} (N)')
        ax.set_title(f'Acceleration Force in {direction} Direction')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    
    return fig


def plot_added_mass_force(time_data, added_mass_force_data, 
                          state_data=None, title="Added Mass Force Over Time"):
    """
    Plot added mass force in X, Y, Z dimensions.
    
    Args:
        time_data: Array of time values
        added_mass_force_data: Array of added mass force values (3D vectors)
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
        
        # Plot added mass force in each direction
        ax.plot(time_data, added_mass_force_data[:, i], color=color, linewidth=2, 
                label=f'Added Mass Force {direction}', zorder=3)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Added Mass Force {direction} (N)')
        ax.set_title(f'Added Mass Force in {direction} Direction')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    
    return fig


def plot_all_forces(time_data, jet_force_data, drag_force_data, coriolis_force_data, 
                    added_mass_force_data, state_data=None, title="All Forces Comparison Over Time"):
    """
    Plot jet force, drag force, Coriolis force, and added mass force together.
    
    Args:
        time_data: Array of time values
        jet_force_data: Array of jet force values (3D vectors)
        drag_force_data: Array of drag force values (3D vectors)
        coriolis_force_data: Array of Coriolis force values (3D vectors)
        added_mass_force_data: Array of added mass force values (3D vectors)
        state_data: Optional array of state values (Phase enum values)
        title: Plot title
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    directions = ['X', 'Y', 'Z']
    
    for i, (ax, direction) in enumerate(zip(axes, directions)):
        # Add phase backgrounds if state_data provided
        if state_data is not None:
            _add_phase_backgrounds(ax, time_data, state_data)
        
        # Plot all four forces in each direction
        ax.plot(time_data, jet_force_data[:, i], color='red', linewidth=2, 
                label='Jet Force', zorder=3)
        ax.plot(time_data, drag_force_data[:, i], color='blue', linewidth=2, 
                label='Drag Force', zorder=3)
        ax.plot(time_data, coriolis_force_data[:, i], color='green', linewidth=2, 
                label='Coriolis Force', zorder=3)
        ax.plot(time_data, added_mass_force_data[:, i], color='orange', linewidth=2, 
                label='Added Mass Force', zorder=3)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Force {direction} (N)')
        ax.set_title(f'All Forces in {direction} Direction')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    
    return fig


def plot_coriolis_torque(time_data, coriolis_torque_data, 
                         state_data=None, title="Coriolis Torque Over Time"):
    """
    Plot Coriolis torque in X, Y, Z dimensions.
    
    Args:
        time_data: Array of time values
        coriolis_torque_data: Array of Coriolis torque values (3D vectors)
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
        
        # Plot Coriolis torque in each direction
        ax.plot(time_data, coriolis_torque_data[:, i], color=color, linewidth=2, 
                label=f'Coriolis Torque {direction}', zorder=3)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Coriolis Torque {direction} (N·m)')
        ax.set_title(f'Coriolis Torque in {direction} Direction')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    
    return fig


def plot_deform_torque(time_data, deform_torque_data, 
                       state_data=None, title="Deformation Torque Over Time"):
    """
    Plot deformation torque in X, Y, Z dimensions.
    
    Args:
        time_data: Array of time values
        deform_torque_data: Array of deformation torque values (3D vectors)
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
        
        # Plot deformation torque in each direction
        ax.plot(time_data, deform_torque_data[:, i], color=color, linewidth=2, 
                label=f'Deform Torque {direction}', zorder=3)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Deform Torque {direction} (N·m)')
        ax.set_title(f'Deformation Torque in {direction} Direction')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    
    return fig


def plot_added_mass_torque(time_data, added_mass_torque_data, 
                           state_data=None, title="Added Mass Torque Over Time"):
    """
    Plot added mass torque in X, Y, Z dimensions.
    
    Args:
        time_data: Array of time values
        added_mass_torque_data: Array of added mass torque values (3D vectors)
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
        
        # Plot added mass torque in each direction
        ax.plot(time_data, added_mass_torque_data[:, i], color=color, linewidth=2, 
                label=f'Added Mass Torque {direction}', zorder=3)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Added Mass Torque {direction} (N·m)')
        ax.set_title(f'Added Mass Torque in {direction} Direction')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    
    return fig


def plot_asymmetry_torque(time_data, asymmetry_torque_data, 
                          state_data=None, title="Asymmetry Torque Over Time"):
    """
    Plot asymmetry torque in X, Y, Z dimensions.
    
    Args:
        time_data: Array of time values
        asymmetry_torque_data: Array of asymmetry torque values (3D vectors)
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
        
        # Plot asymmetry torque in each direction
        ax.plot(time_data, asymmetry_torque_data[:, i], color=color, linewidth=2, 
                label=f'Asymmetry Torque {direction}', zorder=3)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Asymmetry Torque {direction} (N·m)')
        ax.set_title(f'Asymmetry Torque in {direction} Direction')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    
    return fig


def _extract_principal_axes_series(tensor_data):
    """Normalize inertia tensor history to an Nx3 principal-axis series."""
    tensor_data = np.asarray(tensor_data)

    if tensor_data.ndim == 3 and tensor_data.shape[1:] == (3, 3):
        return np.diagonal(tensor_data, axis1=1, axis2=2)

    if tensor_data.ndim == 2 and tensor_data.shape[1] == 3:
        return tensor_data

    raise ValueError(
        "Expected inertia tensor data with shape (N, 3) or (N, 3, 3); "
        f"received shape {tensor_data.shape}."
    )


def plot_inertia_tensor(time_data, inertia_tensor_data, 
                        state_data=None, title="Inertia Tensor Principal Axes Over Time"):
    """
    Plot the 3 principal axes inertia tensor values over time.
    
    Args:
        time_data: Array of time values
        inertia_tensor_data: Array of inertia tensor diagonal values (Nx3 for I_xx, I_yy, I_zz)
        state_data: Optional array of state values (Phase enum values)
        title: Plot title
    """
    principal_axes_data = _extract_principal_axes_series(inertia_tensor_data)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axes_labels = ['I_xx (X-axis)', 'I_yy (Y-axis)', 'I_zz (Z-axis)']
    colors = ['r', 'g', 'b']
    
    for i, (ax, label, color) in enumerate(zip(axes, axes_labels, colors)):
        # Add phase backgrounds if state_data provided
        if state_data is not None:
            _add_phase_backgrounds(ax, time_data, state_data)
        
        # Plot inertia tensor component for this axis
        ax.plot(time_data, principal_axes_data[:, i], color=color, linewidth=2, 
                label=label, zorder=3)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{label} (kg·m²)')
        ax.set_title(f'Inertia Tensor Component {label}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    
    return fig


def plot_inertia_tensor_rate(time_data, inertia_tensor_rate_data,
                             state_data=None, title="Inertia Tensor Rate Over Time"):
    """
    Plot the rate of change of the 3 principal inertia tensor axes over time.

    Args:
        time_data: Array of time values
        inertia_tensor_rate_data: Array of inertia tensor rate diagonal values
            (Nx3 for dI_xx/dt, dI_yy/dt, dI_zz/dt)
        state_data: Optional array of state values (Phase enum values)
        title: Plot title
    """
    principal_axes_rate_data = _extract_principal_axes_series(inertia_tensor_rate_data)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes_labels = ['dI_xx/dt (X-axis)', 'dI_yy/dt (Y-axis)', 'dI_zz/dt (Z-axis)']
    colors = ['r', 'g', 'b']

    for i, (ax, label, color) in enumerate(zip(axes, axes_labels, colors)):
        if state_data is not None:
            _add_phase_backgrounds(ax, time_data, state_data)

        ax.plot(time_data, principal_axes_rate_data[:, i], color=color, linewidth=2,
                label=label, zorder=3)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{label} (kg·m²/s)')
        ax.set_title(f'Inertia Tensor Rate Component {label}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

    return fig


def plot_center_of_mass(time_data, center_of_mass_data, 
                        state_data=None, title="Center of Mass Over Time"):
    """
    Plot the center of mass position in X, Y, Z dimensions over time.
    
    Args:
        time_data: Array of time values
        center_of_mass_data: Array of center of mass positions (Nx3 for X, Y, Z coordinates)
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
        
        # Plot center of mass component in each direction
        ax.plot(time_data, center_of_mass_data[:, i], color=color, linewidth=2, 
                label=f'CoM {direction}', zorder=3)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Center of Mass {direction} (m)')
        ax.set_title(f'Center of Mass Position - {direction} Direction')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    
    return fig


def plot_center_of_mass_rate(time_data, center_of_mass_rate_data,
                             state_data=None, title="Rate of Center of Mass Over Time"):
    """
    Plot the rate of center of mass in X, Y, Z dimensions over time.

    Args:
        time_data: Array of time values
        center_of_mass_rate_data: Array of center of mass rate values (Nx3 for X, Y, Z coordinates)
        state_data: Optional array of state values (Phase enum values)
        title: Plot title
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    directions = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']

    for i, (ax, direction, color) in enumerate(zip(axes, directions, colors)):
        if state_data is not None:
            _add_phase_backgrounds(ax, time_data, state_data)

        ax.plot(time_data, center_of_mass_rate_data[:, i], color=color, linewidth=2,
                label=f'dCoM_{direction}/dt', zorder=3)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'CoM Rate {direction} (m/s)')
        ax.set_title(f'Rate of Center of Mass - {direction} Direction')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

    return fig


def plot_center_of_mass_acc_rate(time_data, center_of_mass_acc_rate_data,
                                 state_data=None, title="Acceleration Rate of Center of Mass Over Time"):
    """
    Plot the acceleration rate of center of mass in X, Y, Z dimensions over time.

    This computes the second time derivative of center-of-mass position:
    d²(CoM)/dt².

    Args:
        time_data: Array of time values
        center_of_mass_acc_rate_data: Array of center of mass acceleration rate values (Nx3 for X, Y, Z coordinates)
        state_data: Optional array of state values (Phase enum values)
        title: Plot title
    """

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    directions = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']

    for i, (ax, direction, color) in enumerate(zip(axes, directions, colors)):
        if state_data is not None:
            _add_phase_backgrounds(ax, time_data, state_data)

        ax.plot(time_data, center_of_mass_acc_rate_data[:, i], color=color, linewidth=2,
                label=f'd²CoM_{direction}/dt²', zorder=3)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'CoM Acc Rate {direction} (m/s²)')
        ax.set_title(f'Acceleration Rate of Center of Mass - {direction} Direction')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

    return fig


def plot_front_position_body_frame(time_data, front_position_data,
                                   state_data=None, title="Robot Front Position in Body Frame"):
    """
    Plot the robot front position in the body frame over time.

    Args:
        time_data: Array of time values
        front_position_data: Array of front positions (Nx3 for X, Y, Z coordinates)
        state_data: Optional array of state values (Phase enum values)
        title: Plot title
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    directions = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']

    for i, (ax, direction, color) in enumerate(zip(axes, directions, colors)):
        if state_data is not None:
            _add_phase_backgrounds(ax, time_data, state_data)

        ax.plot(time_data, front_position_data[:, i], color=color, linewidth=2,
                label=f'Front Position {direction}', zorder=3)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Front Position {direction} (m)')
        ax.set_title(f'Front Position in Body Frame - {direction} Direction')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

    return fig


def plot_front_position_world_frame(time_data, front_position_world_data,
                                    state_data=None, title="Robot Front Position in World Frame"):
    """
    Plot the robot front position in the world frame over time.

    Args:
        time_data: Array of time values
        front_position_world_data: Array of front positions in world frame (Nx3 for X, Y, Z coordinates)
        state_data: Optional array of state values (Phase enum values)
        title: Plot title
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    directions = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']

    for i, (ax, direction, color) in enumerate(zip(axes, directions, colors)):
        if state_data is not None:
            _add_phase_backgrounds(ax, time_data, state_data)

        ax.plot(time_data, front_position_world_data[:, i], color=color, linewidth=2,
                label=f'Front World Position {direction}', zorder=3)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Front World Position {direction} (m)')
        ax.set_title(f'Front Position in World Frame - {direction} Direction')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

    return fig
