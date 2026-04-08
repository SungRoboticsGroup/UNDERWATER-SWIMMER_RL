"""Experiment utilities.

Provides a small helper that uses the shared Excel parser to load the
first 9 columns of an attached measurement file. Includes a CLI for
quick local testing.
"""

import os
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import RotationSpline
from scipy.interpolate import CubicSpline


# Absolute path to local mocap data. Use a raw string to preserve backslashes.
datapath = r"E:\model_calibration_data\mocap_data"


def lowpass_filter(data: np.ndarray, cutoff: float = 5.0, fs: float = 120.0, order: int = 2) -> np.ndarray:
    """Apply a low-pass Butterworth filter to the data.
    
    Args:
        data: Input array (N,) or (N, M) where filtering is applied along axis 0.
        cutoff: Cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order.
    
    Returns:
        Filtered data with same shape as input.
    """
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    # Apply filter along axis 0 for multi-dimensional arrays
    if data.ndim == 1:
        return filtfilt(b, a, data)
    else:
        return np.apply_along_axis(lambda x: filtfilt(b, a, x), 0, data)


def quaternion_to_euler(quaternions: np.ndarray) -> np.ndarray:
    """Convert quaternions (N x 4: x, y, z, w) to Euler angles (N x 3: roll, pitch, yaw) in radians."""
    # scipy expects [x, y, z, w] order which matches our CSV columns
    rot = Rotation.from_quat(quaternions)

    init_rot = rot[0]
    rot = init_rot.inv() * rot  # Normalize to start at zero rotation
    # Returns roll (x), pitch (y), yaw (z) in radians
    euler_angles = rot.as_euler('xyz', degrees=False)

    return euler_angles



def compute_velocities(time: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Compute velocities from time and positions using finite differences.
    
    Returns an (N x 3) array of velocities. The first velocity is set to zero.
    """

    pos_spline = CubicSpline(time, positions)

    # 2. Extract the exact First Derivative (Linear Velocity in meters/sec or mm/sec)
    # pos_spline.derivative(1) creates a new function representing the velocity,
    # which we immediately evaluate at your given 'times'.
    velocities = pos_spline.derivative(1)(time)
    
    
    # dt = np.diff(time)
    # dp = np.diff(positions, axis=0)
    # # Avoid division by zero
    # dt = np.where(dt == 0, 1e-8, dt)
    # velocities = dp / dt[:, np.newaxis]
    # # Prepend zero velocity for the first time step
    # velocities = np.vstack([np.zeros(3), velocities])
    return velocities


def read_file(file_name: str = "", start_time: int = 0) -> tuple:
    """Read mocap CSV and return time, positions, velocities, euler_angles."""
    df = pd.read_csv(
        os.path.join(datapath, file_name),
        usecols=range(9),
        skiprows=6
    )

    time = df['Time (Seconds)'].to_numpy()
    positions = df[['X.1', 'Y.1', 'Z.1']].to_numpy()
    orientations = df[['X', 'Y', 'Z', 'W']].to_numpy()

    idx = time >= start_time
    time = time[idx]
    positions = positions[idx]
    orientations = orientations[idx]
    
    # Filter out zero quaternions (all four components are zero)
    quat_norms = np.linalg.norm(orientations, axis=1)
    nonzero_mask = quat_norms > 1e-8
    time = time[nonzero_mask]
    positions = positions[nonzero_mask]
    orientations = orientations[nonzero_mask]
    
    positions -= positions[0, :]  # Normalize to start at origin
    
    # Compute velocities and apply low-pass filter to reduce noise
    velocities = compute_velocities(time, positions)
    velocities = lowpass_filter(velocities, cutoff=5.0, fs=120.0, order=2)
    
    # Convert quaternions to Euler angles (roll, pitch, yaw)
    euler_angles = quaternion_to_euler(orientations)

    return time, positions, velocities, euler_angles


def plot_all_trajectories_xz(file_names: list, start_time: list, marker_step: int = 200, marker_size: float = 0.02, align_to_x: bool = False) -> None:
    """Plot all trajectory positions in the X-Z plane with yaw direction triangles.
    
    Args:
        file_names: List of CSV file names to plot.
        start_time: List of start times for each file.
        marker_step: Step interval for drawing yaw direction triangles.
        marker_size: Size of the direction triangles.
        align_to_x: If True, rotate all trajectories so the first trajectory
                    aligns as closely as possible to the x-axis.
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    # Compute rotation angle from first trajectory if align_to_x is True
    rotation_angle = 0.0
    if align_to_x and len(file_names) > 0:
        time_ref, positions_ref, _, _ = read_file(file_names[0], start_time=start_time[0])
        x_ref, z_ref = positions_ref[:, 0], positions_ref[:, 2]
        # Compute angle from start to end point
        dx = x_ref[-1] - x_ref[0]
        dz = z_ref[-1] - z_ref[0]
        rotation_angle = -np.arctan2(dz, dx)  # Negative to rotate towards x-axis

    for file_name, start in zip(file_names, start_time):
        time, positions, velocities, euler_angles = read_file(file_name, start_time=start)

        x, z = positions[:, 0], positions[:, 2]
        yaw = euler_angles[:, 1]  # yaw is the third column (rotation about z)

        # Apply rotation if align_to_x is True
        if align_to_x:
            c, s = np.cos(rotation_angle), np.sin(rotation_angle)
            x_rot = c * x - s * z
            z_rot = s * x + c * z
            x, z = x_rot, z_rot
            yaw = yaw + rotation_angle  # Adjust yaw angles as well

        # Use filename (without extension) as label
        label = os.path.splitext(file_name)[0]
        line, = ax.plot(x, z, label=label)
        color = line.get_color()

        # Draw triangles showing yaw direction at sampled points
        indices = np.arange(0, len(x), marker_step)
        for i in indices:
            # Create a triangle pointing in the yaw direction
            angle = yaw[i]
            # Triangle vertices (pointing right, then rotated)
            tri = np.array([
                [1, 0],
                [-0.5, 0.5],
                [-0.5, -0.5],
            ]) * marker_size
            # Rotation matrix
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s], [s, c]])
            tri_rot = tri @ R.T
            # Translate to position
            tri_rot[:, 0] += x[i]
            tri_rot[:, 1] += z[i]
            triangle = plt.Polygon(tri_rot, closed=True, facecolor=color, edgecolor=color)
            ax.add_patch(triangle)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Trajectories (X-Z Plane) with Yaw Direction')
    ax.legend(loc='upper left', fontsize='small')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_velocity_magnitudes(file_names: list, start_time: list) -> None:
    """Plot velocity magnitudes over time for all trajectories."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for file_name, start in zip(file_names, start_time):
        time, positions, velocities, euler_angles = read_file(file_name, start_time=start)
        # Compute velocity magnitude
        speed = np.linalg.norm(velocities, axis=1)
        # Normalize time to start at 0
        time = time - time[0]
        
        # Use filename (without extension) as label
        label = os.path.splitext(file_name)[0]
        ax.plot(time, speed, label=label)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity Magnitude (m/s)')
    ax.set_title('Velocity Magnitudes Over Time')
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_velocity_components_xz(file_names: list, start_time: list) -> None:
    """Plot X and Z velocity components over time for all trajectories."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for file_name, start in zip(file_names, start_time):
        time, positions, velocities, euler_angles = read_file(file_name, start_time=start)
        vx, vz = velocities[:, 0], velocities[:, 2]
        # Normalize time to start at 0
        time = time - time[0]
        
        # Use filename (without extension) as label
        label = os.path.splitext(file_name)[0]
        axes[0].plot(time, vx, label=label)
        axes[1].plot(time, vz, label=label)

    axes[0].set_ylabel('Vx (m/s)')
    axes[0].set_title('X Velocity Component Over Time')
    axes[0].legend(loc='upper right', fontsize='small')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Vz (m/s)')
    axes[1].set_title('Z Velocity Component Over Time')
    axes[1].legend(loc='upper right', fontsize='small')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def compute_average_velocities(file_names: list, start_time: list) -> dict:
    """Compute average velocity magnitude for each trajectory.
    
    Returns a dict mapping filename to average speed (m/s).
    """
    avg_velocities = {}
    print("\n--- Average Velocities ---")
    for file_name, start in zip(file_names, start_time):
        time, positions, velocities, euler_angles = read_file(file_name, start_time=start)
        speed = np.linalg.norm(velocities, axis=1)
        avg_speed = np.mean(speed)
        label = os.path.splitext(file_name)[0]
        avg_velocities[label] = avg_speed
        print(f"{label}: {avg_speed:.4f} m/s")
    print("--------------------------\n")
    return avg_velocities


def compute_yaw_rate(time: np.ndarray, euler_angles: np.ndarray) -> np.ndarray:
    """Compute yaw rate (time derivative of yaw angle) using finite differences.
    
    Args:
        time: Time array (N,)
        euler_angles: Euler angles array (N, 3) with columns [roll, pitch, yaw]
    
    Returns:
        Yaw rate array (N,) in rad/s. First value is set to zero.
    """

    spline = RotationSpline(time, Rotation.from_euler('xyz', euler_angles))

    # 2. Extract the true Angular Velocity (First Derivative)
    # The '1' requests the first derivative. 
    # Returns an array of 3D vectors: [omega_x, omega_y, omega_z] in rad/s
    angular_velocities_rad = spline(time, 1)

    # 3. Convert to degrees per second
    angular_velocities_deg = np.degrees(angular_velocities_rad)

    # 4. Isolate the Yaw Rate (Assuming Z is your vertical axis. Change to 1 if Y is up)
    yaw_rate = angular_velocities_deg[:, 1]


    # yaw = euler_angles[:, 1]  # yaw is the third column
    # dt = np.diff(time)
    # dyaw = np.diff(yaw)
    # # Handle angle wrapping (discontinuities at +/- pi)
    # dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))
    # # Avoid division by zero
    # dt = np.where(dt == 0, 1e-8, dt)
    # yaw_rate = dyaw / dt
    # # Prepend zero for the first time step
    # yaw_rate = np.concatenate([[0], yaw_rate])
    return yaw_rate


def plot_yaw_rate(file_names: list, start_time: list) -> None:
    """Plot yaw rate over time for all trajectories."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for file_name, start in zip(file_names, start_time):
        time, positions, velocities, euler_angles = read_file(file_name, start_time=start)
        yaw_rate = compute_yaw_rate(time, euler_angles)
        # Apply low-pass filter to reduce noise
        yaw_rate = lowpass_filter(yaw_rate, cutoff=5.0, fs=120.0, order=2)
        # Normalize time to start at 0
        time = time - time[0]
        
        # Use filename (without extension) as label
        label = os.path.splitext(file_name)[0]
        ax.plot(time, yaw_rate, label=label)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Yaw Rate (deg/s)')
    ax.set_title('Yaw Rate Over Time')
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_yaw_angle(file_names: list, start_time: list) -> None:
    """Plot yaw angle over time for all trajectories."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for file_name, start in zip(file_names, start_time):
        time, positions, velocities, euler_angles = read_file(file_name, start_time=start)
        euler_angles = Rotation.from_euler('xyz', euler_angles).as_euler('yxz', degrees=False)  # Re-normalize to handle wrapping
        yaw = np.unwrap(euler_angles[:, 0])  # yaw angle in radians
        # Normalize yaw to start at 0
        yaw = yaw - yaw[0]
        # Convert to degrees
        yaw_deg = np.degrees(yaw)
        # Normalize time to start at 0
        time = time - time[0]
        
        # Use filename (without extension) as label
        label = os.path.splitext(file_name)[0]
        ax.plot(time, yaw_deg, label=label)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Yaw Angle (deg)')
    ax.set_title('Yaw Angle Over Time')
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_nozzle_angle_comparison():

    start_time = [22, 165, 118, 22, 55]

    file_names = [
        'compression_3cm_coast_2s_nozzle_0deg.csv',
        'compression_3cm_coast_2s_nozzle_-30deg.csv',
        'compression_3cm_coast_2s_nozzle_30deg.csv',
        'compression_3cm_coast_2s_nozzle_90deg.csv',
        'compression_3cm_coast_2s_nozzle_-90deg.csv',
    ]

    plot_all_trajectories_xz(file_names, start_time, align_to_x=True)


if __name__ == "__main__":

    start_time = [22, 25, 22, 22, 22, 28, 165, 118, 22, 55]

    file_names = [
        'compression_1cm_coast_2s_nozzle_0deg.csv',
        'compression_2cm_coast_2s_nozzle_0deg.csv',
        'compression_3cm_coast_2s_nozzle_0deg.csv',
        'compression_4cm_coast_2s_nozzle_0deg.csv',
        'compression_3cm_coast_1s_nozzle_0deg.csv',
        'compression_3cm_coast_3s_nozzle_0deg.csv',
        'compression_3cm_coast_2s_nozzle_-30deg.csv',
        'compression_3cm_coast_2s_nozzle_30deg.csv',
        'compression_3cm_coast_2s_nozzle_90deg.csv',
        'compression_3cm_coast_2s_nozzle_-90deg.csv',
    ]

    # plot_all_trajectories_xz(file_names, start_time)
    # plot_velocity_magnitudes(file_names, start_time)
    # plot_velocity_components_xz(file_names, start_time)
    # compute_average_velocities(file_names, start_time)
    plot_yaw_rate(file_names, start_time)
    # plot_yaw_angle(file_names, start_time)

    # plot_nozzle_angle_comparison()