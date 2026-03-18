from numba import jit
import numpy as np 

# checked
@jit(nopython=True, cache=True)
def compute_linear_acceleration_jit(mass_matrix, jet_force, drag_force, added_mass_force, coriolis_force, noise_force, acceleraiton_force):
    """Fast compiled computation of Newton's equations."""
    total_force = jet_force + drag_force + added_mass_force + coriolis_force + noise_force + acceleraiton_force
    # np.linalg.solve is faster and more stable than inv() @ vector
    return np.linalg.solve(mass_matrix, total_force)

# checked
@jit(nopython=True, cache=True)
def compute_angular_acceleration_jit(inertia_matrix, jet_torque, drag_torque, coriolis_torque, asymmetry_torque, deform_torque, added_mass_torque, noise_torque):
    """Fast compiled computation of Euler's equations."""
    total_torque = jet_torque + drag_torque + coriolis_torque + asymmetry_torque + deform_torque + added_mass_torque + noise_torque
    return np.linalg.solve(inertia_matrix, total_torque)

# checked
@jit(nopython=True, cache=True)
def to_euler_angle_rate_jit(euler_angle, angular_velocity):
    """Fast compiled conversion to Euler angle rates."""
    phi, theta, _ = euler_angle
    
    # Pre-allocate array for speed
    T = np.array([
        [1.0, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
        [0.0, np.cos(phi), -np.sin(phi)],
        [0.0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
    ])
    return T @ angular_velocity

# checked
@jit(nopython=True, cache=True)
def to_world_frame_jit(euler_angle, vector):
    """Fast compiled rotation from body frame to world frame."""
    phi, theta, psi = euler_angle
    
    R_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(phi), -np.sin(phi)],
        [0.0, np.sin(phi), np.cos(phi)]
    ])
    
    R_y = np.array([
        [np.cos(theta), 0.0, np.sin(theta)],
        [0.0, 1.0, 0.0],
        [-np.sin(theta), 0.0, np.cos(theta)]
    ])
    
    R_z = np.array([
        [np.cos(psi), -np.sin(psi), 0.0],
        [np.sin(psi), np.cos(psi), 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    R = R_z @ R_y @ R_x
    return R @ vector

@jit(nopython=True, cache=True)
def to_body_frame_jit(euler_angle, vector):
    """Fast compiled rotation from body frame to world frame."""
    phi, theta, psi = euler_angle
    
    R_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(phi), -np.sin(phi)],
        [0.0, np.sin(phi), np.cos(phi)]
    ])
    
    R_y = np.array([
        [np.cos(theta), 0.0, np.sin(theta)],
        [0.0, 1.0, 0.0],
        [-np.sin(theta), 0.0, np.cos(theta)]
    ])
    
    R_z = np.array([
        [np.cos(psi), -np.sin(psi), 0.0],
        [np.sin(psi), np.cos(psi), 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    R = R_z @ R_y @ R_x
    return R.T @ vector

# ==================== Numba Force & Torque Calculations ====================
@jit(nopython=True, cache=True)
def compute_jet_velocity_jit(state_val, volume, prev_water_volume, dt, nozzle_area, nozzle_direction):
    """Fast compiled jet velocity calculation."""
    if state_val != 1:  # Only produce jet velocity during JET phase
        return np.zeros(3)
    volume_rate = (volume - prev_water_volume) / dt
    jet_speed = volume_rate / nozzle_area
    return nozzle_direction * jet_speed

# checked
@jit(nopython=True, cache=True)
def compute_jet_force_jit(discharge_coeff, mass_rate, jet_velocity):
    """Fast compiled calculation of jet propulsion force."""
    # mass_rate is a 3x3 diagonal matrix, jet_velocity is a 1D array (3,)
    return -discharge_coeff * (mass_rate @ jet_velocity)

# checked 
@jit(nopython=True, cache=True)
def compute_jet_torque_jit(moment_arm, jet_force):
    """Fast compiled calculation of torque from jet force."""
    return np.cross(moment_arm, jet_force)

# checked
@jit(nopython=True, cache=True)
def compute_drag_force_jit(density, area, trans_drag_coeff, velocity, drag_force_ratio):
    """Fast compiled calculation of translational drag force."""
    v_norm = np.linalg.norm(velocity)
    F_quadratic = -0.5 * density * area * trans_drag_coeff * v_norm * velocity
    F_linear = -0.5 * density * area * trans_drag_coeff * velocity
    return F_quadratic + drag_force_ratio * F_linear

# checked
@jit(nopython=True, cache=True)
def compute_drag_torque_jit(density, rot_drag_coeff, area, angular_velocity, width, length, drag_torque_ratio):
    """Fast compiled calculation of rotational drag torque."""
    w_norm = np.linalg.norm(angular_velocity)
    # Dimensions array for the quadratic drag term
    dims = np.array([width**3, length**3, length**3])
    
    T_quadratic = -0.5 * density * rot_drag_coeff * area * w_norm * angular_velocity * dims
    T_linear = -0.5 * density * rot_drag_coeff * area * angular_velocity * width
    return T_quadratic + drag_torque_ratio * T_linear

# checked
@jit(nopython=True, cache=True)
def compute_added_mass_force_jit(mass, added_mass_coeff, mass_rate, added_mass_rate_coeff, acceleration, angular_velocity, velocity):
    """Fast compiled calculation of added mass force."""
    added_mass = mass @ added_mass_coeff
    added_mass_rate = mass_rate @ added_mass_rate_coeff
    
    term1 = added_mass @ acceleration
    term2 = np.cross(angular_velocity, added_mass @ velocity)
    term3 = added_mass_rate @ velocity
    
    return -(term1 + term2 + term3)

# checked
@jit(nopython=True, cache=True)
def compute_added_mass_torque_jit(I, added_mass_coeff_torque, I_rate, added_mass_rate_coeff_torque, mass, added_mass_coeff_force, angular_acceleration, angular_velocity, velocity):
    """Fast compiled calculation of added mass torque."""
    added_mass = I @ added_mass_coeff_torque
    added_mass_rate = I_rate @ added_mass_rate_coeff_torque
    added_mass_force_matrix = mass @ added_mass_coeff_force
    
    term1 = added_mass @ angular_acceleration
    term2 = np.cross(angular_velocity, added_mass @ angular_velocity)
    term3 = added_mass_rate @ angular_velocity
    term4 = np.cross(velocity, added_mass_force_matrix @ velocity)
    
    return -(term1 + term2 + term3 + term4)

# checked
@jit(nopython=True, cache=True)
def compute_coriolis_force_jit(angular_velocity, mass, velocity):
    """Fast compiled calculation of Coriolis force."""
    return -np.cross(angular_velocity, mass @ velocity)

# checked
@jit(nopython=True, cache=True)
def compute_coriolis_torque_jit(angular_velocity, I):
    """Fast compiled calculation of Coriolis torque."""
    return -np.cross(angular_velocity, I @ angular_velocity)

# checked
@jit(nopython=True, cache=True)
def compute_deform_torque_jit(I_rate, angular_velocity):
    """Fast compiled calculation of deformation torque."""
    return -(I_rate @ angular_velocity)

# checked
@jit(nopython=True, cache=True)
def compute_asymmetry_torque_jit(velocity):
    """Fast compiled calculation of asymmetry torque."""
    v_norm = np.linalg.norm(velocity)
    return np.array([0.0, 0.0, 0.00 * v_norm])