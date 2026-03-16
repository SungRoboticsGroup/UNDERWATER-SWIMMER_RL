from numba import jit
import numpy as np

# ==================== Numba Geometry & Property Calculations ====================
# data-driven refill time 
def fit_compression_refill_time_relation_jit():
    compression = np.array([0.01, 0.02, 0.03, 0.04])  # Example lengths during contraction
    refill_time = np.array([0.4, 1.0, 1.8, 2.2])   # Corresponding widths to maintain constant volume
    coefficients = np.polyfit(compression, refill_time, 2)  # Fit a polynomial of degree 2
    return coefficients

@jit(nopython=True, cache=True)
def refill_time_from_compression_jit(compression, coefficients):
    return coefficients[0] * compression**2 + coefficients[1] * compression + coefficients[2]  # Evaluate the polynomial at the given length

# data-driven propulsion time
def fit_compression_propulsion_time_relation_jit():
    compression = np.array([0.01, 0.02, 0.03, 0.04])  # Example lengths during contraction
    propulsion_time = np.array([0.1, 0.3, 0.4, 0.5])   # Corresponding widths to maintain constant volume
    coefficients = np.polyfit(compression, propulsion_time, 2)  # Fit a polynomial of degree 2
    return coefficients

@jit(nopython=True, cache=True)
def propulsion_time_from_compression_jit(compression, coefficients):
    return coefficients[0] * compression**2 + coefficients[1] * compression + coefficients[2]  # Evaluate the polynomial at the given length

# data-driven geometry relation
def fit_length_width_relation_jit():
    lengths = np.array([0.26, 0.25, 0.24, 0.23, 0.22, 0.21, 0.20])  # Example lengths during contraction
    widths = np.array([0.14, 0.16, 0.175, 0.18, 0.20, 0.21, 0.22])   # Corresponding widths to maintain constant volume
    coefficients = np.polyfit(lengths, widths, 2)  # Fit a polynomial of degree 2
    return coefficients

@jit(nopython=True, cache=True)
def width_from_length_jit(length, coefficients):
    return coefficients[0] * length**2 + coefficients[1] * length + coefficients[2]  # Evaluate the polynomial at the given length

# checked 
@jit(nopython=True, cache=True)
def compute_length_jit(state_val, cycle_time, refill_time, turn_time, init_length, contraction, contract_rate, release_rate):
    """Fast compiled current body length calculation."""
    if state_val == 0:  # REFILL phase
        if cycle_time < refill_time:
            return init_length - cycle_time * contract_rate
        else:
            return init_length - contraction
    elif state_val == 1:  # JET phase
        return init_length - contraction + (cycle_time - max(refill_time, turn_time)) * release_rate
    else:
        return init_length

# checked
@jit(nopython=True, cache=True)
def compute_width_jit(state_val, cycle_time, refill_time, turn_time, init_width, contraction, contract_rate, release_rate):
    """Fast compiled current body width calculation."""
    if state_val == 0:  # REFILL phase
        if cycle_time < refill_time:
            return init_width + cycle_time * contract_rate
        else:
            return init_width + contraction
    elif state_val == 1:  # JET phase
        return init_width + contraction - (cycle_time - max(refill_time, turn_time)) * release_rate
    else:
        return init_width

# checked
@jit(nopython=True, cache=True)
def compute_cross_sectional_area_jit(length, width):
    """Fast compiled cross-sectional area calculation."""
    w_half = width / 2.0
    l_half = length / 2.0
    A_yz = np.pi * w_half * w_half
    A_xz = np.pi * l_half * w_half
    A_xy = np.pi * l_half * w_half
    return np.array([A_yz, A_xz, A_xy])

# checked
@jit(nopython=True, cache=True)
def compute_water_volume_jit(length, width):
    """Fast compiled water volume calculation."""
    return (4.0 / 3.0) * np.pi * (length / 2.0) * (width / 2.0)**2

# checked
@jit(nopython=True, cache=True)
def compute_water_mass_jit(density, volume):
    """Fast compiled water mass calculation."""
    return density * volume

# checked 
@jit(nopython=True, cache=True)
def compute_mass_matrix_jit(dry_mass, water_mass, nozzle_mass):
    """Fast compiled mass matrix calculation."""
    total_mass = dry_mass + water_mass + nozzle_mass
    return np.diag(np.array([total_mass, total_mass, total_mass]))

# checked
@jit(nopython=True, cache=True)
def compute_mass_rate_jit(water_mass, prev_water_mass, dt):
    """Fast compiled mass rate matrix calculation."""
    rate = (water_mass - prev_water_mass) / dt
    return np.diag(np.array([rate, rate, rate]))

# checked
@jit(nopython=True, cache=True)
def compute_drag_coefficient_jit(length, width, init_length, init_width, max_contraction, ranges):
    """Fast compiled drag coefficient interpolation."""
    aspect_ratio = length / width
    init_aspect_ratio = init_length / init_width
    contracted_length = init_length - max_contraction
    contracted_width = init_length - contracted_length + init_width
    end_aspect_ratio = contracted_length / contracted_width
    
    normalized_ratio = (aspect_ratio - end_aspect_ratio) / (init_aspect_ratio - end_aspect_ratio)
    
    # Fast manual clip to avoid Python overhead
    if normalized_ratio < 0.0: normalized_ratio = 0.0
    if normalized_ratio > 1.0: normalized_ratio = 1.0
    
    drag_coeff = np.zeros(3)
    for i in range(3):
        drag_coeff[i] = ranges[i, 1] - normalized_ratio * (ranges[i, 1] - ranges[i, 0])
        
    return drag_coeff

# checked
@jit(nopython=True, cache=True)
def compute_jet_moment_arm_jit(nozzle_middle_pos, length):
    """Fast compiled jet moment arm calculation."""
    r_robot = np.array([-length / 2.0, 0.0, 0.0])
    return nozzle_middle_pos + r_robot

# checked
@jit(nopython=True, cache=True)
def compute_inertia_matrix_jit(mass_scalar, length, width, nozzle_mass, jet_moment_arm):
    """Fast compiled inertia matrix calculation."""
    r_norm = np.linalg.norm(jet_moment_arm)
    I_nozzle = nozzle_mass * (r_norm ** 2) * np.diag(np.array([0.0, 1.0, 1.0]))

    w_half = width / 2.0
    l_half = length / 2.0
    I_xx = 0.2 * mass_scalar * (w_half**2 + w_half**2)
    I_yy = 0.2 * mass_scalar * (l_half**2 + w_half**2)
    I_zz = 0.2 * mass_scalar * (w_half**2 + l_half**2)

    I_robot = np.diag(np.array([I_xx, I_yy, I_zz]))
    return I_robot + I_nozzle

@jit(nopython=True, cache=True)
def randomize_scalar_jit(value, uncertainty=0.1, lower_bound=np.nan, upper_bound=np.nan):
    """Fast compiled randomization for a single scalar value."""
    lower_sample_bound = value * (1.0 - uncertainty)
    upper_sample_bound = value * (1.0 + uncertainty)

    # In Numba, use np.isnan() instead of "is None"
    if np.isnan(lower_bound):
        lower_bound = lower_sample_bound
    if np.isnan(upper_bound):
        upper_bound = upper_sample_bound

    sample = np.random.uniform(lower_sample_bound, upper_sample_bound)

    # Manual min/max is highly optimized in Numba for scalars
    return min(max(sample, lower_bound), upper_bound)