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
    weights = np.array([1e10, 1.0, 1.0, 1.0, 1.0, 1.0, 1e10])
    coefficients = np.polyfit(lengths, widths, deg=2, w=weights)  # Fit a polynomial of degree 2
    return coefficients

@jit(nopython=True, cache=True)
def width_from_length_jit(length, coefficients):
    return coefficients[0] * length**2 + coefficients[1] * length + coefficients[2]  # Evaluate the polynomial at the given length

# checked 
@jit(nopython=True, cache=True)
def compute_length_jit(state_val, cycle_time, refill_time, propulsion_time, turn_time, init_length, contraction, contract_rate, release_rate, refill_coefficient, propulsion_coefficient):
    """Fast compiled current body length calculation."""
    # if state_val == 0:  # REFILL phase
    #     if cycle_time < refill_time:
    #         return init_length - cycle_time * contract_rate
    #     else:
    #         return init_length - contraction
    # elif state_val == 1:  # JET phase
    #     return init_length - contraction + (cycle_time - max(refill_time, turn_time)) * release_rate
    # else:
    #     return init_length

    if state_val == 0:  # REFILL phase
        if cycle_time < refill_time:
            return compute_length_from_time_jit(cycle_time, init_length - contraction, refill_time, refill_coefficient)
        else:
            return init_length - contraction
    elif state_val == 1:  # JET phase
        return compute_length_from_time_jit(cycle_time - max(refill_time, turn_time), init_length, propulsion_time, propulsion_coefficient)
    else:
        return init_length

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

@jit(nopython=True, cache=True)
def fit_length_time_function_jit(init_length, end_length, time):

    a = (init_length - end_length) / (0 - time)**2

    return a

@jit(nopython=True, cache=True)
def compute_length_from_time_jit(time, end_length, end_time, a):
    return a*(time - end_time)**2 + end_length

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
def compute_mass_rate_jit(water_volume, prev_water_volume, dt, loss_coeff, density=1000):
    """Fast compiled mass rate matrix calculation."""
    rate = compute_volume_rate_jit(water_volume, prev_water_volume, dt, loss_coeff)
    mass_rate = rate * density
    return np.diag(np.array([mass_rate, mass_rate, mass_rate]))

@jit(nopython=True, cache=True)
def compute_volume_rate_jit(water_volume, prev_water_volume, dt, loss_coeff):
    """Fast compiled volume rate calculation."""
    volume_rate = (water_volume - prev_water_volume) / dt
    return volume_rate * (1 - loss_coeff)

# checked
@jit(nopython=True, cache=True)
def compute_drag_coefficient_jit(length, width, init_length, init_width, end_length, end_width, ranges):
    """Fast compiled drag coefficient interpolation."""
    aspect_ratio = length / width
    init_aspect_ratio = init_length / init_width
    end_aspect_ratio = end_length / end_width
    
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
def compute_inertia_matrix_jit(length, width, tube_volume, mass_buoy, mass_tube, mass_skin, mass_nozzle):
    """Fast compiled inertia matrix calculation."""
    # buoy inertia tensor
    l_buoy = 0.36
    w_buoy = 0.15
    h_buoy = 0.02

    I_xx_buoy = 1/12 * mass_buoy * (l_buoy**2 + h_buoy**2)
    I_yy_buoy = 1/12 * mass_buoy * (w_buoy**2 + h_buoy**2)
    I_zz_buoy = 1/12 * mass_buoy * (l_buoy**2 + w_buoy**2)

    I_buoy = np.diag(np.array([I_xx_buoy, I_yy_buoy, I_zz_buoy])) + mass_buoy * np.diag(np.array([0.0, (length/2)**2, (length/2)**2]) )

    # tube inertia tensor
    density = 1000
    l_tube = 0.15
    r_tube = 0.03
    

    tube_area_mass = mass_tube - tube_volume * density

    I_xx_tube = 1/2 * tube_area_mass * r_tube**2
    I_yy_tube = 1/12 * tube_area_mass * (3*r_tube**2 + l_tube**2)
    I_zz_tube = 1/12 * tube_area_mass * (3*r_tube**2 + l_tube**2)

    I_tube = np.diag(np.array([I_xx_tube, I_yy_tube, I_zz_tube])) + tube_area_mass * np.diag(np.array([0.0, (length/2 - 0.08)**2, (length/2 - 0.08)**2])) # skin inertia tensor 
    
    # skin inertia tensor
    I_xx_skin = 1/3 * mass_skin * ((width/2)**2 + (width/2)**2)
    I_yy_skin = 1/3 * mass_skin * ((length/2)**2 + (width/2)**2)
    I_zz_skin = 1/3 * mass_skin * ((length/2)**2 + (width/2)**2)

    I_skin = np.diag(np.array([I_xx_skin, I_yy_skin, I_zz_skin]))

    # water inertia tensor
    water_mass_ellipsoid = compute_water_mass_jit(density=1000, volume=compute_water_volume_jit(length, width))
    I_xx_water = 1/5 * water_mass_ellipsoid * ((width/2)**2 + (width/2)**2)
    I_yy_water = 1/5 * water_mass_ellipsoid * ((length/2)**2 + (width/2)**2)
    I_zz_water = 1/5 * water_mass_ellipsoid * ((length/2)**2 + (width/2)**2)

    I_water = np.diag(np.array([I_xx_water, I_yy_water, I_zz_water]))

    # nozzle inertia tensor
    l_nozzle = 0.08
    w_nozzle = 0.06
    h_nozzle = 0.12

    I_xx_nozzle = 1/12 * mass_nozzle * (l_nozzle**2 + h_nozzle**2)
    I_yy_nozzle = 1/12 * mass_nozzle * (w_nozzle**2 + h_nozzle**2)
    I_zz_nozzle = 1/12 * mass_nozzle * (l_nozzle**2 + w_nozzle**2)

    I_nozzle = np.diag(np.array([I_xx_nozzle, I_yy_nozzle, I_zz_nozzle])) + mass_nozzle * np.diag(np.array([0.0, (length/2 + 0.025)**2, (length/2 + 0.025)**2]) )
    
    return I_buoy + I_tube + I_skin + I_water + I_nozzle


@jit(nopython=True, cache=True)
def compute_center_of_mass_jit(length, width, tube_volume, nozzle_mass, buoy_mass, skin_mass, tube_mass, water_mass):
    """Fast compiled center of mass calculation."""

    # body frame is mounted on center of geometry
    pos_buoy = np.array([length / 2, 0.0, 0.0])
    pos_skin = np.array([0.0, 0.0, 0.0])
    pos_tube = np.array([length / 2 -0.08, 0.0, 0.0])
    pos_nozzle = np.array([-length / 2 - 0.025 + 0.05, 0.0, 0.0])

    # get water center of mass
    water_mass_ellipsoid = compute_water_mass_jit(density=1000, volume=compute_water_volume_jit(length, width))
    pos_water = (water_mass_ellipsoid * np.array([0.0, 0.0, 0.0]) - 1000 * tube_volume * pos_tube)/ (water_mass_ellipsoid - 1000 * tube_volume)

    total_mass = tube_mass + nozzle_mass + buoy_mass + skin_mass + water_mass
    center_of_mass = (tube_mass * pos_tube + nozzle_mass * pos_nozzle + buoy_mass * pos_buoy + skin_mass * pos_skin + water_mass * pos_water) / total_mass
    
    return center_of_mass

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