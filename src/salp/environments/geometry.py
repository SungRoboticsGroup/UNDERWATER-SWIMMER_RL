from numba import jit
import numpy as np

# ==================== Numba Geometry & Property Calculations ====================
# data-driven refill time 
def fit_compression_refill_time_relation_jit():
    compression = np.array([0, 0.01, 0.02, 0.03, 0.04])  # Example lengths during contraction
    refill_time = np.array([0, 0.4, 1.0, 1.8, 2.2])   # Corresponding widths to maintain constant volume
    weights = np.array([1e10, 1.0, 1.0, 1.0, 1.0])
    coefficients = np.polyfit(compression, refill_time, 2, w=weights)  # Fit a polynomial of degree 2
    return coefficients

@jit(nopython=True, cache=True)
def refill_time_from_compression_jit(compression, coefficients):
    return coefficients[0] * compression**2 + coefficients[1] * compression + coefficients[2]  # Evaluate the polynomial at the given length


def visualize_compression_refill_time_relation(compression=None, refill_time=None, coefficients=None,
                                               num_points=200, title="Compression Length vs Refill Time",
                                               save_path=None, show=True):
    """Visualize the relation between compression length and refill time.

    Args:
        compression: Optional array-like compression length values.
        refill_time: Optional array-like refill time values paired with ``compression``.
        coefficients: Optional polynomial coefficients [a, b, c]. If None, uses fitted prototype relation.
        num_points: Number of points used to draw the fitted curve.
        title: Figure title.
        save_path: Optional path to save the figure image.
        show: If True, shows the figure interactively.

    Returns:
        Tuple of (fig, ax) for further customization.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualize_compression_refill_time_relation. "
            "Install it with: pip install matplotlib"
        ) from exc

    default_compression = np.array([0, 0.01, 0.02, 0.03, 0.04], dtype=np.float64)
    default_refill_time = np.array([0, 0.4, 1.0, 1.8, 2.2], dtype=np.float64)

    if compression is None and refill_time is None:
        compression_arr = default_compression
        refill_time_arr = default_refill_time
    elif compression is not None and refill_time is not None:
        compression_arr = np.asarray(compression, dtype=np.float64)
        refill_time_arr = np.asarray(refill_time, dtype=np.float64)
        if compression_arr.shape != refill_time_arr.shape:
            raise ValueError("compression and refill_time must have the same shape")
    else:
        raise ValueError("Provide both compression and refill_time, or leave both as None")

    if coefficients is None:
        coefficients = fit_compression_refill_time_relation_jit()
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape[0] != 3:
        raise ValueError("coefficients must contain 3 values [a, b, c]")

    x_min = np.min(compression_arr)
    x_max = np.max(compression_arr)
    curve_compression = np.linspace(x_min, x_max, int(max(2, num_points)))
    curve_refill_time = refill_time_from_compression_jit(curve_compression, coefficients)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(compression_arr, refill_time_arr, color="tab:blue", s=55, label="Measured points", zorder=3)
    ax.plot(curve_compression, curve_refill_time, color="tab:red", linewidth=2, label="Quadratic fit")

    ax.set_xlabel("Compression Length (m)")
    ax.set_ylabel("Refill Time (s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show(block=True)

    return fig, ax

# data-driven propulsion time
def fit_compression_propulsion_time_relation_jit():
    compression = np.array([0, 0.01, 0.02, 0.03, 0.04])  # Example lengths during contraction
    propulsion_time = np.array([0, 0.1, 0.3, 0.4, 0.5])   # Corresponding widths to maintain constant volume
    # propulsion_time = np.array([0, 0.2, 0.6, 0.8, 1.0])   # Corresponding widths to maintain constant volume
    weights = np.array([1e10, 1.0, 1.0, 1.0, 1.0])
    coefficients = np.polyfit(compression, propulsion_time, 2, w=weights)  # Fit a polynomial of degree 2
    return coefficients

@jit(nopython=True, cache=True)
def propulsion_time_from_compression_jit(compression, coefficients):
    return coefficients[0] * compression**2 + coefficients[1] * compression + coefficients[2]  # Evaluate the polynomial at the given length

@jit(nopython=True, cache=True)
def compute_length_from_refill_time_jit(time, coefficients, init_length):
    """
    Analytically solves the quadratic equation to find compression.
    coefficients: [A, B, C] from np.polyfit(x, y, 2)
    """
    A = coefficients[0]
    B = coefficients[1]
    C = coefficients[2]
    
    # We are solving: A*c^2 + B*c + (C - time) = 0
    C_adjusted = C - time
    
    # Calculate the discriminant
    discriminant = B**2 - 4 * A * C_adjusted
    
    if discriminant < 0:
        # Prevent math domain errors if asked to extrapolate too far 
        # below the vertex of the parabola
        return 0.0 
        
    # Quadratic formula: (-B +/- sqrt(discriminant)) / 2A
    # For your monotonically increasing data, the positive root is correct
    c_1 = (-B + np.sqrt(discriminant)) / (2 * A)
    c_2 = (-B - np.sqrt(discriminant)) / (2 * A)
    
    # Return the root that actually makes physical sense (positive compression)
    if c_1 >= 0:
        return init_length - c_1
    else:
        return init_length - c_2

@jit(nopython=True, cache=True)
def compute_length_from_propulsion_time_jit(time, coefficients, init_length, contraction):
    """
    Analytically solves the quadratic equation to find compression.
    coefficients: [A, B, C] from np.polyfit(x, y, 2)
    """
    A = coefficients[0]
    B = coefficients[1]
    C = coefficients[2]
    
    # We are solving: A*c^2 + B*c + (C - time) = 0
    C_adjusted = C - time
    
    # Calculate the discriminant
    discriminant = B**2 - 4 * A * C_adjusted
    
    if discriminant < 0:
        # Prevent math domain errors if asked to extrapolate too far 
        # below the vertex of the parabola
        return 0.0 
        
    # Quadratic formula: (-B +/- sqrt(discriminant)) / 2A
    # For your monotonically increasing data, the positive root is correct
    c_1 = (-B + np.sqrt(discriminant)) / (2 * A)
    c_2 = (-B - np.sqrt(discriminant)) / (2 * A)
    
    # Return the root that actually makes physical sense (positive compression)
    if c_1 >= 0:
        return init_length + c_1 - contraction
    else:
        return init_length + c_2 - contraction

def visualize_compression_propulsion_time_relation(compression=None, propulsion_time=None, coefficients=None,
                                                   num_points=200, title="Compression Length vs Propulsion Time",
                                                   save_path=None, show=True):
    """Visualize the relation between compression length and propulsion time.

    Args:
        compression: Optional array-like compression length values.
        propulsion_time: Optional array-like propulsion time values paired with ``compression``.
        coefficients: Optional polynomial coefficients [a, b, c]. If None, uses fitted prototype relation.
        num_points: Number of points used to draw the fitted curve.
        title: Figure title.
        save_path: Optional path to save the figure image.
        show: If True, shows the figure interactively.

    Returns:
        Tuple of (fig, ax) for further customization.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualize_compression_propulsion_time_relation. "
            "Install it with: pip install matplotlib"
        ) from exc

    default_compression = np.array([0, 0.01, 0.02, 0.03, 0.04], dtype=np.float64)
    default_propulsion_time = np.array([0, 0.2, 0.6, 0.8, 1.0], dtype=np.float64)

    if compression is None and propulsion_time is None:
        compression_arr = default_compression
        propulsion_time_arr = default_propulsion_time
    elif compression is not None and propulsion_time is not None:
        compression_arr = np.asarray(compression, dtype=np.float64)
        propulsion_time_arr = np.asarray(propulsion_time, dtype=np.float64)
        if compression_arr.shape != propulsion_time_arr.shape:
            raise ValueError("compression and propulsion_time must have the same shape")
    else:
        raise ValueError("Provide both compression and propulsion_time, or leave both as None")

    if coefficients is None:
        coefficients = fit_compression_propulsion_time_relation_jit()
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape[0] != 3:
        raise ValueError("coefficients must contain 3 values [a, b, c]")

    x_min = np.min(compression_arr)
    x_max = np.max(compression_arr)
    curve_compression = np.linspace(x_min, x_max, int(max(2, num_points)))
    curve_propulsion_time = propulsion_time_from_compression_jit(curve_compression, coefficients)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(compression_arr, propulsion_time_arr, color="tab:blue", s=55, label="Measured points", zorder=3)
    ax.plot(curve_compression, curve_propulsion_time, color="tab:red", linewidth=2, label="Quadratic fit")

    ax.set_xlabel("Compression Length (m)")
    ax.set_ylabel("Propulsion Time (s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show(block=True)

    return fig, ax

# data-driven geometry relation
def fit_length_width_relation_jit():
    ### Measurements from physical prototype 
    lengths = np.array([0.26, 0.25, 0.24, 0.23, 0.22])  # Example lengths during contraction
    widths = np.array([0.135, 0.157, 0.176, 0.190, 0.198])   # Corresponding widths to maintain constant volume
    ### Measurements from physical prototype
    weights = np.array([1e10, 1.0, 1.0, 1.0, 1e10])
    coefficients = np.polyfit(lengths, widths, deg=2, w=weights)  # Fit a polynomial of degree 2
    return coefficients

@jit(nopython=True, cache=True)
def width_from_length_jit(length, coefficients):
    return coefficients[0] * length**2 + coefficients[1] * length + coefficients[2]  # Evaluate the polynomial at the given length


def visualize_length_width_relation(lengths=None, widths=None, coefficients=None, num_points=200,
                                    title="Robot Length-Width Relation", save_path=None, show=True):
    """Visualize how robot width changes with robot length.

    Args:
        lengths: Optional array-like of measured lengths. If None, uses prototype data.
        widths: Optional array-like of measured widths paired with ``lengths``.
        coefficients: Optional polynomial coefficients [a, b, c]. If None, uses fitted prototype relation.
        num_points: Number of points used to draw the fitted curve.
        title: Figure title.
        save_path: Optional path to save the figure image.
        show: If True, shows the figure interactively.

    Returns:
        Tuple of (fig, ax) for further customization.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualize_length_width_relation. "
            "Install it with: pip install matplotlib"
        ) from exc

    default_lengths = np.array([0.26, 0.25, 0.24, 0.23, 0.22], dtype=np.float64)
    default_widths = np.array([0.135, 0.157, 0.176, 0.190, 0.198], dtype=np.float64)

    if lengths is None and widths is None:
        lengths_arr = default_lengths
        widths_arr = default_widths
    elif lengths is not None and widths is not None:
        lengths_arr = np.asarray(lengths, dtype=np.float64)
        widths_arr = np.asarray(widths, dtype=np.float64)
        if lengths_arr.shape != widths_arr.shape:
            raise ValueError("lengths and widths must have the same shape")
    else:
        raise ValueError("Provide both lengths and widths, or leave both as None")

    if coefficients is None:
        coefficients = fit_length_width_relation_jit()
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape[0] != 3:
        raise ValueError("coefficients must contain 3 values [a, b, c]")

    x_min = np.min(lengths_arr)
    x_max = np.max(lengths_arr)
    curve_lengths = np.linspace(x_min, x_max, int(max(2, num_points)))
    curve_widths = width_from_length_jit(curve_lengths, coefficients)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(lengths_arr, widths_arr, color="tab:blue", s=55, label="Measured points", zorder=3)
    ax.plot(curve_lengths, curve_widths, color="tab:red", linewidth=2, label="Quadratic fit")

    ax.set_xlabel("Length (m)")
    ax.set_ylabel("Width (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show(block=True)

    return fig, ax

# checked 
@jit(nopython=True, cache=True)
def compute_length_jit(state_val, cycle_time, refill_time, turn_time, init_length, contraction, refill_coefficient, propulsion_coefficient):
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
            return compute_length_from_refill_time_jit(cycle_time, refill_coefficient, init_length)
        else:
            return init_length - contraction
    elif state_val == 1:  # JET phase
        return compute_length_from_propulsion_time_jit(cycle_time - max(refill_time, turn_time), propulsion_coefficient, init_length, contraction)
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

# @jit(nopython=True, cache=True)
# def fit_length_time_function_jit(init_length, end_length, time):

#     a = (init_length - end_length) / (0 - time)**2

#     return a

# @jit(nopython=True, cache=True)
# def compute_length_from_time_jit(time, end_length, end_time, a):
#     return a*(time - end_time)**2 + end_length

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
def compute_mass_rate_jit(water_volume, prev_water_volume, dt, density=1000):
    """Fast compiled mass rate matrix calculation."""
    rate = compute_volume_rate_jit(water_volume, prev_water_volume, dt)
    mass_rate = rate * density
    return np.diag(np.array([mass_rate, mass_rate, mass_rate]))

# checked
@jit(nopython=True, cache=True)
def compute_effective_mass_rate_jit(water_volume, prev_water_volume, dt, effective_coeff, density=1000):
    """Fast compiled mass rate matrix calculation."""
    rate = compute_effective_volume_rate_jit(water_volume, prev_water_volume, dt, effective_coeff)
    mass_rate = rate * density
    return np.diag(np.array([mass_rate, mass_rate, mass_rate]))

@jit(nopython=True, cache=True)
def compute_volume_rate_jit(water_volume, prev_water_volume, dt):
    # print(effective_coeff)
    """Fast compiled volume rate calculation."""
    volume_rate = (water_volume - prev_water_volume) / dt
    return volume_rate

@jit(nopython=True, cache=True)
def compute_effective_volume_rate_jit(water_volume, prev_water_volume, dt, effective_coeff):
    # print(effective_coeff)
    """Fast compiled volume rate calculation."""
    volume_rate = (water_volume - prev_water_volume) / dt
    return volume_rate * effective_coeff

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


@jit(nopython=True, cache=True)
def compute_buoy_inertia_maxtrix_jit(length, width, height, length_com, mass):

    I_xx = 1/12 * mass * (length**2 + height**2)
    I_yy = 1/12 * mass * (width**2 + height**2)
    I_zz = 1/12 * mass * (length**2 + width**2)

    I_buoy = np.diag(np.array([I_xx, I_yy, I_zz])) + mass * np.diag(np.array([0.0, length_com**2, length_com**2]))

    return I_buoy

@jit(nopython=True, cache=True)
def compute_tube_inertia_maxtrix_jit(length, radius, length_com, mass):

    density = 1000
    tube_area_mass = mass - np.pi * radius**2 * length * density

    I_xx = 1/2 * tube_area_mass * radius**2
    I_yy = 1/12 * tube_area_mass * (3*radius**2 + length**2)
    I_zz = 1/12 * tube_area_mass * (3*radius**2 + length**2)

    I_tube = np.diag(np.array([I_xx, I_yy, I_zz])) + tube_area_mass * np.diag(np.array([0.0, length_com**2, length_com**2])) # tube inertia tensor 

    return I_tube

@jit(nopython=True, cache=True)
def compute_skin_inertia_maxtrix_jit(length, width, length_com, mass):

    I_xx = 1/3 * mass * ((width/2)**2 + (width/2)**2)
    I_yy = 1/3 * mass * ((length/2)**2 + (width/2)**2)
    I_zz = 1/3 * mass * ((length/2)**2 + (width/2)**2)

    I_skin = np.diag(np.array([I_xx, I_yy, I_zz])) + mass * np.diag(np.array([0.0, length_com**2, length_com**2])) # skin inertia tensor

    return I_skin

@jit(nopython=True, cache=True)
def compute_water_inertia_maxtrix_jit(length, width, length_com, mass):

    I_xx = 1/5 * mass * ((width/2)**2 + (width/2)**2)
    I_yy = 1/5 * mass * ((length/2)**2 + (width/2)**2)
    I_zz = 1/5 * mass * ((length/2)**2 + (width/2)**2)

    I_water = np.diag(np.array([I_xx, I_yy, I_zz])) + mass * np.diag(np.array([0.0, length_com**2, length_com**2])) # water inertia tensor

    return I_water

@jit(nopython=True, cache=True)
def compute_nozzle_inertia_maxtrix_jit(length, radius, radius_inner, length_com, mass):

    I_xx = 1/2 * mass * (radius**2 + radius_inner**2)
    I_yy = 1/12 * mass * (3*(radius**2 + radius_inner**2) + length**2)
    I_zz = 1/12 * mass * (3*(radius**2 + radius_inner**2) + length**2)

    I_nozzle = np.diag(np.array([I_xx, I_yy, I_zz])) + mass * np.diag(np.array([0.0, length_com**2, length_com**2])) # nozzle inertia tensor

    return I_nozzle

@jit(nopython=True, cache=True)
def compute_nozzle_water_inertia_maxtrix_jit(length, radius_inner, length_com, mass):

    I_xx = 1/2 * mass * (radius_inner**2)
    I_yy = 1/12 * mass * (3*(radius_inner**2) + length**2)
    I_zz = 1/12 * mass * (3*(radius_inner**2) + length**2)

    I_nozzle = np.diag(np.array([I_xx, I_yy, I_zz])) + mass * np.diag(np.array([0.0, length_com**2, length_com**2])) # nozzle inertia tensor

    return I_nozzle

# checked
@jit(nopython=True, cache=True)
def compute_inertia_matrix_jit(length, width, l_buoy, w_buoy, h_buoy, 
                               l_tube, r_tube, l_nozzle, r_nozzle, r_nozzle_inner, 
                               l_com, mass_water, mass_buoy, mass_tube, mass_skin, mass_nozzle, mass_nozzle_water):
    """Fast compiled inertia matrix calculation."""

    # buoy inertia tensor
    length_com = length / 2.0
    I_buoy = compute_buoy_inertia_maxtrix_jit(l_buoy, w_buoy, h_buoy, length_com, mass_buoy)

    # tube inertia tensor
    length_com = length / 2.0 - l_tube / 2.0
    I_tube = compute_tube_inertia_maxtrix_jit(l_tube, r_tube, length_com, mass_tube)
    
    # skin inertia tensor
    length_com = l_com[0]
    I_skin = compute_skin_inertia_maxtrix_jit(length, width, length_com, mass_skin)

    # water inertia tensor
    length_com = l_com[0]
    I_water = compute_water_inertia_maxtrix_jit(length, width, length_com, mass_water)

    # nozzle inertia tensor
    length_com = length / 2.0 + l_nozzle / 2.0
    I_nozzle = compute_nozzle_inertia_maxtrix_jit(l_nozzle, r_nozzle, r_nozzle_inner, length_com, mass_nozzle)

    # nozzle water inertia tensor
    length_com = length / 2.0 + l_nozzle / 2.0
    I_nozzle_water = compute_nozzle_water_inertia_maxtrix_jit(l_nozzle, r_nozzle_inner, length_com, mass_nozzle_water)

    # print("I_buoy:", I_buoy)
    # print("I_tube:", I_tube)
    # print("I_skin:", I_skin)
    # print("I_water:", I_water)
    # print("I_nozzle:", I_nozzle)
    # print("I_nozzle_water:", I_nozzle_water)
    # print("Total inertia matrix:", I_buoy + I_tube + I_skin + I_water + I_nozzle + I_nozzle_water)

    return I_buoy + I_tube + I_skin + I_water + I_nozzle + I_nozzle_water

@jit(nopython=True, cache=True)
def compute_center_of_mass_jit(pos_buoy, pos_skin, pos_tube, pos_nozzle, pos_water, pos_nozzle_water, 
                               buoy_mass, skin_mass, tube_mass, nozzle_mass, water_mass, nozzle_water_mass):
    """Fast compiled center of mass calculation."""

    # # body frame is mounted on center of geometry
    # pos_buoy = np.array([length / 2, 0.0, 0.0])
    # pos_skin = np.array([0.0, 0.0, 0.0])
    # pos_tube = np.array([length / 2 -0.08, 0.0, 0.0])
    # pos_nozzle = np.array([-length / 2 - 0.025 + 0.05, 0.0, 0.0])

    # # get water center of mass
    # water_mass_ellipsoid = compute_water_mass_jit(density=1000, volume=compute_water_volume_jit(length, width))
    # pos_water = (water_mass_ellipsoid * np.array([0.0, 0.0, 0.0]) - 1000 * tube_volume * pos_tube)/ (water_mass_ellipsoid - 1000 * tube_volume)

    total_mass = tube_mass + nozzle_mass + buoy_mass + skin_mass + water_mass + nozzle_water_mass
    center_of_mass = (tube_mass * pos_tube + nozzle_mass * pos_nozzle + buoy_mass * pos_buoy + 
                      skin_mass * pos_skin + water_mass * pos_water + nozzle_water_mass * pos_nozzle_water) / total_mass

    # print("Center of mass:", center_of_mass) 
    # print(total_mass)
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


if __name__ == "__main__":

    # Use built-in prototype measurements and fitted relation
    fig, ax = visualize_length_width_relation()
    # fig, ax = visualize_compression_refill_time_relation()
    # fig, ax = visualize_compression_propulsion_time_relation()
