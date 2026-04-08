"""Demo script for the Robot and Nozzle classes.

Runs multiple swim cycles and plots selected diagnostics.
Toggle plots on/off via the ENABLED_PLOTS set in main().
"""
import numpy as np
import matplotlib.pyplot as plt

from robot import Robot, Nozzle
import plotting


# ==================== Robot Parameters (DO NOT CHANGE) ====================
NOZZLE_LENGTH1 = 0.052
NOZZLE_LENGTH2 = 0.038
NOZZLE_LENGTH3 = 0.050
NOZZLE_AREA = np.pi * 0.01**2
NOZZLE_MASS = 0.428
NOZZLE_RADIUS = 0.1
NOZZLE_INNER_RADIUS = 0.022

ROBOT_MASS = 0.738
ROBOT_INIT_LENGTH = 0.26
ROBOT_INIT_WIDTH = 0.135

# ==================== Available Plots ====================
# Each entry maps a short tag to (plotting_function, [data_keys...]).
# The first two keys are always (time, state); only extra data keys are listed.
PLOT_REGISTRY: dict[str, tuple] = {
    # Geometry & body shape
    "geometry":              (plotting.plot_robot_geometry,         ["length_history", "width_history"]),
    "cross_section":         (plotting.plot_cross_sectional_area,  ["area_history"]),

    # Mass & volume
    "mass":                  (plotting.plot_robot_mass,            ["mass_history"]),
    "mass_bbox":             (plotting.plot_robot_mass,            ["bounding_box_mass_history"]),
    "volume_rate":           (plotting.plot_volume_rate,           ["volume_rate_history"]),
    "mass_rate":             (plotting.plot_mass_rate,             ["mass_rate_history"]),
    "mass_rate_bbox":        (plotting.plot_mass_rate,             ["bounding_box_mass_rate_history"]),
    "mass_rate_eff":         (plotting.plot_mass_rate,             ["effective_mass_rate_history"]),

    # Inertia
    "inertia":               (plotting.plot_inertia_tensor,        ["inertia_matrix_history"]),
    "inertia_rate":          (plotting.plot_inertia_tensor_rate,   ["inertia_matrix_rate_history"]),
    "inertia_bbox":          (plotting.plot_inertia_tensor,        ["bounding_box_inertia_matrix_history"]),
    "inertia_rate_bbox":     (plotting.plot_inertia_tensor_rate,   ["bounding_box_inertia_matrix_rate_history"]),

    # Center of mass
    "com":                   (plotting.plot_center_of_mass,          ["center_of_mass_history"]),
    "com_rate":              (plotting.plot_center_of_mass_rate,     ["center_of_mass_rate_history"]),
    "com_acc":               (plotting.plot_center_of_mass_acc_rate, ["center_of_mass_acc_rate_history"]),

    # Translational dynamics – forces
    "all_forces":            (plotting.plot_all_forces,            ["jet_force_history", "drag_force_history",
                                                                    "coriolis_force_history", "added_mass_force_history"]),
    "jet_velocity":          (plotting.plot_jet_properties,        ["jet_velocity_history"]),
    "jet_force":             (plotting.plot_jet_properties,        ["jet_force_history"]),
    "drag_force":            (plotting.plot_drag_properties,       ["drag_force_history"]),
    "drag_coeff":            (plotting.plot_drag_coefficient,      ["trans_drag_coefficient_history"]),
    "coriolis_force":        (plotting.plot_coriolis_force,        ["coriolis_force_history"]),
    "added_mass_force":      (plotting.plot_added_mass_force,      ["added_mass_force_history"]),
    "accel_force":           (plotting.plot_acceleration_force,    ["acceleration_force_history"]),

    # Translational dynamics – kinematics
    "velocity":              (plotting.plot_robot_velocity,        ["velocity_history"]),
    "velocity_world":        (plotting.plot_robot_velocity,        ["velocity_world_history"]),
    "acceleration":          (plotting.plot_robot_acceleration,    ["acceleration_history"]),
    "position":              (plotting.plot_robot_position,        ["position_world_history"]),

    # Rotational dynamics – torques
    "jet_torque":            (plotting.plot_jet_torque,            ["jet_torque_history"]),
    "drag_torque":           (plotting.plot_drag_torque,           ["drag_torque_history"]),
    "coriolis_torque":       (plotting.plot_coriolis_torque,       ["coriolis_torque_history"]),
    "deform_torque":         (plotting.plot_deform_torque,         ["deform_torque_history"]),
    "added_mass_torque":     (plotting.plot_added_mass_torque,     ["added_mass_torque_history"]),
    "asymmetry_torque":      (plotting.plot_asymmetry_torque,      ["asymmetry_torque_history"]),

    # Rotational dynamics – kinematics
    "angular_velocity":      (plotting.plot_angular_velocity,      ["angular_velocity_history"]),
    "angular_acceleration":  (plotting.plot_angular_acceleration,  ["angular_acceleration_history"]),
    "euler_angles":          (plotting.plot_euler_angles,          ["euler_angle_history"]),

    # Nozzle & front position
    "nozzle_yaw":            (plotting.plot_nozzle_yaw_angle,      ["nozzle_yaw_history"]),
    "front_pos_body":        (plotting.plot_front_position_body_frame,  ["position_front_history"]),
    "front_pos_world":       (plotting.plot_front_position_world_frame, ["position_front_world_history"]),
}


def _collect_cycles(robot: Robot, n_cycles: int = 6) -> dict[str, np.ndarray]:
    """Run *n_cycles* swim cycles and return aggregated history arrays."""
    accumulators: dict[str, list] = {}

    for _ in range(n_cycles):
        robot.nozzle.set_yaw_angle(yaw_angle=-1.0 * np.pi / 6)
        robot.nozzle.solve_angles()
        robot.set_control(
            contraction=0.03,
            coast_time=2,
            nozzle_angles=np.array([robot.nozzle.angle1, robot.nozzle.angle2]),
        )
        robot.step_through_cycle()

        # Build time array for this cycle
        cycle_start_time = robot.time - robot.cycle_time
        time_array = np.arange(cycle_start_time, robot.time, robot.dt)[
            : len(robot.length_history) - 1
        ]
        accumulators.setdefault("time", []).extend(time_array)

        # State / property histories (trim trailing duplicate)
        for attr_name in Robot._HISTORY_BUFFER_NAMES:
            buf = getattr(robot, attr_name)
            if isinstance(buf, (np.ndarray, list)) and len(buf) > len(time_array):
                data = buf[:-1]
            else:
                data = buf
            accumulators.setdefault(attr_name, []).extend(data)

    return {k: np.array(v) for k, v in accumulators.items()}


def _run_plots(data: dict[str, np.ndarray], enabled: set[str]):
    """Dispatch enabled plot tags through the registry."""
    t = data["time"]
    s = data["state_history"]

    for tag in enabled:
        if tag == "trajectory_xy":
            # Special case: different call signature
            plotting.plot_trajectory_xy(
                data["position_front_world_history"],
                s,
                data["euler_angle_history"],
                data["nozzle_yaw_history"],
            )
            continue

        if tag not in PLOT_REGISTRY:
            print(f"Warning: unknown plot tag '{tag}', skipping.")
            continue

        func, keys = PLOT_REGISTRY[tag]
        func(t, *(data[k] for k in keys), s)


def main():
    # ---- Setup ----
    nozzle = Nozzle(
        length1=NOZZLE_LENGTH1,
        length2=NOZZLE_LENGTH2,
        length3=NOZZLE_LENGTH3,
        area=NOZZLE_AREA,
        mass=NOZZLE_MASS,
        radius=NOZZLE_RADIUS,
        inner_radius=NOZZLE_INNER_RADIUS,
    )
    nozzle.set_angles(angle1=0.0, angle2=0.0)

    robot = Robot(
        dry_mass=ROBOT_MASS,
        init_length=ROBOT_INIT_LENGTH,
        init_width=ROBOT_INIT_WIDTH,
        max_contraction=0.04,
        nozzle=nozzle,
    )
    robot.set_environment(density=1000)
    robot.enable_history_recording()
    robot.reset()

    # ---- Collect data ----
    data = _collect_cycles(robot, n_cycles=6)

    # ---- Choose plots ----
    # Add / remove tags from this set to toggle plots.
    # Available tags: print(sorted(PLOT_REGISTRY)) or "trajectory_xy"
    enabled_plots = {
        "euler_angles",
        "trajectory_xy",
    }

    _run_plots(data, enabled_plots)
    plt.show(block=True)


if __name__ == "__main__":
    main()
