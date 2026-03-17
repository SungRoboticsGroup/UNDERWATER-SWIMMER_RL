from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from sympy import false
import dynamics
import geometry

class Nozzle:
    """Represents a steerable nozzle for jet propulsion.
    
    Attributes:
        length1: First segment length of the nozzle
        length2: Second segment length of the nozzle
        length3: Third segment length of the nozzle
        area: Area of nozzle opening
        angle1: Rotation angle around y axis
        angle2: Rotation angle around z axis
        mass: Mass of the nozzle
    """
    
    def __init__(self, length1: float = 0.0, length2: float = 0.0, 
                 length3: float = 0.0, area: float = 0.0, mass: float = 0.0):
        """Initialize nozzle with geometric and control parameters."""
        # Geometric properties
        self.length1 = length1
        self.length2 = length2
        self.length3 = length3
        self.area = area
        self.mass = mass
        
        # Angle parameters
        self.angle1 = 0.0
        self.angle2 = 0.0
        self.prev_angle1 = 0.0
        self.prev_angle2 = 0.0
        
        # Yaw control parameters
        self.yaw = 0.0
        self.prev_yaw = 0.0
        self.current_yaw = 0.0
        
        # Fixed parameters
        self.gamma = np.pi / 4  # fixed tilt angle of nozzle downwards
        self.angle_speed = 31 * np.pi / 30  # rad/s
        self.turn_time = 0.0
        
        # Rotation matrices
        self.R_nm = None
        self.R_mb = None
        self.R_br = None

    # ==================== Control Methods ====================
    def set_angles(self, angle1: float, angle2: float):
        """Set the nozzle angles and update rotation matrices.
        
        Args:
            angle1: Rotation angle around y axis
            angle2: Rotation angle around z axis
        """
        self.angle1 = angle1
        self.angle2 = angle2
        self.turn_time = self._nozzle_turn_time()
        self._get_rotation_matrices()
    
    def set_yaw_angle(self, yaw_angle: float):
        """Set the nozzle yaw angle (around z axis).
        
        Args:
            yaw_angle: Rotation angle around z axis
        """
        self.prev_yaw = self.yaw
        self.yaw = yaw_angle

    def solve_angles(self):
        """Solve inverse kinematics to find nozzle angles for target direction."""
        self.prev_angle1 = self.angle1
        self.prev_angle2 = self.angle2

        target_direction = -np.array([np.cos(self.yaw), np.sin(self.yaw), 0])
        target_direction = self.R_br.transpose() @ target_direction

        val2 = np.clip(2 * target_direction[2] - 1, -1.0, 1.0)
        self.angle2 = np.arccos(val2)
        if self.angle2 <= -np.pi:
            self.angle2 += 2 * np.pi
        elif self.angle2 > np.pi:
            self.angle2 -= 2 * np.pi

        if self.angle2 == 0:
            self.angle1 = 0.0
        else:
            a = 0.5 * (np.cos(self.angle2) - 1)
            b = np.sqrt(2) * np.sin(self.angle2) / 2
            c = target_direction[1]
            val1 = np.clip(c / np.sqrt(a**2 + b**2), -1.0, 1.0)
            self.angle1 = np.arcsin(val1) - np.arctan2(b, a)

        if self.angle1 <= -np.pi:
            self.angle1 += 2 * np.pi
        elif self.angle1 > np.pi:
            self.angle1 -= 2 * np.pi

    # ==================== Update Methods ====================
    def step(self, time: float):
        """Update current yaw interpolation during turning phase."""
        # Interpolate yaw angle during nozzle turn time
        if time < self.turn_time:
            ratio = time / self.turn_time
            self.current_yaw = self.prev_yaw + ratio * (self.yaw - self.prev_yaw)
        else:
            self.current_yaw = self.yaw

    # ==================== Geometry Methods ====================
    def get_nozzle_position(self) -> np.ndarray:
        """Calculate the nozzle tip position in world frame.
        
        Returns:
            3D position vector of the nozzle tip
        """
        # Nozzle tip position in nozzle frame
        pos_x3 = self.length3 * np.cos(self.gamma)
        pos_y3 = 0
        pos_z3 = self.length3 * np.sin(self.gamma)
        nozzle_position = np.array([pos_x3, pos_y3, pos_z3])

        # Middle section tip position in body frame
        pos_x2 = 0
        pos_y2 = 0
        pos_z2 = self.length2
        middle_position = np.array([pos_x2, pos_y2, pos_z2])

        # Base section tip position in base frame
        pos_x1 = 0
        pos_y1 = 0
        pos_z1 = self.length1
        base_position = np.array([pos_x1, pos_y1, pos_z1])

        position = self.R_br @ (base_position + self.R_mb @ (middle_position + self.R_nm @ nozzle_position))
        return position
    
    def get_nozzle_direction(self) -> np.ndarray:
        """Calculate the direction vector of the nozzle.
        
        Returns:
            3D direction unit vector in world frame
        """
        pos_x = np.cos(self.gamma)
        pos_y = 0
        pos_z = np.sin(self.gamma)
        nozzle_direction = np.array([pos_x, pos_y, pos_z])

        direction = self.R_br @ self.R_mb @ self.R_nm @ nozzle_direction
        return direction
    
    def get_middle_position(self) -> np.ndarray:
        """Get the position of the second nozzle joint.
        
        Returns:
            3D position vector in body frame
        """
        pos_x = 0
        pos_y = 0
        pos_z = self.length1
        base_position = np.array([pos_x, pos_y, pos_z])

        # Middle section tip position in body frame
        pos_x2 = 0
        pos_y2 = 0
        pos_z2 = self.length2
        middle_position = np.array([pos_x2, pos_y2, pos_z2])

        position = self.R_br @ (base_position + self.R_mb @ middle_position)
        return position

    # ==================== Private Helper Methods ====================
    def _nozzle_turn_time(self) -> float:
        """Calculate time required to turn to new angles.
        
        Returns:
            Time in seconds to reach new angles
        """
        delta_angle1 = abs(self.angle1 - self.prev_angle1)
        delta_angle2 = abs(self.angle2 - self.prev_angle2)

        time1 = delta_angle1 / self.angle_speed
        time2 = delta_angle2 / self.angle_speed

        return time1 + time2

    def _get_rotation_matrices(self):
        """Calculate the rotation matrices for nozzle orientation."""
        R_theta_fixed = np.array([[np.cos(self.gamma), 0, -np.sin(self.gamma)],
                                  [0, 1, 0],
                                  [np.sin(self.gamma), 0, np.cos(self.gamma)]])
        
        R_nozzle = np.array([[np.cos(self.angle2), -np.sin(self.angle2), 0],
                             [np.sin(self.angle2), np.cos(self.angle2), 0],
                             [0, 0, 1]])
        
        R_middle = np.array([[np.cos(self.angle1), -np.sin(self.angle1), 0],
                             [np.sin(self.angle1), np.cos(self.angle1), 0],
                             [0, 0, 1]])

        # Convert from nozzle frame to body frame
        R_base = np.array([[0, 0, -1],
                           [0, 1, 0],
                           [1, 0, 0]])

        self.R_nm = R_theta_fixed @ R_nozzle
        self.R_mb = R_middle
        self.R_br = R_base

class OUDisturbance:
    """
    Ornstein-Uhlenbeck process for generating temporally correlated physical disturbances.
    Uses the Euler-Maruyama method for accurate time-step integration.
    """
    def __init__(self, size=3, mu=0.0, theta=2.0, sigma=0.1, dt=0.01):
        """
        Args:
            size: Dimension of the vector (3 for 3D force/torque).
            mu: The mean value the noise returns to (0.0 for calm water).
            theta: The stiffness/pull-back force (higher = snaps back to mu faster).
            sigma: The volatility/randomness (higher = larger maximum disturbances).
            dt: The physics time step of your simulation.
        """
        self.size = size
        self.mu = np.full(size, mu)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = np.copy(self.mu)

    def reset(self):
        """Resets the disturbance back to the calm equilibrium state."""
        self.state = np.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Calculates and returns the disturbance vector for the current time step."""
        # dx = theta * (mu - x) * dt + sigma * sqrt(dt) * random_noise
        dx = self.theta * (self.mu - self.state) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.randn(self.size)
        
        self.state = self.state + dx
        return self.state
        
         
class Robot:
    """Simulates a jet-propelled robot with deformable body.
    
    The robot uses water jet propulsion and can contract/expand its body.
    Supports different phases: REFILL, JET, COAST, and REST.
    """
    
    class Phase(Enum):
        """Cycle phases for robot locomotion."""
        REFILL = 0
        JET = 1
        COAST = 2
        REST = 3

    phase = [Phase.REFILL, Phase.JET, Phase.COAST, Phase.REST]

    def __init__(self, dry_mass: float, init_length: float, init_width: float,
                 max_contraction: float, nozzle: Nozzle):
        """Initialize the robot.
        
        Args:
            dry_mass: Mass of the robot without water (kg)
            init_length: Initial length of the robot (m)
            init_width: Initial width of the robot (m)
            max_contraction: Maximum contraction distance (m)
            nozzle: Nozzle object for jet propulsion
        """

        # data-driven models
        self.geometric_coefficients = geometry.fit_length_width_relation_jit()
        self.refill_time_coefficients = geometry.fit_compression_refill_time_relation_jit()
        self.propulsion_time_coefficients = geometry.fit_compression_propulsion_time_relation_jit()

        # noise
        self.force_disturbance = OUDisturbance(size=3, mu=0.0, theta=2.0, sigma=0.05, dt=0.01)
        self.torque_disturbance = OUDisturbance(size=3, mu=0.0, theta=2.0, sigma=0.01, dt=0.01)
        self.force_noise = np.zeros(3)
        self.torque_noise = np.zeros(3)

        # ==================== Physical Parameters ====================
        self.dry_mass = dry_mass
        self.init_length = init_length
        self.init_width = init_width
        self.max_contraction = max_contraction
        self.density = 1000  # kg/m^3, density of water
        self.dt = 0.01  # time step
        self.nozzle = nozzle
        self.tube_volume = np.pi * (0.058 / 2)**2 * 0.15
        
        # ==================== Coefficient Parameters ====================
        self.dynamics_randomization = False
        self.disturbances = False
        self.discharge_coefficient_mean = 0.3 # should definite be lower than 0.6 maybe around 0.4 - 0.5
        self.drag_force_ratio_mean = 0.25 
        self.drag_torque_ratio_mean = 0.1
        self.added_mass_coefficient_force_mean = np.diag([0.5, 0.6, 0.6])
        self.added_mass_rate_coefficient_force_mean = np.diag([0.2, 0.2, 0.2])
        self.added_mass_coefficient_torque_mean = np.diag([0.3, 0.6, 0.6])
        self.added_mass_rate_coefficient_torque_mean = np.diag([0.2, 0.2, 0.2])
        self.trans_drag_coefficient_range = self._get_trans_drag_coefficient_range()
        self.rot_drag_coefficient_range = self._get_rot_drag_coefficient_range()
        
        # ==================== Control Parameters ====================
        self.contraction = 0.0
        self._contract_rate = 0.0
        self._release_rate = 0.0
        self.refill_time = 0.0
        self.jet_time = 0.0
        self.coast_time = 0.0
        
        # ==================== Cycle Tracking ====================
        self.state = self.phase[3]  # initial state is rest
        self.cycle = 0
        self.time = 0.0
        self.cycle_time = 0.0
        
        # ==================== Dynamic Properties ====================
        self.length = self.init_length
        self.width = self.init_width
        self.area = self._get_cross_sectional_area()
        self.volume = self._get_water_volume()
        self.water_mass = self._get_water_mass()
        self.prev_water_volume = self.volume
        self.prev_water_mass = self.water_mass
        self.mass = self.get_mass()
        self.mass_rate = self.get_mass_rate()
        self.prev_I = None
        
        # ==================== Force/Torque Vectors ====================
        self.jet_velocity = np.zeros(3)
        self.jet_force = np.zeros(3)
        self.jet_torque = np.zeros(3)
        self.drag_force = np.zeros(3)
        self.drag_torque = np.zeros(3)
        self.coriolis_force = np.zeros(3)
        self.coriolis_torque = np.zeros(3)
        self.added_mass_force = np.zeros(3)
        self.added_mass_torque = np.zeros(3)
        self.asymmetry_torque = np.zeros(3)
        self.deform_torque = np.zeros(3)
        self.trans_drag_coefficient = self._get_trans_drag_coefficient()
        self.rot_drag_coefficient = self._get_rot_drag_coefficient()

        # ==================== State Variables ====================
        self.position_world = np.zeros(3)
        self.position = np.zeros(3)
        self.prev_position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.velocity_world = np.zeros(3)
        self.avg_cycle_velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        
        self.euler_angle = np.zeros(3)
        self.euler_angle_rate = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.avg_cycle_angular_velocity = np.zeros(3)
        self.angular_acceleration = np.zeros(3)
        self.angle = np.zeros(3)
        self.prev_angle = np.zeros(3)
        # ==================== History Buffers ====================
        self.record = False
        self.state_history = []
        self.position_world_history = []
        self.velocity_history = []
        self.velocity_world_history = []
        self.acceleration_history = []
        self.euler_angle_history = []
        self.euler_angle_rate_history = []
        self.angular_velocity_history = []
        self.angular_acceleration_history = []
        self.length_history = []
        self.width_history = []
        self.area_history = []
        self.volume_history = []
        self.mass_history = []
        self.jet_velocity_history = []
        self.jet_force_history = []
        self.jet_torque_history = []
        self.coriolis_force_history = []
        self.coriolis_torque_history = []
        self.added_mass_force_history = []
        self.added_mass_torque_history = []
        self.deform_torque_history = []
        self.asymmetry_torque_history = []
        self.trans_drag_coefficient_history = []
        self.rot_drag_coefficient_history = []
        self.drag_force_history = []
        self.drag_torque_history = []
        self.nozzle_yaw_history = []
        self.inertia_tensor_history = []
        self.debug_buffer = []

    # ==================== Configuration Methods ====================
    def _get_trans_drag_coefficient_range(self):
        """Construct drag coefficient ranges for different body deformations."""
        # Different drag coefficients for along x, y, z directions
        # initial and end of deformation drag coefficients
        trans_x = [1.5, 2.5]
        trans_y = [1.2, 0.7]
        trans_z = [1.2, 0.7]

        return np.array([trans_x, trans_y, trans_z])

    def _get_rot_drag_coefficient_range(self):

        """Construct rotational drag coefficient ranges for different body deformations."""
        # Different drag coefficients for rotational x, y, z directions
        # initial and end of deformation drag coefficients
        rot_x = [0.1, 0.3]
        rot_y = [1.2, 0.5]
        rot_z = [1.2, 0.5]

        return np.array([rot_x, rot_y, rot_z])

    def enable_dynamic_randomization(self):
        """Enable domain randomization."""
        self.dynamics_randomization = True
    
    def enable_disturbances(self):
        self.disturbances = True

    def set_environment(self, density: float):
        """Set the environment properties.
        
        Args:
            density: Fluid density (kg/m^3)
        """
        self.density = density

    # ==================== Reset and Initialization ====================
    def reset(self):

        self.force_disturbance.reset()
        self.torque_disturbance.reset()
        self.force_noise = np.zeros(3)
        self.torque_noise = np.zeros(3)

        """Reset the robot to initial state."""
        self.time = 0.0
        self.cycle_time = 0.0
        self.cycle = 0
        self.state = self.phase[3]

        # Reset state variables
        self.position_world = np.zeros(3)
        self.position = np.zeros(3)
        self.prev_position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.velocity_world = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.euler_angle = np.zeros(3)
        self.euler_angle_rate = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.angular_acceleration = np.zeros(3)
        self.angle = np.zeros(3)
        self.prev_angle = np.zeros(3)

        # Reset body properties
        self.length = self.init_length
        self.width = self.init_width
        self.area = self._get_cross_sectional_area()
        self.volume = self._get_water_volume()
        self.water_mass = self._get_water_mass()
        self.mass = self.get_mass()
        self.prev_water_volume = self.volume
        self.prev_water_mass = self.water_mass
        self.mass_rate = self.get_mass_rate()
        self.prev_I = self.get_inertia_matrix()
        self.trans_drag_coefficient = self._get_trans_drag_coefficient()
        self.rot_drag_coefficient = self._get_rot_drag_coefficient()

        # Clear history
        self.clear_history()

    def clear_history(self):

        self.state_history = []
        self.position_world_history = []
        self.velocity_history = []
        self.velocity_world_history = []
        self.acceleration_history = []
        self.euler_angle_history = []
        self.euler_angle_rate_history = []
        self.angular_velocity_history = []
        self.angular_acceleration_history = []
        self.debug_buffer = []
        self.length_history = []
        self.width_history = []
        self.area_history = []
        self.volume_history = []
        self.mass_history = []
        self.jet_velocity_history = []
        self.jet_force_history = []
        self.jet_torque_history = []
        self.coriolis_force_history = []
        self.coriolis_torque_history = []
        self.added_mass_force_history = []
        self.added_mass_torque_history = []
        self.deform_torque_history = []
        self.asymmetry_torque_history = []
        self.trans_drag_coefficient_history = []
        self.rot_drag_coefficient_history = []
        self.drag_force_history = []
        self.drag_torque_history = []
        self.nozzle_yaw_history = []
        self.inertia_tensor_history = []


    # ==================== Control Methods ====================
    def set_control(self, contraction: float, coast_time: float, nozzle_angles: np.ndarray):
        """Set control inputs for the robot.
        
        Args:
            contraction: Desired contraction distance (m)
            coast_time: Duration of coast phase (s)
            nozzle_angles: Nozzle steering angles [angle1, angle2]
        """
        if self.dynamics_randomization:
            self._randomize_parameters()
        else:
            self.discharge_coefficient = self.discharge_coefficient_mean
            self.drag_force_ratio = self.drag_force_ratio_mean
            self.drag_torque_ratio = self.drag_torque_ratio_mean
            self.added_mass_coefficient_force = self.added_mass_coefficient_force_mean
            self.added_mass_rate_coefficient_force = self.added_mass_rate_coefficient_force_mean
            self.added_mass_coefficient_torque = self.added_mass_coefficient_torque_mean
            self.added_mass_rate_coefficient_torque = self.added_mass_rate_coefficient_torque_mean
        
        # if self.disturbances: 
        #     if self.cycle % 10 == 0:  # inject disturbances every 50 cycles
        #         self.force_noise = np.random.uniform(-0.04, 0.04, size=3)
        #         self.torque_noise = np.random.uniform(-0.01, 0.01, size=3)
        #         self.force_noise[-1] = 0  # no vertical force disturbance
        #         self.torque_noise[0:2] = 0  # no roll disturbance
        #         # print(f"Cycle {self.cycle}: Injecting disturbances - Force Noise: {self.force_noise}, Torque Noise: {self.torque_noise}")
        #     else:
        #         self.force_noise = np.zeros(3)
        #         self.torque_noise = np.zeros(3)
        # else:
        #     self.force_noise = np.zeros(3)
        #     self.torque_noise = np.zeros(3)

        self.clear_history()
        self.avg_cycle_velocity = np.zeros(3)
        self.avg_cycle_angular_velocity = np.zeros(3)
        self.contraction = contraction
        self.coast_time = coast_time
        self.nozzle.set_angles(angle1=nozzle_angles[0], angle2=nozzle_angles[1])

        # Proceed to next cycle
        self.cycle += 1
        # print(f"cycle number: {self.cycle}")
        self.cycle_time = 0.0

        self.refill_time = geometry.refill_time_from_compression_jit(self.contraction, self.refill_time_coefficients)
        self.jet_time = geometry.propulsion_time_from_compression_jit(self.contraction, self.propulsion_time_coefficients)
        self._contract_rate = self.contraction / self.refill_time if self.refill_time > 0 else 0.0
        self._release_rate = self.contraction / self.jet_time if self.jet_time > 0 else 0.0

    def _randomize_parameters(self):
        """Randomize robot parameters for domain randomization."""
        
        # Randomize discharge coefficient
        uncertainty = 0.5
        self.discharge_coefficient = geometry.randomize_scalar_jit(self.discharge_coefficient_mean, uncertainty, 0, 1)

        # Randomize drag ratios
        uncertainty = 0.5
        self.drag_force_ratio = geometry.randomize_scalar_jit(self.drag_force_ratio_mean, uncertainty)
        
        uncertainty = 0.5
        self.drag_torque_ratio = geometry.randomize_scalar_jit(self.drag_torque_ratio_mean, uncertainty)
        
        # Randomize added mass coefficients (force)
        uncertainty = 0.5
        upper_bound = self.added_mass_coefficient_force_mean * (1 + uncertainty)
        lower_bound = self.added_mass_coefficient_force_mean * (1 - uncertainty)
        self.added_mass_coefficient_force = np.random.uniform(lower_bound, upper_bound)
        
        uncertainty = 0.5
        upper_bound = self.added_mass_rate_coefficient_force_mean * (1 + uncertainty)
        lower_bound = self.added_mass_rate_coefficient_force_mean * (1 - uncertainty)
        self.added_mass_rate_coefficient_force = np.random.uniform(lower_bound, upper_bound)
        
        # Randomize added mass coefficients (torque)
        uncertainty = 0.5
        upper_bound = self.added_mass_coefficient_torque_mean * (1 + uncertainty)
        lower_bound = self.added_mass_coefficient_torque_mean * (1 - uncertainty)
        self.added_mass_coefficient_torque = np.random.uniform(lower_bound, upper_bound)
        
        uncertainty = 0.5
        upper_bound = self.added_mass_rate_coefficient_torque_mean * (1 + uncertainty)
        lower_bound = self.added_mass_rate_coefficient_torque_mean * (1 - uncertainty)
        self.added_mass_rate_coefficient_torque = np.random.uniform(lower_bound, upper_bound)

        # print(f"Domain Randomization Applied:")
        # print(f"  Discharge Coefficient: {self.discharge_coefficient:.4f}")
        # print(f"  Drag Force Ratio: {self.drag_force_ratio:.4f}")
        # print(f"  Drag Torque Ratio: {self.drag_torque_ratio:.4f}")
        # print(f"  Added Mass Coefficient (Force): {self.added_mass_coefficient_force}")
        # print(f"  Added Mass Rate Coefficient (Force): {self.added_mass_rate_coefficient_force}")
        # print(f"  Added Mass Coefficient (Torque): {self.added_mass_coefficient_torque}")
        # print(f"  Added Mass Rate Coefficient (Torque): {self.added_mass_rate_coefficient_torque}")

    # ==================== Stepping and State Management ====================
    def update_state(self):
        """Determine current phase based on cycle time."""
        if self.cycle_time <= max(self.refill_time, self.nozzle.turn_time):
            self.state = self.phase[0]  # contract
        elif self.cycle_time <= max(self.refill_time, self.nozzle.turn_time) + self.jet_time:
            self.state = self.phase[1]  # release
        elif self.cycle_time <= max(self.refill_time, self.nozzle.turn_time) + self.jet_time + self.coast_time:
            self.state = self.phase[2]  # coast
        else:
            self.state = self.phase[3]  # reset to rest
    
    def update_properties(self):
        """Update robot properties based on current state."""
        self.prev_water_volume = self.volume
        self.prev_water_mass = self.prev_water_volume * self.density

        self.length = self.get_current_length()
        self.width = self.get_current_width()
        self.area = self._get_cross_sectional_area()
        self.volume = self._get_water_volume()
        self.mass = self.get_mass()
        self.mass_rate = self.get_mass_rate()
        self.trans_drag_coefficient = self._get_trans_drag_coefficient()
        self.rot_drag_coefficient = self._get_rot_drag_coefficient()

    def step(self):
        """Advance simulation by one time step."""
        # I need to update dynamics first and then states?
        self.update_dynamics()
        self.cycle_time += self.dt
        self.time += self.dt
        self.nozzle.step(self.cycle_time)
        self.update_state()
        self.update_properties()

    # ==================== History and Cycle Methods ====================
    def enable_history_recording(self):
        self.record = True

    def disable_history_recording(self):
        self.record = False

    def _get_state_values(self):
        """Get dictionary of current state values for history tracking.
        
        Returns:
            Dictionary mapping history attribute names to current values
        """
        return {
            'state_history': self.state,
            'position_world_history': self.position_world.copy(),
            'velocity_history': self.velocity.copy(),
            'acceleration_history': self.acceleration.copy(),
            'euler_angle_history': self.euler_angle.copy(),
            'euler_angle_rate_history': self.euler_angle_rate.copy(),
            'angular_velocity_history': self.angular_velocity.copy(),
            'angular_acceleration_history': self.angular_acceleration.copy(),
            'length_history': self.length,
            'width_history': self.width,
            'area_history': self.area,
            'volume_history': self.volume,
            'mass_history': self.mass[0, 0],
            'mass_rate_history': self.mass_rate,
            'nozzle_yaw_history': self.nozzle.current_yaw,
            'inertia_tensor_history': np.diag(self.get_inertia_matrix()).copy(),
            'trans_drag_coefficient_history': self.trans_drag_coefficient,
            'rot_drag_coefficient_history': self.rot_drag_coefficient,
        }
    
    def _get_force_values(self):
        """Get dictionary of current force values for history tracking.
        
        Returns:
            Dictionary mapping history attribute names to current values
        """
        return {
            'jet_velocity_history': self.jet_velocity,
            'jet_force_history': self.jet_force,
            'jet_torque_history': self.jet_torque,
            'drag_force_history': self.drag_force,
            'drag_torque_history': self.drag_torque,
            'coriolis_force_history': self.coriolis_force,
            'coriolis_torque_history': self.coriolis_torque,
            'added_mass_force_history': self.added_mass_force,
            'added_mass_torque_history': self.added_mass_torque,
            'deform_torque_history': self.deform_torque,
            'asymmetry_torque_history': self.asymmetry_torque,
        }

    def step_through_cycle(self):
        """Step through an entire breathing cycle and collect state history."""
        total_cycle_time = max(self.refill_time, self.nozzle.turn_time) + self.jet_time + self.coast_time

        self.avg_cycle_velocity = (self.position - self.prev_position) / total_cycle_time
        self.avg_cycle_angular_velocity = (self.angle - self.prev_angle) / total_cycle_time

        self.prev_position = self.position.copy()
        self.prev_angle = self.angle.copy()


        if self.record:
            # Initialize history lists with current values
            for attr_name, initial_value in self._get_state_values().items():
                setattr(self, attr_name, [initial_value])

        while self.cycle_time < total_cycle_time:
            self.step()
            if not self.record:
                continue
            # Append force values to history lists
            for attr_name, current_value in self._get_force_values().items():
                getattr(self, attr_name).append(current_value)

            # Append current values to history lists
            for attr_name, current_value in self._get_state_values().items():
                getattr(self, attr_name).append(current_value)

        # # Convert histories to numpy arrays
        if self.record:
            history_names = self._get_state_values().keys()
            for attr_name in history_names:
                setattr(self, attr_name, np.array(getattr(self, attr_name)))
            
            history_names = self._get_force_values().keys()
            for attr_name in history_names:
                setattr(self, attr_name, np.array(getattr(self, attr_name)))


    # ==================== Coordinate Transformations ====================
    def _to_euler_angle_rate(self) -> np.ndarray:
        """Convert angular velocity to Euler angle rates using Numba."""
        return dynamics.to_euler_angle_rate_jit(self.euler_angle, self.angular_velocity)

    def _to_world_frame(self, vector: np.ndarray) -> np.ndarray:
        """Convert a vector from body frame to world frame using Numba."""
        return dynamics.to_world_frame_jit(self.euler_angle, vector)
    
    # ==================== Dynamics Update Methods ====================
    def _newton_equations(self) -> np.ndarray:
        """Compute translational accelerations using Numba."""
        self.coriolis_force = self._get_coriolis_force()
        self.drag_force = self._get_drag_force()
        self.jet_force = self._get_jet_force()
        self.added_mass_force = self._get_added_mass_force()

        if self.disturbances:
            self.force_noise = self.force_disturbance.sample()
            self.force_noise[-1] = 0  # no vertical force disturbance
        else:
            self.force_noise = np.zeros(3)

        self.mass = self.get_mass()

        return dynamics.compute_linear_acceleration_jit(
            self.mass, 
            self.jet_force, 
            self.drag_force, 
            self.added_mass_force, 
            self.coriolis_force,
            self.force_noise
        )

    def _euler_equations(self) -> np.ndarray:
        """Compute angular accelerations using Numba."""
        self.asymmetry_torque = self._asymmetry_torque_model()
        self.coriolis_torque = self._get_coriolis_torque()
        self.drag_torque = self._get_drag_torque()
        self.jet_torque = self._get_jet_torque()
        self.deform_torque = self._get_deform_torque()
        self.added_mass_torque = self._get_added_mass_torque()

        if self.disturbances:
            self.torque_noise = self.torque_disturbance.sample()
            self.torque_noise[0:2] = 0  # no roll disturbance
        else:
            self.torque_noise = np.zeros(3)

        I = self.get_inertia_matrix()

        return dynamics.compute_angular_acceleration_jit(
            I, 
            self.jet_torque, 
            self.drag_torque, 
            self.coriolis_torque, 
            self.asymmetry_torque, 
            self.deform_torque, 
            self.added_mass_torque,
            self.torque_noise
        )

    # ==================== Dynamics Update Methods ====================
    def update_dynamics(self):
        """Update acceleration and motion states."""
        self.acceleration = self._newton_equations()
        self.angular_acceleration = self._euler_equations()
        self._update_motion_states()

    def _update_motion_states(self):
        """Update robot state variables based on accelerations."""

        # states data and forces data are off by one time step dt

        self.velocity += self.acceleration * self.dt
        self.angular_velocity += self.angular_acceleration * self.dt

        self.euler_angle_rate = self._to_euler_angle_rate()
        self.euler_angle += self.euler_angle_rate * self.dt
        self.velocity_world = self._to_world_frame(self.velocity)
        self.position_world += self.velocity_world * self.dt

        # for average velocity and angular velocity
        self.position += self.velocity * self.dt
        self.angle += self.angular_velocity * self.dt

    # ==================== Inertia Methods ====================
    def get_inertia_matrix(self) -> np.ndarray:
        mass_scalar = self.mass[0, 0] # Extract raw float to pass to Numba
        jet_moment_arm = self._get_jet_moment_arm()
        return geometry.compute_inertia_matrix_jit(
            mass_scalar, self.length, self.width, self.nozzle.mass, jet_moment_arm
        )
    
    def get_inertia_matrix_rate(self) -> np.ndarray:
        """Calculate rate of change of inertia matrix.
        
        Returns:
            3x3 inertia matrix rate
        """
        I_rate = (self.get_inertia_matrix() - self.prev_I) / self.dt
        self.prev_I = self.get_inertia_matrix()
        return I_rate

    # ==================== Jet Force Methods ====================
    def _get_jet_moment_arm(self) -> np.ndarray:
        return geometry.compute_jet_moment_arm_jit(self.nozzle.get_middle_position(), self.length)

    def _get_jet_torque(self) -> np.ndarray:
        return dynamics.compute_jet_torque_jit(self._get_jet_moment_arm(), self.jet_force)
    
    def _get_jet_force(self) -> np.ndarray:

        self.jet_velocity = self._get_jet_velocity()
        if self.state != self.phase[1]:  # only produce jet force during release phase
            return np.zeros(3)
        
        mass_rate = self.get_mass_rate()
            
        return dynamics.compute_jet_force_jit(self.discharge_coefficient, mass_rate, self.jet_velocity)
    
    def _get_jet_velocity(self) -> np.ndarray:
        return dynamics.compute_jet_velocity_jit(
            self.state.value, self.volume, self.prev_water_volume, self.dt, 
            self.nozzle.area, self.nozzle.get_nozzle_direction()
        )

    # ==================== Drag Force and Torque Methods ====================
    def _get_drag_coefficient(self, ranges) -> np.ndarray:
        return geometry.compute_drag_coefficient_jit(
            self.length, self.width, self.init_length, self.init_width, self.max_contraction, ranges
        )
    
    def _get_rot_drag_coefficient(self) -> float:
        return self._get_drag_coefficient(self.rot_drag_coefficient_range)

    def _get_trans_drag_coefficient(self) -> float:
        return self._get_drag_coefficient(self.trans_drag_coefficient_range)

    def _get_drag_torque(self) -> np.ndarray:
        return dynamics.compute_drag_torque_jit(
            self.density, 
            self.rot_drag_coefficient, 
            self.area, 
            self.angular_velocity, 
            self.width, 
            self.length, 
            self.drag_torque_ratio
        )
    
    def _get_drag_force(self) -> np.ndarray:
        return dynamics.compute_drag_force_jit(
            self.density, 
            self.area, 
            self.trans_drag_coefficient, 
            self.velocity, 
            self.drag_force_ratio
        )

    # # ==================== Added Mass Methods ====================
    def _get_added_mass_force(self) -> np.ndarray:
        return dynamics.compute_added_mass_force_jit(
            self.mass, 
            self.added_mass_coefficient_force, 
            self.mass_rate, 
            self.added_mass_rate_coefficient_force, 
            self.acceleration, 
            self.angular_velocity, 
            self.velocity
        )
    
    def _get_added_mass_torque(self) -> np.ndarray:
        return dynamics.compute_added_mass_torque_jit(
            self.get_inertia_matrix(), 
            self.added_mass_coefficient_torque, 
            self.get_inertia_matrix_rate(), 
            self.added_mass_rate_coefficient_torque, 
            self.get_mass(), 
            self.added_mass_coefficient_force, 
            self.angular_acceleration, 
            self.angular_velocity, 
            self.velocity
        )

    # ==================== Coriolis Force and Torque Methods ====================
    def _get_coriolis_force(self) -> np.ndarray:
        return dynamics.compute_coriolis_force_jit(self.angular_velocity, self.get_mass(), self.velocity)

    def _get_coriolis_torque(self) -> np.ndarray:
        return dynamics.compute_coriolis_torque_jit(self.angular_velocity, self.get_inertia_matrix())

    # ==================== Deformation Methods ====================
    def _get_deform_torque(self) -> np.ndarray:
        return dynamics.compute_deform_torque_jit(self.get_inertia_matrix_rate(), self.angular_velocity)
    
    def _asymmetry_torque_model(self) -> np.ndarray:

        return dynamics.compute_asymmetry_torque_jit(self.velocity)


    # ==================== Geometry and Body Shape Methods ====================

    def get_current_length(self) -> float:
        return geometry.compute_length_jit(
            self.state.value, self.cycle_time, self.refill_time, self.nozzle.turn_time, 
            self.init_length, self.contraction, self._contract_rate, self._release_rate
        )

    def get_current_width(self) -> float:
        return geometry.compute_width_jit(
            self.state.value, self.cycle_time, self.refill_time, self.nozzle.turn_time, 
            self.init_width, self.contraction, self._contract_rate, self._release_rate
        )

    def _length_width_relation(self, length: float) -> float:
        """Calculate width based on length (volume conservation).
        
        Args:
            length: Current body length
            
        Returns:
            Corresponding body width
        """
        return geometry.width_from_length_jit(length, self.geometric_coefficients)

    def _get_cross_sectional_area(self) -> np.ndarray:
        return geometry.compute_cross_sectional_area_jit(self.length, self.width)

    # ==================== Mass and Volume Methods ====================
    def _get_water_volume(self) -> float:
        return geometry.compute_water_volume_jit(self.length, self.width) - self.tube_volume

    def _get_water_mass(self) -> float:
        return geometry.compute_water_mass_jit(self.density, self._get_water_volume())

    def get_mass(self) -> np.ndarray:
        self.water_mass = self._get_water_mass()
        return geometry.compute_mass_matrix_jit(self.dry_mass, self.water_mass, self.nozzle.mass)

    def get_mass_rate(self) -> np.ndarray:
        return geometry.compute_mass_rate_jit(self.water_mass, self.prev_water_mass, self.dt)

    # ==================== Timing Methods ====================
    # def _contract_model(self) -> float:
    #     """Calculate contraction time based on contraction distance.
        
    #     Returns:
    #         Time duration in seconds
    #     """
    #     self._contract_rate = 0.06 / 3  # m/s
    #     return self.contraction / self._contract_rate

    # def _release_model(self) -> float:
    #     """Calculate release time based on contraction distance.
        
    #     Returns:
    #         Time duration in seconds
    #     """
    #     self._release_rate = 0.06 / 1.5  # m/s
    #     return self.contraction / self._release_rate


if __name__ == "__main__":
    from plotting import (
        plot_angular_velocity, plot_drag_torque, plot_angular_acceleration,
        plot_euler_angles, plot_robot_geometry, plot_robot_mass, plot_mass_rate,
        plot_volume_rate, plot_cross_sectional_area, plot_jet_velocity,
        plot_jet_properties, plot_drag_coefficient, plot_drag_properties,
        plot_robot_position, plot_robot_velocity, plot_jet_torque, plot_trajectory_xy,
        plot_nozzle_direction, plot_nozzle_yaw_angle, plot_coriolis_force,
        plot_added_mass_force, plot_all_forces, plot_coriolis_torque,
        plot_deform_torque, plot_added_mass_torque, plot_asymmetry_torque,
        plot_inertia_tensor, plot_robot_acceleration
    )

    # Test the Robot and Nozzle classes
    nozzle = Nozzle(length1=0.052, length2=0.039, length3=0.031, area=np.pi*0.01**2, mass=0.440)
    robot = Robot(dry_mass=0.756, init_length=0.26, init_width=0.14, 
                  max_contraction=0.04, nozzle=nozzle)
    robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
    # robot.enable_domain_randomization()
    robot.set_environment(density=1000)
    robot.enable_history_recording()
    robot.reset()
    
    # Step through multiple cycles and collect state data
    n_cycles = 6
    
    # Initialize accumulators for all cycle data
    all_time_data = []
    all_state_data = []
    all_position_data = []
    all_velocity_data = []
    all_acceleration_data = []
    all_euler_angle_data = []
    all_euler_angle_rate_data = []
    all_angular_velocity_data = []
    all_angular_acceleration_data = []
    all_length_data = []
    all_width_data = []
    all_area_data = []
    all_volume_data = []
    all_mass_data = []
    all_jet_velocity_data = []
    all_jet_force_data = []
    all_jet_torque_data = []
    all_coriolis_force_data = []
    all_coriolis_torque_data = []
    all_added_mass_force_data = []
    all_added_mass_torque_data = []
    all_deform_torque_data = []
    all_asymmetry_torque_data = []
    all_drag_coefficient_data = []
    all_drag_force_data = []
    all_drag_torque_data = []
    all_nozzle_yaw_data = []
    all_inertia_tensor_data = []

    for i in range(n_cycles):

        robot.nozzle.set_yaw_angle(yaw_angle= 0 )
        robot.nozzle.solve_angles()
        robot.set_control(contraction=0.03, coast_time=3, 
                          nozzle_angles=np.array([robot.nozzle.angle1, robot.nozzle.angle2]))
        robot.step_through_cycle()
    
        # Create time array for this cycle
        cycle_start_time = robot.time - robot.cycle_time
        time_array = np.arange(cycle_start_time, robot.time, robot.dt)[:len(robot.length_history)-1]
        

        # # debugging
        # robot.velocity_history *= np.linalg.norm(robot.velocity_history, axis=1, keepdims=True)
        # robot.velocity_history *= robot.trans_drag_coefficient_history
        # # robot.debug_buffer.append(robot.debug_buffer[-1])
        # # robot.debug_buffer = np.array(robot.debug_buffer)
        # # robot.velocity_history *= robot.debug_buffer
        # robot.velocity_history *= robot.area_history
        # robot.velocity_history *= -0.5*robot.density

        # Accumulate data from each cycle
        all_time_data.extend(time_array)
        all_state_data.extend(robot.state_history[0:-1])
        all_position_data.extend(robot.position_world_history[0:-1])
        all_velocity_data.extend(robot.velocity_history[0:-1])
        all_acceleration_data.extend(robot.acceleration_history[0:-1])
        all_euler_angle_data.extend(robot.euler_angle_history[0:-1])
        all_euler_angle_rate_data.extend(robot.euler_angle_rate_history[0:-1])
        all_angular_velocity_data.extend(robot.angular_velocity_history[0:-1])
        all_angular_acceleration_data.extend(robot.angular_acceleration_history[0:-1])
        all_length_data.extend(robot.length_history[0:-1])
        all_width_data.extend(robot.width_history[0:-1])
        all_area_data.extend(robot.area_history[0:-1])
        all_volume_data.extend(robot.volume_history[0:-1])
        all_mass_data.extend(robot.mass_history[0:-1])
        all_jet_velocity_data.extend(robot.jet_velocity_history)
        all_jet_force_data.extend(robot.jet_force_history)
        all_jet_torque_data.extend(robot.jet_torque_history)
        all_coriolis_force_data.extend(robot.coriolis_force_history)
        all_coriolis_torque_data.extend(robot.coriolis_torque_history)
        all_added_mass_force_data.extend(robot.added_mass_force_history)
        all_added_mass_torque_data.extend(robot.added_mass_torque_history)
        all_deform_torque_data.extend(robot.deform_torque_history)
        all_asymmetry_torque_data.extend(robot.asymmetry_torque_history)
        all_drag_coefficient_data.extend(robot.trans_drag_coefficient_history[0:-1])
        all_drag_force_data.extend(robot.drag_force_history)
        all_drag_torque_data.extend(robot.drag_torque_history)
        all_nozzle_yaw_data.extend(robot.nozzle_yaw_history[0:-1])
        all_inertia_tensor_data.extend(robot.inertia_tensor_history[0:-1])

    # Convert accumulated data to numpy arrays
    all_time_data = np.array(all_time_data)
    all_state_data = np.array(all_state_data)
    all_position_data = np.array(all_position_data)
    all_velocity_data = np.array(all_velocity_data)
    all_acceleration_data = np.array(all_acceleration_data)
    all_euler_angle_data = np.array(all_euler_angle_data)
    all_euler_angle_rate_data = np.array(all_euler_angle_rate_data)
    all_angular_velocity_data = np.array(all_angular_velocity_data)
    all_angular_acceleration_data = np.array(all_angular_acceleration_data)
    all_length_data = np.array(all_length_data)
    all_width_data = np.array(all_width_data)
    all_area_data = np.array(all_area_data)
    all_volume_data = np.array(all_volume_data)
    all_mass_data = np.array(all_mass_data)
    all_jet_velocity_data = np.array(all_jet_velocity_data)
    all_jet_force_data = np.array(all_jet_force_data)
    all_jet_torque_data = np.array(all_jet_torque_data)
    all_coriolis_force_data = np.array(all_coriolis_force_data)
    all_coriolis_torque_data = np.array(all_coriolis_torque_data)
    all_added_mass_force_data = np.array(all_added_mass_force_data)
    all_added_mass_torque_data = np.array(all_added_mass_torque_data)
    all_deform_torque_data = np.array(all_deform_torque_data)
    all_asymmetry_torque_data = np.array(all_asymmetry_torque_data)
    all_drag_coefficient_data = np.array(all_drag_coefficient_data)
    all_drag_force_data = np.array(all_drag_force_data)
    all_drag_torque_data = np.array(all_drag_torque_data)
    all_nozzle_yaw_data = np.array(all_nozzle_yaw_data)
    all_inertia_tensor_data = np.array(all_inertia_tensor_data)

    # Plot results
    # plot_robot_geometry(all_time_data, all_length_data, all_width_data, all_state_data)
    # plot_cross_sectional_area(all_time_data, all_area_data, all_state_data)  
    # plot_robot_mass(all_time_data, all_mass_data, all_state_data) 
    # plot_volume_rate(all_time_data, all_volume_data, all_state_data)   
    # plot_mass_rate(all_time_data, all_mass_data, all_state_data)
    # plot_inertia_tensor(all_time_data, all_inertia_tensor_data, all_state_data)

    ## Translational Dynamics
    # plot_all_forces(all_time_data, all_jet_force_data, all_drag_force_data, 
    #                 all_coriolis_force_data, all_added_mass_force_data, all_state_data)
    # plot_jet_properties(all_time_data, all_jet_force_data, all_state_data)
    # plot_coriolis_force(all_time_data, all_coriolis_force_data, all_state_data)
    # plot_added_mass_force(all_time_data, all_added_mass_force_data, all_state_data)
    # plot_drag_coefficient(all_time_data, all_drag_coefficient_data, all_state_data)
    # plot_drag_properties(all_time_data, all_drag_force_data, all_state_data)

    plot_robot_velocity(all_time_data, all_velocity_data, all_state_data)  
    plot_robot_position(all_time_data, all_position_data, all_state_data)
    plot_robot_acceleration(all_time_data, all_acceleration_data, all_state_data)

    ## Rotational Dynamics
    # plot_angular_velocity(all_time_data, all_angular_velocity_data, all_state_data)
    # plot_angular_acceleration(all_time_data, all_angular_acceleration_data, all_state_data)
    # plot_euler_angles(all_time_data, all_euler_angle_data, all_state_data)

    # plot_jet_torque(all_time_data, all_jet_torque_data, all_state_data)
    # plot_drag_torque(all_time_data, all_drag_torque_data, all_state_data)
    # plot_coriolis_torque(all_time_data, all_coriolis_torque_data, all_state_data)
    # plot_deform_torque(all_time_data, all_deform_torque_data, all_state_data)
    # plot_added_mass_torque(all_time_data, all_added_mass_torque_data, all_state_data)
    # plot_asymmetry_torque(all_time_data, all_asymmetry_torque_data, all_state_data)
    # plot_nozzle_yaw_angle(all_time_data, all_nozzle_yaw_data, all_state_data)

    # plot_trajectory_xy(all_position_data, all_state_data, all_euler_angle_data)

    plt.show(block=True)
    