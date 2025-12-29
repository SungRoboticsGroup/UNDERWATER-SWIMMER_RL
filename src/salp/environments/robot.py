from enum import Enum
import numpy as np


class Nozzle:
    """Represents a steerable nozzle for jet propulsion.
    
    Attributes:
        length1: First segment length of the nozzle
        length2: Second segment length of the nozzle
        length3: Third segment length of the nozzle
        area: Area of nozzle openning
        angle1: Rotation angle around y axis
        angle2: Rotation angle around z axis
        mass: Mass of the nozzle
        pos_com: Position of the nozzle center of mass
    """
    
    def __init__(self, length1: float = 0.0, length2: float = 0.0, 
                 length3: float = 0.0, area: float = 0.0, mass: float = 0.0):
        self.length1 = length1
        self.length2 = length2
        self.length3 = length3
        self.mass = mass    
        self.area = area
        self.angle1 = 0.0  # angle around y axis
        self.angle2 = 0.0  # angle around z axis 
        self.gamma = np.pi / 4  # fixed tilt angle of nozzle downwards

        # Rotation matrices
        self.R_nm = None
        self.R_mb = None
        self.R_br = None

    def set_angles(self, angle1: float, angle2: float):
        """Set the nozzle angles.
        
        Args:
            angle1: Rotation angle around y axis
            angle2: Rotation angle around z axis
        """
        self.angle1 = angle1
        self.angle2 = angle2
        self._get_rotation_matrices()
    
    def get_nozzle_position(self) -> np.ndarray:
        """Calculate the nozzle position in world frame.
        
        Returns:
            3D position vector of the nozzle tip
        """
        # Nozzle position in nozzle frame
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
        # Nozzle tilted at 45 degrees downwards
        
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

        position = self.R_br @ self.R_mb @ base_position

        return position

    def _get_rotation_matrices(self) -> np.ndarray:
        """Calculate the overall rotation matrix of the nozzle.
        
        Returns:
            3x3 rotation matrix
        """
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
        R_base = np.array([[0, 0, 1],
                           [0, -1, 0],
                           [-1, 0, 0]])

        self.R_nm = R_theta_fixed @ R_nozzle
        self.R_mb = R_middle
        self.R_br = R_base
        
         
class Robot:
    """Simulates a jet-propelled robot with deformable body.
    
    The robot uses water jet propulsion and can contract/expand its body.
    Supports different phases: REFILL, JET, COAST, and REST.
    """
    
    class Phase(Enum):
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
        self.dry_mass = dry_mass  # kg
        self.mass = 0.0
        self.init_length = init_length  # meters
        self.init_width = init_width  # meters
        self.max_contraction = max_contraction  # max contraction length

        self.state = self.phase[3]  # initial state is rest
        self.contraction = 0.0  # contraction level
        self.nozzle_angles = np.zeros(2)  # current angle of nozzle
        self.cycle = 0
        self.dt = 0.01
        self.time = 0.0
        self.cycle_time = 0.0
        self.positions = np.zeros(3)  # x, y positions
        self.euler_angles = np.zeros(3)  #yaw
        self.euler_angle_rates = np.zeros(3)  # yaw rates
        self.velocities = np.zeros(3)  # x, y, z velocities
        self.accelerations = np.zeros(3)  # x, y, z accelerations
        self.angular_velocity = np.zeros(3)  # yaw rate
        self.angular_acceleration = np.zeros(3)  # yaw acceleration
        self.previous_water_volume = 0.0
        self.previous_water_mass = 0.0
        self.density = 0  # kg/m^3, density of water
        self.contract_time = 0.0
        self.release_time = 0.0
        self.coast_time = 0.0
        self.area = 0.0 # cross-sectional area
        self.jet_force = np.zeros(3)  # jet force vector
        self.jet_velocity = np.zeros(3)  # jet velocity vector
        self.volume = 0.0  # water volume inside the robot
        self.drag_force = np.zeros(3)  # drag force vector
        self.drag_coefficient = 0.0
        self.drag_torque = np.zeros(3)  # drag torque vector
        self.jet_torque = np.zeros(3)  # jet torque vector

        self._contract_rate = 0.0
        self._release_rateS = 0.0

        self._drag_coefficents = [0.2, 0.5] # min and max drag coefficients for different shapes

        self.length = 0.0
        self.width = 0.0
        self.nozzle = nozzle
    
    def set_environment(self, density: float):
        """Set the environment properties.
        
        Args:
            density: Fluid density (kg/m^3)
        """
        self.density = density

    def reset(self):
        """Reset the robot to initial state."""
        self.time = 0.0
        self.cycle_time = 0.0

        self.positions = np.zeros(3)  # x, y, z positions
        self.velocities = np.zeros(3)  # x, y, z velocities
        self.accelerations = np.zeros(3)  # x, y, z accelerations
        self.euler_angles = np.zeros(3)  # roll, pitch, yaw
        self.angular_velocity = np.zeros(3)  # yaw rate

        self.length = self.init_length
        self.width = self.init_width
        self.previous_water_volume = 0.0
        self.cycle = 0

        self.volume = self._get_water_volume()
        self.mass = self.get_mass()[0,0]
        self.previous_water_mass = self._get_water_mass()
        self.previous_water_volume = self._get_water_volume()

    def set_control(self, contraction: float, coast_time: float, angle: float):
        """Set control inputs for the robot.
        
        Args:
            contraction: Desired contraction distance (m)
            coast_time: Duration of coast phase (s)
            angle: Nozzle steering angle
        """
        self.contraction = contraction
        self.coast_time = coast_time

        self.nozzle_angle = angle
        self.cycle += 1
        self.cycle_time = 0.0

        self.contract_time = self._contract_model()
        self.release_time = self._release_model()

    def get_state(self) -> str:
        """Determine current phase based on cycle time.
        
        Returns:
            Current phase state
        """
        if self.cycle_time < self.contract_time:
            self.state = self.phase[0]  # contract
        elif self.cycle_time < self.contract_time + self.release_time:
            self.state = self.phase[1]  # release
        elif self.cycle_time < self.contract_time + self.release_time + self.coast_time:
            self.state = self.phase[2]  # coast
        else:
            self.state = self.phase[3]  # reset to contract
        return self.state

    def step(self):
        """Advance simulation by one time step."""
        self.cycle_time += self.dt
        self.time += self.dt
        self.get_state()

        self.length = self.get_current_length()
        self.width = self.get_current_width()
        if self.state == self.phase[0]:  # contract
            self.contract()
        elif self.state == self.phase[1]:  # release
            self.release()
        elif self.state == self.phase[2]:  # coast
            self.coast()
        else:
            pass  # rest
    
    def step_through_cycle(self):
        """Step through an entire breathing cycle and collect state history.
        
        Returns:
            Tuple of arrays containing time history of various state variables
        """

        total_cycle_time = self.contract_time + self.release_time + self.coast_time
        positions_history = [self.positions.copy()]
        euler_angles_history = [self.euler_angles.copy()]
        length_history = [self.length]
        width_history = [self.width]
        mass_history = [self.get_mass()[0,0]]  # store only scalar mass value
        area_data = [self._get_cross_sectional_area()]
        state_data = [self.state]
        jet_force_data = [self.jet_force]
        jet_velocity_data = [self.jet_velocity]
        volume_data = [self.volume]
        drag_data = [self.drag_force]
        drag_coefficient_data = [self.drag_coefficient]
        velocity_data = [self.velocities.copy()]
        angular_velocity_data = [self.angular_velocity.copy()]
        angular_acceleration_data = [self.angular_acceleration.copy()]
        drag_torque_data = [self.drag_torque.copy()]
        jet_torque_data = [self.jet_torque.copy()]

        while self.cycle_time < total_cycle_time:
            self.step()
            positions_history.append(self.positions.copy())
            euler_angles_history.append(self.euler_angles.copy())
            length_history.append(self.length)
            width_history.append(self.width)
            mass_history.append(self.mass)
            area_data.append(self.area)
            state_data.append(self.state)
            jet_force_data.append(self.jet_force)
            jet_velocity_data.append(self.jet_velocity)
            volume_data.append(self.volume)
            drag_data.append(self.drag_force)
            drag_coefficient_data.append(self.drag_coefficient)
            velocity_data.append(self.velocities.copy())
            angular_velocity_data.append(self.angular_velocity.copy())
            angular_acceleration_data.append(self.angular_acceleration.copy())
            drag_torque_data.append(self.drag_torque.copy())
            jet_torque_data.append(self.jet_torque.copy())

        return np.array(positions_history), np.array(euler_angles_history), \
                np.array(length_history), np.array(width_history), \
                np.array(mass_history), np.array(area_data), \
                np.array(state_data), np.array(jet_force_data), \
                np.array(jet_velocity_data), np.array(volume_data), \
                np.array(drag_data), np.array(drag_coefficient_data), \
                np.array(velocity_data), np.array(angular_velocity_data), \
                np.array(angular_acceleration_data), np.array(drag_torque_data), \
                np.array(jet_torque_data)

    def _to_euler_angle_rates(self) -> np.ndarray:
        """Convert angular velocity to Euler angle rates.
        
        Returns:
            Euler angle rate vector
        """
        phi, theta, psi = self.euler_angles

        T = np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                      [0, np.cos(phi), -np.sin(phi)],
                      [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]])

        self.euler_angle_rates = T @ self.angular_velocity

    def _to_world_frame(self, vector: np.ndarray) -> np.ndarray:
        """Convert a vector from body frame to world frame.
        
        Args:
            vector: 3D vector in body frame
            
        Returns:
            3D vector in world frame
        """
        phi, theta, psi = self.euler_angles

        R_x = np.array([[1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)]])
        
        R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])
        
        R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                        [np.sin(psi), np.cos(psi), 0],
                        [0, 0, 1]])
        
        R = R_z @ R_y @ R_x

        return R @ vector

    def _update_states(self):
        """Update robot state variables based on accelerations."""
        self.velocities += self.accelerations * self.dt
        self.angular_velocity += self.angular_acceleration * self.dt

        self._to_euler_angle_rates()
        self.euler_angles += self.euler_angle_rates * self.dt
        self.positions += self._to_world_frame(self.velocities) * self.dt

    def contract(self):
        """Execute contraction phase dynamics."""
        self.accelerations = self._newton_equations()
        self.angular_acceleration = self._euler_equations()
        self._update_states()

    def release(self):
        """Execute release (jet) phase dynamics."""
        self.accelerations = self._newton_equations()
        self.angular_acceleration = self._euler_equations()
        self._update_states()

    def coast(self):
        """Execute coast phase dynamics."""
        self.accelerations = self._newton_equations()
        self.angular_acceleration = self._euler_equations()
        self._update_states()

    def get_current_length(self) -> float:
        """Calculate current body length based on phase.
        
        Returns:
            Current length in meters
        """
        if self.state == self.phase[0]:  # inhale
            length = self.init_length - self.cycle_time*self._contract_rate
        elif self.state == self.phase[1]:  # exhale
            length = self.init_length - self.max_contraction + (self.cycle_time - self.contract_time)*self._release_rate
        else:
            length = self.init_length

        return length
    
    def get_current_width(self) -> float:
        """Calculate current body width based on phase.
        
        Returns:
            Current width in meters
        """
        if self.state == self.phase[0]:  # inhale
            width = self.init_width + self.cycle_time*self._contract_rate
        elif self.state == self.phase[1]:  # exhale
            width = self.init_width + self.max_contraction - (self.cycle_time - self.contract_time)*self._release_rate
        else:
            width = self.init_width

        return width

    def get_mass(self) -> float:
        """Calculate total mass including water.
        
        Returns:
            Mass matrix (diagonal)
        """
        mass = self.dry_mass + self._get_water_mass()
        mass *= np.diag(np.ones(3))
        self.mass = mass[0,0]

        return mass
    
    def get_inertia_matrix(self) -> np.ndarray:
        """Calculate moment of inertia matrix.
        
        Note: Currently only considers water inertia.
        
        Returns:
            3x3 inertia matrix
        """
        I_xx = 0.2 * self.mass * ((self.width/2) ** 2 + (self.width ** 2))
        I_yy = 0.2 * self.mass * ((self.length/2) ** 2 + (self.width/2) ** 2)
        I_zz = 0.2 * self.mass * ((self.width/2) ** 2 + (self.length/2) ** 2)

        return np.diag([I_xx, I_yy, I_zz])

    def _get_jet_moment_arm(self) -> np.ndarray:
        """Calculate moment arm for jet force.
        
        Returns:
            3D moment arm vector
        """
        r_nozzle = self.nozzle.get_2nd_position()
        r_robot = np.array([-self.length/2, 0.0, 0.0])  # center of mass at origin
        return r_nozzle + r_robot
    
    def _get_jet_torque(self) -> np.ndarray:
        """Calculate torque from jet force.
        
        Returns:
            3D torque vector
        """

        T_jet = np.cross(self._get_jet_moment_arm(), self.jet_force)
        self.jet_torque = T_jet
        return T_jet
    
    def _get_jet_force(self) -> np.ndarray:
        """Calculate jet propulsion force.
        
        Returns:
            3D force vector
        """
        if self.state != self.phase[1]:  # only produce jet force during release phase
            self._get_jet_velocity()
            water_mass = self._get_water_mass()
            self.previous_water_mass = water_mass
            return np.zeros(3)

        water_mass = self._get_water_mass()
        mass_rate = (water_mass - self.previous_water_mass) / self.dt
        self.previous_water_mass = water_mass
        jet_velocity = self._get_jet_velocity()
        jet_force = mass_rate * jet_velocity

        C_discharge = 0.1  # discharge coefficient
        self.jet_force = jet_force * C_discharge

        return self.jet_force
    
    def _get_jet_velocity(self) -> np.ndarray:
        """Calculate jet velocity vector.
        
        Returns:
            3D velocity vector in robot frame
        """
        water_volume = self._get_water_volume()
        volume_rate = (water_volume - self.previous_water_volume) / self.dt
        self.previous_water_volume = water_volume
        jet_speed = volume_rate / self.nozzle.area
        
        direction = self.nozzle.get_nozzle_direction()
        jet_velocity = -direction * jet_speed
        self.jet_velocity = jet_velocity
        self.volume = water_volume
        return jet_velocity
    
    def _length_width_relation(self, length: float) -> float:
        """Calculate width based on length (volume conservation).
        
        Args:
            length: Current body length
            
        Returns:
            Corresponding body width
        """
        return self.init_length - length + self.init_width

    def _get_drag_coefficient(self) -> float:
        """Calculate drag coefficient based on body shape.
        
        More elongated (contracted) = lower drag, more spherical = higher drag.
        
        Returns:
            Drag coefficient
        """

        aspect_ratio = self.length / self.width
        
        init_aspect_ratio = self.init_length / self.init_width  # most elongated
        contracted_length = self.init_length - self.max_contraction
        contracted_width = self._length_width_relation(contracted_length)
        min_aspect_ratio = contracted_length / contracted_width  # most spherical
        
        # Normalize to [0, 1]: 0 = most spherical, 1 = most elongated
        normalized_ratio = (aspect_ratio - min_aspect_ratio) / (init_aspect_ratio - min_aspect_ratio)
        normalized_ratio = np.clip(normalized_ratio, 0, 1)
        
        C_d = self._drag_coefficents[1] - normalized_ratio * (self._drag_coefficents[1] - self._drag_coefficents[0])
        self.drag_coefficient = C_d
        return C_d

    def _get_drag_torque(self) -> np.ndarray:
        """Calculate drag torque on the robot.
        
        Returns:
            3D torque vector
        """
        T_drag = - 4.0 / 15.0 * self.density * self.drag_coefficient * \
                (self.width/2) ** 2 * (self.length / 2) ** 4 * abs(self.angular_velocity) * self.angular_velocity
        self.drag_torque = T_drag
        return T_drag
    
    def _get_drag_force(self) -> np.ndarray:
        """Calculate drag force on the robot.
        
        Returns:
            3D force vector
        """
        C_d = self._get_drag_coefficient()
        self.area = self._get_cross_sectional_area()
        F_drag = -0.5 * self.density * self.area * C_d * abs(self.velocities) * self.velocities
        self.drag_force = F_drag
        return F_drag
    
    def _get_added_mass(self) -> float:
        """Calculate added mass from surrounding fluid.
        
        Returns:
            Added mass (currently not implemented)
        """
        # TODO: Implement added mass calculation
        # added_mass = 0.5 * self.density * self._get_water_volume()
        return 0.0
    
    def _get_coriolis_force(self) -> np.ndarray:
        """Calculate Coriolis force.
        
        Returns:
            3D force vector
        """
        return self.get_mass() @ self.angular_velocity * self.velocities

    def _newton_equations(self) -> np.ndarray:
        """Compute translational accelerations using Newton's equations.
        
        Returns:
            3D acceleration vector
        """

        F_coriolis = self._get_coriolis_force()
        F_drag = self._get_drag_force()
        F_jet = self._get_jet_force()
        mass = self.get_mass()
        return np.linalg.inv(mass) @ (F_jet + F_drag + F_coriolis)

    def _get_coriolis_torque(self) -> np.ndarray:
        """Calculate Coriolis torque.
        
        Returns:
            3D torque vector
        """
        return -np.cross(self.angular_velocity, self.get_inertia_matrix() @ self.angular_velocity)

    def _euler_equations(self) -> np.ndarray:
        """Compute angular accelerations using Euler's equations.
        
        Returns:
            3D angular acceleration vector
        """
        T_asymmetry = np.zeros(3)  # TODO: Implement asymmetry torque
        T_coriolis = self._get_coriolis_torque()
        T_drag = self._get_drag_torque()
        T_jet = self._get_jet_torque()
        # T_jet = np.array([0.0, 0.0, 0.0])  # test orientation

        I = self.get_inertia_matrix()
        # print(T_coriolis)
        alpha = np.linalg.inv(I) @ (T_jet + T_drag + T_coriolis + T_asymmetry)

        return alpha


    def _get_water_volume(self) -> float:
        
        volume = 4/3 * np.pi * (self.length/2) * (self.width/2) ** 2

        return volume

    def _get_water_mass(self) -> float:

        mass = self.density * self._get_water_volume()

        return mass

    def _get_cross_sectional_area(self) -> float:

        area = np.pi * (self.length/2) * (self.width/2)

        return area

    def _contract_model(self) -> float:
        """Calculate contraction time based on contraction distance.
        
        Returns:
            Time duration in seconds
        """
        self._contract_rate = 0.06/3  # m/s
        return self.contraction / self._contract_rate

    def _release_model(self) -> float:
        """Calculate release time based on contraction distance.
        
        Returns:
            Time duration in seconds
        """
        self._release_rate = 0.06/1.5  # m/s
        return self.contraction / self._release_rate


if __name__ == "__main__":
    from plotting import (
        plot_angular_velocity, plot_drag_torque, plot_angular_acceleration,
        plot_euler_angles, plot_robot_geometry, plot_robot_mass, plot_mass_rate,
        plot_volume_rate, plot_cross_sectional_area, plot_jet_velocity,
        plot_jet_properties, plot_drag_coefficient, plot_drag_properties,
        plot_robot_position, plot_robot_velocity, plot_jet_torque
    )

    # Test the Robot and Nozzle classes
    nozzle = Nozzle(length1=0.01, length2=0.01, area=0.00009)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                  max_contraction=0.06, nozzle=nozzle)
    robot.nozzle.set_angles(angle1=0.0, angle2=np.pi)  # set nozzle angles
    
    robot.set_environment(density=1000)  # water density in kg/m^3
    robot.reset()
    
    # Set control inputs
    robot.set_control(contraction=0.06, coast_time=1, angle= 0.0)
    
    # Step through a cycle and collect state data
    positions_history, euler_angles_history, length_history, \
    width_history, mass_history, area_data, state_data, \
    jet_force_data, jet_velocity_data, volume_data, drag_data, \
    drag_coefficient_data, velocity_data, angular_velocity_data, \
    angular_acceleration_data, drag_torque_data, jet_torque_data = robot.step_through_cycle()
    
    # Create time array
    time_array = np.arange(0, robot.time + robot.dt, robot.dt)[:len(length_history)]
    
    # Plot with phase backgrounds
    # plot_robot_geometry(time_array, length_history, width_history, state_data) 
    # plot_robot_mass(time_array, mass_history, state_data) 
    # plot_mass_rate(time_array, mass_history, state_data)
    # plot_volume_rate(time_array, volume_data, state_data)   

    # plot_cross_sectional_area(time_array, area_data, state_data)  
    # plot_jet_velocity(time_array, jet_velocity_data, state_data)  # approximate jet velocity
    # plot_jet_properties(time_array, jet_force_data, state_data)
    # plot_drag_coefficient(time_array, drag_coefficient_data, state_data)
    # plot_drag_properties(time_array, drag_data, state_data)
    plot_robot_position(time_array, positions_history, state_data)
    # print("Velocity data shape:", velocity_data)
    plot_robot_velocity(time_array, velocity_data, state_data)  

    plot_angular_velocity(time_array, angular_velocity_data, state_data)
    # plot_jet_torque(time_array, jet_torque_data, state_data)
    # plot_drag_torque(time_array, drag_torque_data, state_data)
    # plot_angular_acceleration(time_array, angular_acceleration_data, state_data)
    plot_euler_angles(time_array, euler_angles_history, state_data)