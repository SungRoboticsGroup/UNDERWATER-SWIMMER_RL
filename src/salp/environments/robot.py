from matplotlib.pylab import Enum
import numpy as np

class Nozzle():
    def __init__(self, length1: float = 0.0, length2: float = 0.0, area: float = 0.0):
        self.length1 = length1  
        self.length2 = length2
        self.area = area    
        self.angle1 = 0.0  # angle around y axis
        self.angle2 = 0.0  # angle around z axis 
    
    def set_angles(self, angle1: float, angle2: float):
        self.angle1 = angle1
        self.angle2 = angle2
    
    def get_position(self) -> np.ndarray:
        # compute the nozzle direction vector based on angles
        # placeholder for now

        # nozzle tilted at 45 degrees downwards
        theta = -np.pi / 4
        pos_x = 0
        pos_y = self.length2 * np.sin(theta)
        pos_z = self.length2 * np.cos(theta)
        nozzle_position = np.array([pos_x, pos_y, pos_z])  # base position of the nozzle
        # print("nozzle position before rotation:", nozzle_position)

        pos_x1 = 0
        pos_y1 = 0
        pos_z1 = self.length1
        base_position = np.array([pos_x1, pos_y1, pos_z1])  # base position of the nozzle

        R_theta_fixed = np.array([[1, 0, 0],
                                  [0, np.cos(theta), -np.sin(theta)],
                                  [0, np.sin(theta), np.cos(theta)]])
        # print(R_theta_fixed)
        
        R_alpha = np.array([[np.cos(self.angle2), -np.sin(self.angle2), 0], 
                            [np.sin(self.angle2), np.cos(self.angle2), 0], 
                            [0, 0, 1]])  
        
        R_beta = np.array([[np.cos(self.angle1), -np.sin(self.angle1), 0],   
                           [np.sin(self.angle1), np.cos(self.angle1), 0],
                           [0, 0, 1]])

        # print(nozzle_position)
        # print(R_alpha @ nozzle_position)
        # print(R_theta_fixed @ R_alpha @ nozzle_position)
        position = R_beta @ (base_position + R_theta_fixed @ R_alpha @ nozzle_position)

        return position
    
    def get_nozzle_direction(self) -> np.ndarray:
        # placeholder for now

        # nozzle tilted at 45 degrees downwards
        theta = -np.pi / 4
        pos_x = 0
        pos_y = np.sin(theta)
        pos_z = np.cos(theta)
        nozzle_direction = np.array([pos_x, pos_y, pos_z])  # base position of the nozzle

        R_theta_fixed = np.array([[1, 0, 0],
                                  [0, np.cos(theta), -np.sin(theta)],
                                  [0, np.sin(theta), np.cos(theta)]])
        
        R_alpha = np.array([[np.cos(self.angle2), -np.sin(self.angle2), 0], 
                            [np.sin(self.angle2), np.cos(self.angle2), 0], 
                            [0, 0, 1]])  
        
        R_beta = np.array([[np.cos(self.angle1), -np.sin(self.angle1), 0],   
                           [np.sin(self.angle1), np.cos(self.angle1), 0],
                           [0, 0, 1]])

        # convert from nozzle frame to body frame
        R_body = np.array([[0, 0, -1],
                           [0, 1, 0],
                           [1, 0, 0]])

        direction = R_body @ R_beta @ R_theta_fixed @ R_alpha @ nozzle_direction
 
        return direction
    
    
    def get_2nd_position(self):

        pos_x1 = 0
        pos_y1 = 0
        pos_z1 = self.length1
        base_position = np.array([pos_x1, pos_y1, pos_z1])  # base position of the nozzle

        R_beta = np.array([[np.cos(self.angle1), -np.sin(self.angle1), 0],   
                           [np.sin(self.angle1), np.cos(self.angle1), 0],
                           [0, 0, 1]])

        # print(nozzle_position)
        # print(R_alpha @ nozzle_position)
        # print(R_theta_fixed @ R_alpha @ nozzle_position)
        # convert from nozzle frame to body frame
        R_body = np.array([[0, 0, -1],
                           [0, 1, 0],
                           [1, 0, 0]])

        position = R_body @ R_beta @ base_position 
        return position

class Robot():
    
    # TODO:
    # 1. implement nozzle steering behavior
    # 2. scale the actions from RL inputs to robot control inputs
    # 3. normalize the observation space numbers
    #  
    
    class Phase(Enum):
        REFILL = 0
        JET = 1
        COAST = 2
        REST = 3

    phase = [Phase.REFILL, Phase.JET, Phase.COAST, Phase.REST]

    def __init__(self, dry_mass: float, init_length: float, init_width: float, 
                 max_contraction: float, nozzle: Nozzle):
        
        self.dry_mass = dry_mass # kg
        self.mass = 0.0
        self.init_length = init_length # meters
        self.init_width = init_width # meters
        self.max_contraction = max_contraction  # max contraction length

        self.state = self.phase[3]  # initial state is rest
        self.contraciton = 0.0  # contraction level
        self.nozzle_angles = np.zeros(2)  # current angle of nozzle
        self.cycle = 0
        self.dt = 0.01
        self.time = 0.0
        self.cycle_time = 0.0
        self.positions = np.zeros(3)  # x, y positions
        self.euler_angles = np.zeros(3)  #yaw
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

        self.density = density

    def reset(self):

        self.time = 0.0
        self.cycle_time = 0.0

        self.positions = np.zeros(3)  # x, y, z positions
        self.velocities = np.zeros(3)  # x, y, z velocities
        self.accelerations = np.zeros(3)  # x, y, z acceler 
        self.euler_angles = np.zeros(3)  # yaw
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

        self.contraciton = contraction
        self.coast_time = coast_time

        self.nozzle_angle = angle
        self.cycle += 1
        self.cycle_time = 0.0

        self.contract_time = self._contract_model()
        self.release_time = self._release_model() 

    def get_state(self) -> str:

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
    
    def _update_states(self):
        # print(self.velocities)
        self.velocities += self.accelerations * self.dt  # update velocities
        self.positions += self.velocities * self.dt  # update positions
        
        self.angular_velocity += self.angular_acceleration * self.dt
        self.euler_angles += self.angular_velocity * self.dt

    def contract(self):

        self.accelerations = self._newton_equations()
        self.accelerations = np.zeros(3)  # test orientation
        self.angular_acceleration = self._euler_equations()
        self._update_states()
        
    def release(self):
        self.accelerations = self._newton_equations()
        self.accelerations = np.zeros(3)  # test orientation
        self.angular_acceleration = self._euler_equations()
        self._update_states()

    def coast(self):
        self.accelerations = self._newton_equations()
        self.accelerations = np.zeros(3)  # test orientation
        self.angular_acceleration = self._euler_equations()
        self._update_states()

    def get_current_length(self) -> float:

        if self.state == self.phase[0]:  # inhale
            length = self.init_length - self.cycle_time*self._contract_rate
        elif self.state == self.phase[1]:  # exhale
            length = self.init_length - self.max_contraction + (self.cycle_time - self.contract_time)*self._release_rate
        else:
            length = self.init_length

        return length
    
    def get_current_width(self) -> float:

        if self.state == self.phase[0]:  # inhale
            width = self.init_width + self.cycle_time*self._contract_rate
        elif self.state == self.phase[1]:  # exhale
            width = self.init_width + self.max_contraction - (self.cycle_time - self.contract_time)*self._release_rate
        else:
            width = self.init_width

        return width

    def get_mass(self) -> float:

        mass = self.dry_mass + self._get_water_mass()
        mass *= np.diag(np.ones(3))
        self.mass = mass[0,0]

        return mass
    
    def get_inertia_matrix(self) -> float:
        
        # this only consider the water inertia

        I_xx = 0.2 * self.mass * ((self.width/2) ** 2 + (self.width ** 2))  # placeholder for now
        I_yy = 0.2 * self.mass * ((self.length/2) ** 2 + (self.width/2) ** 2)
        I_zz = 0.2 * self.mass * ((self.width/2) ** 2 + (self.length/2) ** 2)

        # inertia matrix is changing as the robot contracts and releases
        I = np.diag([I_xx, I_yy, I_zz])

        return I

    def _get_jet_moment_arm(self) -> float:

        # placeholder for now
        r_nozzle = self.nozzle.get_2nd_position()
        r_robot = np.array([-self.length/2, 0.0, 0.0])  # center of mass at origin

        return r_nozzle + r_robot 
    
    def _get_jet_torque(self) -> float:

        # placeholder for now
        T_jet = np.cross(self._get_jet_moment_arm(), self._get_jet_force())
        self.jet_torque = T_jet
        return T_jet
    
    def _get_jet_force(self) -> float:

        if self.state != self.phase[1]:  # only produce jet force during release phase
            self._get_jet_velocity()
            water_mass = self._get_water_mass()
            self.previous_water_mass = water_mass
            return np.zeros(3)      

        water_mass = self._get_water_mass()
        mass_rate = (water_mass - self.previous_water_mass) / self.dt
        # print("mass rate:", mass_rate)
        self.previous_water_mass = water_mass
        jet_velocity = self._get_jet_velocity()
        jet_force = mass_rate * jet_velocity
        # jet_force = 0.0  # placeholder for now 
        # print("jet force:", jet_force)

        C_discharge = 0.1  # discharge coefficient
        self.jet_force = jet_force * C_discharge

        return self.jet_force
    
    def _get_jet_velocity(self) -> float:
        
        # velocity is with respect to the robot frame
        water_volume = self._get_water_volume()
        # print(water_volume)
        volume_rate = (water_volume - self.previous_water_volume) / self.dt
        self.previous_water_volume = water_volume
        jet_speed = volume_rate / self.nozzle.area  # m/s
        
        direction = self.nozzle.get_nozzle_direction()
        jet_velocity = -direction * jet_speed  # jet velocity in robot frameo
        self.jet_velocity = jet_velocity
        self.volume = water_volume

        # print(direction)
        return jet_velocity
    
    def _length_width_relation(self, length) -> float:
        
        # simple linear relation for now
        width = self.init_length - length + self.init_width

        return width

    def _get_drag_coefficient(self) -> float:
        # Map drag coefficient based on aspect ratio (length/width)
        # More elongated (contracted) = lower drag, more spherical = higher drag
        aspect_ratio = self.length / self.width
        
        # Calculate aspect ratio range
        init_aspect_ratio = self.init_length / self.init_width  # most elongated
        contracted_length = self.init_length - self.max_contraction
        contracted_width = self._length_width_relation(contracted_length)
        min_aspect_ratio = contracted_length / contracted_width  # most spherical
        
        # Normalize current aspect ratio to [0, 1]
        # 0 = most spherical (max drag), 1 = most elongated (min drag)
        normalized_ratio = (aspect_ratio - min_aspect_ratio) / (init_aspect_ratio - min_aspect_ratio)
        normalized_ratio = np.clip(normalized_ratio, 0, 1)
        
        # Interpolate: elongated -> low drag, spherical -> high drag
        C_d = self._drag_coefficents[1] - normalized_ratio * (self._drag_coefficents[1] - self._drag_coefficents[0])
        
        self.drag_coefficient = C_d
        return C_d 

    def _get_drag_torque(self) -> float:

        # placeholder for now
        
        T_drag = - 4.0 / 15.0 * self.density * self.drag_coefficient * \
                (self.width/2) ** 2 * (self.length / 2) ** 4 * abs(self.angular_velocity) * self.angular_velocity
        
        # T_drag = np.zeros(3)  # placeholder for now
        self.drag_torque = T_drag
        return T_drag
    
    def _get_drag_force(self) -> float:

        C_d = self._get_drag_coefficient()
        self.area = self._get_cross_sectional_area()
        F_drag = - 0.5 * self.density * self.area * C_d * abs(self.velocities) * self.velocities 
        # print(self.velocities)
        # drag_force = 0.0  # placeholder for now
        self.drag_force = F_drag  # drag force opposes motion
        return F_drag
    
    def _get_added_mass(self) -> float:

        # added_mass = 0.5 * self.density * self._get_water_volume()  # added mass for sphere is 0.5 * density * volume
        added_mass = 0.0  # placeholder for now

        return added_mass
    
    def _get_coriolis_force(self) -> float:
        
        # placeholder for now
        F_coriolis = self.get_mass() @ self.angular_velocity * self.velocities

        return F_coriolis

    def _newton_equations(self):
        
        # Forces presented here
        # 1. coriolis force 
        # 2. drag force
        # 3. jet force
        
        F_coriolis = self._get_coriolis_force()
        F_drag = self._get_drag_force()
        F_jet = self._get_jet_force()
        # print("F_jet:", F_jet)
        # print("F_drag:", F_drag)

        mass = self.get_mass()

        a = np.linalg.inv(mass) @ (F_jet + F_drag + F_coriolis)

        return a

    def _get_coriolis_torque(self) -> float:
        
        # placeholder for now
        T_coriolis = - np.cross(self.angular_velocity, self.get_inertia_matrix() @ self.angular_velocity)

        return T_coriolis

    def _euler_equations(self) -> float:
        
        # Torques presented here
        # 1. coriolis torque
        # 2. drag torque
        # 3. jet torque
        T_asymmetry = np.array([0.0, 0.0, 0.01])  # placeholder for now
        T_coriolis = self._get_coriolis_torque()
        T_drag = self._get_drag_torque()
        T_jet = self._get_jet_torque()

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
        # Simple model for contraction over time
        self._contract_rate = 0.06/3  # m/s
        time = self.contraciton / self._contract_rate
        # print("contraction rate:", self._contract_rate)
        # print("contraction time:", time)
        # print("contraction:", self.contraciton)
        return time

    def _release_model(self) -> float:
        # Simple model for release over time
        self._release_rate = 0.06/1.5  # m/s
        time = self.contraciton / self._release_rate
        return time 


    # def _test_compression_speed(self):
    #     # function is designed to model a constant force presses on a spring 
    #     # with a mass on it
    #     F = 1  # N
    #     k = 1  # N/m
    #     m = 70   # kg
    #     T = 5  # s
    #     n = 500  # steps
    #     x = np.zeros(n)  # m # displacement
    #     v = np.zeros(n)  # m/s # velocity
    #     a = np.zeros(n)  # m/s^2 # acceleration
    #     dt = T / n  # s # time step
    #     for i in range(n-1):
    #         a[i] = (F - k*x[i]) / m
    #         v[i+1] = v[i] + a[i]*dt
    #         x[i+1] = x[i] + v[i]*dt

    #     plt.plot(np.arange(0, T, dt), x*1000, label='Displacement')
    #     # plt.plot(np.arange(0, T, dt), v, color='orange', label='Velocity')
    #     # plt.plot(np.arange(0, T, dt), a, color='green', label='Acceleration')
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Displacement (mm)')
    #     plt.title('Compression Speed Test')
    #     plt.legend()
    #     plt.grid()
    #     plt.show()


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
    robot.nozzle.set_angles(angle1=0.0, angle2=np.pi)
    
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
    plot_jet_velocity(time_array, jet_velocity_data, state_data)  # approximate jet velocity
    plot_jet_properties(time_array, jet_force_data, state_data)
    # plot_drag_coefficient(time_array, drag_coefficient_data, state_data)
    # plot_drag_properties(time_array, drag_data, state_data)
    # plot_robot_position(time_array, positions_history, state_data)
    # print("Velocity data shape:", velocity_data)
    # plot_robot_velocity(time_array, velocity_data, state_data)  

    plot_angular_velocity(time_array, angular_velocity_data, state_data)
    plot_jet_torque(time_array, jet_torque_data, state_data)
    # plot_drag_torque(time_array, drag_torque_data, state_data)
    # plot_angular_acceleration(time_array, angular_acceleration_data, state_data)
    # plot_euler_angles(time_array, euler_angles_history, state_data)