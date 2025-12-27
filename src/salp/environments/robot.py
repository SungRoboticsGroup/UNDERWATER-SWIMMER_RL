import numpy as np

class Nozzle():
    def __init__(self, length1: float = 0.0, length2: float = 0.0):
        self.length1 = length1  
        self.length2 = length2
        self.angle1 = 0.0  # angle around y axis
        self.angle2 = 0.0  # angle around z axis 
    
    def set_angles(self, angle1: float, angle2: float):
        self.angle1 = angle1
        self.angle2 = angle2
    
    def get_position(self) -> np.ndarray:
        # compute the nozzle direction vector based on angles
        # placeholder for now
        direction = np.array([self.length1 + self.length2, 0.0, 0.0])  # pointing along x axis

        R1 = np.array([[np.cos(self.angle1), 0, np.sin(self.angle1)],
                       [0, 1, 0],
                       [-np.sin(self.angle1), 0, np.cos(self.angle1)]])
        
        R2 = np.array([[np.cos(self.angle2), -np.sin(self.angle2), 0],   
                       [np.sin(self.angle2), np.cos(self.angle2), 0],
                       [0, 0, 1]])
        

        return direction

class Robot():
    
    # TODO:
    # 1. implement nozzle steering behavior
    # 2. scale the actions from RL inputs to robot control inputs
    # 3. normalize the observation space numbers
    #  
    phase = ["refill", "jet", "coast", "rest"]

    def __init__(self, dry_mass: float, init_length: float, init_width: float, max_contraction: float, nozzle_area: float):
        self.dry_mass = dry_mass # kg
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
        self.velocities = np.zeros(3)  # x, y velocities
        self.angular_velocity = np.zeros(3)  # yaw rate
        self.previous_water_volume = 0.0
        self.nozzle_area = nozzle_area  # m^2, cross-sectional area of the nozzle
        self.density = 0  # kg/m^3, density of water
        self.contract_time = 0.0
        self.release_time = 0.0
        self.coast_time = 0.0

        self._contract_rate = 0.0
        self._release_rateS = 0.0

        self._drag_coefficents = [0.2, 0.5] # min and max drag coefficients for different shapes

        self.length = 0.0
        self.width = 0.0
    
    def set_environment(self, density: float):

        self.density = density

    def reset(self):

        self.time = 0.0
        self.cycle_time = 0.0
        self.positions = np.zeros(3)  # x, y, z positions
        # print(self.positions)
        self.euler_angles = np.zeros(3)  # yaw
        self.velocities = np.zeros(3)  # x, y, z velocities
        self.angular_velocity = np.zeros(3)  # yaw rate
        self.previous_water_volume = 0.0
        self.cycle = 0

    def set_control(self, contraction: float, coast_time: float, angle: float):

        self.contraciton = contraction
        self.coast_time = coast_time
        # print("contraction set to:", self.contraciton)
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

        self.get_state()
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
        positions_history = []
        euler_angles_history = []
        length_history = []
        width_history = []
        while self.cycle_time < total_cycle_time:
            self.step()
            positions_history.append(self.positions)
            euler_angles_history.append(self.euler_angles)
            length_history.append(self.get_current_length())
            width_history.append(self.get_current_width())
        
        return np.array(positions_history), np.array(euler_angles_history), np.array(length_history), np.array(width_history)

    def contract(self):
        # computes
        # a = F/m
        # v = v + a*dt
        # x = x + v*dt
        # print("contracting")
        jet_force = self._get_jet_force()
        added_mass = self._get_added_mass()
        drag_force = self._get_drag_force()
        a = (jet_force - drag_force - added_mass) / self.get_mass()  # acceleration
        # self.velocities[0] += a * self.dt  # update velocities
        self.velocities[0] = 0 
        self.velocities[1] = 0
        self.positions[0] += self.velocities[0] * self.dt  # update positions
        self.positions[1] += self.velocities[1] * self.dt  # update positions

        self.angular_velocity[0] = 0.001 
        self.euler_angles[0] += self.angular_velocity[0] * self.dt  # update yaw angle
    
        self.cycle_time += self.dt
        self.time += self.dt

    def release(self):
        # print("releasing")
        jet_force = self._get_jet_force()
        added_mass = self._get_added_mass()
        drag_force = self._get_drag_force()
        a = (jet_force - drag_force - added_mass) / self.get_mass()  # acceleration
        # self.velocities[0] += a * self.dt  # update velocities
        self.velocities[0] = 0.001 # m/s
        self.velocities[1] = 0.0
        self.positions[0] += self.velocities[0] * self.dt  # update positions
        self.positions[1] += self.velocities[1] * self.dt  # update positions
        # self.positions[0] = 1
        # print(self.velocities[0]*self.dt)
        # a = self.velocities[0]*self.dt
        # # print(a)
        # print(self.positions[0])
        self.cycle_time += self.dt
        self.time += self.dt 

    def coast(self):
        # print("coasting")
        jet_force = self._get_jet_force()
        added_mass = self._get_added_mass()
        drag_force = self._get_drag_force()
        a = (- drag_force) / self.get_mass()  # acceleration
        # self.velocities[0] += a * self.dt  # update velocities
        self.velocities[0] = 0.0  # m/s
        self.velocities[1] = 0.0    
        self.positions[0] += self.velocities[0] * self.dt  # update positions
        self.positions[1] += self.velocities[1] * self.dt  # update positions

        self.cycle_time += self.dt
        self.time += self.dt 

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

        total_mass = self.dry_mass + self._get_water_mass()

        return total_mass

    def _get_jet_force(self) -> float:

        water_mass = self._get_water_mass()
        mass_rate = (water_mass - self.previous_water_volume) / self.dt
        jet_velocity = self._get_jet_velocity()
        jet_force = mass_rate * jet_velocity
        # jet_force = 0.0  # placeholder for now 

        return jet_force
    
    def _nozzle_angles_to_rotation_matrix(self) -> np.ndarray:
        # placeholder for now
        R = np.eye(3)

        return R

    def _get_jet_velocity(self) -> float:
        
        # velocity is with respect to the robot frame
        water_volume = self._get_water_volume()
        volume_rate = (water_volume - self.previous_water_volume) / self.dt
        jet_speed = volume_rate / self.nozzle_area
        
        R = self._nozzle_angles_to_rotation_matrix(self.nozzle_angles)
        # body frame 
        # alpha = 0 # around y axis
        # beta = 0 # around z axis
        # R_y = np.array([[np.cos(alpha), 0, np.sin(alpha)],
        #                 [0, 1, 0],
        #                 [-np.sin(alpha), 0, np.cos(alpha)]])
        
        # R_z = np.array([[np.cos(beta), -np.sin(beta), 0],   
        #                 [np.sin(beta), np.cos(beta), 0],
        #                 [0, 0, 1]])
        jet_velocity = np.array([-jet_speed, 0, 0])  # jet velocity in body frame
        jet_velocity = R @ jet_velocity  # rotate to world frame

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
        
        return C_d 

    def _get_drag_force(self) -> float:

        C_d = self._get_drag_coefficient()
        A = self._get_cross_sectional_area()
        F_drag = 0.5 * self.density * A * C_d * abs(self.velocities) * self.velocities 
        # drag_force = 0.0  # placeholder for now

        return F_drag
    
    def _get_added_mass(self) -> float:

        # added_mass = 0.5 * self.density * self._get_water_volume()  # added mass for sphere is 0.5 * density * volume
        added_mass = 0.0  # placeholder for now

        return added_mass

    def _newton_equations(self):
        
        # Forces presented here
        # 1. coriolis force 
        # 2. drag force
        # 3. jet force
        
        F_coriolis = self._get_coriolis_force()
        F_drag = self._get_drag_force()
        F_jet = self._get_jet_force()

        mass = np.diag(self.get_mass() * np.ones(3))

        a = np.linalg.inv(mass) @ (F_jet + F_drag + F_coriolis)

        return a
    
    def _euler_equations(self) -> float:
        
        # Torques presented here
        # 1. coriolis torque
        # 2. drag torque
        # 3. jet torque

        T_coriolis = self._get_coriolis_torque()
        T_drag = self._get_drag_torque()
        T_jet = self._get_jet_torque()

        I = self._get_inertia_matrix()

        alpha = np.linalg.inv(I) @ (T_jet + T_drag + T_coriolis)

        return alpha


    def _get_water_volume(self) -> float:
        
        volume = 4/3 * np.pi * (self.length/2) * (self.width/2) ** 2

        return volume

    def _get_water_mass(self) -> float:

        mass = self.density * self._get_water_volume()

        return mass

    def _get_cross_sectional_area(self) -> float:
        
        length = self.get_current_length()
        width = self.get_current_width()
        area = np.pi * (length/2) * (width/2)

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
