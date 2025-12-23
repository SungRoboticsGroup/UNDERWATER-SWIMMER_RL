import numpy as np

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
        self.nozzle_angle = 0.0  # current angle of nozzle
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
        self.velocities[0] = 0.0 # m/s
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

        # water_mass = self._get_water_volume()
        # mass_rate = (water_mass - self.previous_water_volume) / self.dt
        # jet_velocity = self._get_jet_velocity()
        # jet_force = mass_rate * jet_velocity
        jet_force = 0.0  # placeholder for now 

        return jet_force
    
    def _get_jet_velocity(self) -> float:
        
        # velocity is with respect to the robot frame
        water_volume = self._get_water_volume()
        volume_rate = (water_volume - self.previous_water_volume) / self.dt
        jet_velocity = volume_rate / self.nozzle_area

        return jet_velocity
    
    def _get_drag_force(self) -> float:

        # drag_force = 0.5 * self.density * (self.velocities**2) * self._get_cross_sectional_area() * self.drag_coefficient  # drag coefficient for sphere is 0.47
        drag_force = 0.0  # placeholder for now

        return drag_force
    
    def _get_added_mass(self) -> float:

        # added_mass = 0.5 * self.density * self._get_water_volume()  # added mass for sphere is 0.5 * density * volume
        added_mass = 0.0  # placeholder for now

        return added_mass

    def _get_water_volume(self) -> float:
        
        length = self.get_current_length()
        width = self.get_current_width()
        volume = 4/3*np.pi*(length/2)*(width/2)**2

        return volume

    def _get_water_mass(self) -> float:

        density = 997  # kg/m^3
        mass = density * self._get_water_volume()

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
