"""
SALP Robot Simulation
Bio-inspired soft underwater robot with steerable rear nozzle.
Based on research from University of Pennsylvania Sung Robotics Lab.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from robot import Robot, Nozzle

class SalpRobotEnv(gym.Env):
    """
    SALP-inspired robot environment with steerable nozzle.
    
    Features:
    - Slow, realistic breathing cycles (2-3 seconds per phase)
    - Hold-to-inhale control scheme
    - Steerable rear nozzle (not body rotation)
    - Realistic underwater physics and momentum
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode: Optional[str] = None, width: int = 900, height: int = 700, robot: Optional[Robot] = None):
        super().__init__()
        
        # Environment parameters
        self.width = width
        self.height = height
        self.pos_init = np.array([width / 2, height / 2])  # Start in center
        self.tank_margin = 50
        
        # Pygame setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # # Robot state
        self.robot = robot
        self.action = np.array([0.0, 0.0, 0.0])  # Current action
        
        # Action space: [inhale_control (0/1), nozzle_direction (-1 to 1)]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        scale = 20.0  # pixels per meter
        # Observation space:
        # pos_x_limits = [(-self.width + self.tank_margin) / 2 / scale, (self.width - self.tank_margin) / 2 / scale]
        # pos_y_limits = [(-self.height + self.tank_margin) / 2 / scale, (self.height - self.tank_margin) / 2 / scale]
        pos_x_limits = [-np.inf, np.inf]
        pos_y_limits = [-np.inf, np.inf]
        vel_x_limits = [-np.inf, np.inf]
        vel_y_limits = [-np.inf, np.inf]
        yaw_limits = [-np.inf, np.inf]
        angular_vel_limits = [-np.inf, np.inf] 
        self.observation_space = spaces.Box(
            low=np.array([pos_x_limits[0], pos_y_limits[0], vel_x_limits[0], vel_y_limits[0], yaw_limits[0], angular_vel_limits[0]]),
            high=np.array([pos_x_limits[1], pos_y_limits[1], vel_x_limits[1], vel_y_limits[1], yaw_limits[1], angular_vel_limits[1]]),
            dtype=np.float32
        )
        # Movement history for the current action/breathing cycle (robot-frame meters)
        self.cycle_positions = []
        self.cycle_lengths = []
        self.cycle_widths = []
        self.cycle_euler_angles = []
        self.cycle_nozzle_yaws = []
        self._history_color = (255, 200, 0)
        # index of the history sample to draw (one ellipse at a time)
        self._history_draw_index = 0
        # whether to loop the history animation and how many samples to advance each frame
        self._history_loop = True
        self._history_step = 1
        # Animation control
        self._animation_start_time = None
        self._animation_complete = True
        self._animation_speed = 20  # milliseconds per frame

        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # initialize a target point
        self.target_point = self.generate_target_point(strategy="random")
        # print(f"New target point: ({self.target_point[0]:.2f}, {self.target_point[1]:.2f}) meters")
        
        # Reset robot to center
        self.robot.reset()
        self.pos_init = np.array([self.width / 2, self.height / 2])
        self.prev_dist = np.linalg.norm(self.robot.position[0:-1] - self.target_point)
        self.prev_action = np.array([0.0, 0.0, 0.0])
       
        # self.body_radius = self.base_radius  # Current body radius
        self.ellipse_a = self.robot.get_current_length()    # Semi-major axis for ellipse
        self.ellipse_b = self.robot.get_current_width()    # Semi-minor axis for ellipse

        # clear any previously recorded cycle history
        self.cycle_positions = []
        self.cycle_lengths = []
        self.cycle_widths = []
        self.cycle_euler_angles = []
        self.cycle_nozzle_yaws = []
        self._history_draw_index = 0
        self._history_loop = True
        self._history_step = 1

        return self._get_observation(), {}

    def _rescale_action(self, action: np.ndarray) -> np.ndarray:

        """Rescale action from [-1, 1] to robot input ranges."""
        rescaled = np.zeros_like(action)
        rescaled[0] = action[0] * 0.06  # inhale_control
        rescaled[1] = action[1] * 10.0   # coast_time
        rescaled[2] = action[2] * (np.pi / 2)  # nozzle yaw angle

        return rescaled
     
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:

        rescaled_action = self._rescale_action(action) 

        # print(f"Action taken: Inhale: {action[0]:.2f}, Coast Time: {action[1]:.2f}, Nozzle Yaw: {action[2]:.2f} rad")
        self.robot.nozzle.set_yaw_angle(yaw_angle = rescaled_action[2])  # Map -1 to 1 to -pi/2 to pi/2
        self.robot.nozzle.solve_angles()
        self.robot.set_control(rescaled_action[0], rescaled_action[1], np.array([self.robot.nozzle.angle1, self.robot.nozzle.angle2]))  # contraction, coast_time, nozzle angle
        self.robot.step_through_cycle()

        # store the most recent breathing-cycle histories (meters)
        if self.render_mode == "human":
            try:
                # convert to Python lists for easier use in render
                self.cycle_positions = [np.array(p) for p in self.robot.position_history]
                self.cycle_euler_angles = [np.array(ea) for ea in self.robot.euler_angle_history]
                self.cycle_lengths = [float(l) for l in self.robot.length_history]
                self.cycle_widths = [float(w) for w in self.robot.width_history]
                self.cycle_nozzle_yaws = [float(ny) for ny in self.robot.nozzle_yaw_history]
                # start drawing from the first recorded sample
                self._history_draw_index = 0
                # Reset animation for new cycle
                self._animation_start_time = None
                self._animation_complete = False
            except Exception:
                self.cycle_positions = []
                self.cycle_euler_angles = []
                self.cycle_lengths = []
                self.cycle_widths = []
                self.cycle_nozzle_yaws = []
                self._animation_complete = True

        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        done = False
        truncated = False

        distance_to_target = np.linalg.norm(self.robot.position[0:-1] - self.target_point)
        if distance_to_target < 0.01:
            done = True
            reward += 10.0  # big reward for reaching target
        elif distance_to_target > 5.0:
            truncated = True
            reward -= 5.0  # penalty for going out of bounds

        # reset after a certain number of steps
        if self.robot.cycle >= 500:
            truncated = True
        
        observation = self._get_observation()
        # print(f"Obs: {observation}")

        # info = self._get_info()
        info = {
            'position_history': self.robot.position_history,
            'length_history': self.robot.length_history,
            'width_history': self.robot.width_history
        }
        
        self.prev_action = self.action
        return observation, reward, done, truncated, info
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on realistic movement and efficiency."""
        
        current_diff = self.robot.position[0:-1] - self.target_point
        current_dist = np.linalg.norm(current_diff)
        dist_improvement = - current_dist + self.prev_dist   # Negative distance as improvement
        # print(f"Distance to target: {current_dist:.3f} m, Improvement: {dist_improvement:.3f} m")
        r_track = dist_improvement * 100
        self.prev_dist = current_dist
        # print(r_track)
        
        # 2. Heading (Dot Product)
        # Normalize vectors first!

        error_direction = - (current_diff / (np.linalg.norm(current_diff) + 1e-6))
        heading = self.robot.velocity_world[0:-1] / (np.linalg.norm(self.robot.velocity_world[0:-1]) + 1e-6)
        r_heading = np.dot(heading, error_direction)
        # print(r_heading)
        
        # 3. Energy (Thrust + Coasting) I don't care about this for now
        _, _, nozzle_yaw = self.action
        # # Penalize high thrust, Reward long coasting
        # r_energy = -0.1 * (thrust ** 2) - 0.01 / (coast_time + 1e-6)
        r_energy = 0.0
        # print(r_energy)
        
        # 4. Smoothness (Action Jerk)
        # Only penalize the nozzle angle change, not the thrust change
        angle_change = abs(nozzle_yaw - self.prev_action[2])
        r_smooth = -0.1 * (angle_change ** 2)
        # print(r_smooth)
        
        # Total
        # Note: Weights are critical. Tracking is usually the most important.
        total_reward = (1.0 * r_track) + (0.5 * r_heading) + r_energy + r_smooth
        # print(total_reward)

        # print(f"Reward components: Track={r_track:.3f}, Heading={r_heading:.3f}, Energy={r_energy:.3f}, Smoothness={r_smooth:.3f}, Total={total_reward:.3f}")
        
        return float(total_reward)
    
    def generate_target_point(self, strategy: str = "random", 
                             center: Optional[np.ndarray] = None,
                             max_distance: float = 2.0) -> np.ndarray:
        """
        Generate a target point for the robot to reach.
        
        Args:
            strategy: Target generation strategy:
                - "random": Uniform random point within tank bounds
                - "relative": Point relative to robot's current position
                - "circle": Point on a circle around a center point
                - "corridor": Point along a horizontal corridor
                
            center: Center point for relative/circle strategies. 
                   Defaults to robot's current position or tank center.
                   
            max_distance: Maximum distance from center (for relative/circle strategies).
                         Default is 2.0 meters.
        
        Returns:
            Target point as [x, y] in meters (robot frame coordinates)
        """
        scale = 200.0  # pixels to meters conversion
        
        # Get current robot position
        current_pos = self.robot.position[0:-1] if hasattr(self.robot, 'position') else np.array([0.0, 0.0])
        
        if strategy == "random":
            # Generate random point within tank bounds
            # Convert pixel bounds to meters
            x_min = (-self.width / 2 + self.tank_margin) / scale
            x_max = (self.width / 2 - self.tank_margin) / scale
            y_min = (-self.height / 2 + self.tank_margin) / scale
            y_max = (self.height / 2 - self.tank_margin) / scale
            
            target = np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max)
            ])
            
        elif strategy == "relative":
            # Generate point relative to current position
            if center is None:
                center = current_pos
            
            # Random distance and angle
            distance = np.random.uniform(0.1, max_distance)
            angle = np.random.uniform(0, 2 * np.pi)
            
            target = center + distance * np.array([np.cos(angle), np.sin(angle)])
            
        elif strategy == "circle":
            # Generate point on circle around center
            if center is None:
                center = current_pos
            
            angle = np.random.uniform(0, 2 * np.pi)
            target = center + max_distance * np.array([np.cos(angle), np.sin(angle)])
            
        elif strategy == "corridor":
            # Generate point along a horizontal corridor at robot's y-position
            if center is None:
                center = current_pos
            
            x_min = (-self.width / 2 + self.tank_margin) / scale
            x_max = (self.width / 2 - self.tank_margin) / scale
            
            target = np.array([
                np.random.uniform(x_min, x_max),
                center[1]  # Keep same y-coordinate
            ])
            
        else:
            raise ValueError(f"Unknown target generation strategy: {strategy}")
        
        # Clamp to tank bounds
        x_min = (-self.width / 2 + self.tank_margin) / scale
        x_max = (self.width / 2 - self.tank_margin) / scale
        y_min = (-self.height / 2 + self.tank_margin) / scale
        y_max = (self.height / 2 - self.tank_margin) / scale
        
        target[0] = np.clip(target[0], x_min, x_max)
        target[1] = np.clip(target[1], y_min, y_max)
        
        return target.astype(np.float32)
    
    def sample_random_action(self) -> np.ndarray:
        """
        Sample a random action from the action space.
        
        The action space contains three continuous values:
        - inhale_control: [0.0, 1.0] - Controls water intake
        - coast_time: [0.0, 1.0] - Duration of coasting phase
        - nozzle_direction: [-1.0, 1.0] - Steering angle for nozzle
        
        Returns:
            Random action as numpy array of shape (3,) with dtype float32
        """
        action = self.action_space.sample()

        return action.astype(np.float32)
    
    def _draw_target_point(self, scale: float = 200):
        """
        Draw the target point on the screen.
        
        Args:
            scale: Pixels per meter for coordinate conversion
        """
        if not hasattr(self, 'target_point') or self.target_point is None:
            return
        
        if self.screen is None:
            return
        
        # Convert target point from meters to screen pixels
        target_screen_x = int(self.pos_init[0] + self.target_point[0] * scale)
        target_screen_y = int(self.pos_init[1] + self.target_point[1] * scale)
        # print(f"Drawing target at screen pos: ({target_screen_x}, {target_screen_y})")
        
        # Draw target point as a circle with crosshair
        target_radius = 15
        target_color = (255, 0, 0)  # Bright red
        outline_color = (255, 100, 100)  # Light red outline
        crosshair_color = (200, 0, 0)  # Darker red for crosshair
        
        # Draw filled circle
        pygame.draw.circle(self.screen, target_color, (target_screen_x, target_screen_y), target_radius)
        
        # Draw outline
        pygame.draw.circle(self.screen, outline_color, (target_screen_x, target_screen_y), target_radius, 3)
        
        # Draw crosshair (plus sign)
        crosshair_size = target_radius + 5
        pygame.draw.line(self.screen, crosshair_color, 
                        (target_screen_x - crosshair_size, target_screen_y),
                        (target_screen_x + crosshair_size, target_screen_y), 2)
        pygame.draw.line(self.screen, crosshair_color,
                        (target_screen_x, target_screen_y - crosshair_size),
                        (target_screen_x, target_screen_y + crosshair_size), 2)
        
        # Draw label
        if not (hasattr(pygame, 'font') and pygame.font.get_init()):
            pygame.font.init()
        font = pygame.font.Font(None, 14)
        label = font.render("TARGET", True, outline_color)
        label_rect = label.get_rect(midbottom=(target_screen_x, target_screen_y - target_radius - 10))
        self.screen.blit(label, label_rect)
        
        # Draw distance to target as info
        robot_pos = self.robot.position[0:-1]
        distance_to_target = np.linalg.norm(self.target_point - robot_pos)
        dist_label = font.render(f"d:{distance_to_target:.2f}m", True, crosshair_color)
        dist_label_rect = dist_label.get_rect(midtop=(target_screen_x, target_screen_y + target_radius + 10))
        self.screen.blit(dist_label, dist_label_rect)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Map breathing phase to number
        
        # print( np.array([
        #     self.robot.position[0] - self.target_point[0],  # Normalized position
        #     self.robot.position[1] - self.target_point[1],
        #     self.robot.velocity[0],  # Normalized velocity
        #     self.robot.velocity[1],
        #     self.robot.euler_angle[2],  # Normalized body angle
        #     self.robot.angular_velocity[2],  # Normalized angular velocity
        # ], dtype=np.float32))

        return np.array([
            self.robot.position[0] - self.target_point[0],  # Normalized position
            self.robot.position[1] - self.target_point[1],
            self.robot.velocity[0],  # Normalized velocity
            self.robot.velocity[1],
            self.robot.euler_angle[2],  # Normalized body angle
            self.robot.angular_velocity[2],  # Normalized angular velocity
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional information."""
        # Try to extract yaw (first Euler angle) into a simple list for convenience
        try:
            yaw_hist = [float(ea[0]) for ea in self.cycle_euler_angles]
        except Exception:
            yaw_hist = []
        return {
            "position_history": self.cycle_positions,
            "length_history": self.cycle_lengths,
            "width_history": self.cycle_widths,
            "euler_angle_history": self.cycle_euler_angles,
            "yaw_history": yaw_hist
        }

    # -- Render helper methods -------------------------------------------------
    def _ensure_screen(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()

            if self.width <= 0 or self.height <= 0:
                self.width = 900
                self.height = 700

            if self.render_mode == "human":
                try:
                    self.screen = pygame.display.set_mode((int(self.width), int(self.height)))
                    pygame.display.set_caption("SALP Robot")
                except pygame.error as e:
                    print(f"Pygame display error: {e}")
                    self.width, self.height = 640, 480
                    self.screen = pygame.display.set_mode((self.width, self.height))
            else:
                # we are not using the image to learn for now 
                # self.screen = pygame.Surface((int(self.width), int(self.height)))\
                pass 

        if self.clock is None:
            self.clock = pygame.time.Clock()

    def _draw_background_and_tank(self):
        # Clear screen with deep water color
        self.screen.fill((10, 25, 50))

        # Draw tank boundaries
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (self.tank_margin, self.tank_margin,
                         self.width - 2*self.tank_margin, self.height - 2*self.tank_margin), 3)

    def _draw_history(self, scale: float):
        """Draw real-time animated simulation of the robot moving through the cycle."""
        if len(self.cycle_positions) == 0:
            self._animation_complete = True
            return

        n = len(self.cycle_positions)

        # Sample points to reduce rendering load
        sample_step = max(1, n // 50)
        sampled = list(range(0, n, sample_step))
        if sampled[-1] != n - 1:
            sampled.append(n - 1)

        pts = []
        for idx in sampled:
            try:
                p = self.cycle_positions[idx]
            except Exception:
                continue

            px = int(float(p[0]) * scale) + self.pos_init[0]
            py = int(float(p[1]) * scale) + self.pos_init[1]
            pts.append((px, py, idx))

        if not pts:
            self._animation_complete = True
            return

        # Initialize animation start time
        if self._animation_start_time is None:
            self._animation_start_time = pygame.time.get_ticks()

        # Calculate current frame based on elapsed time since animation start
        elapsed_time = pygame.time.get_ticks() - self._animation_start_time
        current_frame_idx = int(elapsed_time / self._animation_speed)

        # Check if animation is complete
        if current_frame_idx >= len(pts):
            self._animation_complete = True
            current_frame_idx = len(pts) - 1  # Show last frame

        # Draw only the current frame
        px, py, idx = pts[current_frame_idx]
        
        li = min(idx, len(self.cycle_lengths) - 1) if len(self.cycle_lengths) > 0 else 0
        wi = min(idx, len(self.cycle_widths) - 1) if len(self.cycle_widths) > 0 else 0
        ei = min(idx, len(self.cycle_euler_angles) - 1) if len(self.cycle_euler_angles) > 0 else 0
        ni = min(idx, len(self.cycle_nozzle_yaws) - 1) if len(self.cycle_nozzle_yaws) > 0 else 0
        
        try:
            body_len = float(self.cycle_lengths[li])
            body_wid = float(self.cycle_widths[wi])
            body_angle = float(self.cycle_euler_angles[ei][2])
            nozzle_yaw = float(self.cycle_nozzle_yaws[ni])
        except Exception:
            body_len = float(self.robot.init_length)
            body_wid = float(self.robot.init_width)
            body_angle = float(self.robot.euler_angle[2])
            nozzle_yaw = float(self.robot.nozzle.yaw)
            
        ew = max(4, int(scale * body_len)) if body_len <= 10.0 else max(4, int(body_len))
        eh = max(4, int(scale * body_wid)) if body_wid <= 10.0 else max(4, int(body_wid))

        # Draw the current position
        alpha = 180
        
        try:
            ell_surf = pygame.Surface((ew, eh), pygame.SRCALPHA)
            color = (*self._history_color, alpha)
            pygame.draw.ellipse(ell_surf, color, (0, 0, ew, eh))
            rotated_surf = pygame.transform.rotate(ell_surf, -math.degrees(body_angle))
            rect = rotated_surf.get_rect(center=(px, py))
            self.screen.blit(rotated_surf, rect)
            
            # Draw body frame at this historical position
            self._draw_robot_reference_frame_at_position(scale, px, py, body_angle)
            
            # Draw nozzle at this historical position
            self._draw_nozzle_at_position(scale, px, py, body_angle, body_len, nozzle_yaw)
        except Exception:
            pygame.draw.circle(self.screen, (*self._history_color, alpha), (px, py), 2)

    def is_animation_complete(self) -> bool:
        """Check if the current cycle animation has completed."""
        return self._animation_complete

    def wait_for_animation(self):
        """Block until the current cycle animation completes."""
        while not self._animation_complete:
            self.render()
            pygame.event.pump()  # Process pygame events to prevent freezing

    def _draw_body(self, scale: float, robot_x: int, robot_y: int):
        """Draw the current robot body at the end-of-cycle position with current dimensions."""
        # Body color - yellow
        body_color = (255, 200, 0)  # Yellow
        outline_color = (255, 255, 255)  # White outline

        # Get current robot dimensions at end of cycle
        try:
            body_length = float(self.robot.get_current_length())
            body_width = float(self.robot.get_current_width())
            body_angle = float(self.robot.euler_angle[2])
        except Exception:
            body_length = float(self.robot.init_length)
            body_width = float(self.robot.init_width)
            body_angle = 0.0

        # Convert to pixels
        ellipse_width = max(4, int(scale * body_length))
        ellipse_height = max(4, int(scale * body_width))

        # Create and draw the ellipse
        ellipse_surf = pygame.Surface((ellipse_width, ellipse_height), pygame.SRCALPHA)
        pygame.draw.ellipse(ellipse_surf, body_color, (0, 0, ellipse_width, ellipse_height))
        pygame.draw.ellipse(ellipse_surf, outline_color, (0, 0, ellipse_width, ellipse_height), 2)

        # Rotate according to robot's current yaw angle
        rotated_surf = pygame.transform.rotate(ellipse_surf, -math.degrees(body_angle))
        rect = rotated_surf.get_rect(center=(robot_x, robot_y))
        self.screen.blit(rotated_surf, rect)

    def _draw_rulers(self, scale: float):
        """Draw axis rulers and faint grid lines showing meters relative to the screen center."""
        left = int(self.tank_margin)
        right = int(self.width - self.tank_margin)
        top = int(self.tank_margin)
        bottom = int(self.height - self.tank_margin)

        # Choose a tick spacing that results in roughly 50-80 pixels between ticks
        target_px = 50
        step = target_px / scale # 0.25m per tick
        # nice_steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        # pick the nicest step closest to desired_m
        # step = min(nice_steps, key=lambda s: abs(s - desired_m))

        meters_left = (left - self.pos_init[0]) / scale
        meters_right = (right - self.pos_init[0]) / scale
        meters_top = (top - self.pos_init[1]) / scale
        meters_bottom = (bottom - self.pos_init[1]) / scale

        # prepare font
        if not (hasattr(pygame, 'font') and pygame.font.get_init()):
            pygame.font.init()
        font = pygame.font.Font(None, 16)

        tick_color = (220, 220, 220)
        grid_color = (30, 45, 70)

        # X axis ticks (top)
        first_x = math.ceil(meters_left / step) * step
        num_x = int(max(0, math.floor((meters_right - first_x) / step))) + 1
        for i in range(num_x):
            x_m = first_x + i * step
            px = int(self.pos_init[0] + x_m * scale)
            # tick on top edge
            pygame.draw.line(self.screen, tick_color, (px, top), (px, top + 8), 1)
            # vertical grid line
            pygame.draw.line(self.screen, grid_color, (px, top + 9), (px, bottom - 9), 1)
            # label
            label = f"{x_m:.1f}m"
            text = font.render(label, True, tick_color)
            text_rect = text.get_rect(center=(px, top - 10))
            self.screen.blit(text, text_rect)

        # Y axis ticks (left)
        first_y = math.ceil(meters_top / step) * step
        num_y = int(max(0, math.floor((meters_bottom - first_y) / step))) + 1
        for i in range(num_y):
            y_m = first_y + i * step
            py = int(self.pos_init[1] + y_m * scale)
            # tick on left edge
            pygame.draw.line(self.screen, tick_color, (left, py), (left + 8, py), 1)
            # horizontal grid line
            pygame.draw.line(self.screen, grid_color, (left + 9, py), (right - 9, py), 1)
            # label (positive downward)
            label = f"{y_m:.1f}m"
            text = font.render(label, True, tick_color)
            text_rect = text.get_rect(center=(left - 36, py))
            self.screen.blit(text, text_rect)

    def _draw_reference_frame(self, scale: float, axis_len_m: float = 0.25):
        """Draw a small x/y reference frame at the center of the tank (in meters).
        X points to the right, Y points downward (screen coordinates).
        """
        cx = int(self.pos_init[0])
        cy = int(self.pos_init[1])
        axis_px = max(8, int(axis_len_m * scale))

        # Colors for axes
        x_color = (220, 60, 60)
        y_color = (60, 200, 80)
        origin_color = (240, 240, 240)

        # Draw axes lines
        pygame.draw.line(self.screen, x_color, (cx, cy), (cx + axis_px, cy), 2)
        pygame.draw.line(self.screen, y_color, (cx, cy), (cx, cy + axis_px), 2)

        # Arrowheads (small triangles)
        ah = max(6, axis_px // 6)
        # X arrowhead (pointing right)
        pygame.draw.polygon(self.screen, x_color, [
            (cx + axis_px, cy),
            (cx + axis_px - ah, cy - ah // 2),
            (cx + axis_px - ah, cy + ah // 2)
        ])
        # Y arrowhead (pointing down)
        pygame.draw.polygon(self.screen, y_color, [
            (cx, cy + axis_px),
            (cx - ah // 2, cy + axis_px - ah),
            (cx + ah // 2, cy + axis_px - ah)
        ])

        # Origin marker
        pygame.draw.circle(self.screen, origin_color, (cx, cy), 3)

        # Labels
        if not (hasattr(pygame, 'font') and pygame.font.get_init()):
            pygame.font.init()
        font = pygame.font.Font(None, 18)
        tx = font.render('x', True, x_color)
        ty = font.render('y', True, y_color)
        self.screen.blit(tx, tx.get_rect(center=(cx + axis_px + 12, cy)))
        self.screen.blit(ty, ty.get_rect(center=(cx, cy + axis_px + 12)))

    def _draw_robot_reference_frame(self, scale: float, robot_x: int, robot_y: int, axis_len_m: float = 0.25):
        """Draw a small x/y frame attached to the robot and rotated by its yaw (in meters)."""
        axis_px = max(8, int(axis_len_m * scale))

        try:
            yaw = float(self.robot.euler_angles[0])
        except Exception:
            yaw = 0.0

        # basis vectors for robot frame in screen coordinates (x forward, y to robot's left)
        ux = math.cos(yaw)
        uy = math.sin(yaw)
        vx = math.cos(yaw + math.pi/2)
        vy = math.sin(yaw + math.pi/2)

        x_end = robot_x + ux * axis_px
        y_end = robot_y + uy * axis_px
        x2_end = robot_x + vx * axis_px
        y2_end = robot_y + vy * axis_px

        x_color = (60, 160, 220)
        y_color = (220, 160, 60)
        origin_color = (240, 240, 240)

        # draw axes
        pygame.draw.line(self.screen, x_color, (int(robot_x), int(robot_y)), (int(x_end), int(y_end)), 2)
        pygame.draw.line(self.screen, y_color, (int(robot_x), int(robot_y)), (int(x2_end), int(y2_end)), 2)

        # arrowheads
        ah = max(6, axis_px // 4)
        perp_x = -uy
        perp_y = ux
        tip_x = x_end
        tip_y = y_end
        base_x = tip_x - ux * ah
        base_y = tip_y - uy * ah
        left = (base_x + perp_x * (ah/2), base_y + perp_y * (ah/2))
        right = (base_x - perp_x * (ah/2), base_y - perp_y * (ah/2))
        pygame.draw.polygon(self.screen, x_color, [(int(tip_x), int(tip_y)), (int(left[0]), int(left[1])), (int(right[0]), int(right[1]))])

        perp2_x = -vy
        perp2_y = vx
        tip2_x = x2_end
        tip2_y = y2_end
        base2_x = tip2_x - vx * ah
        base2_y = tip2_y - vy * ah
        left2 = (base2_x + perp2_x * (ah/2), base2_y + perp2_y * (ah/2))
        right2 = (base2_x - perp2_x * (ah/2), base2_y - perp2_y * (ah/2))
        pygame.draw.polygon(self.screen, y_color, [(int(tip2_x), int(tip2_y)), (int(left2[0]), int(left2[1])), (int(right2[0]), int(right2[1]))])

        # origin marker and angle label
        pygame.draw.circle(self.screen, origin_color, (int(robot_x), int(robot_y)), 3)
        if not (hasattr(pygame, 'font') and pygame.font.get_init()):
            pygame.font.init()
        font = pygame.font.Font(None, 16)
        # show yaw degrees
        yaw_label = font.render(f"{math.degrees(yaw):.0f}°", True, origin_color)
        self.screen.blit(yaw_label, yaw_label.get_rect(center=(int(robot_x), int(robot_y - axis_px - 12))))

    def _draw_robot_reference_frame_at_position(self, scale: float, x: int, y: int, angle: float, axis_len_m: float = 0.25):
        """Draw a small x/y frame at a specific position with a specific angle.
        
        Args:
            scale: pixels per meter
            x: x position in pixels
            y: y position in pixels
            angle: yaw angle in radians
            axis_len_m: length of axis in meters
        """
        axis_px = max(8, int(axis_len_m * scale))

        # basis vectors for robot frame in screen coordinates (x forward, y to robot's left)
        ux = math.cos(angle)
        uy = math.sin(angle)
        vx = math.cos(angle + math.pi/2)
        vy = math.sin(angle + math.pi/2)

        x_end = x + ux * axis_px
        y_end = y + uy * axis_px
        x2_end = x + vx * axis_px
        y2_end = y + vy * axis_px

        # Use semi-transparent colors for historical frames
        x_color = (60, 160, 220, 150)
        y_color = (220, 160, 60, 150)
        origin_color = (240, 240, 240, 150)

        # draw axes
        pygame.draw.line(self.screen, x_color[:3], (int(x), int(y)), (int(x_end), int(y_end)), 2)
        pygame.draw.line(self.screen, y_color[:3], (int(x), int(y)), (int(x2_end), int(y2_end)), 2)

        # arrowheads for x-axis
        ah = max(6, axis_px // 4)
        perp_x = -uy
        perp_y = ux
        tip_x = x_end
        tip_y = y_end
        base_x = tip_x - ux * ah
        base_y = tip_y - uy * ah
        left = (base_x + perp_x * (ah/2), base_y + perp_y * (ah/2))
        right = (base_x - perp_x * (ah/2), base_y - perp_y * (ah/2))
        pygame.draw.polygon(self.screen, x_color[:3], [(int(tip_x), int(tip_y)), (int(left[0]), int(left[1])), (int(right[0]), int(right[1]))])

        # arrowheads for y-axis
        perp2_x = -vy
        perp2_y = vx
        tip2_x = x2_end
        tip2_y = y2_end
        base2_x = tip2_x - vx * ah
        base2_y = tip2_y - vy * ah
        left2 = (base2_x + perp2_x * (ah/2), base2_y + perp2_y * (ah/2))
        right2 = (base2_x - perp2_x * (ah/2), base2_y - perp2_y * (ah/2))
        pygame.draw.polygon(self.screen, y_color[:3], [(int(tip2_x), int(tip2_y)), (int(left2[0]), int(left2[1])), (int(right2[0]), int(right2[1]))])

        # origin marker
        pygame.draw.circle(self.screen, origin_color[:3], (int(x), int(y)), 3)

    def _draw_nozzle_at_position(self, scale: float, x: int, y: int, yaw: float, body_len: float, nozzle_angle: float = 0.0):
        """Draw the nozzle at a specific position with specific angle.
        
        Args:
            scale: pixels per meter
            x: x position in pixels
            y: y position in pixels
            yaw: robot yaw angle in radians
            body_len: robot body length in meters
            nozzle_angle: nozzle steering angle in radians (relative to robot)
        """
        # Rear of robot in meters (half body length behind center)
        rear_offset_m = body_len / 2
        rear_angle = yaw + math.pi  # opposite direction
        rear_x = x + math.cos(rear_angle) * rear_offset_m * scale
        rear_y = y + math.sin(rear_angle) * rear_offset_m * scale

        # 1. Straight connector from rear of robot
        connector_len_m = 0.05  # 5cm straight connector
        connector_len_px = connector_len_m * scale
        joint_x = rear_x + math.cos(rear_angle) * connector_len_px
        joint_y = rear_y + math.sin(rear_angle) * connector_len_px
        pygame.draw.line(self.screen, (150, 150, 150), 
                        (int(rear_x), int(rear_y)), (int(joint_x), int(joint_y)), 2)

        # 2. Revolute joint (small circle) - semi-transparent
        joint_radius = max(3, int(0.015 * scale))  # 1.5cm radius joint
        pygame.draw.circle(self.screen, (180, 180, 80), (int(joint_x), int(joint_y)), joint_radius)
        pygame.draw.circle(self.screen, (100, 100, 50), (int(joint_x), int(joint_y)), joint_radius, 1)

        # 3. Nozzle part (rotates around joint by nozzle_angle)
        nozzle_len_m = 0.08  # 8cm nozzle
        nozzle_len_px = nozzle_len_m * scale
        # Nozzle angle is relative to the robot body (rear_angle)
        nozzle_world_angle = rear_angle + nozzle_angle
        nozzle_end_x = joint_x + math.cos(nozzle_world_angle) * nozzle_len_px
        nozzle_end_y = joint_y + math.sin(nozzle_world_angle) * nozzle_len_px
        
        # Draw nozzle as a tapered line (semi-transparent for history)
        pygame.draw.line(self.screen, (180, 180, 80),
                        (int(joint_x), int(joint_y)), (int(nozzle_end_x), int(nozzle_end_y)), 3)
        # Draw tip
        pygame.draw.circle(self.screen, (160, 160, 70), (int(nozzle_end_x), int(nozzle_end_y)), 2)

    def _draw_nozzle(self, scale: float, robot_x: int, robot_y: int):
        """Draw the nozzle at the rear of the robot: straight connector + revolute joint + steerable nozzle."""
        try:
            yaw = float(self.robot.euler_angles[0])
            nozzle_angle = float(self.robot.nozzle_angle)
        except Exception:
            yaw = 0.0
            nozzle_angle = 0.0

        # Get robot body dimensions
        try:
            body_len = float(self.robot.get_current_length())
        except Exception:
            body_len = float(self.robot.init_length)

        # Rear of robot in meters (half body length behind center)
        rear_offset_m = body_len / 2
        rear_angle = yaw + math.pi  # opposite direction
        rear_x = robot_x + math.cos(rear_angle) * rear_offset_m * scale
        rear_y = robot_y + math.sin(rear_angle) * rear_offset_m * scale

        # 1. Straight connector from rear of robot
        connector_len_m = 0.05  # 5cm straight connector
        connector_len_px = connector_len_m * scale
        joint_x = rear_x + math.cos(rear_angle) * connector_len_px
        joint_y = rear_y + math.sin(rear_angle) * connector_len_px
        pygame.draw.line(self.screen, (180, 180, 180), 
                        (int(rear_x), int(rear_y)), (int(joint_x), int(joint_y)), 3)

        # 2. Revolute joint (small circle)
        joint_radius = max(4, int(0.015 * scale))  # 1.5cm radius joint
        pygame.draw.circle(self.screen, (200, 200, 100), (int(joint_x), int(joint_y)), joint_radius)
        pygame.draw.circle(self.screen, (120, 120, 60), (int(joint_x), int(joint_y)), joint_radius, 2)

        # 3. Nozzle part (rotates around joint by nozzle_angle)
        nozzle_len_m = 0.08  # 8cm nozzle
        nozzle_len_px = nozzle_len_m * scale
        # Nozzle angle is relative to the robot body (rear_angle)
        nozzle_world_angle = rear_angle + nozzle_angle
        nozzle_end_x = joint_x + math.cos(nozzle_world_angle) * nozzle_len_px
        nozzle_end_y = joint_y + math.sin(nozzle_world_angle) * nozzle_len_px
        
        # Draw nozzle as a tapered line (thicker at joint, thinner at tip)
        pygame.draw.line(self.screen, (200, 200, 100),
                        (int(joint_x), int(joint_y)), (int(nozzle_end_x), int(nozzle_end_y)), 5)
        # Draw tip
        pygame.draw.circle(self.screen, (180, 180, 80), (int(nozzle_end_x), int(nozzle_end_y)), 3)

    def _draw_cycle_info(self):
        """Draw cycle count and robot state information overlay."""
        if not (hasattr(pygame, 'font') and pygame.font.get_init()):
            pygame.font.init()
        
        font = pygame.font.Font(None, 28)
        small_font = pygame.font.Font(None, 20)
        
        # Cycle count
        cycle_text = font.render(f"Cycle: {self.robot.cycle}", True, (255, 255, 255))
        self.screen.blit(cycle_text, (10, 10))
        
        # Current state
        state_text = small_font.render(f"State: {self.robot.update_state()}", True, (200, 200, 200))
        self.screen.blit(state_text, (10, 40))
        
        # Position
        pos_text = small_font.render(f"Position: ({self.robot.position[0]:.3f}, {self.robot.position[1]:.3f}) m", True, (200, 200, 200))
        self.screen.blit(pos_text, (10, 65))
        
        # Angle
        angle_deg = math.degrees(self.robot.euler_angle[2])
        angle_text = small_font.render(f"Yaw: {angle_deg:.1f}°", True, (200, 200, 200))
        self.screen.blit(angle_text, (10, 90))

    def get_cycle_count(self) -> int:
        """Get the current cycle count from the robot."""
        return self.robot.cycle

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return

        # ensure pygame screen and clock are initialized
        self._ensure_screen()

        # background and tank
        self._draw_background_and_tank()

        # scaling between meters and pixels (pixels per meter)
        scale = 200

        # robot screen center in pixels (convert robot meter positions to screen coordinates)
        # print(f"Robot world pos: ({self.robot.position[0]}, {self.robot.position[1]})")
        robot_x = int(self.pos_init[0] + self.robot.position[0] * scale)
        robot_y = int(self.pos_init[1] + self.robot.position[1] * scale)
        # print(f"Robot screen pos: ({robot_x}, {robot_y})")

        # draw rulers and grid to visualize meters in both x and y
        self._draw_rulers(scale)

        # draw a small reference frame at the tank center (x/y axes)
        self._draw_reference_frame(scale)

        self._draw_target_point(scale)
        # draw historical path and sized ellipses
        # draw real-time animated history of the current cycle
        self._draw_history(scale)

        # draw target point

        # draw current robot body at end-of-cycle position
        # self._draw_body(scale, robot_x, robot_y)

        # draw robot-attached reference frame (rotated with robot yaw)
        # self._draw_robot_reference_frame(scale, robot_x, robot_y)

        # draw nozzle (straight connector + revolute joint + steerable nozzle)
        # self._draw_nozzle(scale, robot_x, robot_y)

        # draw cycle info overlay
        self._draw_cycle_info()

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(pygame.surfarray.array3d(self.screen), axes=(1, 0, 2))
    
    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    
    # TODO: need to fix the scale issues with the robot size and movement speed
    nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.00016, mass=1.0)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                  max_contraction=0.06, nozzle=nozzle)
    robot.nozzle.set_angles(angle1=0.0, angle2=np.pi)
    robot.set_environment(density=1000)  # water density in kg/m^3
    env = SalpRobotEnv(render_mode="human", robot=robot)
    obs, info = env.reset()
    
    done = False
    cnt = 0
    test_actions = np.array([
        [0.22891083, 0.06766406, -0.9850989],
        [0.2842933, 0.963629, 0.9741967],
        [0.8862339, 0.32421368, -0.9328714],
        [0.05561769, 0.91966885, 0.85212207],
        [0.25341812, 0.6691348, 0.7325938],
        [0.8321035, 0.23156995, -0.92043316],
        [0.05115855, 0.96011114, 0.8534517],
        [0.24099252, 0.71873295, 0.7108506],
        [0.6869794, 0.04822099, -0.86440706],
        [0.2337848, 0.9906088, 0.99013484],
        [0.6945975, 0.09169137, -0.8268971],
        [0.06978488, 0.9933312, 0.9369149],
        [0.06852683, 0.7448164, 0.8688922],
        [0.4922041, 0.10394484, 0.8375015],
        [0.68481743, 0.00496772, -0.99800324],
        [0.9857271, 0.647208, 0.99998224],
        [0.7742654, 0.83240354, -0.66497386],
        [0.78678393, 0.01270097, 0.9582412],
        [3.6354065e-03, 1.2317300e-04, -9.9999970e-01],
        [6.2082112e-03, 3.0893087e-04, -9.9999964e-01],
        [8.4684789e-03, 2.1675229e-04, -9.9999875e-01],
        [1.7061472e-02, 3.0627847e-04, -9.9999666e-01],
        [6.9575727e-02, 5.1766634e-04, -9.9997485e-01],
        [0.6683986, 0.02849919, -0.984029],
        [0.99468315, 0.3537972, 0.9998045],
        [0.6911941, 0.04722786, -0.96879447],
        [0.9897277, 0.33548862, 0.9979936],
        [0.89258796, 0.00411212, -0.96228147],
    ])

    while not done:
        # action = [0.06, 0.0, 0.0]  # inhale with no nozzle steering
        # For every step in the environment, there are multiple internal robot steps
        # action = env.sample_random_action()
        action = test_actions[cnt % len(test_actions)]
        obs, reward, done, truncated, info = env.step(action)
        # print("Step:", cnt, "Action:", action, "Obs:", obs, "Reward:", reward, "Done:", done)
        # print(reward)
        cnt += 1
        # Wait for the animation to complete before next step
        env.wait_for_animation()
        # env.render()
    env.close()
    