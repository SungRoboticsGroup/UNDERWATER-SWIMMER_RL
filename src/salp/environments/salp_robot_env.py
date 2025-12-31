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
        
        # Action space: [inhale_control (0/1), nozzle_direction (-1 to 1)]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: [pos_x, pos_y, vel_x, vel_y, body_angle, angular_vel, body_size, breathing_phase, water_volume, nozzle_angle]
        # TODO: pull some of these limits out from the robot 
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -10, -10, -math.pi, -0.1, 0.5, 0, 0, -1]),
            high=np.array([width, height, 10, 10, math.pi, 0.1, 2.0, 2, 1, 1]),
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
        
        # Reset robot to center
        self.robot.reset()
        self.pos_init = np.array([self.width / 2, self.height / 2])
       
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
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.robot.nozzle.set_yaw_angle(yaw_angle=-np.pi / 2)  # Map -1 to 1 to -pi/2 to pi/2
        self.robot.nozzle.solve_angles()
        self.robot.set_control(action[0], action[1], np.array([self.robot.nozzle.angle1, self.robot.nozzle.angle2]))  # contraction, coast_time, nozzle angle
        # self.robot.set_control(action[0], action[1], np.array([0.0, 0.0]))  # contraction, coast_time, nozzle angle
        self.robot.step_through_cycle()
        # print(position_history)

        # store the most recent breathing-cycle histories (meters)
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
        
        observation = self._get_observation()
        # info = self._get_info()
        info = {
            'position_history': self.robot.position_history,
            'length_history': self.robot.length_history,
            'width_history': self.robot.width_history
        }
        
        return observation, reward, done, truncated, info
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on realistic movement and efficiency."""
        # Reward for smooth movement
        # speed = math.sqrt(self.robot_velocity[0]**2 + self.robot_velocity[1]**2)
        # movement_reward = min(speed * 0.08, 0.6)
        
        # # Reward for efficient breathing (not too frequent)
        # breathing_efficiency = 0.15 if self.breathing_phase == "rest" else 0.08
        
        # # Small penalty for excessive nozzle movement (energy cost)
        # nozzle_penalty = abs(self.nozzle_angle) * 0.02
        
        # # Small reward for staying in bounds
        # bounds_reward = 0.05
        
        return 0
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Map breathing phase to number
        
        # return np.array([
        #     self.robot_pos[0] / self.width,  # Normalized position
        #     self.robot_pos[1] / self.height,
        #     self.robot_velocity[0] / 5.0,  # Normalized velocity
        #     self.robot_velocity[1] / 5.0,
        #     self.robot_angle / math.pi,  # Normalized body angle
        #     self.robot_angular_velocity / 0.1,  # Normalized angular velocity
        #     max(self.ellipse_a, self.ellipse_b),  # Normalized body size
        #     phase_num / 2.0,  # Normalized breathing phase
        #     self.water_volume,  # Water volume (0-1)
        #     self.nozzle_angle  # Normalized nozzle angle
        # ], dtype=np.float32)
        return 0
    
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
        robot_x = int(self.pos_init[0] + self.robot.position[0] * scale)
        robot_y = int(self.pos_init[1] + self.robot.position[1] * scale)
        # print(f"Robot screen pos: ({robot_x}, {robot_y})")

        # draw rulers and grid to visualize meters in both x and y
        self._draw_rulers(scale)

        # draw a small reference frame at the tank center (x/y axes)
        self._draw_reference_frame(scale)

        # draw historical path and sized ellipses
        # draw real-time animated history of the current cycle
        self._draw_history(scale)

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
    nozzle = Nozzle(length1=0.01, length2=0.01, area=0.00009)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                  max_contraction=0.06, nozzle=nozzle)
    robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
    env = SalpRobotEnv(render_mode="human", robot=robot)
    obs, info = env.reset()
    
    done = False
    cnt = 0
    while not done:
        action = [0.06, 0.0, 0.0]  # inhale with no nozzle steering
        # For every step in the environment, there are multiple internal robot steps
        obs, reward, done, truncated, info = env.step(action)
        cnt += 1
        # Wait for the animation to complete before next step
        env.wait_for_animation()
        # env.render()
    env.close()
    