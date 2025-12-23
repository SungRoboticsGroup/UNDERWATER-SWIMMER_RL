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
from robot import Robot

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
        self._history_color = (255, 200, 0)
        # index of the history sample to draw (one ellipse at a time)
        self._history_draw_index = 0
        # whether to loop the history animation and how many samples to advance each frame
        self._history_loop = True
        self._history_step = 1

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
        self._history_draw_index = 0
        self._history_loop = True
        self._history_step = 1

        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        
        self.robot.set_control(action[0], action[1], action[2])  # contraction, coast_time, nozzle angle
        position_history, euler_angles_history, length_history, width_history = self.robot.step_through_cycle()
        # print(position_history)

        # store the most recent breathing-cycle histories (meters)
        try:
            # convert to Python lists for easier use in render
            self.cycle_positions = [np.array(p) for p in position_history]
            self.cycle_euler_angles = [np.array(ea) for ea in euler_angles_history]
            self.cycle_lengths = [float(l) for l in length_history]
            self.cycle_widths = [float(w) for w in width_history]
            # start drawing from the first recorded sample
            self._history_draw_index = 0
        except Exception:
            self.cycle_positions = []
            self.cycle_euler_angles = []
            self.cycle_lengths = []
            self.cycle_widths = []

        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        done = False
        truncated = False
        
        observation = self._get_observation()
        # info = self._get_info()
        info = {
            'position_history': position_history,
            'length_history': length_history,
            'width_history': width_history
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

    def _draw_history(self, scale: float, robot_x: int, robot_y: int):

        if len(self.cycle_positions) == 0:
            return

        n = len(self.cycle_positions)

        # --- 1. Data Preparation (Keep this the same) ---
        # max_coord = 0.0
        # for p in self.cycle_positions:
        #     try:
        #         max_coord = max(max_coord, abs(float(p[0])), abs(float(p[1])))
        #     except Exception:
        #         continue
        # positions_in_pixels = max_coord > 50.0

        # Sample points (Keep this to reduce jitter)
        sample_step = max(1, n // 80)
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
            # print(px, py)
            pts.append((px, py, idx))

        if not pts:
            return

        # --- 2. THE NEW ANIMATION LOGIC ---
        
        # Calculate which frame to show based on system time.
        # pygame.time.get_ticks() gives milliseconds. 
        # Dividing by 50 means "change frame every 50ms" (approx 20 FPS).
        # The % len(pts) makes it loop back to the start automatically.
        animation_speed = 50 
        current_frame_idx = int(pygame.time.get_ticks() / animation_speed) % len(pts)

        # Pick ONLY the one point for this frame
        px, py, idx = pts[current_frame_idx]

        # --- 3. Draw the Single Ellipse (Logic moved out of loop) ---
        li = min(idx, len(self.cycle_lengths) - 1) if len(self.cycle_lengths) > 0 else 0
        wi = min(idx, len(self.cycle_widths) - 1) if len(self.cycle_widths) > 0 else 0
        ei = min(idx, len(self.cycle_euler_angles) - 1) if len(self.cycle_euler_angles) > 0 else 0
        
        try:
            body_len = float(self.cycle_lengths[li])
            body_wid = float(self.cycle_widths[wi])
            body_angle = float(self.cycle_euler_angles[ei][0])
        except Exception:
            body_len = float(self.robot.init_length)
            body_wid = float(self.robot.init_width)
            body_angle = float(self.robot.euler_angles[0])

        if body_len > 10.0:
            ew = max(4, int(body_len))
        else:
            ew = max(4, int(scale * body_len))

        if body_wid > 10.0:
            eh = max(4, int(body_wid))
        else:
            eh = max(4, int(scale * body_wid))

        # Draw it! (I removed the alpha fade since it's just one solid ghost now)
        try:
            # Use a solid color or semi-transparent for the single ghost
            # 150 alpha makes it look like a "ghost" but clearly visible
            ell_surf = pygame.Surface((ew, eh), pygame.SRCALPHA)
            color = (*self._history_color, 150)
            pygame.draw.ellipse(ell_surf, color, (0, 0, ew, eh))
            # rotate according to recorded yaw (first Euler angle) for this sample (fallback to current robot yaw)
            # sample_angle = (float(self.cycle_euler_angles[li][0]) if len(self.cycle_euler_angles) > li else float(self.robot.euler_angles[0]))
            rotated_surf = pygame.transform.rotate(ell_surf, -math.degrees(body_angle))
            rect = rotated_surf.get_rect(center=(px, py))
            self.screen.blit(rotated_surf, rect)
        except Exception:
            pygame.draw.circle(self.screen, self._history_color, (px, py), 3)

    def _draw_body(self, scale: float, robot_x: int, robot_y: int):
        # Body color based on robot internal state
        phase_colors = {
            "refill": (100, 140, 180),
            "jet": (70, 100, 150),
            "coast": (150, 100, 70),
            "rest": (100, 140, 180)
        }
        body_color = phase_colors.get(self.robot.get_state(), (100, 140, 180))

        # Draw morphing body
        ellipse_width = max(2, int(scale * float(self.robot.get_current_length())))
        ellipse_height = max(2, int(scale * float(self.robot.get_current_width())))

        ellipse_surf = pygame.Surface((ellipse_width, ellipse_height), pygame.SRCALPHA)
        pygame.draw.ellipse(ellipse_surf, body_color, (0, 0, ellipse_width, ellipse_height))

        # Robot body rotation uses robot.angle
        rotated_surf = pygame.transform.rotate(ellipse_surf, -math.degrees(self.robot.angle))
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

    # def _draw_nozzle_and_jet(self, scale: float, robot_x: int, robot_y: int):
    #     # Draw front indicator
    #     front_distance = max(self.ellipse_a, self.ellipse_b) * 0.8
    #     front_x = robot_x + math.cos(self.robot.angle) * front_distance
    #     front_y = robot_y + math.sin(self.robot.angle) * front_distance
    #     pygame.draw.circle(self.screen, (255, 255, 255), (int(front_x), int(front_y)), 4)

    #     # Draw steerable nozzle
    #     back_distance = max(self.ellipse_a, self.ellipse_b) * 0.9
    #     back_x = robot_x + math.cos(self.robot.angle + math.pi) * back_distance
    #     back_y = robot_y + math.sin(self.robot.angle + math.pi) * back_distance

    #     nozzle_world_angle = self.robot.angle + math.pi + self.robot.nozzle_angle
    #     nozzle_length = 15
    #     nozzle_end_x = back_x + math.cos(nozzle_world_angle) * nozzle_length
    #     nozzle_end_y = back_y + math.sin(nozzle_world_angle) * nozzle_length

    #     pygame.draw.line(self.screen, (200, 200, 100),
    #                     (int(back_x), int(back_y)), (int(nozzle_end_x), int(nozzle_end_y)), 4)

    #     # Draw water jet during 'jet' phase
    #     if self.robot.get_state() == "jet":
    #         num_particles = 8
    #         for i in range(num_particles):
    #             base_distance = nozzle_length + 5 + i * 4
    #             curve_factor = abs(self.robot.nozzle_angle) * 0.5
    #             curve_offset = curve_factor * (i * 0.3)

    #             perpendicular_angle = nozzle_world_angle + math.pi/2
    #             if self.robot.nozzle_angle > 0:
    #                 curve_offset = -curve_offset

    #             straight_x = back_x + math.cos(nozzle_world_angle) * base_distance
    #             straight_y = back_y + math.sin(nozzle_world_angle) * base_distance

    #             curved_x = straight_x + math.cos(perpendicular_angle) * curve_offset
    #             curved_y = straight_y + math.sin(perpendicular_angle) * curve_offset

    #             spread_variation = (i - num_particles/2) * 0.08
    #             spread_x = curved_x + math.cos(perpendicular_angle) * spread_variation * 3
    #             spread_y = curved_y + math.sin(perpendicular_angle) * spread_variation * 3

    #             particle_size = max(1, 5 - i)
    #             blue_intensity = max(100, 200 - i * 15)
    #             particle_color = (80, 120, blue_intensity)

    #             pygame.draw.circle(self.screen, particle_color,
    #                              (int(spread_x), int(spread_y)), particle_size)

    # def _draw_ui(self):
    #     if hasattr(pygame, 'font') and pygame.font.get_init():
    #         font = pygame.font.Font(None, 24)
    #         info_lines = [
    #             "SALP Robot - Steerable Nozzle",
    #             f"Phase: {self.robot.get_state().title()}",
    #             f"Body Size: {max(self.ellipse_a, self.ellipse_b):.2f}",
    #             f"Speed: {self.robot.velocities[0]:.2f}",
    #             f"Nozzle: {math.degrees(self.robot.nozzle_angle):.0f}°",
    #         ]

    #         for i, line in enumerate(info_lines):
    #             text = font.render(line, True, (255, 255, 255))
    #             self.screen.blit(text, (10, 10 + i * 25))
    
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
        robot_x = int(self.pos_init[0] + self.robot.positions[0] * scale)
        robot_y = int(self.pos_init[1] + self.robot.positions[1] * scale)

        # draw rulers and grid to visualize meters in both x and y
        self._draw_rulers(scale)

        # draw historical path and sized ellipses
        self._draw_history(scale, robot_x, robot_y)

        # draw the current robot body
        # self._draw_body(scale, robot_x, robot_y)

        # draw nozzle and any jet particles
        # self._draw_nozzle_and_jet(scale, robot_x, robot_y)

        # draw UI overlay
        # self._draw_ui()

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
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, max_contraction=0.06, nozzle_area=0.001)
    env = SalpRobotEnv(render_mode="human", robot=robot)
    obs, info = env.reset()
    
    done = False
    while not done:
        action = [0.06, 0.0, 0.0]  # inhale with no nozzle steering
        # For every step in the environment, there are multiple internal robot steps
        obs, reward, done, truncated, info = env.step(action)
        env.render()    
    env.close()
    