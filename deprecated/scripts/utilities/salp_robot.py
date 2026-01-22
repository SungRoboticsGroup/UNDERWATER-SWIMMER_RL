"""
Realistic SALP Robot Simulation
Bio-inspired soft underwater robot with steerable rear nozzle and realistic breathing cycles.
Based on research from University of Pennsylvania Sung Robotics Lab.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from typing import Tuple, Optional, Dict, Any


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
    
    def __init__(self, render_mode: Optional[str] = None, width: int = 800, height: int = 600):
        super().__init__()
        
        # Environment parameters
        self.width = width
        self.height = height
        self.tank_margin = 50
        
        # SALP robot parameters
        self.base_radius = 30  # Base body radius (circle at rest)
        self.max_thrust_force = 100  # Thrust per expansion
        self.drag_coefficient = 0.98  # Underwater drag
        
        # Constant surface area parameters
        # Circle surface area = 2πr, Ellipse surface area ≈ π(a + b)
        # For constant surface area: 2πr = π(a + b), so a + b = 2r
        self.base_surface_area = 2 * math.pi * self.base_radius
        
        # Nozzle parameters
        self.max_nozzle_angle = math.pi / 3  # ±60 degrees nozzle steering
        self.nozzle_response_rate = 0.05  # How fast nozzle moves
        
        # Realistic breathing cycle parameters (much slower)
        self.inhale_duration = 120  # 2 seconds at 60fps
        self.exhale_duration = 150  # 2.5 seconds at 60fps
        self.rest_duration = 60    # 1 second rest between cycles
        
        # Pygame setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # Robot state
        self.robot_pos = np.array([width/2, height/2], dtype=float)
        self.robot_velocity = np.array([0.0, 0.0], dtype=float)
        self.robot_angle = 0.0  # Body orientation (changes slowly due to physics)
        self.robot_angular_velocity = 0.0
        
        # Nozzle state
        self.nozzle_angle = 0.0  # Relative to body orientation (-max_nozzle_angle to +max_nozzle_angle)
        self.target_nozzle_angle = 0.0
        
        # Breathing state
        self.breathing_phase = "rest"  # "rest", "inhaling", "exhaling"
        self.breathing_timer = 0
        self.body_radius = self.base_radius  # Current body radius
        self.ellipse_a = self.base_radius    # Semi-major axis for ellipse
        self.ellipse_b = self.base_radius    # Semi-minor axis for ellipse
        self.is_inhaling = False  # True when space is held down
        self.water_volume = 0.0  # Amount of water inhaled (0-1)
        
        # Action space: [inhale_control (0/1), nozzle_direction (-1 to 1)]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: [pos_x, pos_y, vel_x, vel_y, body_angle, angular_vel, body_size, breathing_phase, water_volume, nozzle_angle]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -10, -10, -math.pi, -0.1, 0.5, 0, 0, -1]),
            high=np.array([width, height, 10, 10, math.pi, 0.1, 2.0, 2, 1, 1]),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Reset robot to center
        self.robot_pos = np.array([self.width/2, self.height/2], dtype=float)
        self.robot_velocity = np.array([0.0, 0.0], dtype=float)
        self.robot_angle = 0.0
        self.robot_angular_velocity = 0.0
        
        # Reset nozzle
        self.nozzle_angle = 0.0
        self.target_nozzle_angle = 0.0
        
        # Reset breathing state
        self.breathing_phase = "rest"
        self.breathing_timer = 0
        self.body_radius = self.base_radius  # Current body radius
        self.ellipse_a = self.base_radius    # Semi-major axis for ellipse
        self.ellipse_b = self.base_radius    # Semi-minor axis for ellipse
        self.is_inhaling = False
        self.water_volume = 0.0
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Check if we're in forced breathing mode (passed from environment)
        forced_breathing = getattr(self, 'forced_breathing', False)
        
        if forced_breathing:
            # FORCED BREATHING MODE: Only nozzle control
            nozzle_direction = float(action[0])  # -1 to 1 (single action)
            # Force automatic breathing cycle
            self.is_inhaling = self._get_forced_breathing_state()
        else:
            # NORMAL MODE: Full control
            inhale_control = float(action[0])  # 0 to 1
            nozzle_direction = float(action[1])  # -1 to 1
            self.is_inhaling = inhale_control > 0.5
        
        # Update control inputs
        self.target_nozzle_angle = nozzle_direction * self.max_nozzle_angle
        
        # Update nozzle position (smooth movement)
        self._update_nozzle()
        
        # Update breathing cycle
        self._update_breathing_cycle()
        
        # Update physics
        self._update_physics()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        done = False
        truncated = False
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, truncated, info
    
    def _get_forced_breathing_state(self) -> bool:
        """Determine breathing state for forced breathing mode."""
        # Create automatic breathing cycle based on timer
        cycle_length = self.inhale_duration + self.exhale_duration + self.rest_duration
        cycle_position = self.breathing_timer % cycle_length
        
        if cycle_position < self.inhale_duration:
            return True  # Inhaling
        else:
            return False  # Exhaling or resting
    
    def _update_nozzle(self):
        """Update nozzle angle smoothly."""
        angle_diff = self.target_nozzle_angle - self.nozzle_angle
        if abs(angle_diff) > self.nozzle_response_rate:
            if angle_diff > 0:
                self.nozzle_angle += self.nozzle_response_rate
            else:
                self.nozzle_angle -= self.nozzle_response_rate
        else:
            self.nozzle_angle = self.target_nozzle_angle
        
        # Clamp nozzle angle
        self.nozzle_angle = max(-self.max_nozzle_angle, 
                              min(self.max_nozzle_angle, self.nozzle_angle))
    
    def _update_breathing_cycle(self):
        """Update the breathing cycle with correct sequence: ellipsoid → sphere → ellipsoid."""
        if self.breathing_phase == "rest":
            # At rest: ellipsoid shape (natural resting state)
            self.body_radius = self.base_radius
            # Start with moderate ellipsoid shape
            self.ellipse_a = self.base_radius * 1.3  # Slightly elongated
            self.ellipse_b = self.base_radius * 0.8  # Slightly compressed
            
            if self.is_inhaling:
                # Start inhaling
                self.breathing_phase = "inhaling"
                self.breathing_timer = 0
            
        elif self.breathing_phase == "inhaling":
            if self.is_inhaling and self.breathing_timer < self.inhale_duration:
                # Continue inhaling - body becomes more spherical (filling with water)
                self.breathing_timer += 1
                progress = self.breathing_timer / self.inhale_duration
                
                # Transition from ellipsoid to sphere as water fills
                # Start: ellipse_a = 1.3*r, ellipse_b = 0.8*r
                # End: ellipse_a = 1.1*r, ellipse_b = 1.1*r (slightly larger sphere)
                start_a = self.base_radius * 1.3
                start_b = self.base_radius * 0.8
                end_a = self.base_radius * 1.1  # Slightly expanded sphere
                end_b = self.base_radius * 1.1
                
                self.ellipse_a = start_a + (end_a - start_a) * progress
                self.ellipse_b = start_b + (end_b - start_b) * progress
                
                self.water_volume = progress
                
            else:
                # Stop inhaling (either released early or reached max inhale)
                if self.water_volume > 0.05:  # Only exhale if we inhaled some water
                    self.breathing_phase = "exhaling"
                    self.breathing_timer = 0
                    # Scale exhale duration based on how much water was inhaled
                    self.current_exhale_duration = int(self.exhale_duration * max(self.water_volume, 0.3))
                else:
                    # Return to rest if barely inhaled (early release with minimal water)
                    self.breathing_phase = "rest"
                    self.breathing_timer = 0
                    self.water_volume = 0.0
        
        elif self.breathing_phase == "exhaling":
            self.breathing_timer += 1
            # Use scaled exhale duration based on water volume
            exhale_duration = getattr(self, 'current_exhale_duration', self.exhale_duration)
            progress = self.breathing_timer / exhale_duration
            
            if progress <= 1.0:
                # Exhale - body transitions from sphere back to ellipsoid (expelling water)
                # Start: sphere (ellipse_a = 1.1*r, ellipse_b = 1.1*r)
                # End: ellipsoid (ellipse_a = 1.3*r, ellipse_b = 0.8*r)
                start_a = self.base_radius * 1.1
                start_b = self.base_radius * 1.1
                end_a = self.base_radius * 1.3
                end_b = self.base_radius * 0.8
                
                self.ellipse_a = start_a + (end_a - start_a) * progress
                self.ellipse_b = start_b + (end_b - start_b) * progress
                
                # Apply thrust during early expansion, scaled by water volume
                if 0.1 <= progress <= 0.5:
                    self._apply_jet_thrust()
                
                # Reduce water volume
                self.water_volume = max(0, self.water_volume * (1.0 - progress))
                
            else:
                # Exhale complete, return to rest
                self.breathing_phase = "rest"
                self.breathing_timer = 0
                self.water_volume = 0.0
    
    def _apply_jet_thrust(self):
        """Apply jet thrust through steerable nozzle with improved moment physics."""
        # Calculate thrust based on water volume and expansion rate
        thrust_magnitude = self.max_thrust_force * self.water_volume * 0.4
        
        # CORRECT thrust direction calculation:
        # When nozzle_angle = 0 (straight back): robot should move forward (robot_angle)
        # When nozzle points left: robot should move right
        # When nozzle points right: robot should move left
        
        thrust_angle = self.robot_angle - self.nozzle_angle
        
        thrust_x = math.cos(thrust_angle) * thrust_magnitude
        thrust_y = math.sin(thrust_angle) * thrust_magnitude
        
        # Apply thrust to velocity
        self.robot_velocity[0] += thrust_x * 0.012
        self.robot_velocity[1] += thrust_y * 0.012
        
        # Enhanced but corrected torque physics
        # When nozzle points right (+), robot should turn left (-)
        # When nozzle points left (-), robot should turn right (+)
        
        # 1. Primary steering torque (opposite to nozzle angle)
        primary_torque = -self.nozzle_angle * thrust_magnitude * 0.0002
        
        # 2. Moment arm effect - thrust applied at rear creates rotation
        moment_arm = max(self.ellipse_a, self.ellipse_b) * 0.7
        thrust_perpendicular = thrust_magnitude * math.sin(-self.nozzle_angle)  # Corrected sign
        moment_torque = thrust_perpendicular * moment_arm * 0.00005
        
        # 3. Body shape effect during morphing
        shape_torque = -self.nozzle_angle * thrust_magnitude * self.water_volume * 0.00003
        
        # Combine torque effects
        total_torque = primary_torque + moment_torque + shape_torque
        self.robot_angular_velocity += total_torque
        
        # 4. Thrust vectoring effect - angled nozzle creates side force
        # This simulates how real underwater vehicles behave with vectored thrust
        side_thrust_angle = thrust_angle + math.pi/2  # Perpendicular to main thrust
        side_thrust_magnitude = thrust_magnitude * abs(self.nozzle_angle) * 0.3
        
        side_thrust_x = math.cos(side_thrust_angle) * side_thrust_magnitude
        side_thrust_y = math.sin(side_thrust_angle) * side_thrust_magnitude
        
        self.robot_velocity[0] += side_thrust_x * 0.008
        self.robot_velocity[1] += side_thrust_y * 0.008
        
        # Add slight random variation for realism (reduced since we have more physics)
        noise_angle = thrust_angle + (np.random.random() - 0.5) * 0.05
        noise_force = thrust_magnitude * 0.04
        self.robot_velocity[0] += math.cos(noise_angle) * noise_force * 0.002
        self.robot_velocity[1] += math.sin(noise_angle) * noise_force * 0.002
    
    def _update_physics(self):
        """Update robot physics with realistic underwater dynamics."""
        # Apply drag
        self.robot_velocity *= self.drag_coefficient
        self.robot_angular_velocity *= 0.95
        
        # Update position
        self.robot_pos += self.robot_velocity
        
        # Update body angle (changes slowly due to momentum and thrust)
        self.robot_angle += self.robot_angular_velocity
        
        # Normalize angle
        while self.robot_angle > math.pi:
            self.robot_angle -= 2 * math.pi
        while self.robot_angle < -math.pi:
            self.robot_angle += 2 * math.pi
        
        # Keep robot within bounds (bounce off walls)
        margin = self.tank_margin + max(self.ellipse_a, self.ellipse_b)
        if self.robot_pos[0] < margin:
            self.robot_pos[0] = margin
            self.robot_velocity[0] = abs(self.robot_velocity[0]) * 0.4
            self.robot_angular_velocity *= 0.7
        elif self.robot_pos[0] > self.width - margin:
            self.robot_pos[0] = self.width - margin
            self.robot_velocity[0] = -abs(self.robot_velocity[0]) * 0.4
            self.robot_angular_velocity *= 0.7
        
        if self.robot_pos[1] < margin:
            self.robot_pos[1] = margin
            self.robot_velocity[1] = abs(self.robot_velocity[1]) * 0.4
            self.robot_angular_velocity *= 0.7
        elif self.robot_pos[1] > self.height - margin:
            self.robot_pos[1] = self.height - margin
            self.robot_velocity[1] = -abs(self.robot_velocity[1]) * 0.4
            self.robot_angular_velocity *= 0.7
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on realistic movement and efficiency."""
        # Reward for smooth movement
        speed = math.sqrt(self.robot_velocity[0]**2 + self.robot_velocity[1]**2)
        movement_reward = min(speed * 0.08, 0.6)
        
        # Reward for efficient breathing (not too frequent)
        breathing_efficiency = 0.15 if self.breathing_phase == "rest" else 0.08
        
        # Small penalty for excessive nozzle movement (energy cost)
        nozzle_penalty = abs(self.nozzle_angle) * 0.02
        
        # Small reward for staying in bounds
        bounds_reward = 0.05
        
        return movement_reward + breathing_efficiency + bounds_reward - nozzle_penalty
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Map breathing phase to number
        phase_map = {"rest": 0, "inhaling": 1, "exhaling": 2}
        phase_num = phase_map.get(self.breathing_phase, 0)
        
        return np.array([
            self.robot_pos[0] / self.width,  # Normalized position
            self.robot_pos[1] / self.height,
            self.robot_velocity[0] / 5.0,  # Normalized velocity
            self.robot_velocity[1] / 5.0,
            self.robot_angle / math.pi,  # Normalized body angle
            self.robot_angular_velocity / 0.1,  # Normalized angular velocity
            max(self.ellipse_a, self.ellipse_b) / self.base_radius,  # Normalized body size
            phase_num / 2.0,  # Normalized breathing phase
            self.water_volume,  # Water volume (0-1)
            self.nozzle_angle / self.max_nozzle_angle  # Normalized nozzle angle
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional information."""
        return {
            'robot_position': tuple(self.robot_pos),
            'robot_velocity': tuple(self.robot_velocity),
            'robot_angle': self.robot_angle,
            'nozzle_angle': self.nozzle_angle,
            'ellipse_a': self.ellipse_a,
            'ellipse_b': self.ellipse_b,
            'body_radius': self.body_radius,
            'breathing_phase': self.breathing_phase,
            'water_volume': self.water_volume,
            'breathing_timer': self.breathing_timer
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return
        
        if self.screen is None:
            # Initialize pygame with proper error handling
            try:
                pygame.init()
                pygame.display.init()
                
                # Ensure we have valid dimensions
                if self.width <= 0 or self.height <= 0:
                    self.width = 800
                    self.height = 600
                
                if self.render_mode == "human":
                    # Try to create display with error handling
                    try:
                        self.screen = pygame.display.set_mode((int(self.width), int(self.height)))
                        pygame.display.set_caption("SALP Robot - Snake Game")
                    except pygame.error as e:
                        print(f"Pygame display error: {e}")
                        # Fallback to smaller window
                        self.width, self.height = 640, 480
                        self.screen = pygame.display.set_mode((self.width, self.height))
                        pygame.display.set_caption("SALP Robot - Snake Game")
                else:
                    self.screen = pygame.Surface((int(self.width), int(self.height)))
            except Exception as e:
                print(f"Pygame initialization error: {e}")
                # Create a minimal surface as fallback
                self.screen = pygame.Surface((800, 600))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Clear screen with deep water color
        self.screen.fill((10, 25, 50))  # Deep blue water
        
        # Draw tank boundaries
        pygame.draw.rect(self.screen, (30, 60, 100), 
                        (self.tank_margin, self.tank_margin, 
                         self.width - 2*self.tank_margin, self.height - 2*self.tank_margin), 3)
        
        # Draw robot
        robot_x, robot_y = int(self.robot_pos[0]), int(self.robot_pos[1])
        
        # Body color based on breathing phase
        phase_colors = {
            "rest": (100, 140, 180),      # Light blue-gray
            "inhaling": (70, 100, 150),   # Darker blue (contracting)
            "exhaling": (150, 100, 70)    # Orange-brown (thrusting)
        }
        body_color = phase_colors.get(self.breathing_phase, (100, 140, 180))
        
        # Draw morphing body (always draw ellipse based on current ellipse_a and ellipse_b)
        ellipse_width = int(self.ellipse_a * 2)
        ellipse_height = int(self.ellipse_b * 2)
        
        # Create ellipse surface
        ellipse_surf = pygame.Surface((ellipse_width, ellipse_height), pygame.SRCALPHA)
        pygame.draw.ellipse(ellipse_surf, body_color, 
                          (0, 0, ellipse_width, ellipse_height))
        
        # Rotate ellipse to match body orientation
        rotated_surf = pygame.transform.rotate(ellipse_surf, -math.degrees(self.robot_angle))
        rect = rotated_surf.get_rect(center=(robot_x, robot_y))
        self.screen.blit(rotated_surf, rect)
        
        # Draw body outline (use larger dimension for outline)
        outline_radius = int(max(self.ellipse_a, self.ellipse_b))
        pygame.draw.circle(self.screen, (60, 80, 120), (robot_x, robot_y), outline_radius, 2)
        
        # Draw front indicator (direction)
        front_distance = max(self.ellipse_a, self.ellipse_b) * 0.8
        front_x = robot_x + math.cos(self.robot_angle) * front_distance
        front_y = robot_y + math.sin(self.robot_angle) * front_distance
        pygame.draw.circle(self.screen, (255, 255, 255), (int(front_x), int(front_y)), 4)
        
        # Draw steerable nozzle at the back
        back_distance = max(self.ellipse_a, self.ellipse_b) * 0.9
        back_x = robot_x + math.cos(self.robot_angle + math.pi) * back_distance
        back_y = robot_y + math.sin(self.robot_angle + math.pi) * back_distance
        
        # Nozzle direction (relative to body)
        nozzle_world_angle = self.robot_angle + math.pi + self.nozzle_angle
        nozzle_length = 15
        nozzle_end_x = back_x + math.cos(nozzle_world_angle) * nozzle_length
        nozzle_end_y = back_y + math.sin(nozzle_world_angle) * nozzle_length
        
        # Draw nozzle
        pygame.draw.line(self.screen, (200, 200, 100), 
                        (int(back_x), int(back_y)), (int(nozzle_end_x), int(nozzle_end_y)), 4)
        
        # Draw water jet during exhale with curved trajectory
        if self.breathing_phase == "exhaling":
            # Enhanced water jet particles with curved flow
            num_particles = 8
            for i in range(num_particles):
                # Base distance from nozzle
                base_distance = nozzle_length + 5 + i * 4
                
                # Curve effect: water flow curves due to nozzle angle and water dynamics
                curve_factor = abs(self.nozzle_angle) * 0.5  # More curve with larger nozzle angles
                curve_offset = curve_factor * (i * 0.3)  # Curve increases with distance
                
                # Apply curve perpendicular to nozzle direction
                perpendicular_angle = nozzle_world_angle + math.pi/2
                if self.nozzle_angle > 0:  # Nozzle right, water curves right
                    curve_offset = -curve_offset
                
                # Calculate curved particle position
                straight_x = back_x + math.cos(nozzle_world_angle) * base_distance
                straight_y = back_y + math.sin(nozzle_world_angle) * base_distance
                
                curved_x = straight_x + math.cos(perpendicular_angle) * curve_offset
                curved_y = straight_y + math.sin(perpendicular_angle) * curve_offset
                
                # Add natural spread to the jet
                spread_variation = (i - num_particles/2) * 0.08
                spread_x = curved_x + math.cos(perpendicular_angle) * spread_variation * 3
                spread_y = curved_y + math.sin(perpendicular_angle) * spread_variation * 3
                
                # Particle properties
                alpha = max(0, 255 - i * 25)
                particle_size = max(1, 5 - i)
                
                # Color varies with distance (darker blue further out)
                blue_intensity = max(100, 200 - i * 15)
                particle_color = (80, 120, blue_intensity)
                
                pygame.draw.circle(self.screen, particle_color, 
                                 (int(spread_x), int(spread_y)), particle_size)
        
        # Draw velocity vector
        speed = math.sqrt(self.robot_velocity[0]**2 + self.robot_velocity[1]**2)
        if speed > 0.5:
            vel_scale = 8
            vel_end_x = robot_x + self.robot_velocity[0] * vel_scale
            vel_end_y = robot_y + self.robot_velocity[1] * vel_scale
            pygame.draw.line(self.screen, (0, 255, 150), 
                           (robot_x, robot_y), (int(vel_end_x), int(vel_end_y)), 2)
        
        # UI information
        if hasattr(pygame, 'font') and pygame.font.get_init():
            font = pygame.font.Font(None, 24)
            
            info_lines = [
                "Realistic SALP Robot - Steerable Nozzle",
                f"Phase: {self.breathing_phase.title()}",
                f"Body Size: {max(self.ellipse_a, self.ellipse_b):.1f}",
                f"Water: {self.water_volume:.2f}",
                f"Speed: {speed:.1f}",
                f"Nozzle: {math.degrees(self.nozzle_angle):.0f}°",
                "HOLD SPACE: Inhale, ←/→ : Steer Nozzle"
            ]
            
            for i, line in enumerate(info_lines):
                color = (255, 255, 255) if i < 6 else (180, 180, 180)
                text = font.render(line, True, color)
                self.screen.blit(text, (10, 10 + i * 25))
        
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


def main():
    """Demo the realistic SALP robot environment."""
    print("Realistic SALP Robot Demo - Steerable Nozzle & Breathing")
    print("Controls:")
    print("- HOLD SPACE: Inhale water (slow contraction)")
    print("- RELEASE SPACE: Exhale water (slow expansion + thrust)")
    print("- ←/→ Arrow Keys: Steer rear nozzle left/right")
    print("- ESC: Quit")
    
    pygame.init()
    env = SalpRobotEnv(render_mode="human")
    observation, info = env.reset()
    
    running = True
    nozzle_direction = 0.0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Get keyboard input
        keys = pygame.key.get_pressed()
        
        # Inhale control (hold space)
        inhale_control = 1.0 if keys[pygame.K_SPACE] else 0.0
        
        # Nozzle steering (using arrow keys) - Fixed direction
        if keys[pygame.K_LEFT]:  # Left arrow key - steer nozzle left
            nozzle_direction = min(1.0, nozzle_direction + 0.03)
        elif keys[pygame.K_RIGHT]:  # Right arrow key - steer nozzle right
            nozzle_direction = max(-1.0, nozzle_direction - 0.03)
        # Removed auto-centering - nozzle maintains position when no input
        
        # Apply action
        action = np.array([inhale_control, nozzle_direction])
        observation, reward, done, truncated, info = env.step(action)
        
        # Render
        env.render()
        
        if done or truncated:
            observation, info = env.reset()
    
    env.close()


if __name__ == "__main__":
    main()
