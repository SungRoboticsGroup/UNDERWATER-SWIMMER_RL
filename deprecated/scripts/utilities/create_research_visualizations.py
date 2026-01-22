"""
Create research visualizations for SALP Robot project.
Generates physics diagram and trajectory visualization.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, FancyArrowPatch, Circle, Wedge
import numpy as np
import math

def create_physics_diagram():
    """Create a comprehensive physics diagram showing SALP breathing cycle and mechanics."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    
    # === TOP ROW: Breathing Cycle ===
    breathing_phases = [
        ('Rest (Ellipsoid)', 0, 0),
        ('Inhaling (→ Sphere)', 0, 1),
        ('Exhaling (→ Ellipsoid)', 0, 2)
    ]
    
    for title, row, col in breathing_phases:
        ax = axes[row, col]
        ax.set_xlim(-2.8, 2.8)
        ax.set_ylim(-2.8, 2.8)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.axis('off')
        
        if col == 0:  # Rest - Ellipsoid
            # Body ellipse (larger)
            ellipse = Ellipse((0, 0), width=3.2, height=2.0, 
                            angle=0, facecolor='lightblue', 
                            edgecolor='darkblue', linewidth=3, alpha=0.7)
            ax.add_patch(ellipse)
            
            # Front marker
            circle = Circle((1.5, 0), 0.15, color='white', zorder=10)
            ax.add_patch(circle)
            
            # Rear nozzle (straight)
            ax.arrow(-1.5, 0, -0.5, 0, head_width=0.2, head_length=0.2,
                    fc='gold', ec='orange', linewidth=3, zorder=10)
            
            # Labels
            ax.text(0, -2.0, 'Natural resting state\nEllipsoid shape', 
                   ha='center', fontsize=14, style='italic')
            
        elif col == 1:  # Inhaling - Sphere (larger)
            # Body sphere 
            circle = Circle((0, 0), 1.4, facecolor='steelblue', 
                          edgecolor='darkblue', linewidth=3, alpha=0.7)
            ax.add_patch(circle)
            
            # Front marker
            dot = Circle((1.1, 0), 0.15, color='white', zorder=10)
            ax.add_patch(dot)
            
            # Water intake arrows
            for angle in [-30, 0, 30]:
                rad = math.radians(angle)
                start_x = 2.3 * math.cos(rad)
                start_y = 2.3 * math.sin(rad)
                end_x = 1.6 * math.cos(rad)
                end_y = 1.6 * math.sin(rad)
                ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                        head_width=0.2, head_length=0.2, fc='cyan', 
                        ec='blue', linewidth=2, alpha=0.6)
            
            # Rear nozzle (straight, closed)
            ax.plot([-1.4, -1.8], [0, 0], 'gold', linewidth=4)
            
            # Labels
            ax.text(0, -2.0, 'Water intake\nBody expands → Sphere', 
                   ha='center', fontsize=14, style='italic', color='blue')
            
        else:  # Exhaling - Ellipsoid with thrust (larger)
            # Body ellipse (returning to rest)
            ellipse = Ellipse((0, 0), width=3.2, height=2.0,
                            angle=0, facecolor='lightsalmon',
                            edgecolor='red', linewidth=3, alpha=0.7)
            ax.add_patch(ellipse)
            
            # Front marker
            circle = Circle((1.5, 0), 0.15, color='white', zorder=10)
            ax.add_patch(circle)
            
            # Rear nozzle (straight)
            ax.arrow(-1.5, 0, -0.5, 0, head_width=0.2, head_length=0.2,
                    fc='gold', ec='orange', linewidth=3, zorder=10)
            
            # Water jet particles (larger, with more visibility)
            for i in range(8):
                x_pos = -2.0 - i * 0.3
                size = max(0.08, 0.18 - i * 0.015)
                alpha = max(0.2, 0.8 - i * 0.1)
                circle = Circle((x_pos, 0), size, color='dodgerblue', alpha=alpha)
                ax.add_patch(circle)
            
            # Thrust arrow
            ax.arrow(-0.6, -1.6, 1.8, 0, head_width=0.25, head_length=0.25,
                    fc='red', ec='darkred', linewidth=4, zorder=5)
            ax.text(0.3, -1.25, 'THRUST', fontsize=14, fontweight='bold',
                   color='darkred', ha='center')
            
            # Labels
            ax.text(0, -2.2, 'Water expulsion\nBody contracts → Thrust',
                   ha='center', fontsize=14, style='italic', color='red')
    
    # === BOTTOM ROW: Mechanics Details ===
    
    # Bottom Left: Steerable Nozzle Mechanism
    ax = axes[1, 0]
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.8, 2.8)
    ax.set_aspect('equal')
    ax.set_title('Steerable Nozzle Control', fontsize=20, fontweight='bold')
    ax.axis('off')
    
    # Central body (larger)
    ellipse = Ellipse((0, 0), width=3.2, height=2.0,
                     angle=0, facecolor='lightblue',
                     edgecolor='darkblue', linewidth=3, alpha=0.7)
    ax.add_patch(ellipse)
    
    # Nozzle at different angles
    nozzle_angles = [-40, 0, 40]
    nozzle_colors = ['green', 'gold', 'purple']
    nozzle_labels = ['Left (-40°)', 'Center (0°)', 'Right (+40°)']
    
    for angle_deg, color, label in zip(nozzle_angles, nozzle_colors, nozzle_labels):
        angle_rad = math.radians(180 + angle_deg)
        start_x = -1.5
        start_y = 0
        end_x = start_x + 0.6 * math.cos(angle_rad)
        end_y = start_y + 0.6 * math.sin(angle_rad)
        
        ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                head_width=0.2, head_length=0.15, fc=color, ec=color,
                linewidth=3, alpha=0.7)
    
    # Angle arc showing the steering range
    wedge = Wedge((-1.5, 0), 1.0, 180-60, 180+60, 
                  facecolor='yellow', alpha=0.2, edgecolor='orange', linewidth=2.5)
    ax.add_patch(wedge)
    
    # ±60° label positioned to the right of the arrow base, at the axis (y=0)
    ax.text(-0.7, 0, '±60°', fontsize=16, fontweight='bold', color='black', ha='left', va='center')
    
    # Control info
    ax.text(0, -2.2, 'Nozzle steering range: ±60°\nControls thrust direction',
           ha='center', fontsize=14, style='italic')
    
    # Bottom Middle: Thrust Vectoring Physics
    ax = axes[1, 1]
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.8, 2.8)
    ax.set_aspect('equal')
    ax.set_title('Thrust Vectoring Physics', fontsize=20, fontweight='bold')
    ax.axis('off')
    
    # Body with angled nozzle (right turn, larger)
    ellipse = Ellipse((0, 0), width=3.2, height=2.0,
                     angle=0, facecolor='lightsalmon',
                     edgecolor='red', linewidth=3, alpha=0.7)
    ax.add_patch(ellipse)
    
    # Angled nozzle (pointing down-left for right turn)
    nozzle_angle = math.radians(180 + 30)
    ax.arrow(-1.5, 0, 0.5 * math.cos(nozzle_angle), 0.5 * math.sin(nozzle_angle),
            head_width=0.2, head_length=0.15, fc='gold', ec='orange', linewidth=3)
    
    # Thrust vector (opposite to nozzle)
    thrust_angle = nozzle_angle + math.pi
    ax.arrow(0, 0, 2.0 * math.cos(thrust_angle), 2.0 * math.sin(thrust_angle),
            head_width=0.3, head_length=0.3, fc='red', ec='darkred',
            linewidth=4, alpha=0.8)
    ax.text(0.6, 1.0, 'Forward +\nRight Turn', fontsize=12, fontweight='bold',
           color='darkred')
    
    # Rotation indicator (centered at origin where thrust force is applied)
    # The arc should be at the center of the robot body where rotation occurs
    arc = patches.Arc((0, 0), 2.2, 2.2, angle=0, theta1=-50, theta2=50,
                     color='blue', linewidth=3, linestyle='--')
    ax.add_patch(arc)
    ax.text(1.1, -1.1, '⟲', fontsize=32, color='blue', fontweight='bold')
    
    # Physics info
    ax.text(0, -2.2, 'Angled thrust creates:\n• Forward propulsion\n• Rotational torque',
           ha='center', fontsize=14, style='italic')
    
    # Bottom Right: Key Parameters & Performance
    ax = axes[1, 2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    # Remove separate title since it will be inside the box
    
    # Parameter boxes - larger and more readable
    param_text = """BREATHING CYCLE
  • Rest: 1 sec (ellipsoid)
  • Inhale: 2 sec (→ sphere)
  • Exhale: 2.5 sec (→ ellipsoid)

CONTROL SYSTEM
  • Action Space: [inhale, nozzle_angle]
  • Nozzle Range: ±60°
  • Steering Rate: 5% per frame

PHYSICS
  • Drag Coefficient: 0.98
  • Max Thrust: 100 units
  • Water Volume: 0-1 (normalized)"""
    
    # Center the text box with larger font for better slide readability
    ax.text(5, 9.2, param_text, fontsize=15, family='monospace',
           verticalalignment='top', linespacing=1.6, ha='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=1.4, linewidth=2.5))
    
    # # Bio-inspired note
    # ax.text(5, 0.8, 'Bio-Inspired Design\nBased on marine salps\n(barrel-shaped invertebrates)',
    #        ha='center', fontsize=12, style='italic',
    #        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6, pad=0.7, linewidth=2))
    
    plt.tight_layout(pad=1.0, h_pad=1.5, w_pad=1.0)
    plt.savefig('salp_physics_diagram.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    print("✓ Created: salp_physics_diagram.png")
    plt.close()


def create_trajectory_comparison():
    """Create trajectory visualization comparing untrained vs trained behavior."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('SALP Robot Learning: Trajectory Comparison', fontsize=14, fontweight='bold')
    
    # Simulation parameters
    np.random.seed(42)
    
    # === LEFT: Untrained (Random) Behavior ===
    ax = axes[0]
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 600)
    ax.set_aspect('equal')
    ax.set_title('Untrained Agent (Random Policy)', fontsize=12, fontweight='bold')
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    
    # Tank boundaries
    ax.add_patch(patches.Rectangle((50, 50), 700, 500, fill=False,
                                   edgecolor='darkblue', linewidth=2))
    
    # Random trajectory (wandering, inefficient)
    x_random = [400]
    y_random = [300]
    angle = 0
    
    for i in range(150):
        # Random movement with frequent direction changes
        angle += np.random.uniform(-0.5, 0.5)
        speed = np.random.uniform(1, 4)
        
        x_new = x_random[-1] + speed * np.cos(angle)
        y_new = y_random[-1] + speed * np.sin(angle)
        
        # Bounce off walls
        if x_new < 80:
            x_new = 80
            angle = np.pi - angle
        elif x_new > 720:
            x_new = 720
            angle = np.pi - angle
        if y_new < 80:
            y_new = 80
            angle = -angle
        elif y_new > 520:
            y_new = 520
            angle = -angle
        
        x_random.append(x_new)
        y_random.append(y_new)
    
    # Plot trajectory with color gradient
    points = np.array([x_random, y_random]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, cmap='Reds', alpha=0.6)
    lc.set_array(np.linspace(0, 1, len(x_random)))
    lc.set_linewidth(2)
    ax.add_collection(lc)
    
    # Start and end markers
    ax.plot(x_random[0], y_random[0], 'go', markersize=10, label='Start')
    ax.plot(x_random[-1], y_random[-1], 'ro', markersize=10, label='End')
    
    # Add some random breathing cycles (circles)
    for i in range(0, len(x_random), 20):
        circle = Circle((x_random[i], y_random[i]), 15, 
                       fill=False, edgecolor='red', alpha=0.3, linestyle='--')
        ax.add_patch(circle)
    
    ax.legend(loc='upper right')
    ax.text(400, 30, 'Erratic movement • Inefficient breathing • No goal-directed behavior',
           ha='center', fontsize=9, style='italic', color='darkred')
    
    # === RIGHT: Trained Behavior ===
    ax = axes[1]
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 600)
    ax.set_aspect('equal')
    ax.set_title('Trained Agent (SAC+GAIL)', fontsize=12, fontweight='bold')
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    
    # Tank boundaries
    ax.add_patch(patches.Rectangle((50, 50), 700, 500, fill=False,
                                   edgecolor='darkblue', linewidth=2))
    
    # Smooth trajectory (efficient, goal-directed)
    x_trained = [100]
    y_trained = [100]
    
    # Create smooth curve to target
    target_x, target_y = 700, 500
    
    for i in range(100):
        t = i / 100
        # Smooth S-curve trajectory
        x_new = x_trained[0] + (target_x - x_trained[0]) * t
        y_new = y_trained[0] + (target_y - y_trained[0]) * t
        
        # Add gentle sinusoidal variation for realistic swimming
        phase = t * 4 * np.pi
        x_new += 30 * np.sin(phase) * (1 - t)
        y_new += 20 * np.cos(phase * 1.3) * (1 - t)
        
        x_trained.append(x_new)
        y_trained.append(y_new)
    
    # Plot trajectory with color gradient
    points = np.array([x_trained, y_trained]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc = LineCollection(segments, cmap='Greens', alpha=0.7)
    lc.set_array(np.linspace(0, 1, len(x_trained)))
    lc.set_linewidth(3)
    ax.add_collection(lc)
    
    # Start and end markers
    ax.plot(x_trained[0], y_trained[0], 'go', markersize=12, label='Start', zorder=10)
    ax.plot(x_trained[-1], y_trained[-1], 'rs', markersize=12, label='Goal Reached', zorder=10)
    
    # Add breathing cycle markers (showing learned rhythm)
    breathing_intervals = [0, 20, 40, 60, 80]
    for i in breathing_intervals:
        if i < len(x_trained):
            circle = Circle((x_trained[i], y_trained[i]), 25,
                          fill=False, edgecolor='green', linewidth=2, alpha=0.5)
            ax.add_patch(circle)
            # Arrow showing direction
            if i < len(x_trained) - 5:
                dx = x_trained[i+5] - x_trained[i]
                dy = y_trained[i+5] - y_trained[i]
                ax.arrow(x_trained[i], y_trained[i], dx*0.3, dy*0.3,
                        head_width=15, head_length=10, fc='green', ec='darkgreen',
                        alpha=0.6, linewidth=1.5)
    
    ax.legend(loc='upper right')
    ax.text(400, 30, 'Smooth trajectory • Rhythmic breathing • Goal-directed navigation',
           ha='center', fontsize=9, style='italic', color='darkgreen')
    
    plt.tight_layout()
    plt.savefig('salp_trajectory_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Created: salp_trajectory_comparison.png")
    plt.close()


def main():
    """Generate all research visualizations."""
    print("Generating SALP Robot Research Visualizations...")
    print()
    
    print("Creating physics diagram...")
    create_physics_diagram()
    print()
    
    print("=" * 60)
    print("✓ Visualization created successfully!")
    print("=" * 60)
    print()
    print("Generated file:")
    print("  • salp_physics_diagram.png - Comprehensive mechanics & breathing cycle")
    print()
    print("This image is ready for your research update!")


if __name__ == "__main__":
    main()
