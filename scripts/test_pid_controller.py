#!/usr/bin/env python3
"""
Test and compare PID controller vs legacy controller for paddle control.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import airhockey
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airhockey.sims.airhockey_box2d import AirHockeyBox2D


def test_controller(use_pid=False, pid_kp=1000.0, pid_ki=50.0, pid_kd=100.0, scenario_name="test"):
    """
    Test controller with a specific configuration.
    
    Args:
        use_pid: Whether to use PID controller
        pid_kp: Proportional gain
        pid_ki: Integral gain  
        pid_kd: Derivative gain
        scenario_name: Name of the scenario
    
    Returns:
        positions: List of paddle positions over time
        velocities: List of paddle velocities over time
        accelerations: List of paddle accelerations over time
        actions: List of actions taken
    """
    # Create simulator
    sim_params = {
        'absorb_target': False,
        'block_density': 500,
        'block_width': 0.0254,
        'force_scaling': 1,
        'gravity': -0.5,
        'length': 1.9304,
        'max_force_timestep': 100,
        'paddle_damping': 3,
        'paddle_density': 1500,
        'paddle_radius': 0.0508,
        'puck_damping': 0.5,
        'puck_density': 250,
        'puck_radius': 0.03175,
        'render_size': 360,
        'wall_bounce_scale': 0.02,
        'width': 0.8636,
        'seed': 0,
        'use_pid': use_pid,
        'pid_kp': pid_kp,
        'pid_ki': pid_ki,
        'pid_kd': pid_kd,
    }
    
    simulator = AirHockeyBox2D(**sim_params)
    
    # Reset and spawn paddle at center
    simulator.reset(seed=0)
    paddle_start_pos = (0.0, 0.0)
    paddle_start_vel = (0.0, 0.0)
    simulator.spawn_paddle(paddle_start_pos, paddle_start_vel, 'paddle_ego', affected_by_gravity=False)
    
    # Test scenarios
    actions = []
    
    # Scenario 1: Step input - move 0.2m to the right
    for _ in range(50):
        actions.append(np.array([0.02, 0.0]))  # Constant delta position
    
    # Scenario 2: Return to center
    for _ in range(50):
        actions.append(np.array([-0.02, 0.0]))
    
    # Track data
    positions = []
    velocities = []
    accelerations = []
    jerks = []
    
    print(f"\n{'='*80}")
    print(f"Testing: {scenario_name}")
    print(f"  PID enabled: {use_pid}")
    if use_pid:
        print(f"  Kp={pid_kp}, Ki={pid_ki}, Kd={pid_kd}")
    print(f"{'='*80}")
    
    # Run simulation
    for i, action in enumerate(actions):
        state_info = simulator.get_current_state()
        
        if 'paddles' in state_info and 'paddle_ego' in state_info['paddles']:
            pos = state_info['paddles']['paddle_ego']['position']
            vel = state_info['paddles']['paddle_ego']['velocity']
            acc = state_info['paddles']['paddle_ego']['acceleration']
            jrk = state_info['paddles']['paddle_ego']['jerk']
            
            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)
            jerks.append(jrk)
        
        simulator.get_transition(action)
        
        if i % 25 == 0:
            print(f"  Step {i:3d}: pos={pos[0]:7.4f}, vel={vel[0]:7.4f}, acc={acc[0]:8.2f}")
    
    # Final state
    final_state = simulator.get_current_state()
    if 'paddles' in final_state and 'paddle_ego' in final_state['paddles']:
        final_pos = final_state['paddles']['paddle_ego']['position']
        final_vel = final_state['paddles']['paddle_ego']['velocity']
        print(f"\nFinal state:")
        print(f"  Position: {final_pos}")
        print(f"  Velocity: {final_vel}")
    
    return positions, velocities, accelerations, jerks, actions


def plot_comparison(legacy_data, pid_data, output_path="controller_comparison.png"):
    """
    Plot comparison between legacy and PID controllers.
    
    Args:
        legacy_data: Tuple of (positions, velocities, accelerations, jerks, actions) for legacy
        pid_data: Tuple of (positions, velocities, accelerations, jerks, actions) for PID
        output_path: Path to save the plot
    """
    legacy_pos, legacy_vel, legacy_acc, legacy_jerk, legacy_actions = legacy_data
    pid_pos, pid_vel, pid_acc, pid_jerk, pid_actions = pid_data
    
    # Convert to numpy arrays
    legacy_pos = np.array(legacy_pos)
    legacy_vel = np.array(legacy_vel)
    legacy_acc = np.array(legacy_acc)
    legacy_jerk = np.array(legacy_jerk)
    
    pid_pos = np.array(pid_pos)
    pid_vel = np.array(pid_vel)
    pid_acc = np.array(pid_acc)
    pid_jerk = np.array(pid_jerk)
    
    timesteps = np.arange(len(legacy_pos))
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Plot position (X-axis only)
    axes[0].plot(timesteps, legacy_pos[:, 0], 'b-', label='Legacy Controller', linewidth=2)
    axes[0].plot(timesteps, pid_pos[:, 0], 'r-', label='PID Controller', linewidth=2)
    axes[0].set_ylabel('X Position (m)')
    axes[0].set_title('Paddle Control Comparison: Legacy vs PID')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot velocity (X-axis only)
    axes[1].plot(timesteps, legacy_vel[:, 0], 'b-', label='Legacy Controller', linewidth=2)
    axes[1].plot(timesteps, pid_vel[:, 0], 'r-', label='PID Controller', linewidth=2)
    axes[1].set_ylabel('X Velocity (m/s)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot acceleration (X-axis only)
    axes[2].plot(timesteps, legacy_acc[:, 0], 'b-', label='Legacy Controller', linewidth=2)
    axes[2].plot(timesteps, pid_acc[:, 0], 'r-', label='PID Controller', linewidth=2)
    axes[2].set_ylabel('X Acceleration (m/s²)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot jerk (X-axis only)
    axes[3].plot(timesteps, legacy_jerk[:, 0], 'b-', label='Legacy Controller', linewidth=2)
    axes[3].plot(timesteps, pid_jerk[:, 0], 'r-', label='PID Controller', linewidth=2)
    axes[3].set_ylabel('X Jerk (m/s³)')
    axes[3].set_xlabel('Timestep')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to: {output_path}")
    
    # Print statistics
    print("\nStatistics (X-axis):")
    print(f"{'Metric':<20} {'Legacy':<15} {'PID':<15}")
    print(f"{'-'*50}")
    print(f"{'Max Position':<20} {np.max(np.abs(legacy_pos[:, 0])):<15.4f} {np.max(np.abs(pid_pos[:, 0])):<15.4f}")
    print(f"{'Max Velocity':<20} {np.max(np.abs(legacy_vel[:, 0])):<15.4f} {np.max(np.abs(pid_vel[:, 0])):<15.4f}")
    print(f"{'Max Acceleration':<20} {np.max(np.abs(legacy_acc[:, 0])):<15.4f} {np.max(np.abs(pid_acc[:, 0])):<15.4f}")
    print(f"{'Max Jerk':<20} {np.max(np.abs(legacy_jerk[:, 0])):<15.4f} {np.max(np.abs(pid_jerk[:, 0])):<15.4f}")
    print(f"{'Avg |Acceleration|':<20} {np.mean(np.abs(legacy_acc[:, 0])):<15.4f} {np.mean(np.abs(pid_acc[:, 0])):<15.4f}")
    print(f"{'Avg |Jerk|':<20} {np.mean(np.abs(legacy_jerk[:, 0])):<15.4f} {np.mean(np.abs(pid_jerk[:, 0])):<15.4f}")


def main():
    """Main function to test and compare controllers."""
    print("="*80)
    print("PID CONTROLLER TEST")
    print("="*80)
    
    # Test legacy controller
    legacy_data = test_controller(
        use_pid=False,
        scenario_name="Legacy Controller"
    )
    
    # Test PID controller with default gains
    pid_data = test_controller(
        use_pid=True,
        pid_kp=1000.0,
        pid_ki=50.0,
        pid_kd=100.0,
        scenario_name="PID Controller (default gains)"
    )
    
    # Plot comparison
    output_dir = os.path.join(os.path.dirname(__file__), 'controller_comparison')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'legacy_vs_pid.png')
    
    plot_comparison(legacy_data, pid_data, output_path)
    
    print("\n" + "="*80)
    print("✓ Test complete!")
    print("="*80)


if __name__ == '__main__':
    main()

