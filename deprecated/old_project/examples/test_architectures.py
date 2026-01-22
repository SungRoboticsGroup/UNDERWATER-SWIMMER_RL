"""
Example: Testing Different Network Architectures with SB3

This demonstrates how you can easily test different neural network architectures
without writing any core RL code. SB3 handles all the algorithm implementation.
"""

import torch.nn as nn
from salp.environments import SalpSnakeEnv
from salp.agents import SB3SACAgent
from salp.config.base_config import AgentConfig


def test_small_network():
    """Test a small, fast network (good for quick experiments)."""
    print("\n=== Testing Small Network (64-64) ===")
    
    env = SalpSnakeEnv(render_mode=None, num_food_items=3)
    
    config = AgentConfig(
        name="SAC",
        hidden_sizes=[64, 64],  # Small network
        learning_rate=0.0003,
        batch_size=64,
        buffer_size=50000,
        gamma=0.99,
        tau=0.005
    )
    
    agent = SB3SACAgent(config, env)
    print("✓ Small network created successfully")
    env.close()


def test_medium_network():
    """Test a medium network (balanced performance/capacity)."""
    print("\n=== Testing Medium Network (256-256) ===")
    
    env = SalpSnakeEnv(render_mode=None, num_food_items=3)
    
    config = AgentConfig(
        name="SAC",
        hidden_sizes=[256, 256],  # Medium network
        learning_rate=0.0003,
        batch_size=256,
        buffer_size=100000,
        gamma=0.99,
        tau=0.005
    )
    
    agent = SB3SACAgent(config, env)
    print("✓ Medium network created successfully")
    env.close()


def test_large_deep_network():
    """Test a large, deep network (more capacity for complex tasks)."""
    print("\n=== Testing Large Deep Network (512-512-256) ===")
    
    env = SalpSnakeEnv(render_mode=None, num_food_items=5)
    
    config = AgentConfig(
        name="SAC",
        hidden_sizes=[512, 512, 256],  # Large, deep network
        learning_rate=0.0001,  # Lower LR for larger network
        batch_size=512,
        buffer_size=200000,
        gamma=0.99,
        tau=0.005
    )
    
    agent = SB3SACAgent(config, env)
    print("✓ Large deep network created successfully")
    env.close()


def test_asymmetric_network():
    """Test asymmetric networks (different sizes for actor/critic)."""
    print("\n=== Testing Asymmetric Network (Actor: 256-256, Critic: 512-512) ===")
    
    env = SalpSnakeEnv(render_mode=None, num_food_items=3)
    
    # For asymmetric networks, we need to pass policy_kwargs directly to SB3
    from stable_baselines3 import SAC
    
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        buffer_size=100000,
        batch_size=256,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],      # Actor (policy) network
                qf=[512, 512]       # Critic (Q-function) networks
            )
        ),
        verbose=1
    )
    
    print("✓ Asymmetric network created successfully")
    print("  • Actor: [256, 256]")
    print("  • Critic: [512, 512]")
    env.close()


def test_different_activation():
    """Test different activation functions."""
    print("\n=== Testing Different Activation Functions ===")
    
    # Test with ELU activation
    print("\n1. ELU Activation:")
    env = SalpSnakeEnv(render_mode=None, num_food_items=3)
    
    from stable_baselines3 import SAC
    
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=nn.ELU  # ELU activation
        ),
        verbose=1
    )
    print("✓ ELU activation network created")
    env.close()
    
    # Test with Tanh activation
    print("\n2. Tanh Activation:")
    env = SalpSnakeEnv(render_mode=None, num_food_items=3)
    
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=nn.Tanh  # Tanh activation
        ),
        verbose=1
    )
    print("✓ Tanh activation network created")
    env.close()


def compare_architectures_training():
    """
    Compare different architectures with actual training.
    This shows how to set up an architecture comparison experiment.
    """
    print("\n=== Architecture Comparison Setup ===")
    
    architectures = {
        "small": [64, 64],
        "medium": [256, 256],
        "large": [512, 512, 256],
        "deep": [256, 256, 256, 256]
    }
    
    print("\nReady to compare these architectures:")
    for name, layers in architectures.items():
        print(f"  • {name}: {layers}")
    
    print("\nTo run comparison:")
    print("1. Train each architecture for same number of episodes")
    print("2. Evaluate on same test episodes")
    print("3. Compare: training time, sample efficiency, final performance")
    
    # Example training setup for one architecture
    env = SalpSnakeEnv(render_mode=None, num_food_items=5)
    
    config = AgentConfig(
        name="SAC",
        hidden_sizes=architectures["medium"],
        learning_rate=0.0003,
        batch_size=256,
        buffer_size=100000,
        gamma=0.99,
        tau=0.005
    )
    
    agent = SB3SACAgent(config, env)
    print("\n✓ Example agent created for medium architecture")
    print("  Ready to train with: agent.learn(total_timesteps=100000)")
    env.close()


def main():
    """Run all architecture tests."""
    print("=" * 60)
    print("SB3 Architecture Testing Examples")
    print("=" * 60)
    print("\nThese examples show how to test different architectures")
    print("using Stable Baselines3 WITHOUT writing core RL code.")
    print("\nSB3 handles all the complex RL algorithm implementation:")
    print("  ✓ Replay buffer management")
    print("  ✓ Target network updates")
    print("  ✓ Gradient computation")
    print("  ✓ Entropy tuning")
    print("  ✓ Training loop")
    print("\nYou only need to specify the architecture!")
    
    try:
        # Test different architectures
        test_small_network()
        test_medium_network()
        test_large_deep_network()
        test_asymmetric_network()
        test_different_activation()
        compare_architectures_training()
        
        print("\n" + "=" * 60)
        print("✅ All architecture tests passed!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Choose an architecture to start with")
        print("2. Train it: agent.learn(total_timesteps=100000)")
        print("3. Compare with other architectures")
        print("4. Use the best performing one for your research")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
