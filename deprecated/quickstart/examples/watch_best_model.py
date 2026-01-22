#!/usr/bin/env python3
"""
Zero-configuration script to watch the best trained model.

Perfect for newcomers - just run and watch!

Usage:
    python quickstart/examples/watch_best_model.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the main watch agent script
from quickstart.watch_agent import find_best_model, watch_agent


def main():
    print("🎬 SALP Agent Quick Demo")
    print("=" * 70)
    print("This will automatically find and run your best trained model.")
    print("=" * 70 + "\n")
    
    # Find best model
    print("🔍 Finding best model...")
    model_path = find_best_model()
    
    if model_path is None:
        print("\n❌ No trained models found!")
        print("\nTo train a model, run:")
        print("  python examples/demo.py train")
        sys.exit(1)
    
    print(f"✓ Found: {model_path.parent.name}/{model_path.name}\n")
    print("Press Ctrl+C to stop at any time.\n")
    
    # Watch the agent for 3 episodes (quick demo)
    watch_agent(
        model_path=model_path,
        n_episodes=3,
        deterministic=True,
        render=True
    )
    
    print("\n🎉 Demo complete!")
    print("\nTo explore more options:")
    print("  python quickstart/list_models.py          # See all available models")
    print("  python quickstart/watch_agent.py --help   # See all options")
    print()


if __name__ == "__main__":
    main()
