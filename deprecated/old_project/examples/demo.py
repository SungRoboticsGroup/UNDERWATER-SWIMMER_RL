"""
Demo script for SALP RL experiments.
Provides easy access to training, evaluation, and environment testing.
"""

import argparse
import os
import sys
import time
from typing import Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.base_config import get_sac_snake_config, ExperimentConfig
from training.trainer import Trainer, EvaluationRunner
from training.continuous_trainer import ContinuousTrainer
from environments.salp_snake_env import SalpSnakeEnv
import pygame


def demo_environment():
    """Demo the SALP Snake environment with manual control."""
    print("=== SALP Snake Environment Demo ===")
    print("Controls:")
    print("- HOLD SPACE: Inhale water")
    print("- RELEASE SPACE: Exhale water (thrust)")
    print("- ←/→ Arrow Keys: Steer nozzle")
    print("- Collect green food items!")
    print("- ESC: Quit")
    print()
    
    pygame.init()
    env = SalpSnakeEnv(render_mode="human", num_food_items=5)
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
        
        # Inhale control
        inhale_control = 1.0 if keys[pygame.K_SPACE] else 0.0
        
        # Nozzle steering
        if keys[pygame.K_LEFT]:
            nozzle_direction = min(1.0, nozzle_direction + 0.03)
        elif keys[pygame.K_RIGHT]:
            nozzle_direction = max(-1.0, nozzle_direction - 0.03)
        
        # Apply action
        action = [inhale_control, nozzle_direction]
        observation, reward, done, truncated, info = env.step(action)
        
        # Render
        env.render()
        
        if done or truncated:
            print(f"Episode ended! Score: {info['score']}, Food collected: {info['food_collected']}")
            observation, info = env.reset()
    
    env.close()


def train_agent(episodes: Optional[int] = None, continuous: bool = False, 
                from_scratch: bool = False, model_name: Optional[str] = None, resume_model: Optional[str] = None):
    """Train a SALP agent."""
    if continuous:
        mode = "Continuous Visual Training"
    else:
        mode = "Training"
    
    print(f"=== {mode} SALP Agent ===")
    
    # Get configuration (only sac_snake is available)
    config = get_sac_snake_config()
    
    # Handle training mode selection
    if not from_scratch and resume_model is None:
        # Interactive mode - ask user what they want to do
        print("\n🤖 Training Mode Selection:")
        print("1. Start from scratch (new model)")
        print("2. Resume from existing model")
        
        while True:
            choice = input("\nChoose option (1 or 2): ").strip()
            if choice == "1":
                from_scratch = True
                break
            elif choice == "2":
                # Show available models
                available_models = _get_available_models()
                if not available_models:
                    print("❌ No existing models found. Starting from scratch.")
                    from_scratch = True
                    break
                
                print("\n📁 Available models:")
                for i, (exp_name, model_path, size_mb) in enumerate(available_models, 1):
                    print(f"  {i}. {exp_name} ({size_mb:.1f} MB)")
                
                while True:
                    try:
                        model_choice = input(f"\nChoose model (1-{len(available_models)}): ").strip()
                        model_idx = int(model_choice) - 1
                        if 0 <= model_idx < len(available_models):
                            resume_model = available_models[model_idx][1]
                            print(f"✅ Selected: {available_models[model_idx][0]}")
                            break
                        else:
                            print(f"❌ Please enter a number between 1 and {len(available_models)}")
                    except ValueError:
                        print("❌ Please enter a valid number")
                break
            else:
                print("❌ Please enter 1 or 2")
    
    # Handle model naming for new training
    if from_scratch and model_name is None:
        print(f"\n📝 Creating new model experiment")
        default_name = f"salp_snake_{int(time.time())}"  # Add timestamp for uniqueness
        model_name = input(f"Enter experiment name (default: {default_name}): ").strip()
        if not model_name:
            model_name = default_name
        print(f"✅ New experiment: {model_name}")
    
    # Update configuration based on training mode
    if from_scratch:
        config.training.experiment_name = model_name
        print(f"🆕 Starting fresh training: {model_name}")
    elif resume_model:
        # Extract experiment name from model path
        exp_name = os.path.basename(os.path.dirname(resume_model))
        config.training.experiment_name = f"{exp_name}_continued"
        print(f"🔄 Resuming from: {resume_model}")
        print(f"📁 New experiment: {config.training.experiment_name}")
    
    # Override episodes if specified
    if episodes is not None:
        config.training.max_episodes = episodes
    
    print(f"\n📊 Training Configuration:")
    print(f"  Episodes: {config.training.max_episodes}")
    print(f"  Environment: {config.environment.name}")
    print(f"  Agent: {config.agent.name}")
    print(f"  Experiment: {config.training.experiment_name}")
    
    if continuous:
        print("  🎮 Continuous visual display with real-time model updates!")
    print()
    
    # Create trainer and start training
    if resume_model and not from_scratch:
        # Use the continue training approach
        if continuous:
            from scripts.continue_training import ContinuousTrainerFromCheckpoint
            trainer = ContinuousTrainerFromCheckpoint(config, resume_model)
        else:
            # Create a basic trainer that can load from checkpoint
            trainer = Trainer(config)
            # Load the model if it exists
            if os.path.exists(resume_model):
                print(f"🔄 Loading checkpoint: {resume_model}")
                # We'll need to create the agent first, then load
                from environments.salp_snake_env import SalpSnakeEnv
                from agents.sac_agent import SACAgent
                
                env = SalpSnakeEnv(render_mode=None, **config.environment.params)
                agent = SACAgent(
                    config=config.agent,
                    obs_dim=env.observation_space.shape[0],
                    action_dim=env.action_space.shape[0],
                    action_space=env.action_space
                )
                agent.load(resume_model)
                trainer.agent = agent
                env.close()
    else:
        # Fresh training
        if continuous:
            trainer = ContinuousTrainer(config)
        else:
            trainer = Trainer(config)
    
    trainer.train()


def _get_available_models():
    """Get list of available trained models."""
    models = []
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        return models
    
    for exp_name in os.listdir(models_dir):
        exp_path = os.path.join(models_dir, exp_name)
        if os.path.isdir(exp_path):
            best_model_path = os.path.join(exp_path, "best_model.pth")
            if os.path.exists(best_model_path):
                size_mb = os.path.getsize(best_model_path) / (1024 * 1024)
                models.append((exp_name, best_model_path, size_mb))
    
    return sorted(models)


def evaluate_agent(model_path: str, episodes: int = 5):
    """Evaluate a trained SALP agent."""
    print(f"=== Evaluating SALP Agent ===")
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print()
    
    # Get configuration (only sac_snake is available)
    config = get_sac_snake_config()
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Create evaluation runner
    runner = EvaluationRunner(config, model_path)
    runner.run_episodes(episodes, deterministic=True)


def list_models(experiment_name: str = "salp_snake_sac"):
    """List available trained models."""
    print(f"=== Available Models for {experiment_name} ===")
    
    model_dir = os.path.join("models", experiment_name)
    
    if not os.path.exists(model_dir):
        print(f"No models found in {model_dir}")
        return
    
    models = [f for f in os.listdir(model_dir) if f.endswith('.pth') and not f.endswith('_training_state.pth')]
    
    if not models:
        print("No model files found")
        return
    
    print("Available models:")
    for model in sorted(models):
        model_path = os.path.join(model_dir, model)
        size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"  - {model} ({size:.1f} MB)")
    
    print(f"\nTo evaluate a model, use:")
    print(f"python demo.py evaluate --model models/{experiment_name}/best_model.pth")


def quick_test():
    """Quick test to verify the implementation works."""
    print("=== Quick Implementation Test ===")
    
    try:
        # Test environment creation
        print("Testing environment creation...")
        env = SalpSnakeEnv(render_mode=None, num_food_items=3)
        obs, info = env.reset()
        print(f"✓ Environment created. Observation shape: {obs.shape}")
        
        # Test agent creation
        print("Testing agent creation...")
        config = get_sac_snake_config()
        from agents.sac_agent import SACAgent
        
        agent = SACAgent(
            config=config.agent,
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            action_space=env.action_space
        )
        print(f"✓ Agent created. Device: {agent.device}")
        
        # Test action selection
        print("Testing action selection...")
        action = agent.select_action(obs, deterministic=True)
        print(f"✓ Action selected: {action}")
        
        # Test environment step
        print("Testing environment step...")
        next_obs, reward, done, truncated, info = env.step(action)
        print(f"✓ Environment step completed. Reward: {reward:.3f}")
        
        # Test replay buffer
        print("Testing replay buffer...")
        from core.base_agent import ReplayBuffer
        buffer = ReplayBuffer(1000, obs.shape[0], action.shape[0])
        buffer.add(obs, action, reward, next_obs, done)
        print(f"✓ Replay buffer working. Size: {len(buffer)}")
        
        env.close()
        print("\n✅ All tests passed! Implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main demo function with command line interface."""
    parser = argparse.ArgumentParser(description="SALP RL Demo Script")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Environment demo
    env_parser = subparsers.add_parser('env', help='Demo the environment with manual control')
    
    # Training
    train_parser = subparsers.add_parser('train', help='Train a SALP agent')
    train_parser.add_argument('--episodes', type=int, help='Number of episodes to train')
    train_parser.add_argument('--continuous', action='store_true', help='Continuous visual display with real-time model updates')
    train_parser.add_argument('--from-scratch', action='store_true', help='Start training from scratch (skip interactive prompt)')
    train_parser.add_argument('--model-name', type=str, help='Name for new experiment (only with --from-scratch)')
    train_parser.add_argument('--resume-model', type=str, help='Path to model to resume from (skip interactive prompt)')
    
    # Evaluation
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained agent')
    eval_parser.add_argument('--model', required=True, help='Path to model file')
    eval_parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to evaluate')
    
    # List models
    list_parser = subparsers.add_parser('list', help='List available trained models')
    list_parser.add_argument('--experiment', default='salp_snake_sac', help='Experiment name')
    
    # Quick test
    test_parser = subparsers.add_parser('test', help='Quick test of the implementation')
    
    args = parser.parse_args()
    
    if args.command == 'env':
        demo_environment()
    elif args.command == 'train':
        train_agent(
            episodes=args.episodes, 
            continuous=getattr(args, 'continuous', False),
            from_scratch=getattr(args, 'from_scratch', False),
            model_name=getattr(args, 'model_name', None),
            resume_model=getattr(args, 'resume_model', None)
        )
    elif args.command == 'evaluate':
        evaluate_agent(args.model, episodes=args.episodes)
    elif args.command == 'list':
        list_models(args.experiment)
    elif args.command == 'test':
        quick_test()
    else:
        # No command specified, show help and run quick test
        parser.print_help()
        print("\n" + "="*50)
        quick_test()


if __name__ == "__main__":
    main()
