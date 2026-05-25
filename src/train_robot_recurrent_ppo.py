"""
RecurrentPPO Training Script for SALP Robot Navigation

This script trains a RecurrentPPO agent with LSTM memory for underwater robot navigation.
The LSTM enables the agent to learn temporal patterns in the breathing cycle dynamics.

Comparison with SAC (train_robot.py):
- RecurrentPPO: On-policy, memory-enabled (LSTM)
- SAC: Off-policy, memoryless

Usage:
    python train_robot_recurrent_ppo.py

Monitor training:
    tensorboard --logdir ../experiments/v5_recurrent/logs
"""

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle
from tensorboard_callback import DetailedMetricsCallback
import numpy as np
import os

def make_env():
    """Create and return the SalpRobotEnv environment.

    Uses Dongsheng's real-robot-calibrated parameters (origin/reward_shaping).
    """
    nozzle = Nozzle(length1=0.052, length2=0.038, length3=0.050,
                    area=np.pi * 0.01 ** 2, mass=0.428,
                    radius=0.1, inner_radius=0.022)
    nozzle.set_angles(angle1=0.0, angle2=0.0)

    robot = Robot(dry_mass=0.738, init_length=0.26, init_width=0.135,
                  max_contraction=0.04, nozzle=nozzle)
    robot.set_environment(density=1000)
    robot.enable_dynamic_randomization()
    robot.enable_disturbances()

    env = SalpRobotEnv(render_mode=None, robot=robot)
    return env

if __name__ == "__main__":
    
    print("="*70)
    print("RECURRENT PPO TRAINING - VERSION 5")
    print("="*70)
    print("\n🧠 Algorithm: RecurrentPPO with LSTM Memory")
    print("   - On-policy learning (requires more samples than SAC)")
    print("   - LSTM captures temporal patterns in breathing cycles")
    print("   - Better handling of sequential decision making")
    print("\n📊 Same environment and reward as SAC v4")
    print("="*70 + "\n")

    # Create experiment directories
    exp_dir = "../experiments/v5_recurrent"
    os.makedirs(f"{exp_dir}/logs", exist_ok=True)
    os.makedirs(f"{exp_dir}/models", exist_ok=True)
    os.makedirs(f"{exp_dir}/models/best_model", exist_ok=True)
    os.makedirs(f"{exp_dir}/recordings", exist_ok=True)
    print(f"✅ Experiment directory created: {exp_dir}\n")

    # Create vectorized environments
    # NOTE: RecurrentPPO works best with DummyVecEnv (not SubprocVecEnv)
    # because LSTM states need to be managed carefully
    num_cpu = 4  # Reduced from 8 for stability with RecurrentPPO
    print(f"Creating {num_cpu} parallel environments...")
    vec_env = make_vec_env(make_env, n_envs=num_cpu, vec_env_cls=DummyVecEnv)
    print("✅ Environments created\n")

    # Create separate evaluation environment
    eval_env = make_env()
    print("✅ Evaluation environment created\n")

    # Initialize RecurrentPPO model
    print("Initializing RecurrentPPO model...")
    print("\n📋 Hyperparameters:")
    print("   - Policy: MlpLstmPolicy (with LSTM memory)")
    print("   - Learning rate: 3e-4")
    print("   - n_steps: 2048 (steps per env before update)")
    print("   - batch_size: 64")
    print("   - n_epochs: 10 (optimization epochs per update)")
    print("   - LSTM hidden size: 256")
    print("   - Gamma: 0.99 (discount factor)")
    print("   - GAE lambda: 0.95")
    print("   - Clip range: 0.2")
    
    model = RecurrentPPO(
        "MlpLstmPolicy",  # Policy with LSTM
        vec_env,
        verbose=1,
        tensorboard_log=f'{exp_dir}/logs',
        learning_rate=3e-4,
        n_steps=2048,  # Steps per environment before update
        batch_size=64,  # Minibatch size for optimization
        n_epochs=10,  # Number of epochs when optimizing the surrogate loss
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # Factor for trade-off of bias vs variance for GAE
        clip_range=0.2,  # Clipping parameter for PPO
        ent_coef=0.0,  # Entropy coefficient for exploration
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Max gradient norm for clipping
        policy_kwargs=dict(
            lstm_hidden_size=256,  # LSTM hidden state size
            n_lstm_layers=1,  # Number of LSTM layers
            enable_critic_lstm=True,  # Use LSTM for critic too
            shared_lstm=False,  # Separate LSTMs for actor and critic
        ),
        device="auto"
    )
    print("\n✅ RecurrentPPO model initialized!\n")

    # Setup Callbacks
    print("Setting up callbacks...")
    
    # Detailed metrics callback (logs custom metrics every 1000 steps)
    metrics_callback = DetailedMetricsCallback(log_freq=1000, verbose=1)
    
    # Evaluation callback (saves best model)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'{exp_dir}/models/best_model/',
        log_path=f'{exp_dir}/logs/eval_logs/',
        eval_freq=10000,  # Evaluate every 10k steps (more frequent than SAC)
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )
    
    # Checkpoint callback (save model periodically)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # Save every 50k steps
        save_path=f'{exp_dir}/models/',
        name_prefix="salp_robot_v5_recurrent",
        verbose=1
    )
    
    # Combine callbacks
    callback_list = CallbackList([metrics_callback, eval_callback, checkpoint_callback])
    
    print("✅ Callbacks configured:")
    print("   - Detailed metrics logged every 1000 steps")
    print("   - Evaluation every 10000 steps")
    print("   - Checkpoints every 50000 steps\n")

    # Train
    print("="*70)
    print("STARTING TRAINING - 500k timesteps")
    print("="*70)
    print("📊 Monitor progress: tensorboard --logdir ../experiments/v5_recurrent/logs")
    print("📖 See METRICS.md for metric documentation")
    print("\n⚠️  NOTE: RecurrentPPO needs more samples than SAC!")
    print("   - SAC v4: 200k timesteps")
    print("   - RecurrentPPO v5: 500k timesteps (2.5x more)")
    print("   - This is expected for on-policy algorithms")
    print("="*70 + "\n")
    
    model.learn(
        total_timesteps=500000,  # More than SAC due to on-policy nature
        callback=callback_list,
        tb_log_name="salp_robot_v5_recurrent",
        progress_bar=True
    )

    # Save Final Model
    print("\n✅ Training complete!")
    model.save(f"{exp_dir}/models/salp_robot_v5_recurrent_final")
    print(f"💾 Final model saved: {exp_dir}/models/salp_robot_v5_recurrent_final")
    print(f"💾 Best model saved: {exp_dir}/models/best_model/")
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Algorithm: RecurrentPPO with LSTM")
    print(f"Total timesteps: 500,000")
    print(f"Parallel envs: {num_cpu}")
    print(f"Logs: {exp_dir}/logs")
    print(f"Models: {exp_dir}/models")
    print("="*70)
