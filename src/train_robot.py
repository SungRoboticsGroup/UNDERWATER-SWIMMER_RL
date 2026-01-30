from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement, CallbackList
from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle
import numpy as np
import time
import os

def make_env():
    # Create and return the SalpRobotEnv environment
    nozzle = Nozzle(
        length1=0.05,
        length2=0.05,
        length3=0.05,
        area=0.00016,
        mass=1.0)
    robot = Robot(
        dry_mass=1.0,
        init_length=0.3,
        init_width=0.15,
        max_contraction=0.06,
        nozzle=nozzle)
    robot.nozzle.set_angles(angle1=0.0, angle2=0.0)  # set nozzle angles
    robot.set_environment(density=1000)  # water density in kg/m^3

    def _make():
        return SalpRobotEnv(render_mode=None, robot=robot)
    return _make

if __name__ == "__main__":
    # 1. Create Vectorized Environment (4 parallel environments)
    vec_env = make_vec_env(make_env(), n_envs=4)
    print("Created vectorized environment with 4 parallel workers")

    # 2. Sanity Check (CRITICAL)
    # This checks if your observation/action spaces match what the step() function returns.
    # It will crash here if you made a mistake, saving you hours of debugging.
    print("\nValidating environment structure...")
    nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.00016, mass=1.0)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, max_contraction=0.06, nozzle=nozzle)
    robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
    robot.set_environment(density=1000)
    test_env = SalpRobotEnv(render_mode=None, robot=robot)
    check_env(test_env)
    print("✅ Environment structure validated!\n")

    # 3. Define the Model (SAC)
    # Option 1: Continue training from Dongsheng's model
    # Option 2: Train from scratch
    
    # Set this to the path of the model you want to continue from, or None to train from scratch
    CONTINUE_FROM_MODEL = "../salp_robot_finalv2.zip"  # Change to None for fresh training
    
    if CONTINUE_FROM_MODEL and os.path.exists(CONTINUE_FROM_MODEL):
        print(f"🔄 Loading existing model from: {CONTINUE_FROM_MODEL}")
        print("   Continuing training from checkpoint...")
        model = SAC.load(CONTINUE_FROM_MODEL, env=vec_env, device="cpu")
        print("   ✅ Model loaded successfully!\n")
    else:
        if CONTINUE_FROM_MODEL:
            print(f"⚠️  Model file not found: {CONTINUE_FROM_MODEL}")
            print("   Starting fresh training instead...\n")
        else:
            print("🆕 Training from scratch\n")
        
        model = SAC(
            "MlpPolicy",           # Use standard Dense Neural Network
            vec_env,
            verbose=1,
            tensorboard_log="./sac_salp_robot_tensorboard/",

            # --- Tuning for Robotics ---
            learning_rate=3e-4,
            buffer_size=100000,    # Big memory for off-policy
            batch_size=512,        # Mini-batch size
            ent_coef='auto',       # Automatically adjust exploration (Temperature)
            gamma=0.99,            # Discount factor
            tau=0.005,             # Polyak averaging (Soft update)
            device="cpu"           # Using CPU (change to "cuda" if you have GPU)
        )

    # 4. Setup Detailed Logging Callback
    class DetailedLoggingCallback(BaseCallback):
        """
        Custom callback for detailed training progress logging.
        Shows timesteps, episodes, rewards, and saves checkpoints.
        """
        def __init__(self, check_freq, save_path, name_prefix, total_timesteps):
            super().__init__()
            self.check_freq = check_freq
            self.save_path = save_path
            self.name_prefix = name_prefix
            self.total_timesteps = total_timesteps
            self.episode_count = 0
            self.start_time = None

        def _on_training_start(self):
            self.start_time = time.time()
            print("\n" + "="*70)
            print("🚀 TRAINING STARTED")
            print("="*70)
            print(f"Total timesteps: {self.total_timesteps:,}")
            print(f"Progress updates: every {self.check_freq:,} steps")
            print(f"Parallel environments: {vec_env.num_envs}")
            print(f"Only best model will be saved (no regular checkpoints)")
            print("="*70 + "\n")

        def _on_step(self):
            # Print progress every 1000 steps
            if self.n_calls % 1000 == 0 and self.n_calls > 0:
                elapsed = time.time() - self.start_time
                progress = (self.n_calls / self.total_timesteps) * 100
                steps_per_sec = self.n_calls / elapsed if elapsed > 0 else 0
                eta_seconds = (self.total_timesteps - self.n_calls) / steps_per_sec if steps_per_sec > 0 else 0

                print(f"\n{'='*70}")
                print(f"⏱️  Timestep: {self.n_calls:,} / {self.total_timesteps:,} ({progress:.1f}%)")
                print(f"📊 Speed: {steps_per_sec:.1f} steps/sec")
                print(f"⏳ Elapsed: {elapsed/60:.1f} min | ETA: {eta_seconds/60:.1f} min")

                # Print recent episode stats if available
                if len(self.model.ep_info_buffer) > 0:
                    recent_eps = list(self.model.ep_info_buffer)[-10:]
                    rewards = [ep['r'] for ep in recent_eps]
                    lengths = [ep['l'] for ep in recent_eps]

                    print(f"\n📈 Last {len(recent_eps)} Episodes:")
                    print(f"   Reward: {np.mean(rewards):7.2f} ± {np.std(rewards):6.2f}")
                    print(f"   Length: {np.mean(lengths):7.1f} steps")
                    print(f"   Best:   {np.max(rewards):7.2f}")
                    print(f"   Worst:  {np.min(rewards):7.2f}")

                print("="*70)

            return True

        def _on_training_end(self):
            elapsed = time.time() - self.start_time
            print("\n" + "="*70)
            print("✅ TRAINING COMPLETE!")
            print("="*70)
            print(f"Total time: {elapsed/3600:.2f} hours")
            print(f"Final timesteps: {self.n_calls:,}")
            if len(self.model.ep_info_buffer) > 0:
                all_rewards = [ep['r'] for ep in self.model.ep_info_buffer]
                print(f"Average reward: {np.mean(all_rewards):.2f}")
            print("="*70 + "\n")

    # Create logging callback
    logging_callback = DetailedLoggingCallback(
        check_freq=1000,
        save_path='./logs/',
        name_prefix='salp_robot_model',
        total_timesteps=500000
    )

    # Create evaluation environment (separate from training)
    eval_env = make_vec_env(make_env(), n_envs=1)

    # Setup early stopping: stop if no improvement for N evaluations
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=30,  # Stop if no improvement for 30 evals
        min_evals=10,                 # Need at least 10 evals before considering stopping
        verbose=1
    )

    # Evaluation callback: evaluates model every N steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/best_model/',
        log_path='./logs/eval/',
        eval_freq=1000,               # Evaluate every 1000 steps
        n_eval_episodes=5,            # Run 5 episodes per evaluation
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback,  # Check for early stopping after each eval
        verbose=1
    )

    # Combine callbacks
    callback = CallbackList([logging_callback, eval_callback])

    # 5. Train
    print("Starting training...")
    print("📊 Early stopping enabled: will stop if no improvement for 30 evaluations")
    print("🎯 Best model will be saved to: ./logs/best_model/\n")

    model.learn(
        total_timesteps=500000,  # Run for 500k steps
        callback=callback,
        reset_num_timesteps=True,
        tb_log_name="salp_robot_training"
    )

    # 6. Save Final Model (with timestamp to avoid overwriting)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_model_name = f"salp_robot_trained_{timestamp}"
    model.save(final_model_name)
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Final model saved to: {final_model_name}.zip")
    print(f"Best model saved to: ./logs/best_model/best_model.zip")
    print(f"Original model (salp_robot_finalv2.zip) unchanged")
    print(f"{'='*70}\n")
