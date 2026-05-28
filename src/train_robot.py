import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle
from tensorboard_callback import DetailedMetricsCallback
import numpy as np

def make_env():
    # Dongsheng's real-robot-calibrated parameters (origin/reward_shaping).
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

    parser = argparse.ArgumentParser(description="Train SAC agent for salp robot")
    parser.add_argument("--version", type=str, default="v6",
                        help="Experiment version label (e.g. v6). Controls all output paths.")
    parser.add_argument("--warm-start", type=str, default=None,
                        help="Path to a .zip model to warm-start from. If omitted, trains from scratch.")
    parser.add_argument("--timesteps", type=int, default=500000,
                        help="Total environment timesteps for model.learn().")
    args = parser.parse_args()

    version = args.version

    num_cpu = 8
    vec_env = make_vec_env(make_env, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)

    # Create separate evaluation environment
    eval_env = make_env()
    print("Environment is valid!")

    print("="*70)
    if args.warm_start:
        print(f"TRAINING {version.upper()} - Warm-start from: {args.warm_start}")
        print("="*70)
        print(f"\nLoading model to warm-start {version} training...")
        model = SAC.load(args.warm_start, env=vec_env)
        print(f"✅ Warm-start model loaded successfully!")
    else:
        print(f"TRAINING {version.upper()} - Training from scratch")
        print("="*70)
        model = SAC(
            "MlpPolicy", vec_env, verbose=1,
            tensorboard_log=f'../experiments/{version}/logs',
            learning_rate=3e-4, buffer_size=100000,
            learning_starts=1000, batch_size=256,
            tau=0.005, gamma=0.99,
            train_freq=1, gradient_steps=1, device="auto"
        )

    # Set tensorboard log dir (needed for warm-start case where model was already created)
    model.tensorboard_log = f'../experiments/{version}/logs'

    # Setup Callbacks with Detailed Metrics
    print("\nSetting up callbacks...")

    metrics_callback = DetailedMetricsCallback(log_freq=1000, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'../experiments/{version}/models/best_model/',
        log_path=f'../experiments/{version}/logs/eval_logs/',
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f'../experiments/{version}/models/',
        name_prefix=f"salp_robot_{version}",
        verbose=1
    )

    callback_list = CallbackList([metrics_callback, eval_callback, checkpoint_callback])

    print("✅ Callbacks configured:")
    print("   - Detailed metrics logged every 1000 steps")
    print("   - Evaluation every 5000 steps")
    print("   - Checkpoints every 50000 steps")

    print("\n" + "="*70)
    print(f"STARTING TRAINING - {args.timesteps:,} timesteps")
    print("="*70)
    print(f"📊 Monitor progress: tensorboard --logdir ../experiments/{version}/logs")
    print("📖 See METRICS.md for metric documentation")
    print("="*70 + "\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback_list,
        tb_log_name=f"salp_robot_{version}",
        progress_bar=True
    )

    model.save(f"../experiments/{version}/models/salp_robot_{version}_final")
    print(f"\n✅ Training complete!")
    print(f"💾 Final model saved: salp_robot_{version}_final")
    print(f"💾 Best model saved: ../experiments/{version}/models/best_model/")
