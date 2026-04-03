import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle
from tensorboard_callback import DetailedMetricsCallback
import numpy as np

def make_env():
    # Create and return the SalpRobotEnv environment (no obstacles for clean comparison)
    nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.00016, mass=1.0)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15,
                    max_contraction=0.06, nozzle=nozzle)
    robot.nozzle.set_angles(angle1=0.0, angle2=0.0)  # set nozzle angles
    robot.set_environment(density=1000)  # water density in kg/m^3

    env = SalpRobotEnv(render_mode=None, robot=robot, num_obstacles=0)

    return env

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train PPO agent for salp robot")
    parser.add_argument("--version", type=str, default="v6_ppo",
                        help="Experiment version label (e.g. v6_ppo). Controls all output paths.")
    parser.add_argument("--warm-start", type=str, default=None,
                        help="Path to a .zip model to warm-start from. If omitted, trains from scratch.")
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
        model = PPO.load(args.warm_start, env=vec_env)
        print(f"✅ Warm-start model loaded successfully!")
    else:
        print(f"TRAINING {version.upper()} - Training from scratch")
        print("="*70)
        model = PPO(
            "MlpPolicy", vec_env, verbose=1,
            tensorboard_log=f'../experiments/{version}/logs',
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="auto"
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
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f'../experiments/{version}/models/',
        name_prefix=f"salp_robot_ppo_{version}",
        verbose=1
    )

    callback_list = CallbackList([metrics_callback, eval_callback, checkpoint_callback])

    print("✅ Callbacks configured:")
    print("   - Detailed metrics logged every 1000 steps")
    print("   - Evaluation every 10000 steps")
    print("   - Checkpoints every 50000 steps")

    print("\n" + "="*70)
    print("STARTING TRAINING - 500k timesteps")
    print("="*70)
    print(f"📊 Monitor progress: tensorboard --logdir ../experiments/{version}/logs")
    print("📖 See METRICS.md for metric documentation")
    print("="*70 + "\n")

    model.learn(
        total_timesteps=500000,
        callback=callback_list,
        tb_log_name=f"salp_robot_ppo_{version}",
        progress_bar=True
    )

    model.save(f"../experiments/{version}/models/salp_robot_ppo_{version}_final")
    print(f"\n✅ Training complete!")
    print(f"💾 Final model saved: salp_robot_ppo_{version}_final")
    print(f"💾 Best model saved: ../experiments/{version}/models/best_model/")
