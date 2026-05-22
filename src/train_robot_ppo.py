import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.utils import get_schedule_fn
from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle
from tensorboard_callback import DetailedMetricsCallback
import numpy as np


# Conservative PPO hparams for fine-tuning a BC'd / pretrained policy.
# Rationale (vs from-scratch defaults below):
#   - lr 1e-4 vs 3e-4: slower drift away from the cloned actor
#   - ent_coef 0.0 vs 0.01: don't fight the BC peak by injecting entropy
#   - clip_range 0.1 vs 0.2: tighter ratio bound for an already-confident policy
#   - target_kl 0.05: SB3 early-stops the inner SGD epoch if approx_kl > target_kl,
#                     prevents the runaway-update collapse we saw in ppo_bc_v1
#   - n_epochs 5 vs 10: half as many SGD passes per rollout, less risk of overshoot
FINE_TUNE_HPARAMS = dict(
    learning_rate=1e-4,
    ent_coef=0.0,
    clip_range=0.1,
    target_kl=0.05,
    n_epochs=5,
)

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

    parser = argparse.ArgumentParser(description="Train PPO agent for salp robot")
    parser.add_argument("--version", type=str, default="v6_ppo",
                        help="Experiment version label (e.g. v6_ppo). Controls all output paths.")
    parser.add_argument("--warm-start", type=str, default=None,
                        help="Path to a .zip model to warm-start from. If omitted, trains from scratch.")
    parser.add_argument("--timesteps", type=int, default=2000000,
                        help="Total environment timesteps for model.learn().")
    parser.add_argument("--fine-tune", dest="fine_tune", action="store_true",
                        help="Apply conservative fine-tune-mode PPO hparams "
                             "(default: auto-on when --warm-start is given).")
    parser.add_argument("--no-fine-tune", dest="fine_tune", action="store_false",
                        help="Disable fine-tune mode even when warm-starting.")
    parser.set_defaults(fine_tune=None)  # None -> auto = warm_start is not None
    args = parser.parse_args()
    if args.fine_tune is None:
        args.fine_tune = args.warm_start is not None

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
        if args.fine_tune:
            print(f"\nApplying fine-tune hparam overrides: {FINE_TUNE_HPARAMS}")
            model.learning_rate = FINE_TUNE_HPARAMS["learning_rate"]
            model.lr_schedule   = get_schedule_fn(FINE_TUNE_HPARAMS["learning_rate"])
            model.clip_range    = get_schedule_fn(FINE_TUNE_HPARAMS["clip_range"])
            model.ent_coef      = FINE_TUNE_HPARAMS["ent_coef"]
            model.target_kl     = FINE_TUNE_HPARAMS["target_kl"]
            model.n_epochs      = FINE_TUNE_HPARAMS["n_epochs"]
            for g in model.policy.optimizer.param_groups:
                g["lr"] = FINE_TUNE_HPARAMS["learning_rate"]
    else:
        print(f"TRAINING {version.upper()} - Training from scratch"
              + (" (fine-tune hparams)" if args.fine_tune else ""))
        print("="*70)
        from_scratch_hparams = dict(
            learning_rate=3e-4,
            n_epochs=10,
            clip_range=0.2,
            ent_coef=0.01,
            target_kl=None,
        )
        if args.fine_tune:
            from_scratch_hparams.update(FINE_TUNE_HPARAMS)
        model = PPO(
            "MlpPolicy", vec_env, verbose=1,
            tensorboard_log=f'../experiments/{version}/logs',
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="auto",
            **from_scratch_hparams,
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
        deterministic=False,
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
    print(f"STARTING TRAINING - {args.timesteps:,} timesteps")
    print("="*70)
    print(f"📊 Monitor progress: tensorboard --logdir ../experiments/{version}/logs")
    print("📖 See METRICS.md for metric documentation")
    print("="*70 + "\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback_list,
        tb_log_name=f"salp_robot_ppo_{version}",
        progress_bar=True
    )

    model.save(f"../experiments/{version}/models/salp_robot_ppo_{version}_final")
    print(f"\n✅ Training complete!")
    print(f"💾 Final model saved: salp_robot_ppo_{version}_final")
    print(f"💾 Best model saved: ../experiments/{version}/models/best_model/")
