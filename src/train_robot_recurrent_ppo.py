"""
RecurrentPPO training for the SALP robot, structured to match train_robot_ppo.py:
argparse-controlled `--version`, `--warm-start`, `--timesteps`, `--n-envs`, and
auto fine-tune-mode hparams when warm-starting from a BC'd RecurrentPPO zip.

Env mirrors train_robot_reward_shaping.py (Dongsheng's calibrated params +
domain randomization + disturbances). MlpLstmPolicy with separate actor/critic
LSTMs, hidden size 256.

Run examples:
    # From scratch
    python train_robot_recurrent_ppo.py --version recurrent_v1 --timesteps 2000000

    # From a recurrent-BC warm-start (fine-tune hparams auto-enabled)
    python train_robot_recurrent_ppo.py \
        --version recurrent_ppo_bc_v1 \
        --warm-start ../experiments/recurrent_bc_v1/models/bc_recurrent_ppo.zip \
        --timesteps 2000000
"""

import argparse
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.utils import get_schedule_fn

from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle
from tensorboard_callback import DetailedMetricsCallback


# Conservative RecurrentPPO hparams for fine-tuning a BC'd / pretrained policy.
# Same shape as train_robot_ppo.py's FINE_TUNE_HPARAMS; rationale carries over:
# slower drift away from the cloned actor, no entropy injection, tighter clip,
# target_kl early-stops runaway updates, fewer SGD passes per rollout.
FINE_TUNE_HPARAMS = dict(
    learning_rate=1e-4,
    ent_coef=0.0,
    clip_range=0.1,
    target_kl=0.05,
    n_epochs=5,
)


def make_env():
    nozzle = Nozzle(length1=0.052, length2=0.038, length3=0.050,
                    area=np.pi * 0.01 ** 2, mass=0.428,
                    radius=0.1, inner_radius=0.022)
    nozzle.set_angles(angle1=0.0, angle2=0.0)

    robot = Robot(dry_mass=0.738, init_length=0.26, init_width=0.135,
                  max_contraction=0.04, nozzle=nozzle)
    robot.set_environment(density=1000)
    robot.enable_dynamic_randomization()
    robot.enable_disturbances()

    return SalpRobotEnv(render_mode=None, robot=robot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RecurrentPPO agent for salp robot")
    parser.add_argument("--version", type=str, default="recurrent_v1",
                        help="Experiment version label. Controls all output paths.")
    parser.add_argument("--warm-start", type=str, default=None,
                        help="Path to a .zip RecurrentPPO checkpoint to warm-start from.")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Parallel envs. RecurrentPPO uses DummyVecEnv (LSTM state "
                             "threading is fragile under subprocs).")
    parser.add_argument("--fine-tune", dest="fine_tune", action="store_true",
                        help="Apply conservative fine-tune-mode hparams "
                             "(default: auto-on when --warm-start is given).")
    parser.add_argument("--no-fine-tune", dest="fine_tune", action="store_false",
                        help="Disable fine-tune mode even when warm-starting.")
    parser.set_defaults(fine_tune=None)  # None -> auto = warm_start is not None
    args = parser.parse_args()
    if args.fine_tune is None:
        args.fine_tune = args.warm_start is not None

    version = args.version
    log_dir   = f"../experiments/{version}/logs"
    model_dir = f"../experiments/{version}/models"

    vec_env  = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=DummyVecEnv)
    eval_env = make_env()
    print("Environment is valid!")

    print("=" * 70)
    if args.warm_start:
        print(f"TRAINING {version.upper()} - Warm-start from: {args.warm_start}")
        print("=" * 70)
        model = RecurrentPPO.load(args.warm_start, env=vec_env)
        print("✅ Warm-start RecurrentPPO loaded successfully!")
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
        print("=" * 70)
        from_scratch_hparams = dict(
            learning_rate=3e-4,
            n_epochs=10,
            clip_range=0.2,
            ent_coef=0.0,
            target_kl=None,
        )
        if args.fine_tune:
            from_scratch_hparams.update(FINE_TUNE_HPARAMS)
        model = RecurrentPPO(
            "MlpLstmPolicy", vec_env, verbose=1,
            tensorboard_log=log_dir,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                lstm_hidden_size=256,
                n_lstm_layers=1,
                enable_critic_lstm=True,
                shared_lstm=False,
            ),
            device="auto",
            **from_scratch_hparams,
        )

    # Set tb log dir (needed for warm-start case where model was already constructed)
    model.tensorboard_log = log_dir

    print("\nSetting up callbacks...")
    metrics_callback = DetailedMetricsCallback(log_freq=1000, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best_model/",
        log_path=f"{log_dir}/eval_logs/",
        eval_freq=10000,
        deterministic=False,
        render=False,
        n_eval_episodes=5,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=model_dir,
        name_prefix=f"salp_robot_recurrent_ppo_{version}",
        verbose=1,
    )
    callback_list = CallbackList([metrics_callback, eval_callback, checkpoint_callback])

    print("✅ Callbacks configured:")
    print("   - Detailed metrics logged every 1000 steps")
    print("   - Evaluation every 10000 steps")
    print("   - Checkpoints every 50000 steps")

    print("\n" + "=" * 70)
    print(f"STARTING TRAINING - {args.timesteps:,} timesteps")
    print("=" * 70)
    print(f"📊 Monitor progress: tensorboard --logdir {log_dir}")
    print("=" * 70 + "\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback_list,
        tb_log_name=f"salp_robot_recurrent_ppo_{version}",
        progress_bar=True,
    )

    model.save(f"{model_dir}/salp_robot_recurrent_ppo_{version}_final")
    print(f"\n✅ Training complete!")
    print(f"💾 Final model saved: salp_robot_recurrent_ppo_{version}_final")
    print(f"💾 Best model saved: {model_dir}/best_model/")
