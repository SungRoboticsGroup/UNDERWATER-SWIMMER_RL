"""
SAC training script using Dongsheng's calibrated reward_shaping setup,
wrapped in this repo's argparse + `../experiments/{version}/...` scaffolding.

Source for params and callbacks: origin/reward_shaping:src/salp/environments/train_robot.py
Physics, env, and reward live in src/{robot,dynamics,geometry,salp_robot_env}.py
(replaced wholesale from reward_shaping in this PR).
"""

import argparse
import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from salp_robot_env import SalpRobotEnv
from robot import Nozzle, Robot


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "vec_normalize", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            self.training_env.save(path)
            if self.verbose > 0:
                print(f"Saved VecNormalize to {path}")
        return True


class EpisodeComponentCallback(BaseCallback):
    """Logs per-component episode rewards (`rewards/foo` -> `episode_rewards/foo`)."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_episode_rewards: dict = {}
        self.num_envs: int = 0

    def _init_callback(self) -> None:
        self.num_envs = self.training_env.num_envs
        self.current_episode_rewards = {i: {} for i in range(self.num_envs)}

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i, info in enumerate(infos):
            for key, value in info.items():
                if key.startswith("rewards/"):
                    self.current_episode_rewards[i][key] = (
                        self.current_episode_rewards[i].get(key, 0.0) + value
                    )
            if dones[i]:
                for key, value in self.current_episode_rewards[i].items():
                    log_key = key.replace("rewards/", "episode_rewards/")
                    self.logger.record(log_key, value)
                self.current_episode_rewards[i] = {}
        return True


# Dongsheng's calibrated parameters (origin/reward_shaping).
_NOZZLE_PARAMS = dict(
    length1=0.052,
    length2=0.038,
    length3=0.050,
    area=np.pi * 0.01 ** 2,
    mass=0.428,
    radius=0.1,
    inner_radius=0.022,
)
_ROBOT_PARAMS = dict(
    dry_mass=0.738,
    init_length=0.26,
    init_width=0.135,
    max_contraction=0.04,
)


def make_env():
    nozzle = Nozzle(**_NOZZLE_PARAMS)
    nozzle.set_angles(angle1=0.0, angle2=0.0)

    robot = Robot(**_ROBOT_PARAMS, nozzle=nozzle)
    robot.set_environment(density=1000)
    robot.enable_dynamic_randomization()
    robot.enable_disturbances()

    env = SalpRobotEnv(render_mode=None, robot=robot)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Dongsheng's reward_shaping SAC training")
    parser.add_argument("--version", type=str, default="rs_repro",
                        help="Experiment version label. Controls all output paths.")
    parser.add_argument("--warm-start", type=str, default=None,
                        help="Path to a .zip checkpoint to resume from.")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--save-freq", type=int, default=5_000)
    parser.add_argument("--tb-log-name", type=str, default="salp_robot_body_frame_sideslip",
                        help="TB log subdir name. Defaults to Dongsheng's run name for tb comparison.")
    args = parser.parse_args()

    version = args.version
    log_dir = f"../experiments/{version}/logs"
    model_dir = f"../experiments/{version}/models"
    os.makedirs(model_dir, exist_ok=True)

    vec_env = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)

    if args.warm_start:
        print(f"Warm-start from: {args.warm_start}")
        model = SAC.load(args.warm_start, env=vec_env)
        model.tensorboard_log = log_dir
    else:
        model = SAC(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=512,
            ent_coef="auto",
            gamma=0.99,
            tau=0.005,
            device="auto",
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=model_dir,
        name_prefix=f"salp_robot_{version}",
    )
    episode_callback = EpisodeComponentCallback()
    callbacks = CallbackList([checkpoint_callback, episode_callback])

    print(f"Training {version} for {args.timesteps:,} timesteps")
    print(f"TB: tensorboard --logdir {log_dir}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=(args.warm_start is None),
        tb_log_name=args.tb_log_name,
        progress_bar=True,
    )

    final_path = os.path.join(model_dir, f"salp_robot_{version}_final")
    model.save(final_path)
    print(f"Saved final model: {final_path}")
