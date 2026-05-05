import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from robot import Nozzle, Robot
from salp_robot_env import SalpRobotEnv

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class SaveVecNormalizeCallback(BaseCallback):
    """Saves VecNormalize statistics at the same frequency as CheckpointCallback."""

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
    """
    Accumulates per-step reward components and logs their episode sum to
    TensorBoard, matching the scale of 'ep_rew_mean'.
    """

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

# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

# Robot physical parameters — DO NOT CHANGE
_NOZZLE_PARAMS = dict(
    length1=0.052,
    length2=0.038,
    length3=0.050,
    area=np.pi * 0.01**2,
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
    # env.enable_action_randomization()
    # env.enable_latency()
    # env.enable_observation_randomization()

    return env

# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # 1. Build vectorised environment
    num_cpu = 8
    vec_env = make_vec_env(make_env, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)

    # Optional: observation / reward normalisation
    # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    # vec_env = VecNormalize.load("vec_final_vecnormalize.pkl", vec_env)
    # vec_env.training = True
    # vec_env.norm_reward = True

    # 2. Load or create model
    model = SAC.load("./salp_robot_final_calibrated", env=vec_env)

    # To train from scratch instead, comment the line above and uncomment below:
    # model = SAC(
    #     "MlpPolicy",
    #     vec_env,
    #     verbose=1,
    #     tensorboard_log="./sac_salp_robot_tensorboard/",
    #     learning_rate=3e-4,
    #     buffer_size=100_000,
    #     batch_size=512,
    #     ent_coef="auto",
    #     gamma=0.99,
    #     tau=0.005,
    #     device="cuda",
    # )

    # 3. Callbacks
    save_freq = 12_500
    save_dir = "./logs/"
    prefix = "salp_robot_calibrated"

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_dir,
        name_prefix=prefix,
    )

    # Optional: save VecNormalize stats alongside each checkpoint
    # vec_norm_callback = SaveVecNormalizeCallback(
    #     save_freq=save_freq,
    #     save_path=save_dir,
    #     name_prefix=f"{prefix}_vecnormalize",
    #     verbose=1,
    # )

    episode_callback = EpisodeComponentCallback()

    # 4. Train
    print("Starting training...")
    model.learn(
        total_timesteps=2_000_000,
        callback=[checkpoint_callback, episode_callback],
        reset_num_timesteps=True,
        tb_log_name="salp_robot_calibrated_run",
    )

    # 5. Save final model
    model.save("salp_robot_final_calibratedv2")
    # vec_env.save("vec_final_vecnormalizev2.pkl")

    print("Training finished.")