from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle
import numpy as np
import os 

class SaveVecNormalizeCallback(BaseCallback):
    """
    Saves the VecNormalize statistics at the same frequency as the CheckpointCallback.
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "vec_normalize", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Ensure the save folder exists
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Check if we reached the save frequency
        if self.n_calls % self.save_freq == 0:
            # Create a filename that matches the model checkpoint format
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            
            # Save the VecNormalize stats
            self.training_env.save(path)
            
            if self.verbose > 0:
                print(f"Saved VecNormalize to {path}")
                
        return True

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log all keys starting with "rewards/" found in the info dict
        # We look at the first env's info (assuming vectorized envs)
        infos = self.locals["infos"]
        for info in infos:
            for key, value in info.items():
                if key.startswith("rewards/"):
                    self.logger.record(key, value)
        return True
    
class EpisodeComponentCallback(BaseCallback):
    """
    Accumulates individual reward components and logs their EPISODE SUM 
    to TensorBoard, so they match the scale of 'ep_rew_mean'.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_episode_rewards = {}
        self.num_envs = 0

    def _init_callback(self) -> None:
        # Initialize the buffer for each environment
        self.num_envs = self.training_env.num_envs
        self.current_episode_rewards = {i: {} for i in range(self.num_envs)}

    def _on_step(self) -> bool:
        # Get the infos from all environments
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        
        for i, info in enumerate(infos):
            # 1. Accumulate rewards for this step
            for key, value in info.items():
                if key.startswith("rewards/"):
                    if key not in self.current_episode_rewards[i]:
                        self.current_episode_rewards[i][key] = 0.0
                    self.current_episode_rewards[i][key] += value
            
            # 2. If episode is done, log the TOTAL sum and reset
            if dones[i]:
                for key, value in self.current_episode_rewards[i].items():
                    # Log the Episode Sum (e.g., "episode_rewards/track")
                    # We add 'episode_' prefix to clarify this is a sum
                    log_key = key.replace("rewards/", "episode_rewards/")
                    self.logger.record(log_key, value)
                
                # Reset buffer for this environment
                self.current_episode_rewards[i] = {}
                
        return True

def make_env():
    # Create and return the SalpRobotEnv environment
    nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.0036, mass=1.0)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                    max_contraction=0.06, nozzle=nozzle)
    robot.nozzle.set_angles(angle1=0.0, angle2=0.0)  # set nozzle angles
    robot.set_environment(density=1000)  # water density in kg/m^3
    # robot.enable_dynamic_randomization()  # enable domain randomization
    # robot.enable_disturbances()

    env = SalpRobotEnv(render_mode=None, robot=robot)
    # env.enable_action_randomization()
    # env.enable_latency()
    # env.enable_observation_randomization()

    return env

if __name__ == "__main__":

    num_cpu = 8
    vec_env = make_vec_env(make_env, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    # vec_env = VecNormalize.load("vec_final_vecnormalize.pkl", vec_env)

    # 3. CRITICAL: Turn training mode ON. 
    # Unlike your test script, we WANT the running averages to keep updating as it learns new things.
    # vec_env.training = True
    # vec_env.norm_reward = True

    # 2. Sanity Check (CRITICAL)
    # This checks if your observation/action spaces match what the step() function returns.
    # It will crash here if you made a mistake, saving you hours of debugging.
    print("Checking environment...")
    # check_env(env)
    print("Environment is valid!")

    # 3. Define the model (SAC)
    # model = SAC(
    #     "MlpPolicy",           # Use standard Dense Neural Network
    #     vec_env,
    #     verbose=1,
    #     tensorboard_log="./sac_salp_robot_tensorboard/",
        
    #     # --- Tuning for Robotics ---
    #     learning_rate=3e-4,
    #     buffer_size=100000,    # Big memory for off-policy
    #     batch_size=512,        # Mini-batch size
    #     ent_coef='auto',       # Automatically adjust exploration (Temperature)
    #     gamma=0.99,            # Discount factor
    #     tau=0.005,             # Polyak averaging (Soft update)
    #     device="cuda" 

    # )    
    model = SAC.load("./logs/salp_robot_body_frame_1400000_steps", env=vec_env)   


    # model.learning_rate = 1e-3  # Reset learning rate when loading

    # 4. Setup Saving (Checkpoints)
    # Save the model every 10,000 steps so you don't lose progress if it crashes.
    # 4. Setup Saving (Checkpoints)
    save_freq = 12500
    save_dir = './logs/'
    prefix = 'salp_robot_body_frame_sideslip'

    checkpoint_callback = CheckpointCallback(
        save_freq= save_freq,
        save_path= save_dir,
        name_prefix= prefix,
    )

# Save the normalization stats (.pkl)
    # vec_norm_callback = SaveVecNormalizeCallback(
    #     save_freq=save_freq,
    #     save_path=save_dir,
    #     name_prefix=f"{prefix}_vecnormalize",
    #     verbose=1
    # )
    callback = EpisodeComponentCallback()
    
    # 5. Train
    print("Starting training...")
    model.learn(
        total_timesteps=2000000, # Run for 2M steps
        callback=[checkpoint_callback, callback],
        reset_num_timesteps=True,
        tb_log_name="salp_robot_body_frame_sideslip"
    )
    
    # 6. Save Final Model
    model.save("salp_robot_final_body_frame_sideslip")
    # vec_env.save("vec_final_vecnormalizev2.pkl")

    print("Training finished.")