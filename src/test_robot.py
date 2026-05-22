# test_robot.py
import numpy as np
from stable_baselines3 import SAC, PPO
from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle

# --- Model selection ---
# NOTE: pre-integration checkpoints (SAC v6, PPO v3/v2, PPO BC finetune) were
# trained against the old env (6+2N obs, [0,1] inhale) and will NOT load here.
# Point this at a checkpoint trained on the new env (see train_robot_reward_shaping.py).

ModelClass  = SAC
model_path  = "../experiments/rs_v1/models/salp_robot_rs_v1_final"

nozzle = Nozzle(length1=0.052, length2=0.038, length3=0.050,
                area=np.pi * 0.01 ** 2, mass=0.428,
                radius=0.1, inner_radius=0.022)
nozzle.set_angles(angle1=0.0, angle2=0.0)

robot = Robot(dry_mass=0.738, init_length=0.26, init_width=0.135,
              max_contraction=0.04, nozzle=nozzle)
robot.set_environment(density=1000)

#to visulaize the whole cycle not just start points of the cycle comment in or out as needed
robot.enable_history_recording()
env = SalpRobotEnv(render_mode="human", robot=robot)
model = ModelClass.load(model_path, env=env)

obs, _ = env.reset()
env.start_recording()
for i in range(100):
    action, _states = model.predict(obs, deterministic=False)

    obs, reward, terminated, truncated, info = env.step(action)

    env.wait_for_animation()

    print(f"Step {i}: Action={action}, State={obs}")

    if truncated:
        obs, _ = env.reset()

    if terminated:
        print("Episode finished!")
        break
gif_path = env.stop_recording("test_robot_simulation2.gif")
env.close()
print(f"Simulation GIF saved to: {gif_path}")
