# test_robot.py
from stable_baselines3 import SAC, PPO
from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle

# --- Model selection ---
# SAC v6 (best overall, trained with 2 obstacles):
#   ModelClass, model_path, num_obstacles = SAC, "../experiments/v6/models/best_model/best_model", 2
#
# PPO v3 best (peak reward -253 at 80k steps, no obstacles):
#   ModelClass, model_path, num_obstacles = PPO, "../experiments/ppo_v3/models/best_model/best_model", 0
#
# PPO v3 400k checkpoint:
#   ModelClass, model_path, num_obstacles = PPO, "../experiments/ppo_v3/models/salp_robot_ppo_ppo_v3_400000_steps", 0
#
# PPO v2 best (peak reward -258 at 320k steps, no obstacles):
#   ModelClass, model_path, num_obstacles = PPO, "../experiments/ppo_v2/models/best_model/best_model", 0

ModelClass  = PPO
model_path  = "../experiments/ppo_v4/models/salp_robot_ppo_ppo_v4_1200000_steps"
num_obstacles = 0

nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.00016, mass=1.0)
robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15,
              max_contraction=0.06, nozzle=nozzle)
robot.nozzle.set_angles(angle1=0.0, angle2=0.0)

#to visulaize the whole cycle not just start points of the cycle comment in or out as needed
robot.enable_history_recording()
env = SalpRobotEnv(render_mode="human", robot=robot, num_obstacles=num_obstacles)
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
