# test_robot.py
import numpy as np
from stable_baselines3 import SAC, PPO
from sb3_contrib import RecurrentPPO
from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle

# --- Model selection ---
# NOTE: pre-integration checkpoints (SAC v6, PPO v3/v2, old PPO BC finetune) were
# trained against the old env (6+2N obs, [0,1] inhale) and will NOT load here.
# Point this at a checkpoint trained on the new env (train_robot_reward_shaping.py
# for SAC, train_robot_ppo.py --warm-start for PPO BC-finetune,
# train_robot_recurrent_ppo.py --warm-start for the recurrent variant).
#
# Current options on this branch:
#   SAC rs_v2 final:           SAC,          "../experiments/rs_v2/models/salp_robot_rs_v2_final"
#   PPO BC v3 best:            PPO,          "../experiments/ppo_bc_v3_finetune/models/best_model/best_model"
#   PPO BC v3 final (2M):      PPO,          "../experiments/ppo_bc_v3_finetune/models/salp_robot_ppo_ppo_bc_v3_finetune_final"
#   Recurrent PPO BC v1 best:  RecurrentPPO, "../experiments/recurrent_ppo_bc_v1/models/best_model/best_model"
#   Recurrent PPO BC v1 final: RecurrentPPO, "../experiments/recurrent_ppo_bc_v1/models/salp_robot_recurrent_ppo_recurrent_ppo_bc_v1_final"

ModelClass  = RecurrentPPO
model_path  = "../experiments/recurrent_ppo_bc_v1/models/best_model/best_model"

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

# RecurrentPPO needs LSTM hidden state threaded across steps and reset on
# episode boundaries; feedforward SAC/PPO ignore both kwargs.
lstm_states = None
episode_starts = np.ones((1,), dtype=bool)

for i in range(100):
    if isinstance(model, RecurrentPPO):
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=False,
        )
    else:
        action, _ = model.predict(obs, deterministic=False)

    obs, reward, terminated, truncated, info = env.step(action)
    episode_starts = np.array([terminated or truncated])

    env.wait_for_animation()

    print(f"Step {i}: Action={action}, State={obs}")

    if truncated:
        obs, _ = env.reset()
        lstm_states = None  # fresh episode -> zero LSTM belief

    if terminated:
        print("Episode finished!")
        break
gif_path = env.stop_recording("test_robot_simulation2.gif")
env.close()
print(f"Simulation GIF saved to: {gif_path}")
