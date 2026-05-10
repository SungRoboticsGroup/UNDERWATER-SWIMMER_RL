# SALP — Underwater Swimmer RL

Bio-inspired soft underwater robot simulation for reinforcement-learning research. The agent controls a steerable rear nozzle and a breathing-cycle compression schedule to navigate to a target while avoiding obstacles. Custom 6-DOF rigid-body physics (added mass, drag, Coriolis, jet thrust), wrapped as a Gymnasium environment and trained with Stable-Baselines3 (SAC / PPO / RecurrentPPO).

GRASP Lab — University of Pennsylvania (Sung Robotics Lab).

## Quick start

Requires Python 3.11 (3.8–3.11 supported per `setup.py`). The recipe below uses [`uv`](https://docs.astral.sh/uv/), but plain `python -m venv` + `pip` works too.

```bash
# 1. Install uv (one-time, per machine)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Create the project venv
uv python install 3.11
uv venv --python 3.11

# 3. Install dependencies + the project itself in editable mode
uv pip install -r requirements.txt
uv pip install -e .   # exposes src/*.py as importable top-level modules

# 4. Smoke-test
.venv/bin/python -c "
from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle
nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.00016, mass=1.0)
robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, max_contraction=0.06, nozzle=nozzle)
robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
robot.set_environment(density=1000)
env = SalpRobotEnv(render_mode=None, robot=robot, num_obstacles=0)
o, _ = env.reset(); print('obs', o.shape)
"
```

## Running things

All scripts live in `src/`. After `pip install -e .` the modules are importable from anywhere, but the training scripts write outputs to `../experiments/{version}/…` (a path relative to the current working directory), so still invoke them from inside `src/`.

| Task                | Command                                                                                           |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| Train SAC           | `cd src && python train_robot.py --version v6 [--warm-start path/to/model.zip]`                   |
| Train PPO           | `cd src && python train_robot_ppo.py --version v6_ppo [--warm-start path/to/model.zip]`           |
| Train RecurrentPPO  | `cd src && python train_robot_recurrent_ppo.py --version v6_lstm [--warm-start path/to/model.zip]`|
| Watch a model live  | `cd src && python watch_model.py [--model path/to/model.zip]`                                     |

Outputs (TensorBoard logs, checkpoints, eval logs, recordings) go to `../experiments/{version}/…` relative to `src/`.

For the full TensorBoard metric reference see [METRICS.md](METRICS.md).

## Layout

```
src/
  salp_robot_env.py      Gymnasium env (SalpRobotEnv)
  robot.py               Robot + Nozzle classes (state, kinematics, breathing cycle)
  dynamics.py            Numba-JIT physics (forces, accelerations)
  geometry.py            Data-driven compression / refill-time models
  train_robot*.py        Training entry points (SAC / PPO / RecurrentPPO)
  watch_model.py         Live model viewer (auto-reloads best checkpoint)
  tensorboard_callback.py  Custom metric logger
  plotting.py, compare_trajectories.py  Analysis utilities
```

## Notes

- Apple Silicon: PyTorch uses MPS automatically; CUDA is not available.
- `cv2` (pulled in via `stable-baselines3[extra]`) and `pygame` both bundle SDL2 dylibs, so importing both prints `objc[…] Class SDL_… is implemented in both …` warnings. Cosmetic at present.
