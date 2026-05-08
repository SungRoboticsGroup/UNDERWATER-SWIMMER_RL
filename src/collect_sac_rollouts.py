"""
Phase A of SAC -> PPO behavior cloning: collect demonstration rollouts from the
trained SAC v1 expert.

Run from src/:
    # Smoke (single env, prints fast)
    python collect_sac_rollouts.py --num-transitions 10000 --n-envs 1

    # Real run (parallel)
    python collect_sac_rollouts.py --num-transitions 200000 --n-envs 8

Output npz arrays (transitions are laid out per-env contiguously, then concatenated,
so episode boundaries marked by `dones` are correct for a left-to-right walk):
    obs        (N, obs_dim) float32
    actions    (N, act_dim) float32
    rewards    (N,)         float32
    dones      (N,)         bool      # terminated OR truncated
    next_obs   (N, obs_dim) float32   # true s' even on episode-end (from terminal_observation)
    ep_returns (E,)         float32   # one entry per completed episode (for QA)
"""

import argparse
import os
import time
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle


def make_env():
    nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.00016, mass=1.0)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15,
                  max_contraction=0.06, nozzle=nozzle)
    robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
    robot.set_environment(density=1000)
    return SalpRobotEnv(render_mode=None, robot=robot, num_obstacles=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="../experiments/sac_v1/models/best_model/best_model",
                        help="Path to the SAC zip (without .zip suffix is fine).")
    parser.add_argument("--num-transitions", type=int, default=200_000,
                        help="Total (s, a) transitions to collect.")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Parallel envs. 1 -> DummyVecEnv (no subprocess overhead).")
    parser.add_argument("--out", type=str,
                        default="../experiments/sac_v1/rollouts/expert.npz")
    parser.add_argument("--deterministic", action="store_true",
                        help="If set, sample actions deterministically instead of stochastically.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    n_envs = max(1, args.n_envs)
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=vec_cls, seed=args.seed)

    obs_dim = vec_env.observation_space.shape[0]
    act_dim = vec_env.action_space.shape[0]
    print(f"vec_env: n_envs={n_envs}, obs_dim={obs_dim}, act_dim={act_dim}")

    print(f"loading SAC expert: {args.model}")
    # SAC was saved with SB3 2.8 (uses FloatSchedule); imitation pins SB3 to 2.2.1.
    # Inference doesn't need lr_schedule / clip_range, so stub them out to silence
    # the deserialize warning.
    expert = SAC.load(
        args.model, env=vec_env,
        custom_objects={"lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.2},
    )

    N_target = args.num_transitions
    steps_per_env = (N_target + n_envs - 1) // n_envs
    N_actual = steps_per_env * n_envs
    if N_actual != N_target:
        print(f"  rounded up: collecting {N_actual} transitions "
              f"({steps_per_env} per env x {n_envs}), will trim to {N_target} on save")

    # Buffers laid out as (steps_per_env, n_envs, ...). At save time we transpose to
    # (n_envs, steps_per_env, ...) and flatten so each env's stream is contiguous.
    obs_buf       = np.zeros((steps_per_env, n_envs, obs_dim), dtype=np.float32)
    next_obs_buf  = np.zeros((steps_per_env, n_envs, obs_dim), dtype=np.float32)
    act_buf       = np.zeros((steps_per_env, n_envs, act_dim), dtype=np.float32)
    rew_buf       = np.zeros((steps_per_env, n_envs),          dtype=np.float32)
    done_buf      = np.zeros((steps_per_env, n_envs),          dtype=bool)

    ep_returns_running   = np.zeros(n_envs, dtype=np.float64)
    ep_returns_completed = []

    obs = vec_env.reset()  # (n_envs, obs_dim)
    t0 = time.time()
    # Print every ~1000 wall-clock transitions
    log_every = max(1, 1000 // n_envs)
    print(f"starting parallel rollout collection: target {N_actual} transitions "
          f"({steps_per_env} steps x {n_envs} envs); first ~hundred steps slow due to numba JIT compile")

    for t in range(steps_per_env):
        actions, _ = expert.predict(obs, deterministic=args.deterministic)
        step_obs, rewards, dones, infos = vec_env.step(actions)

        # When VecEnv auto-resets on done, step_obs[i] is the FIRST obs of the next
        # episode, not the terminal obs. Recover the terminal obs from info.
        real_next_obs = step_obs.copy()
        for i in range(n_envs):
            if dones[i] and isinstance(infos[i], dict) and "terminal_observation" in infos[i]:
                real_next_obs[i] = infos[i]["terminal_observation"]

        obs_buf[t]      = obs
        next_obs_buf[t] = real_next_obs
        act_buf[t]      = actions
        rew_buf[t]      = rewards
        done_buf[t]     = dones

        ep_returns_running += rewards
        for i in range(n_envs):
            if dones[i]:
                ep_returns_completed.append(float(ep_returns_running[i]))
                ep_returns_running[i] = 0.0

        obs = step_obs  # auto-reset already done; safe to use as next iter's input

        if (t + 1) % log_every == 0 or (t + 1) == steps_per_env:
            elapsed = time.time() - t0
            transitions_so_far = (t + 1) * n_envs
            rate = transitions_so_far / max(elapsed, 1e-6)
            eta_s = (N_actual - transitions_so_far) / max(rate, 1e-6)
            recent = np.mean(ep_returns_completed[-50:]) if ep_returns_completed else float("nan")
            print(f"  step {t+1:>6}/{steps_per_env}  "
                  f"transitions={transitions_so_far:>7}/{N_actual}  "
                  f"episodes={len(ep_returns_completed):>5}  "
                  f"recent_ep_return={recent:>8.1f}  "
                  f"rate={rate:>5.0f} steps/s  eta={eta_s:>6.0f}s")

    vec_env.close()

    # Flatten: transpose to (n_envs, steps_per_env, ...) so each env's stream is
    # contiguous, then reshape. Episode boundaries marked by `dones` remain valid
    # for a left-to-right walk (boundary between env i and env i+1 looks like an
    # episode end from env i's tail, which is the correct behavior for MC-return
    # computation downstream).
    obs_out      = obs_buf.transpose(1, 0, 2).reshape(N_actual, obs_dim)
    next_obs_out = next_obs_buf.transpose(1, 0, 2).reshape(N_actual, obs_dim)
    act_out      = act_buf.transpose(1, 0, 2).reshape(N_actual, act_dim)
    rew_out      = rew_buf.transpose(1, 0).reshape(N_actual)
    done_out     = done_buf.transpose(1, 0).reshape(N_actual)

    if N_target < N_actual:
        obs_out = obs_out[:N_target]; next_obs_out = next_obs_out[:N_target]
        act_out = act_out[:N_target]; rew_out = rew_out[:N_target]
        done_out = done_out[:N_target]

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        args.out,
        obs=obs_out,
        actions=act_out,
        rewards=rew_out,
        dones=done_out,
        next_obs=next_obs_out,
        ep_returns=np.asarray(ep_returns_completed, dtype=np.float32),
    )
    elapsed = time.time() - t0
    print(f"\nsaved -> {args.out}")
    print(f"  transitions: {len(obs_out)}")
    print(f"  episodes:    {len(ep_returns_completed)}")
    print(f"  wall_clock:  {elapsed:.0f}s  ({len(obs_out)/elapsed:.0f} transitions/s)")
    if ep_returns_completed:
        ep_arr = np.asarray(ep_returns_completed)
        print(f"  ep_return: mean={ep_arr.mean():.1f}  std={ep_arr.std():.1f}  "
              f"min={ep_arr.min():.1f}  max={ep_arr.max():.1f}")


if __name__ == "__main__":
    main()
