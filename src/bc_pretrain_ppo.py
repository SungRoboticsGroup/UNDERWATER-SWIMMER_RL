"""
Phase B of SAC -> PPO behavior cloning.

1. Build a fresh PPO with the same hyperparameters as train_robot_ppo.py.
2. Behavior-clone the PPO actor against the SAC rollouts using imitation.bc.BC,
   training PPO's actual policy object in place (no manual weight transfer).
3. Pretrain the critic on Monte-Carlo discounted returns from the saved rewards
   so the first PPO update doesn't blow up the cloned policy.
4. Save as a PPO zip that the existing train_robot_ppo.py --warm-start path
   can load unchanged.

Env construction mirrors train_robot_reward_shaping.py (Dongsheng's calibrated
params + DR + disturbances) so observation/action spaces and stochasticity
match the SAC expert that produced the rollouts.

Run from src/:
    python bc_pretrain_ppo.py \
        --rollouts ../experiments/rs_v2/rollouts/expert.npz \
        --out ../experiments/bc_v3/models/bc_ppo \
        --bc-epochs 20 \
        --critic-epochs 10
"""

import argparse
import os
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms import bc
from imitation.data.types import Transitions

from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle


def make_env():
    # Mirror train_robot_reward_shaping.py: Dongsheng's calibrated params +
    # domain randomization + disturbances.
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


def compute_mc_returns(rewards: np.ndarray, dones: np.ndarray, gamma: float) -> np.ndarray:
    """Discounted Monte-Carlo returns. No bootstrapping past episode boundaries."""
    N = len(rewards)
    returns = np.zeros(N, dtype=np.float32)
    G = 0.0
    for t in reversed(range(N)):
        if dones[t]:
            G = 0.0
        G = float(rewards[t]) + gamma * G
        returns[t] = G
    return returns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", type=str,
                   default="../experiments/rs_v2/rollouts/expert.npz")
    p.add_argument("--out", type=str,
                   default="../experiments/bc_v3/models/bc_ppo",
                   help="Output PPO zip path (without .zip suffix).")
    p.add_argument("--bc-epochs",     type=int, default=20)
    p.add_argument("--bc-batch-size", type=int, default=256)
    p.add_argument("--critic-epochs", type=int, default=10)
    p.add_argument("--critic-batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr",    type=float, default=3e-4,
                   help="Learning rate for BC actor (and PPO construction).")
    p.add_argument("--critic-lr", type=float, default=1e-3,
                   help="Learning rate for the critic-pretrain regression. "
                        "Higher than BC LR is fine: the critic is starting from "
                        "random init and needs faster convergence.")
    p.add_argument("--seed",  type=int,   default=0)
    p.add_argument("--eval-episodes", type=int, default=10,
                   help="Set to 0 to skip the post-BC evaluation rollouts.")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    th.manual_seed(args.seed)

    # --- 1. Build a fresh PPO with the same hparams as train_robot_ppo.py ----
    vec_env = make_vec_env(make_env, n_envs=1)
    ppo = PPO(
        "MlpPolicy", vec_env, verbose=0,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="auto",
        seed=args.seed,
    )
    device = ppo.device
    print(f"PPO built. device={device}, "
          f"obs_space={vec_env.observation_space}, act_space={vec_env.action_space}")

    # --- 2. Load expert rollouts -------------------------------------------
    print(f"\nloading rollouts: {args.rollouts}")
    data = np.load(args.rollouts)
    # .copy() so the arrays are writable — torch.as_tensor warns on read-only
    # numpy buffers (npz returns memmap-style read-only arrays).
    obs       = data["obs"].astype(np.float32).copy()
    actions   = data["actions"].astype(np.float32).copy()
    rewards   = data["rewards"].astype(np.float32).copy()
    dones     = data["dones"].astype(bool).copy()
    next_obs  = data["next_obs"].astype(np.float32).copy()
    print(f"  N={len(obs)}  ep_returns mean={data['ep_returns'].mean():.1f} "
          f"(std {data['ep_returns'].std():.1f}, n_episodes={len(data['ep_returns'])})")

    transitions = Transitions(
        obs=obs,
        acts=actions,
        next_obs=next_obs,
        dones=dones,
        infos=np.array([{} for _ in range(len(obs))]),
    )

    # --- 3. Behavior-clone the PPO actor ------------------------------------
    print(f"\n=== BC: {args.bc_epochs} epochs, batch={args.bc_batch_size} ===")
    bc_trainer = bc.BC(
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        demonstrations=transitions,
        policy=ppo.policy,                 # train PPO's actual policy in place
        rng=rng,
        batch_size=args.bc_batch_size,
        optimizer_kwargs={"lr": args.lr},
        device=device,
    )

    # Per-epoch loop so we can capture average BC loss per epoch for plotting.
    # Loss = -log p(expert_action | obs) under PPO's stochastic actor (matches
    # what imitation.bc.BC minimizes, plus a small entropy bonus we ignore).
    obs_t_all = th.as_tensor(obs, dtype=th.float32, device=device)
    act_t_all = th.as_tensor(actions, dtype=th.float32, device=device)
    eval_idx_size = min(8192, len(obs))  # sample for fast per-epoch loss probe

    bc_losses = []
    pbar = tqdm(range(args.bc_epochs), desc="BC epochs")
    for epoch in pbar:
        bc_trainer.train(n_epochs=1, progress_bar=False)
        ppo.policy.eval()
        with th.no_grad():
            idx = th.randint(len(obs), (eval_idx_size,), device=device)
            _, logp, _ = ppo.policy.evaluate_actions(obs_t_all[idx], act_t_all[idx])
            loss = -logp.mean().item()
        ppo.policy.train()
        bc_losses.append(loss)
        pbar.set_postfix({"neg_logp": f"{loss:.3f}"})

    # --- 4. Critic pretrain on MC returns -----------------------------------
    print(f"\n=== Critic pretrain: {args.critic_epochs} epochs, "
          f"batch={args.critic_batch_size} ===")
    returns = compute_mc_returns(rewards, dones, gamma=args.gamma)
    print(f"  MC returns: mean={returns.mean():.1f}  std={returns.std():.1f}  "
          f"min={returns.min():.1f}  max={returns.max():.1f}")

    obs_t = th.as_tensor(obs, dtype=th.float32, device=device)
    ret_t = th.as_tensor(returns, dtype=th.float32, device=device)

    # Only train critic-side params: mlp_extractor.value_net + value_net head.
    # FlattenExtractor (default features_extractor for MlpPolicy + vector obs)
    # has no learnable params, so we don't need to include it.
    critic_params = (
        list(ppo.policy.mlp_extractor.value_net.parameters())
        + list(ppo.policy.value_net.parameters())
    )
    print(f"  critic param tensors: {len(critic_params)} "
          f"(total params: {sum(p.numel() for p in critic_params)}, lr={args.critic_lr})")
    opt = th.optim.Adam(critic_params, lr=args.critic_lr)

    # Initial critic MSE for context:
    ppo.policy.eval()
    with th.no_grad():
        v0 = ppo.policy.predict_values(obs_t).squeeze(-1)
        mse0 = F.mse_loss(v0, ret_t).item()
    print(f"  pre-pretrain critic MSE: {mse0:.3f}")

    ppo.policy.train()
    loader = DataLoader(
        TensorDataset(obs_t, ret_t),
        batch_size=args.critic_batch_size, shuffle=True,
    )
    critic_mses = []
    pbar = tqdm(range(args.critic_epochs), desc="Critic epochs")
    for epoch in pbar:
        losses = []
        for ob_b, ret_b in loader:
            v = ppo.policy.predict_values(ob_b).squeeze(-1)
            loss = F.mse_loss(v, ret_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        epoch_mse = float(np.mean(losses))
        critic_mses.append(epoch_mse)
        pbar.set_postfix({"mse": f"{epoch_mse:.3f}"})

    ppo.policy.eval()
    with th.no_grad():
        v1 = ppo.policy.predict_values(obs_t).squeeze(-1)
        mse1 = F.mse_loss(v1, ret_t).item()
    print(f"  post-pretrain critic MSE: {mse1:.3f}  (was {mse0:.3f})")

    # --- 5. Eval the BC'd PPO directly ------------------------------------
    if args.eval_episodes > 0:
        print(f"\n=== Eval: {args.eval_episodes} episodes (stochastic) ===")
        mean_r, std_r = evaluate_policy(
            ppo, vec_env,
            n_eval_episodes=args.eval_episodes,
            deterministic=False,
        )
        print(f"  BC'd PPO mean ep return: {mean_r:.1f} ± {std_r:.1f}")
        print(f"  (SAC expert reference, from rollouts: "
              f"{data['ep_returns'].mean():.1f})")

    # --- 6. Save as a PPO zip the existing --warm-start path can load -------
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ppo.save(args.out)
    print(f"\nsaved -> {args.out}.zip")

    # --- 7. Save metrics + plots -----------------------------------------
    plots_dir = os.path.join(out_dir, "plots") if out_dir else "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Persist raw numbers so they can be re-plotted later if needed.
    metrics_npz = os.path.join(plots_dir, "bc_metrics.npz")
    np.savez(
        metrics_npz,
        bc_losses=np.asarray(bc_losses, dtype=np.float32),
        critic_mses=np.asarray(critic_mses, dtype=np.float32),
        critic_mse_pre=np.float32(mse0),
        critic_mse_post=np.float32(mse1),
    )
    print(f"saved metrics -> {metrics_npz}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(range(1, len(bc_losses) + 1), bc_losses, marker="o", markersize=3)
    axes[0].set_xlabel("BC epoch")
    axes[0].set_ylabel("mean -log p(expert action | obs)")
    axes[0].set_title(f"BC actor loss ({len(bc_losses)} epochs, sampled probe)")
    axes[0].grid(True, alpha=0.3)

    # Critic MSE on a log scale — easier to read the order-of-magnitude drop.
    critic_x = list(range(1, len(critic_mses) + 1))
    axes[1].plot(critic_x, critic_mses, marker="o", markersize=3, label="train MSE")
    axes[1].axhline(mse0, color="red",   linestyle="--", alpha=0.6, label=f"pre-pretrain ({mse0:.0f})")
    axes[1].axhline(mse1, color="green", linestyle="--", alpha=0.6, label=f"post-pretrain ({mse1:.0f})")
    axes[1].set_xlabel("critic epoch")
    axes[1].set_ylabel("MSE(V(s), MC return)")
    axes[1].set_title(f"Critic pretrain ({len(critic_mses)} epochs)")
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    plot_path = os.path.join(plots_dir, "bc_curves.png")
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    print(f"saved plot    -> {plot_path}")
    print(f"\nNext step:")
    print(f"  cd src && python train_robot_ppo.py \\")
    print(f"      --version ppo_bc_v3_finetune \\")
    print(f"      --warm-start {args.out}.zip")


if __name__ == "__main__":
    main()
