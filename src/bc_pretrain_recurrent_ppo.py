"""
Recurrent BC pretrain for RecurrentPPO.

Same three-phase structure as bc_pretrain_ppo.py, but BC and critic warmup are
done over expert episode SEQUENCES (rolling the LSTM forward) instead of
flattened (s, a) pairs. This addresses the feedforward-BC failure mode under
domain randomization, where averaging over physics rolls produces a blurry
policy: the LSTM hidden state can implicitly infer the randomized physics from
the trajectory of (obs, action) up to the current step, breaking the averaging.

Pipeline:
1. Build a fresh RecurrentPPO with the same hparams as train_robot_recurrent_ppo.py.
2. Slice the existing flat rollouts into episodes using the `dones` array (no
   recollection — boundaries were preserved by collect_sac_rollouts.py).
3. Recurrent BC: roll lstm_actor through each episode, compute
   -log p(expert_action_t | obs_{0..t}) summed over valid timesteps, backprop.
4. Recurrent critic warmup: roll lstm_critic through each episode, regress
   V(s_t, h_t) -> Monte-Carlo return-to-go.
5. Save as a RecurrentPPO zip the existing train_robot_recurrent_ppo.py
   --warm-start path can load unchanged.

Run from src/:
    python bc_pretrain_recurrent_ppo.py \
        --rollouts ../experiments/rs_v2/rollouts/expert.npz \
        --out ../experiments/recurrent_bc_v1/models/bc_recurrent_ppo \
        --bc-epochs 50 \
        --critic-epochs 10
"""

import argparse
import os
import numpy as np
import torch as th
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from salp_robot_env import SalpRobotEnv
from robot import Robot, Nozzle


def make_env():
    # Mirror train_robot_recurrent_ppo.py: Dongsheng's calibrated params +
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


def slice_into_episodes(obs, actions, rewards, dones):
    """Walk flat transitions left-to-right, slice on done into per-episode dicts.
    The flat layout from collect_sac_rollouts.py is per-env-contiguous, so done
    boundaries are correct for a left-to-right walk."""
    episodes = []
    start = 0
    for i in range(len(obs)):
        if dones[i]:
            episodes.append({
                "obs":     obs[start : i + 1],
                "actions": actions[start : i + 1],
                "rewards": rewards[start : i + 1],
            })
            start = i + 1
    # Any trailing partial episode (no terminal done at end of buffer) is dropped.
    return episodes


def mc_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Discounted return-to-go for a single episode."""
    G = 0.0
    returns = np.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        G = float(rewards[t]) + gamma * G
        returns[t] = G
    return returns


def pad_batch(episodes, idx_batch, device):
    """Stack a batch of episodes into right-padded tensors.

    Returns:
        obs_pad: (T, B, obs_dim)     -- transposed for LSTM batch_first=False
        act_pad: (T, B, act_dim)
        ret_pad: (T, B)
        mask:    (T, B)              -- 1.0 for valid timesteps, 0.0 for padding
    """
    eps = [episodes[i] for i in idx_batch]
    B = len(eps)
    T = max(len(e["obs"]) for e in eps)
    obs_dim = eps[0]["obs"].shape[-1]
    act_dim = eps[0]["actions"].shape[-1]

    obs_pad = np.zeros((T, B, obs_dim), dtype=np.float32)
    act_pad = np.zeros((T, B, act_dim), dtype=np.float32)
    ret_pad = np.zeros((T, B),          dtype=np.float32)
    mask    = np.zeros((T, B),          dtype=np.float32)

    for j, e in enumerate(eps):
        L = len(e["obs"])
        obs_pad[:L, j] = e["obs"]
        act_pad[:L, j] = e["actions"]
        ret_pad[:L, j] = e["returns"]
        mask[:L, j]    = 1.0

    return (
        th.as_tensor(obs_pad, device=device),
        th.as_tensor(act_pad, device=device),
        th.as_tensor(ret_pad, device=device),
        th.as_tensor(mask,    device=device),
    )


def recurrent_actor_logp(policy, obs_TBO, act_TBA):
    """Roll lstm_actor through the sequence, return per-timestep log p(a|obs_seq).

    obs_TBO: (T, B, obs_dim)
    act_TBA: (T, B, act_dim)
    returns: log_prob (T, B)
    """
    # LSTM with batch_first=False: input (T, B, in), output (T, B, hidden)
    lstm_out, _ = policy.lstm_actor(obs_TBO)
    T, B, H = lstm_out.shape
    latent_pi = policy.mlp_extractor.policy_net(lstm_out.reshape(T * B, H))
    dist = policy._get_action_dist_from_latent(latent_pi)
    log_prob = dist.log_prob(act_TBA.reshape(T * B, -1))  # (TB,)
    return log_prob.reshape(T, B)


def recurrent_critic_values(policy, obs_TBO):
    """Roll lstm_critic through the sequence, return per-timestep value (T, B)."""
    lstm_out, _ = policy.lstm_critic(obs_TBO)
    T, B, H = lstm_out.shape
    latent_vf = policy.mlp_extractor.value_net(lstm_out.reshape(T * B, H))
    values = policy.value_net(latent_vf).squeeze(-1)  # (TB,)
    return values.reshape(T, B)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", type=str,
                   default="../experiments/rs_v2/rollouts/expert.npz")
    p.add_argument("--out", type=str,
                   default="../experiments/recurrent_bc_v1/models/bc_recurrent_ppo",
                   help="Output RecurrentPPO zip path (without .zip suffix).")
    p.add_argument("--bc-epochs",       type=int, default=50)
    p.add_argument("--bc-batch-size",   type=int, default=64,
                   help="Episodes per BC gradient step.")
    p.add_argument("--critic-epochs",   type=int, default=10)
    p.add_argument("--critic-batch-size", type=int, default=64,
                   help="Episodes per critic gradient step.")
    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--critic-lr",  type=float, default=1e-3)
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--lstm-hidden",   type=int, default=256)
    args = p.parse_args()

    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- 1. Build a fresh RecurrentPPO ---------------------------------------
    vec_env = make_vec_env(make_env, n_envs=1, vec_env_cls=DummyVecEnv)
    ppo = RecurrentPPO(
        "MlpLstmPolicy", vec_env, verbose=0,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            lstm_hidden_size=args.lstm_hidden,
            n_lstm_layers=1,
            enable_critic_lstm=True,
            shared_lstm=False,
        ),
        device="auto",
        seed=args.seed,
    )
    device = ppo.device
    print(f"RecurrentPPO built. device={device}")
    print(f"  obs_space={vec_env.observation_space}  act_space={vec_env.action_space}")
    print(f"  lstm_actor: {ppo.policy.lstm_actor}")

    # --- 2. Load rollouts and slice into episodes ----------------------------
    print(f"\nloading rollouts: {args.rollouts}")
    data = np.load(args.rollouts)
    obs     = data["obs"].astype(np.float32).copy()
    actions = data["actions"].astype(np.float32).copy()
    rewards = data["rewards"].astype(np.float32).copy()
    dones   = data["dones"].astype(bool).copy()
    print(f"  N={len(obs)} transitions, "
          f"ep_returns mean={data['ep_returns'].mean():.1f} "
          f"(n_episodes_recorded={len(data['ep_returns'])})")

    episodes = slice_into_episodes(obs, actions, rewards, dones)
    # Attach MC returns per episode for the critic phase.
    for ep in episodes:
        ep["returns"] = mc_returns(ep["rewards"], gamma=args.gamma)

    ep_lens = np.array([len(e["obs"]) for e in episodes])
    print(f"  sliced into {len(episodes)} episodes: "
          f"len mean={ep_lens.mean():.1f}  min={ep_lens.min()}  max={ep_lens.max()}  "
          f"total={ep_lens.sum()}")

    # --- 3. Recurrent BC -----------------------------------------------------
    print(f"\n=== Recurrent BC: {args.bc_epochs} epochs, "
          f"{args.bc_batch_size} episodes/batch ===")
    # Train only actor-side params: lstm_actor + mlp_extractor.policy_net +
    # action_net + log_std. (FlattenExtractor is parameter-free for vector obs.)
    actor_params = (
        list(ppo.policy.lstm_actor.parameters())
        + list(ppo.policy.mlp_extractor.policy_net.parameters())
        + list(ppo.policy.action_net.parameters())
        + [ppo.policy.log_std]
    )
    actor_opt = th.optim.Adam(actor_params, lr=args.lr)

    n_eps = len(episodes)
    bc_losses = []
    pbar = tqdm(range(args.bc_epochs), desc="BC epochs")
    for epoch in pbar:
        perm = np.random.permutation(n_eps)
        epoch_loss_sum = 0.0
        epoch_valid   = 0.0
        for start in range(0, n_eps, args.bc_batch_size):
            idx = perm[start : start + args.bc_batch_size]
            obs_TBO, act_TBA, _, mask = pad_batch(episodes, idx, device)

            log_prob = recurrent_actor_logp(ppo.policy, obs_TBO, act_TBA)
            # mean -log_prob over valid timesteps
            valid = mask.sum()
            loss = -(log_prob * mask).sum() / valid

            actor_opt.zero_grad()
            loss.backward()
            actor_opt.step()

            # accumulate for epoch-mean reporting
            epoch_loss_sum += loss.item() * valid.item()
            epoch_valid   += valid.item()

        epoch_loss = epoch_loss_sum / max(epoch_valid, 1.0)
        bc_losses.append(epoch_loss)
        pbar.set_postfix({"neg_logp": f"{epoch_loss:.3f}"})

    # --- 4. Recurrent critic warmup ------------------------------------------
    print(f"\n=== Recurrent critic warmup: {args.critic_epochs} epochs, "
          f"{args.critic_batch_size} episodes/batch ===")
    critic_params = (
        list(ppo.policy.lstm_critic.parameters())
        + list(ppo.policy.mlp_extractor.value_net.parameters())
        + list(ppo.policy.value_net.parameters())
    )
    critic_opt = th.optim.Adam(critic_params, lr=args.critic_lr)

    # Pre/post MSE on a single deterministic eval batch over the whole dataset
    # (in chunks to fit in memory).
    @th.no_grad()
    def full_critic_mse():
        ppo.policy.eval()
        total_se = 0.0
        total_n  = 0.0
        for s in range(0, n_eps, args.critic_batch_size):
            obs_TBO, _, ret_TB, mask = pad_batch(
                episodes, list(range(s, min(s + args.critic_batch_size, n_eps))),
                device,
            )
            v = recurrent_critic_values(ppo.policy, obs_TBO)
            se = ((v - ret_TB) ** 2 * mask).sum().item()
            total_se += se
            total_n  += mask.sum().item()
        ppo.policy.train()
        return total_se / max(total_n, 1.0)

    mse_pre = full_critic_mse()
    print(f"  pre-pretrain critic MSE: {mse_pre:.3f}")

    critic_mses = []
    pbar = tqdm(range(args.critic_epochs), desc="Critic epochs")
    for epoch in pbar:
        perm = np.random.permutation(n_eps)
        epoch_loss_sum = 0.0
        epoch_valid   = 0.0
        for start in range(0, n_eps, args.critic_batch_size):
            idx = perm[start : start + args.critic_batch_size]
            obs_TBO, _, ret_TB, mask = pad_batch(episodes, idx, device)

            v = recurrent_critic_values(ppo.policy, obs_TBO)
            valid = mask.sum()
            loss = ((v - ret_TB) ** 2 * mask).sum() / valid

            critic_opt.zero_grad()
            loss.backward()
            critic_opt.step()

            epoch_loss_sum += loss.item() * valid.item()
            epoch_valid   += valid.item()

        epoch_mse = epoch_loss_sum / max(epoch_valid, 1.0)
        critic_mses.append(epoch_mse)
        pbar.set_postfix({"mse": f"{epoch_mse:.3f}"})

    mse_post = full_critic_mse()
    print(f"  post-pretrain critic MSE: {mse_post:.3f}  (was {mse_pre:.3f})")

    # --- 5. Eval the BC'd RecurrentPPO ---------------------------------------
    if args.eval_episodes > 0:
        print(f"\n=== Eval: {args.eval_episodes} episodes (stochastic) ===")
        # evaluate_policy handles recurrent policies via the policy's own predict
        # method, which resets LSTM state on episode boundaries internally.
        mean_r, std_r = evaluate_policy(
            ppo, vec_env,
            n_eval_episodes=args.eval_episodes,
            deterministic=False,
        )
        print(f"  BC'd RecurrentPPO mean ep return: {mean_r:.1f} ± {std_r:.1f}")
        print(f"  (SAC expert reference, from rollouts: "
              f"{data['ep_returns'].mean():.1f})")

    # --- 6. Save as a RecurrentPPO zip ---------------------------------------
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ppo.save(args.out)
    print(f"\nsaved -> {args.out}.zip")

    # --- 7. Save metrics + plots ---------------------------------------------
    plots_dir = os.path.join(out_dir, "plots") if out_dir else "plots"
    os.makedirs(plots_dir, exist_ok=True)
    metrics_npz = os.path.join(plots_dir, "bc_recurrent_metrics.npz")
    np.savez(
        metrics_npz,
        bc_losses=np.asarray(bc_losses, dtype=np.float32),
        critic_mses=np.asarray(critic_mses, dtype=np.float32),
        critic_mse_pre=np.float32(mse_pre),
        critic_mse_post=np.float32(mse_post),
    )
    print(f"saved metrics -> {metrics_npz}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(range(1, len(bc_losses) + 1), bc_losses, marker="o", markersize=3)
    axes[0].set_xlabel("BC epoch")
    axes[0].set_ylabel("mean -log p(expert action | obs sequence)")
    axes[0].set_title(f"Recurrent BC actor loss ({len(bc_losses)} epochs)")
    axes[0].grid(True, alpha=0.3)

    critic_x = list(range(1, len(critic_mses) + 1))
    axes[1].plot(critic_x, critic_mses, marker="o", markersize=3, label="train MSE")
    axes[1].axhline(mse_pre,  color="red",   linestyle="--", alpha=0.6,
                    label=f"pre-pretrain ({mse_pre:.0f})")
    axes[1].axhline(mse_post, color="green", linestyle="--", alpha=0.6,
                    label=f"post-pretrain ({mse_post:.0f})")
    axes[1].set_xlabel("critic epoch")
    axes[1].set_ylabel("MSE(V(s,h), MC return)")
    axes[1].set_title(f"Recurrent critic warmup ({len(critic_mses)} epochs)")
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    plot_path = os.path.join(plots_dir, "bc_recurrent_curves.png")
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    print(f"saved plot    -> {plot_path}")

    print(f"\nNext step:")
    print(f"  cd src && python train_robot_recurrent_ppo.py \\")
    print(f"      --version recurrent_ppo_bc_v1 \\")
    print(f"      --warm-start {args.out}.zip")


if __name__ == "__main__":
    main()
