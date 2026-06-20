"""
Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

Small CTDE policy-gradient baseline for an attacker-defender cyber Markov game.

This file keeps the CTDE baseline readable:
  * decentralized categorical actors for defender and attacker;
  * a centralized critic that sees the joint state and both actions;
  * episodic rollouts through the hybrid ODE environment;
  * policy-gradient updates with a shared advantage estimate.

The Markov-game step is sampled-data: both agents observe at t_k, choose joint
actions, the environment applies any impulse jump, integrates the ODE until
t_{k+1}, and returns the next observation and both rewards.

For PPO-style cooperative defenders on node-level SIPRS dynamics, use
``node_siprs_mappo.py``.  This file focuses on the two-player interaction loop
and does not claim to be a full MAPPO implementation.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from cyber_hybrid_env import HybridCyberDefenseEnv

from cybercontrol.torch_utils import configure_torch


class Actor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)

    def sample(self, obs):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy()


class CentralCritic(nn.Module):
    def __init__(self, obs_dim, n_def_actions, n_atk_actions, hidden=128):
        super().__init__()
        in_dim = obs_dim + n_def_actions + n_atk_actions
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.n_def_actions = n_def_actions
        self.n_atk_actions = n_atk_actions

    def forward(self, obs, a_def, a_atk):
        d_onehot = nn.functional.one_hot(a_def, self.n_def_actions).float()
        a_onehot = nn.functional.one_hot(a_atk, self.n_atk_actions).float()
        x = torch.cat([obs, d_onehot, a_onehot], dim=-1)
        return self.net(x).squeeze(-1)


def rollout(env, defender, attacker, critic, horizon, device):
    """Collect one attacker-defender trajectory from the hybrid environment."""
    obs_np = env.reset()
    storage = []
    for k in range(horizon):
        obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        a_def, logp_def, ent_def = defender.sample(obs)
        a_atk, logp_atk, ent_atk = attacker.sample(obs)
        value = critic(obs, a_def, a_atk)
        next_obs, rewards, done, _ = env.step(int(a_def.item()), int(a_atk.item()))
        # scale rewards to avoid large gradients
        r_def = torch.tensor(rewards["defender"] / 10.0, dtype=torch.float32, device=device)
        r_atk = torch.tensor(rewards["attacker"] / 10.0, dtype=torch.float32, device=device)
        storage.append((obs.squeeze(0), a_def.squeeze(0), a_atk.squeeze(0), logp_def.squeeze(0),
                        logp_atk.squeeze(0), ent_def.squeeze(0), ent_atk.squeeze(0), value.squeeze(0), r_def, r_atk))
        obs_np = next_obs
        if done:
            break
    return storage


def discounted_returns(rewards, gamma=0.99, device="cpu"):
    out = []
    G = torch.tensor(0.0, device=device)
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return list(reversed(out))


def train(args):
    """Train compact decentralized actors with centralized critics.

    The function returns defender and attacker actors.  When
    `return_history=True`, it also returns per-episode diagnostics for tutorial
    plots.  This is a readable CTDE skeleton, not a full MAPPO implementation.
    """
    _, resolved_device, _ = configure_torch(
        seed=args.seed,
        device=getattr(args, "device", "auto"),
        threads=getattr(args, "threads", 1),
    )
    device = torch.device(resolved_device)
    np.random.seed(args.seed)
    env = HybridCyberDefenseEnv(seed=args.seed)
    if args.smoke:
        env.cfg.horizon = 10
    if hasattr(args, "horizon") and args.horizon is not None:
        env.cfg.horizon = args.horizon
    defender = Actor(env.obs_dim, env.n_defender_actions, args.hidden).to(device)
    attacker = Actor(env.obs_dim, env.n_attacker_actions, args.hidden).to(device)
    critic_d = CentralCritic(env.obs_dim, env.n_defender_actions, env.n_attacker_actions, args.hidden).to(device)
    critic_a = CentralCritic(env.obs_dim, env.n_defender_actions, env.n_attacker_actions, args.hidden).to(device)
    opt = optim.Adam(list(defender.parameters()) + list(attacker.parameters()) +
                     list(critic_d.parameters()) + list(critic_a.parameters()), lr=args.lr)
    history = []

    for ep in range(args.episodes):
        data = rollout(env, defender, attacker, critic_d, env.cfg.horizon, device)
        obs, ad, aa, logpd, logpa, entd, enta, vd_old, rd, ra = zip(*data)
        obs = torch.stack(obs)
        ad = torch.stack(ad)
        aa = torch.stack(aa)
        logpd = torch.stack(logpd)
        logpa = torch.stack(logpa)
        entd = torch.stack(entd)
        enta = torch.stack(enta)
        rd = list(rd)
        ra = list(ra)
        Gd = torch.stack(discounted_returns(rd, args.gamma, device=device)).detach()
        Ga = torch.stack(discounted_returns(ra, args.gamma, device=device)).detach()
        Vd = critic_d(obs, ad, aa)
        Va = critic_a(obs, ad, aa)
        adv_d = (Gd - Vd.detach())
        adv_a = (Ga - Va.detach())
        # In a general-sum game, each actor optimizes its own objective.
        actor_loss = -(logpd * adv_d).mean() - (logpa * adv_a).mean()
        critic_loss = nn.functional.mse_loss(Vd, Gd) + nn.functional.mse_loss(Va, Ga)
        entropy_bonus = (entd.mean() + enta.mean())
        loss = actor_loss + 0.5 * critic_loss - args.entropy_coef * entropy_bonus
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(defender.parameters()) + list(attacker.parameters()), 5.0)
        opt.step()
        if ep % args.log_every == 0:
            rd_total = sum(x.item() for x in rd)
            ra_total = sum(x.item() for x in ra)
            print(f"ep={ep:04d}, len={len(data):03d}, Rd={rd_total:7.2f}, Ra={ra_total:7.2f}, loss={loss.detach().item():.3f}")
            history.append({
                "episode": ep,
                "rollout_length": len(data),
                "defender_return": float(rd_total),
                "attacker_return": float(ra_total),
                "loss": float(loss.detach().item()),
                "critic_loss": float(critic_loss.detach().item()),
                "entropy": float(entropy_bonus.detach().item()),
            })
    if getattr(args, "return_history", False):
        return defender, attacker, history
    return defender, attacker


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Run a compact CTDE attacker-defender training loop."
    )
    p.add_argument("--smoke", action="store_true", help="Run a tiny execution check.")
    p.add_argument("--episodes", type=int, default=200, help="Number of training episodes.")
    p.add_argument("--horizon", type=int, default=None, help="Decision epochs per episode.")
    p.add_argument("--hidden", type=int, default=128, help="Hidden width for actors and critics.")
    p.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate.")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    p.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient.")
    p.add_argument("--log-every", type=int, default=10, help="Episode interval for console logs and history rows.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto", help="Training device.")
    p.add_argument("--threads", type=int, default=1, help="Torch CPU thread count; use 0 to leave unchanged.")
    args = p.parse_args()
    if args.smoke:
        args.episodes = 3
        args.log_every = 1
        args.hidden = 32
    train(args)
