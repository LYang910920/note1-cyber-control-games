"""
Small CTDE/MADRL example for an attacker-defender cyber Markov game.

This is a deliberately compact MAPPO-like skeleton:
  * decentralized categorical actors for defender and attacker;
  * a centralized critic that sees the joint state and both actions;
  * episodic rollouts through the hybrid ODE environment;
  * policy-gradient updates with a shared advantage estimate.

For serious experiments, replace this with a full MAPPO implementation that uses
GAE, mini-batches, clipped policy ratios, entropy scheduling, and opponent pools.
The value of this file is pedagogical: it makes the interaction loop explicit.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from cyber_hybrid_env import HybridCyberDefenseEnv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def rollout(env, defender, attacker, critic, horizon):
    obs_np = env.reset()
    storage = []
    for k in range(horizon):
        obs = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        a_def, logp_def, ent_def = defender.sample(obs)
        a_atk, logp_atk, ent_atk = attacker.sample(obs)
        value = critic(obs, a_def, a_atk)
        next_obs, rewards, done, _ = env.step(int(a_def.item()), int(a_atk.item()))
        # scale rewards to avoid large gradients
        r_def = torch.tensor(rewards["defender"] / 10.0, dtype=torch.float32, device=DEVICE)
        r_atk = torch.tensor(rewards["attacker"] / 10.0, dtype=torch.float32, device=DEVICE)
        storage.append((obs.squeeze(0), a_def.squeeze(0), a_atk.squeeze(0), logp_def.squeeze(0),
                        logp_atk.squeeze(0), ent_def.squeeze(0), ent_atk.squeeze(0), value.squeeze(0), r_def, r_atk))
        obs_np = next_obs
        if done:
            break
    return storage


def discounted_returns(rewards, gamma=0.99):
    out = []
    G = torch.tensor(0.0, device=DEVICE)
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return list(reversed(out))


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    env = HybridCyberDefenseEnv(seed=args.seed)
    if args.smoke:
        env.cfg.horizon = 10
    if hasattr(args, "horizon") and args.horizon is not None:
        env.cfg.horizon = args.horizon
    defender = Actor(env.obs_dim, env.n_defender_actions, args.hidden).to(DEVICE)
    attacker = Actor(env.obs_dim, env.n_attacker_actions, args.hidden).to(DEVICE)
    critic_d = CentralCritic(env.obs_dim, env.n_defender_actions, env.n_attacker_actions, args.hidden).to(DEVICE)
    critic_a = CentralCritic(env.obs_dim, env.n_defender_actions, env.n_attacker_actions, args.hidden).to(DEVICE)
    opt = optim.Adam(list(defender.parameters()) + list(attacker.parameters()) +
                     list(critic_d.parameters()) + list(critic_a.parameters()), lr=args.lr)
    history = []

    for ep in range(args.episodes):
        data = rollout(env, defender, attacker, critic_d, env.cfg.horizon)
        obs, ad, aa, logpd, logpa, entd, enta, vd_old, rd, ra = zip(*data)
        obs = torch.stack(obs); ad = torch.stack(ad); aa = torch.stack(aa)
        logpd = torch.stack(logpd); logpa = torch.stack(logpa)
        entd = torch.stack(entd); enta = torch.stack(enta)
        rd = list(rd); ra = list(ra)
        Gd = torch.stack(discounted_returns(rd, args.gamma)).detach()
        Ga = torch.stack(discounted_returns(ra, args.gamma)).detach()
        Vd = critic_d(obs, ad, aa)
        Va = critic_a(obs, ad, aa)
        adv_d = (Gd - Vd.detach())
        adv_a = (Ga - Va.detach())
        # In a general-sum game, each actor optimizes its own objective.
        actor_loss = -(logpd * adv_d).mean() - (logpa * adv_a).mean()
        critic_loss = nn.functional.mse_loss(Vd, Gd) + nn.functional.mse_loss(Va, Ga)
        entropy_bonus = (entd.mean() + enta.mean())
        loss = actor_loss + 0.5 * critic_loss - args.entropy_coef * entropy_bonus
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(list(defender.parameters()) + list(attacker.parameters()), 5.0); opt.step()
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
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.smoke:
        args.episodes = 3; args.log_every = 1; args.hidden = 32
    train(args)
