"""
Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

DDQN defender for a sampled-data cyber-defense MDP.

The defender has discrete actions: none, patch, clean, deceive, isolate.  The
attacker is scripted.  This is the simplest RL bridge from a continuous-time ODE
model to an MDP:

    observe x(t_k^-) -> choose a_k -> jump/ODE transition -> reward -> replay

The checked-in MDP uses the fixed decision interval `EnvConfig.dt`.  More
general experiments can use nonuniform action intervals if the environment
records each `Delta t_k`.  RK4 substeps inside `env.step` are only numerical
integration points.
"""
from __future__ import annotations

import argparse
import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from cyber_hybrid_env import HybridCyberDefenseEnv, scripted_attacker
from shared_setup import ensure_foundation_package, resolve_torch_device

ensure_foundation_package()
from cybercontrol.torch_utils import MLP as SharedMLP
from cybercontrol.torch_utils import configure_torch

Transition = namedtuple("Transition", "s a r sp done")


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.data = deque(maxlen=capacity)

    def push(self, *args):
        self.data.append(Transition(*args))

    def sample(self, batch_size, device="cpu"):
        batch = random.sample(self.data, batch_size)
        s = torch.tensor(np.stack([b.s for b in batch]), dtype=torch.float32, device=device)
        a = torch.tensor([b.a for b in batch], dtype=torch.int64, device=device).unsqueeze(1)
        r = torch.tensor([b.r for b in batch], dtype=torch.float32, device=device).unsqueeze(1)
        sp = torch.tensor(np.stack([b.sp for b in batch]), dtype=torch.float32, device=device)
        done = torch.tensor([b.done for b in batch], dtype=torch.float32, device=device).unsqueeze(1)
        return s, a, r, sp, done

    def __len__(self):
        return len(self.data)


def make_q_network(in_dim, out_dim, hidden=128, depth=2):
    """Build the DDQN Q-network using the shared MLP implementation."""

    return SharedMLP(in_dim, out_dim, width=hidden, depth=depth, activation=nn.ReLU)


def evaluate(qnet, episodes=5, seed=1000, horizon=None):
    """Evaluate the greedy defender policy against the scripted attacker."""
    device = next(qnet.parameters()).device
    env = HybridCyberDefenseEnv(seed=seed)
    if horizon is not None:
        env.cfg.horizon = horizon
    returns = []
    for ep in range(episodes):
        s = env.reset()
        total = 0.0
        for k in range(env.cfg.horizon):
            with torch.no_grad():
                obs = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                a = int(qnet(obs).argmax(1).item())
            sp, rewards, done, _ = env.step(a, scripted_attacker(env, k))
            total += rewards["defender"]
            s = sp
            if done:
                break
        returns.append(total)
    return float(np.mean(returns))


def train(args):
    """Train a DDQN defender and optionally return logged history.

    Required arguments are provided by the CLI or by
    `scripts/run_training_iterations.py`.  The returned network maps the current
    pre-jump cyber observation at a decision epoch to Q-values for the five
    defender actions.
    """
    random.seed(args.seed); np.random.seed(args.seed)
    _, resolved_device, _ = resolve_torch_device(
        configure_torch,
        seed=args.seed,
        device=getattr(args, "device", "auto"),
        threads=getattr(args, "threads", 1),
    )
    device = torch.device(resolved_device)
    env = HybridCyberDefenseEnv(seed=args.seed)
    if args.smoke:
        env.cfg.horizon = 10
    if hasattr(args, "horizon") and args.horizon is not None:
        env.cfg.horizon = args.horizon
    depth = getattr(args, "depth", 2)
    q = make_q_network(env.obs_dim, env.n_defender_actions, hidden=args.hidden, depth=depth).to(device)
    target = make_q_network(env.obs_dim, env.n_defender_actions, hidden=args.hidden, depth=depth).to(device)
    target.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.buffer_size)
    global_step = 0
    history = []

    for ep in range(args.episodes):
        s = env.reset()
        ep_return = 0.0
        last_loss = float("nan")
        for k in range(env.cfg.horizon):
            eps = args.eps_end + (args.eps_start - args.eps_end)*np.exp(-global_step/args.eps_decay)
            if random.random() < eps:
                a = random.randrange(env.n_defender_actions)
            else:
                with torch.no_grad():
                    obs = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    a = int(q(obs).argmax(1).item())
            sp, rewards, done, _ = env.step(a, scripted_attacker(env, k))
            r = rewards["defender"] / 10.0  # scale for stable Q values
            replay.push(s, a, r, sp, done)
            s = sp
            ep_return += rewards["defender"]
            global_step += 1

            if len(replay) >= args.batch_size:
                bs, ba, br, bsp, bd = replay.sample(args.batch_size, device=device)
                q_sa = q(bs).gather(1, ba)
                with torch.no_grad():
                    next_a = q(bsp).argmax(1, keepdim=True)
                    next_q = target(bsp).gather(1, next_a)
                    y = br + args.gamma*(1.0-bd)*next_q
                loss = nn.functional.smooth_l1_loss(q_sa, y)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 5.0); opt.step()
                last_loss = float(loss.detach().item())

            if global_step % args.target_update == 0:
                target.load_state_dict(q.state_dict())
            if done:
                break
        if ep % args.log_every == 0:
            eval_horizon = getattr(args, "eval_horizon", None)
            if eval_horizon is None and args.smoke:
                eval_horizon = 10
            val = evaluate(q, episodes=getattr(args, "eval_episodes", 2), horizon=eval_horizon)
            print(f"ep={ep:04d}, return={ep_return:8.2f}, eval={val:8.2f}, eps={eps:.3f}")
            history.append({
                "episode": ep,
                "training_return": float(ep_return),
                "evaluation_return": float(val),
                "epsilon": float(eps),
                "last_td_loss": last_loss,
                "replay_size": len(replay),
                "device": str(device),
                "hidden": int(args.hidden),
                "depth": int(depth),
                "batch_size": int(args.batch_size),
            })
    if getattr(args, "return_history", False):
        return q, history
    return q


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a DDQN defender in the hybrid cyber-defense environment."
    )
    parser.add_argument("--smoke", action="store_true", help="Run a tiny execution check.")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes.")
    parser.add_argument("--horizon", type=int, default=None, help="Decision epochs per episode.")
    parser.add_argument("--eval-horizon", type=int, default=None, help="Decision epochs for evaluation episodes.")
    parser.add_argument("--eval-episodes", type=int, default=2, help="Evaluation episodes per log point.")
    parser.add_argument("--batch-size", type=int, default=128, help="Replay minibatch size.")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden width for the Q-network.")
    parser.add_argument("--depth", type=int, default=2, help="Hidden-layer depth for the Q-network.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay-buffer capacity.")
    parser.add_argument("--target-update", type=int, default=500, help="Target-network sync period in environment steps.")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Initial epsilon for exploration.")
    parser.add_argument("--eps-end", type=float, default=0.05, help="Final epsilon for exploration.")
    parser.add_argument("--eps-decay", type=float, default=20000.0, help="Exponential epsilon-decay time constant.")
    parser.add_argument("--log-every", type=int, default=25, help="Episode interval for console logs and history rows.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto", help="Training device.")
    parser.add_argument("--threads", type=int, default=1, help="Torch CPU thread count; use 0 to leave unchanged.")
    args = parser.parse_args()
    if args.smoke:
        args.episodes = 2
        args.batch_size = 4
        args.log_every = 1
        args.target_update = 10
    train(args)
