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

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .envs import EnvConfig, SampledContinuousImpulseCyberEnv, scripted_attacker

from cybercontrol.nn import MLP as SharedMLP, parameter_count
from cybercontrol.rl import ReplayBuffer
from cybercontrol.torch_utils import configure_torch

LOGGER = logging.getLogger(__name__)


def make_q_network(in_dim, out_dim, hidden=128, depth=2):
    """Build the DDQN Q-network using the shared MLP implementation."""

    return SharedMLP(in_dim, out_dim, width=hidden, depth=depth, activation=nn.ReLU)


def evaluate(qnet, episodes=5, seed=1000, horizon=None):
    """Evaluate the greedy defender policy against the scripted attacker."""
    device = next(qnet.parameters()).device
    env = SampledContinuousImpulseCyberEnv(
        config=EnvConfig(randomize_initial_state=True),
        seed=seed,
    )
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
    `python -m cybergames medium`.  The returned network maps the current
    pre-jump cyber observation at a decision epoch to Q-values for the five
    defender actions.
    """
    _, resolved_device, _ = configure_torch(
        seed=args.seed,
        device=getattr(args, "device", "auto"),
        threads=getattr(args, "threads", 1),
    )
    device = torch.device(resolved_device)
    rng = np.random.default_rng(args.seed)
    env = SampledContinuousImpulseCyberEnv(seed=args.seed)
    if args.smoke:
        env.cfg.horizon = 10
    if hasattr(args, "horizon") and args.horizon is not None:
        env.cfg.horizon = args.horizon
    depth = getattr(args, "depth", 2)
    q = make_q_network(env.obs_dim, env.n_defender_actions, hidden=args.hidden, depth=depth).to(
        device
    )
    target = make_q_network(
        env.obs_dim, env.n_defender_actions, hidden=args.hidden, depth=depth
    ).to(device)
    target.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.buffer_size, seed=args.seed)
    global_step = 0
    history = []

    for ep in range(args.episodes):
        s = env.reset()
        ep_return = 0.0
        last_loss = float("nan")
        for k in range(env.cfg.horizon):
            eps = args.eps_end + (args.eps_start - args.eps_end) * np.exp(
                -global_step / args.eps_decay
            )
            if rng.random() < eps:
                a = int(rng.integers(env.n_defender_actions))
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
                bs, ba, br, bsp, bd = replay.sample_torch(args.batch_size, device=device)
                q_sa = q(bs).gather(1, ba)
                with torch.no_grad():
                    next_a = q(bsp).argmax(1, keepdim=True)
                    next_q = target(bsp).gather(1, next_a)
                    y = br + args.gamma * (1.0 - bd) * next_q
                loss = nn.functional.smooth_l1_loss(q_sa, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()
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
            LOGGER.info("ep=%04d return=%8.2f eval=%8.2f eps=%.3f", ep, ep_return, val, eps)
            history.append(
                {
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
                    "parameters": parameter_count(q),
                }
            )
    if getattr(args, "return_history", False):
        return q, history
    return q
