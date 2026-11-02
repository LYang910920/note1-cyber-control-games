"""DDQN for a sampled defender facing a scripted attacker.

An observation contains S, I, R and decision phase. An action is held through
one jump/ODE transition. Replay stores decision transitions, not ODE substeps.
Copyright (c) 2026 Luxing Yang. Licensed under the repository MIT License.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cybercontrol.experiments import configure_torch
from cybercontrol.nn import MLP as SharedMLP, parameter_count
from cybercontrol.rl import ReplayBuffer

from .envs import EnvConfig, SampledFlowImpulseEnv, scripted_attacker

LOGGER = logging.getLogger(__name__)


def make_q_network(in_dim, out_dim, hidden=128, depth=2):
    """Return one Q value per admissible defender action."""
    return SharedMLP(in_dim, out_dim, width=hidden, depth=depth, activation=nn.ReLU)


def evaluate(qnet, episodes=5, seed=1000, horizon=None):
    """Evaluate a frozen greedy policy on seeded, varied initial conditions."""
    device = next(qnet.parameters()).device
    env = SampledFlowImpulseEnv(config=EnvConfig(randomize_initial_state=True), seed=seed)
    if horizon is not None:
        env.cfg.horizon = horizon
    returns = []
    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        for decision in range(env.cfg.horizon):
            with torch.no_grad():
                observation = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(qnet(observation).argmax(1).item())
            state, rewards, done, _ = env.step(action, scripted_attacker(env, decision))
            total += rewards["defender"]
            if done:
                break
        returns.append(total)
    return float(np.mean(returns))


def train(args):
    """Create networks, collect transitions, update DDQN, and evaluate a frozen policy."""
    _, resolved_device, _ = configure_torch(
        seed=args.seed,
        device=getattr(args, "device", "auto"),
        threads=getattr(args, "threads", 1),
    )
    device = torch.device(resolved_device)
    rng = np.random.default_rng(args.seed)
    env = SampledFlowImpulseEnv(config=EnvConfig(randomize_initial_state=True), seed=args.seed)
    if args.smoke:
        env.cfg.horizon = 10
    if getattr(args, "horizon", None) is not None:
        env.cfg.horizon = args.horizon
    depth = getattr(args, "depth", 2)
    q = make_q_network(env.obs_dim, env.n_defender_actions, args.hidden, depth).to(device)
    target = make_q_network(env.obs_dim, env.n_defender_actions, args.hidden, depth).to(device)
    target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.buffer_size, seed=args.seed)
    global_step, history = 0, []

    for episode in range(args.episodes):
        state = env.reset()
        episode_return, last_loss = 0.0, float("nan")
        for decision in range(env.cfg.horizon):
            epsilon = args.eps_end + (args.eps_start - args.eps_end) * np.exp(
                -global_step / args.eps_decay
            )
            if rng.random() < epsilon:
                action = int(rng.integers(env.n_defender_actions))
            else:
                with torch.no_grad():
                    observation = torch.tensor(state, dtype=torch.float32, device=device)[None, :]
                    action = int(q(observation).argmax(1).item())
            next_state, rewards, done, _ = env.step(action, scripted_attacker(env, decision))
            reward = rewards["defender"] / 10.0  # Uniform scaling stabilizes Q magnitudes.
            replay.push(state, action, reward, next_state, done)
            state = next_state
            episode_return += rewards["defender"]
            global_step += 1

            if len(replay) >= args.batch_size:
                states, actions, rewards_batch, next_states, terminals = replay.sample_torch(
                    args.batch_size, device=device
                )
                predicted = q(states).gather(1, actions)
                with torch.no_grad():
                    next_actions = q(next_states).argmax(1, keepdim=True)
                    next_values = target(next_states).gather(1, next_actions)
                    expected = rewards_batch + args.gamma * (1.0 - terminals) * next_values
                loss = nn.functional.smooth_l1_loss(predicted, expected)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                optimizer.step()
                last_loss = float(loss.detach().item())
            if global_step % args.target_update == 0:
                target.load_state_dict(q.state_dict())
            if done:
                break

        if episode % args.log_every == 0 or episode == args.episodes - 1:
            eval_horizon = getattr(args, "eval_horizon", None)
            if eval_horizon is None and args.smoke:
                eval_horizon = 10
            value = evaluate(q, episodes=getattr(args, "eval_episodes", 2), horizon=eval_horizon)
            LOGGER.info("episode=%d return=%.3f validation=%.3f", episode, episode_return, value)
            history.append(
                {
                    "episode": episode,
                    "training_return": float(episode_return),
                    "evaluation_return": float(value),
                    "epsilon": float(epsilon),
                    "last_td_loss": last_loss,
                    "replay_size": len(replay),
                    "device": str(device),
                    "hidden": int(args.hidden),
                    "depth": int(depth),
                    "batch_size": int(args.batch_size),
                    "parameters": parameter_count(q),
                }
            )
    return (q, history) if getattr(args, "return_history", False) else q
