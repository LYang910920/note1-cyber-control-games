"""
Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

Node-level SIPRS community-defense environment and compact MAPPO baseline.

The environment uses the canonical ``cybercontrol.network_models`` SIPRS
equations.  Agents are regional defenders.  Each chooses one sampled-data mode
(``none``, ``patch``, or ``clean``) per decision epoch; the continuous node ODE
then flows until the next decision point.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from cybercontrol.heterogeneity import node_heterogeneity_summary
from cybercontrol.network_models import (
    community_correlated_node_siprs_params,
    contiguous_community_index,
    node_siprs_rhs_numpy,
    normalize_adjacency,
)
from cybercontrol.numerics import project_compartments, rk4_integrate
from cybercontrol.torch_utils import MLP, configure_torch


@dataclass
class NodeSIPRSEnvConfig:
    """Small deterministic node-SIPRS profile for MAPPO smoke experiments."""

    nodes: int = 48
    communities: int = 3
    horizon: int = 18
    dt: float = 0.5
    substeps: int = 4
    mean_degree: float = 5.0
    initial_infected: float = 0.08
    beta: float = 0.85
    gamma: float = 0.16
    omega_p: float = 0.035
    omega_r: float = 0.025
    patch_rate: float = 0.35
    clean_rate: float = 0.45
    heterogeneity_strength: float = 0.35
    local_weight: float = 1.0
    global_weight: float = 0.5
    action_cost: float = 0.03
    seed: int = 17


def build_community_graph(cfg: NodeSIPRSEnvConfig, rng: np.random.Generator) -> np.ndarray:
    """Build a small graph with stronger within-community connectivity."""

    n = cfg.nodes
    communities = contiguous_community_index(cfg.nodes, cfg.communities)
    A = np.zeros((n, n), dtype=np.float64)
    p_in = min(0.55, cfg.mean_degree / max(1, n // cfg.communities))
    p_out = min(0.08, p_in / 5.0)
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if communities[i] == communities[j] else p_out
            if rng.random() < p:
                A[i, j] = A[j, i] = 1.0
    for i in range(n):
        if A[i].sum() == 0.0:
            j = (i + 1) % n
            A[i, j] = A[j, i] = 1.0
    return normalize_adjacency(A)


class NodeSIPRSEnv:
    """Deterministic node-probability SIPRS environment for regional defenders."""

    ACTIONS = ("none", "patch", "clean")

    def __init__(self, cfg: NodeSIPRSEnvConfig | None = None):
        self.cfg = cfg or NodeSIPRSEnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.community = contiguous_community_index(self.cfg.nodes, self.cfg.communities)
        self.adjacency = build_community_graph(self.cfg, self.rng)
        self.params = community_correlated_node_siprs_params(
            self.community,
            strength=self.cfg.heterogeneity_strength,
            beta=self.cfg.beta,
            gamma=self.cfg.gamma,
            omega_p=self.cfg.omega_p,
            omega_r=self.cfg.omega_r,
        )
        self.resolved_params = self.params.resolve(self.cfg.nodes)
        self.obs_dim = 13
        self.n_agents = self.cfg.communities
        self.n_actions = len(self.ACTIONS)
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset to a deterministic noisy initial infection profile."""

        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.adjacency = build_community_graph(self.cfg, self.rng)
        self.k = 0
        self.prev_actions = np.zeros(self.n_agents, dtype=np.float64)
        x = np.zeros((self.cfg.nodes, 4), dtype=np.float64)
        x[:, 0] = 1.0 - self.cfg.initial_infected
        x[:, 1] = self.cfg.initial_infected
        jitter = self.rng.normal(0.0, 0.01, size=self.cfg.nodes)
        x[:, 1] = np.clip(x[:, 1] + jitter, 0.01, 0.35)
        x[:, 0] = 1.0 - x[:, 1]
        self.state = project_compartments(x)
        return self.observation()

    def observation(self) -> np.ndarray:
        """Return one local observation vector per defender community."""

        obs = []
        infected = self.state[:, 1]
        global_i = float(infected.mean())
        pressure = self.adjacency @ infected
        time_to_go = 1.0 - self.k / max(1, self.cfg.horizon)
        for m in range(self.n_agents):
            mask = self.community == m
            local = self.state[mask].mean(axis=0)
            boundary_pressure = float(pressure[mask].mean())
            risk_summary = node_heterogeneity_summary(self.resolved_params, mask)
            budget_proxy = 1.0
            obs.append(
                np.r_[
                    local,
                    boundary_pressure,
                    global_i,
                    budget_proxy,
                    time_to_go,
                    self.prev_actions[m],
                    risk_summary,
                ]
            )
        return np.asarray(obs, dtype=np.float32)

    def _action_rates(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        patch = np.zeros(self.cfg.nodes, dtype=np.float64)
        clean = np.zeros(self.cfg.nodes, dtype=np.float64)
        for m, action in enumerate(actions):
            mask = self.community == m
            if int(action) == 1:
                patch[mask] = np.minimum(self.cfg.patch_rate, self.resolved_params.patch_bound[mask])
            elif int(action) == 2:
                clean[mask] = np.minimum(self.cfg.clean_rate, self.resolved_params.clean_bound[mask])
        return patch, clean

    def step(self, actions: np.ndarray):
        """Apply community actions and integrate the node SIPRS ODE."""

        actions = np.asarray(actions, dtype=np.int64)
        patch, clean = self._action_rates(actions)

        def rhs_flat(y, t):
            x = y.reshape(self.cfg.nodes, 4)
            return node_siprs_rhs_numpy(x, self.adjacency, self.params, patch=patch, clean=clean).reshape(-1)

        y_next, _ = rk4_integrate(
            rhs_flat,
            self.state.reshape(-1),
            t0=self.k * self.cfg.dt,
            dt=self.cfg.dt,
            substeps=self.cfg.substeps,
            project=lambda y: project_compartments(y.reshape(self.cfg.nodes, 4)).reshape(-1),
        )
        self.state = y_next.reshape(self.cfg.nodes, 4)
        local_rewards = []
        global_i = float(self.state[:, 1].mean())
        for m, action in enumerate(actions):
            mask = self.community == m
            local_i = float(np.average(self.state[mask, 1], weights=self.resolved_params.criticality[mask]))
            if int(action) == 1:
                cost = self.cfg.action_cost * float(np.mean(self.resolved_params.patch_cost[mask]))
            elif int(action) == 2:
                cost = self.cfg.action_cost * float(np.mean(self.resolved_params.clean_cost[mask]))
            else:
                cost = 0.0
            local_rewards.append(-self.cfg.dt * (self.cfg.local_weight * local_i + self.cfg.global_weight * global_i + cost))
        self.prev_actions = actions.astype(np.float64) / max(1, self.n_actions - 1)
        self.k += 1
        done = self.k >= self.cfg.horizon
        info = {
            "global_infected": global_i,
            "mean_patch_rate": float(patch.mean()),
            "mean_clean_rate": float(clean.mean()),
            "mass_error": float(np.max(np.abs(self.state.sum(axis=1) - 1.0))),
            "mean_risk_score": float(self.resolved_params.risk_score().mean()),
            "heterogeneity_strength": float(self.cfg.heterogeneity_strength),
        }
        return self.observation(), np.asarray(local_rewards, dtype=np.float32), done, info


def _community_scores(env: NodeSIPRSEnv, values: np.ndarray) -> np.ndarray:
    """Average node scores over defender communities."""

    return np.asarray([float(np.mean(values[env.community == m])) for m in range(env.n_agents)], dtype=np.float64)


def _single_budget_action(env: NodeSIPRSEnv, community: int) -> np.ndarray:
    """Use one community-level intervention budget in a transparent way."""

    actions = np.zeros(env.n_agents, dtype=np.int64)
    mask = env.community == int(community)
    local_i = float(np.average(env.state[mask, 1], weights=env.resolved_params.criticality[mask]))
    actions[int(community)] = 2 if local_i >= env.cfg.initial_infected * 1.4 else 1
    return actions


def baseline_actions(env: NodeSIPRSEnv, policy: str, rng: np.random.Generator) -> np.ndarray:
    """Return a transparent baseline action vector for the current state.

    The non-learning baselines use one active community per decision epoch so
    that uniform, degree, risk, oracle, and random policies have the same
    community-level action budget.
    """

    if policy == "uniform":
        community = env.k % env.n_agents
    elif policy == "degree":
        degree = np.count_nonzero(env.adjacency > 0.0, axis=1) + np.count_nonzero(env.adjacency.T > 0.0, axis=1)
        community = int(np.argmax(_community_scores(env, degree)))
    elif policy == "risk":
        community = int(np.argmax(_community_scores(env, env.resolved_params.risk_score())))
    elif policy == "oracle":
        score = env.resolved_params.criticality * env.state[:, 1]
        community = int(np.argmax(_community_scores(env, score)))
    elif policy == "budget_random":
        community = int(rng.integers(0, env.n_agents))
    else:
        raise ValueError(f"unknown node-SIPRS baseline policy: {policy}")
    return _single_budget_action(env, community)


def learned_actions(actor, obs: np.ndarray, device: str) -> np.ndarray:
    """Greedy MAPPO actor actions for evaluation."""

    import torch

    with torch.no_grad():
        logits = actor(torch.tensor(obs, dtype=torch.float32, device=device))
        return torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int64)


def rollout_policy(policy: str, cfg: NodeSIPRSEnvConfig, *, actor=None, device: str = "cpu") -> dict[str, float | int | str]:
    """Roll out one policy on a held-out node-SIPRS profile."""

    env = NodeSIPRSEnv(cfg)
    obs = env.reset(seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed + 9_001)
    cumulative_reward = 0.0
    cumulative_infected = 0.0
    peak_infected = float(env.state[:, 1].mean())
    action_count = 0
    done = False
    last_info = {
        "global_infected": peak_infected,
        "mass_error": 0.0,
        "mean_risk_score": float(env.resolved_params.risk_score().mean()),
    }
    while not done:
        if policy == "learned_mappo":
            if actor is None:
                raise ValueError("learned_mappo rollout requires an actor")
            actions = learned_actions(actor, obs, device)
        else:
            actions = baseline_actions(env, policy, rng)
        obs, rewards, done, last_info = env.step(actions)
        global_i = float(last_info["global_infected"])
        cumulative_reward += float(np.mean(rewards))
        cumulative_infected += cfg.dt * global_i
        peak_infected = max(peak_infected, global_i)
        action_count += int(np.count_nonzero(actions))
    return {
        "policy": policy,
        "seed": int(cfg.seed),
        "heterogeneity_strength": float(cfg.heterogeneity_strength),
        "cumulative_reward": cumulative_reward,
        "cumulative_infected_exposure": cumulative_infected,
        "peak_global_infected": peak_infected,
        "final_global_infected": float(last_info["global_infected"]),
        "action_count": int(action_count),
        "mean_risk_score": float(last_info["mean_risk_score"]),
        "mass_error": float(last_info["mass_error"]),
    }


def evaluate_policy_baselines(
    *,
    actor=None,
    base_cfg: NodeSIPRSEnvConfig | None = None,
    seeds: tuple[int, ...] = (101, 102),
    strengths: tuple[float, ...] = (0.2, 0.5),
    device: str = "cpu",
) -> list[dict[str, float | int | str]]:
    """Compare transparent baselines and an optional learned actor on unseen profiles."""

    base_cfg = base_cfg or NodeSIPRSEnvConfig()
    policies = ["uniform", "degree", "risk", "oracle", "budget_random"]
    if actor is not None:
        policies.append("learned_mappo")
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for strength in strengths:
            cfg = replace(base_cfg, seed=seed, heterogeneity_strength=strength)
            for policy in policies:
                rows.append(rollout_policy(policy, cfg, actor=actor, device=device))
    return rows


class RolloutBuffer:
    """MAPPO rollout storage for one vectorized community game."""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.logp = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, actions, logp, rewards, done, value):
        self.obs.append(obs)
        self.actions.append(actions)
        self.logp.append(logp)
        self.rewards.append(rewards)
        self.dones.append(float(done))
        self.values.append(value)


def compute_gae(rewards, values, dones, last_value, gamma: float, lam: float):
    """Generalized advantage estimation for shared cooperative reward."""

    adv = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    return adv, adv + values


def train_mappo(args):
    """Train a compact cooperative MAPPO defender on node SIPRS dynamics."""

    torch, device, _ = configure_torch(seed=args.seed, device=args.device, threads=1)
    import torch.nn.functional as F
    from torch.distributions import Categorical

    cfg = NodeSIPRSEnvConfig(
        nodes=args.nodes,
        communities=args.communities,
        horizon=args.horizon,
        seed=args.seed,
        heterogeneity_strength=args.heterogeneity_strength,
    )
    env = NodeSIPRSEnv(cfg)
    actor = MLP(env.obs_dim, env.n_actions, width=args.hidden, depth=2).to(device)
    critic = MLP(env.obs_dim * env.n_agents, 1, width=args.hidden, depth=2).to(device)
    actor._cybercontrol_device = device
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)
    history = []

    for update in range(args.updates):
        obs = env.reset(seed=args.seed + update)
        buffer = RolloutBuffer()
        for _ in range(args.rollout_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            logits = actor(obs_t)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            logp = dist.log_prob(actions)
            value = critic(obs_t.reshape(1, -1)).squeeze(0).squeeze(-1)
            next_obs, rewards, done, info = env.step(actions.detach().cpu().numpy())
            shared_reward = float(np.mean(rewards))
            buffer.add(obs, actions.detach().cpu().numpy(), logp.detach().cpu().numpy(), shared_reward, done, float(value.detach().cpu()))
            obs = next_obs
            if done:
                obs = env.reset(seed=args.seed + update + 10_000)
        with torch.no_grad():
            last_value = float(critic(torch.tensor(obs, dtype=torch.float32, device=device).reshape(1, -1)).cpu().item())
        rewards = np.asarray(buffer.rewards, dtype=np.float32)
        values = np.asarray(buffer.values, dtype=np.float32)
        dones = np.asarray(buffer.dones, dtype=np.float32)
        advantages, returns = compute_gae(rewards, values, dones, last_value, args.gamma, args.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_arr = torch.tensor(np.asarray(buffer.obs), dtype=torch.float32, device=device)
        actions_arr = torch.tensor(np.asarray(buffer.actions), dtype=torch.int64, device=device)
        old_logp = torch.tensor(np.asarray(buffer.logp), dtype=torch.float32, device=device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
        flat_index = np.arange(args.rollout_steps)
        for _ in range(args.ppo_epochs):
            np.random.shuffle(flat_index)
            for start in range(0, args.rollout_steps, args.minibatch_size):
                idx = flat_index[start : start + args.minibatch_size]
                batch_obs = obs_arr[idx]
                batch_actions = actions_arr[idx]
                logits = actor(batch_obs.reshape(-1, env.obs_dim)).reshape(len(idx), env.n_agents, env.n_actions)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(logp.sum(dim=1) - old_logp[idx].sum(dim=1))
                unclipped = ratio * adv_t[idx]
                clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_t[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()

                value = critic(batch_obs.reshape(len(idx), -1)).squeeze(-1)
                value_loss = F.mse_loss(value, ret_t[idx])
                actor_opt.zero_grad()
                critic_opt.zero_grad()
                (policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy).backward()
                torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), args.max_grad_norm)
                actor_opt.step()
                critic_opt.step()
        history.append(
            {
                "update": update,
                "mean_reward": float(rewards.mean()),
                "final_global_infected": float(info["global_infected"]),
                "mass_error": float(info["mass_error"]),
            }
        )
        if update % args.log_every == 0:
            row = history[-1]
            print(
                f"update={update:03d}, reward={row['mean_reward']:.4f}, "
                f"global_I={row['final_global_infected']:.4f}, mass_error={row['mass_error']:.1e}"
            )
    return actor, critic, history


def build_parser():
    parser = argparse.ArgumentParser(description="Train a compact MAPPO baseline on node-level SIPRS.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--nodes", type=int, default=48)
    parser.add_argument("--communities", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=18)
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--rollout-steps", type=int, default=18)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--minibatch-size", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--heterogeneity-strength", type=float, default=0.35)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--policy-csv", type=Path, default=None, help="Optional held-out policy-baseline comparison CSV.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.device == "auto":
        args.device = None
    if args.smoke:
        args.nodes = 24
        args.communities = 3
        args.horizon = 6
        args.updates = 2
        args.rollout_steps = 6
        args.ppo_epochs = 2
        args.minibatch_size = 3
        args.hidden = 32
    actor, _, history = train_mappo(args)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f"wrote {args.output_csv}")
    if args.policy_csv is not None:
        rows = evaluate_policy_baselines(actor=actor, device=getattr(actor, "_cybercontrol_device", args.device or "cpu"))
        args.policy_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.policy_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.policy_csv}")
