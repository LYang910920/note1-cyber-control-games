"""Budget-matched baselines and held-out evaluation for node-SIPS policies."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .configs import NodeSIPSEnvConfig
from .node_env import NodeSIPSEnv


def _community_scores(env: NodeSIPSEnv, values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [float(np.mean(values[env.community == community])) for community in range(env.n_agents)],
        dtype=np.float64,
    )


def _single_budget_action(env: NodeSIPSEnv, community: int) -> np.ndarray:
    """Allocate one intervention to a selected community."""

    actions = np.zeros(env.n_agents, dtype=np.int64)
    mask = env.community == int(community)
    local_infected = float(
        np.average(
            env.state[mask, 1],
            weights=env.resolved_params.criticality[mask],
        )
    )
    actions[int(community)] = 2 if local_infected >= env.cfg.initial_infected * 1.4 else 1
    return actions


def node_sips_baseline_actions(
    env: NodeSIPSEnv, policy: str, rng: np.random.Generator
) -> np.ndarray:
    """Return one budget-matched transparent community action."""

    if policy == "uniform":
        community = env.k % env.n_agents
    elif policy == "degree":
        degree = np.count_nonzero(env.adjacency > 0.0, axis=1) + np.count_nonzero(
            env.adjacency.T > 0.0, axis=1
        )
        community = int(np.argmax(_community_scores(env, degree)))
    elif policy == "risk":
        community = int(np.argmax(_community_scores(env, env.resolved_params.risk_score())))
    elif policy == "oracle":
        score = env.resolved_params.criticality * env.state[:, 1]
        community = int(np.argmax(_community_scores(env, score)))
    elif policy == "budget_random":
        community = int(rng.integers(0, env.n_agents))
    else:
        raise ValueError(f"unknown node-SIPS baseline policy: {policy}")
    return _single_budget_action(env, community)


def rollout_node_sips_policy(
    policy: str,
    cfg: NodeSIPSEnvConfig,
    *,
    actor=None,
    device: str = "cpu",
) -> dict[str, float | int | str]:
    """Evaluate one policy on a deterministic held-out graph/profile pair."""

    env = NodeSIPSEnv(cfg)
    observation = env.reset(seed=cfg.seed)
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
    community_adjacency = env.community_adjacency()
    while not done:
        if policy == "learned_budgeted_ppo":
            if actor is None:
                raise ValueError("learned_budgeted_ppo rollout requires an actor")
            actions = actor.greedy(observation, device, community_adjacency)
        else:
            actions = node_sips_baseline_actions(env, policy, rng)
        observation, rewards, done, last_info = env.step(actions)
        global_infected = float(last_info["global_infected"])
        cumulative_reward += float(np.mean(rewards))
        cumulative_infected += cfg.dt * global_infected
        peak_infected = max(peak_infected, global_infected)
        action_count += int(np.count_nonzero(actions))
    return {
        "policy": policy,
        "seed": int(cfg.seed),
        "heterogeneity_strength": float(cfg.heterogeneity_strength),
        "cumulative_reward": cumulative_reward,
        "cumulative_infected_exposure": cumulative_infected,
        "peak_global_infected": peak_infected,
        "final_global_infected": float(last_info["global_infected"]),
        "action_count": action_count,
        "mean_risk_score": float(last_info["mean_risk_score"]),
        "mass_error": float(last_info["mass_error"]),
    }


def evaluate_policy_baselines(
    *,
    actor=None,
    base_cfg: NodeSIPSEnvConfig | None = None,
    seeds: tuple[int, ...] = (101, 102, 103, 104, 105),
    strengths: tuple[float, ...] = (0.2, 0.35, 0.5),
    device: str = "cpu",
) -> list[dict[str, float | int | str]]:
    """Compare learned and transparent policies on unseen profile seeds."""

    base_cfg = base_cfg or NodeSIPSEnvConfig()
    policies = ["uniform", "degree", "risk", "oracle", "budget_random"]
    if actor is not None:
        policies.append("learned_budgeted_ppo")
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for strength in strengths:
            cfg = replace(base_cfg, seed=seed, heterogeneity_strength=strength)
            for policy in policies:
                rows.append(
                    rollout_node_sips_policy(
                        policy,
                        cfg,
                        actor=actor,
                        device=device,
                    )
                )
    return rows
