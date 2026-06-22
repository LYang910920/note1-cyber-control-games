"""
Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

Large heterogeneous node-SIPRS attacker-defender benchmark.

This file is a bounded, learning-ready bridge from the cooperative node-SIPRS
MAPPO example to larger attacker-defender graph games.  It keeps the same
foundation SIPRS equations, uses sparse graphs, and trains simple community
softmax policies by self-play.  The policy update is intentionally small so the
environment contract, metrics, and response matrix stay easy to inspect before
replacing the learners with MAPPO, MADDPG, or another MARL library.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse as sp

from cybercontrol.heterogeneity import node_heterogeneity_summary
from cybercontrol.network_models import (
    community_correlated_node_siprs_params,
    contiguous_community_index,
    node_siprs_rhs_numpy,
    normalize_adjacency,
)
from cybercontrol.numerics import project_compartments, rk4_integrate


@dataclass
class LargeAdversarialSIPRSConfig:
    """Large sparse node-SIPRS attacker-defender configuration."""

    nodes: int = 512
    communities: int = 8
    horizon: int = 18
    dt: float = 0.5
    substeps: int = 3
    mean_degree: float = 8.0
    initial_infected: float = 0.06
    beta: float = 0.86
    gamma: float = 0.15
    omega_p: float = 0.03
    omega_r: float = 0.02
    patch_rate: float = 0.32
    clean_rate: float = 0.42
    attack_boost: float = 0.65
    heterogeneity_strength: float = 0.40
    defender_budget: int = 2
    attacker_budget: int = 2
    local_weight: float = 1.0
    global_weight: float = 0.6
    defense_cost: float = 0.035
    attack_cost: float = 0.025
    seed: int = 29


def build_sparse_scale_free_graph(cfg: LargeAdversarialSIPRSConfig) -> sp.csr_matrix:
    """Return a row-normalized sparse Barabasi-Albert graph in model convention."""

    attachment = max(1, min(cfg.nodes - 1, int(round(cfg.mean_degree / 2.0))))
    graph = nx.barabasi_albert_graph(cfg.nodes, attachment, seed=cfg.seed)
    rows: list[int] = []
    cols: list[int] = []
    for i, j in graph.edges():
        rows.extend([i, j])
        cols.extend([j, i])
    data = np.ones(len(rows), dtype=np.float64)
    adjacency = sp.csr_matrix((data, (rows, cols)), shape=(cfg.nodes, cfg.nodes))
    return normalize_adjacency(adjacency).tocsr()


def _top_communities(scores: np.ndarray, budget: int) -> np.ndarray:
    """Return up to ``budget`` highest-scoring community indices."""

    if budget <= 0:
        return np.asarray([], dtype=int)
    count = min(int(budget), len(scores))
    return np.argsort(scores)[-count:][::-1].astype(int)


def _softmax(logits: np.ndarray, available: np.ndarray) -> np.ndarray:
    masked = np.asarray(logits, dtype=np.float64).copy()
    masked[~available] = -1e9
    shifted = masked - float(np.max(masked))
    probs = np.exp(shifted)
    probs[~available] = 0.0
    total = float(probs.sum())
    if total <= 0.0:
        probs = available.astype(np.float64)
        total = float(probs.sum())
    return probs / total


def _sample_without_replacement(
    logits: np.ndarray,
    budget: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample communities and return the REINFORCE gradient of log probability."""

    available = np.ones(len(logits), dtype=bool)
    gradient = np.zeros(len(logits), dtype=np.float64)
    chosen: list[int] = []
    for _ in range(min(int(budget), len(logits))):
        probs = _softmax(logits, available)
        idx = int(rng.choice(len(logits), p=probs))
        chosen.append(idx)
        gradient[idx] += 1.0
        gradient -= probs
        available[idx] = False
    return np.asarray(chosen, dtype=int), gradient


class LargeAdversarialSIPRSEnv:
    """Sparse node-SIPRS Markov game with community attacker/defender actions."""

    DEFENDER_POLICIES = ("none", "uniform", "degree", "risk", "oracle", "budget_random", "learned")
    ATTACKER_POLICIES = ("none", "uniform", "degree", "risk", "oracle", "budget_random", "learned")

    def __init__(self, cfg: LargeAdversarialSIPRSConfig | None = None):
        self.cfg = cfg or LargeAdversarialSIPRSConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.community = contiguous_community_index(self.cfg.nodes, self.cfg.communities)
        self.adjacency = build_sparse_scale_free_graph(self.cfg)
        self.degree = np.asarray(self.adjacency.getnnz(axis=1), dtype=np.float64)
        self.params = community_correlated_node_siprs_params(
            self.community,
            strength=self.cfg.heterogeneity_strength,
            beta=self.cfg.beta,
            gamma=self.cfg.gamma,
            omega_p=self.cfg.omega_p,
            omega_r=self.cfg.omega_r,
        )
        self.resolved_params = self.params.resolve(self.cfg.nodes)
        self.k = 0
        self.state = self._initial_state()

    def _initial_state(self) -> np.ndarray:
        x = np.zeros((self.cfg.nodes, 4), dtype=np.float64)
        x[:, 0] = 1.0 - self.cfg.initial_infected
        x[:, 1] = self.cfg.initial_infected
        high_degree = np.argsort(self.degree)[-max(1, self.cfg.nodes // 25) :]
        x[high_degree, 1] = np.minimum(0.30, x[high_degree, 1] + 0.08)
        x[high_degree, 0] = 1.0 - x[high_degree, 1]
        jitter = self.rng.normal(0.0, 0.008, size=self.cfg.nodes)
        x[:, 1] = np.clip(x[:, 1] + jitter, 0.005, 0.35)
        x[:, 0] = 1.0 - x[:, 1]
        return project_compartments(x)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset state and optionally rebuild the graph with a new seed."""

        if seed is not None:
            self.cfg = replace(self.cfg, seed=int(seed))
            self.rng = np.random.default_rng(seed)
            self.adjacency = build_sparse_scale_free_graph(self.cfg)
            self.degree = np.asarray(self.adjacency.getnnz(axis=1), dtype=np.float64)
        self.k = 0
        self.state = self._initial_state()
        return self.observation()

    def community_mean(self, values: np.ndarray) -> np.ndarray:
        """Average node values over communities."""

        arr = np.asarray(values, dtype=np.float64)
        return np.asarray(
            [float(np.mean(arr[self.community == c])) for c in range(self.cfg.communities)],
            dtype=np.float64,
        )

    def observation(self) -> np.ndarray:
        """Community summaries for learned policies or external MARL wrappers."""

        infected = self.state[:, 1]
        pressure = np.asarray(self.adjacency @ (self.resolved_params.infectivity * infected)).reshape(-1)
        rows = []
        for c in range(self.cfg.communities):
            mask = self.community == c
            risk_summary = node_heterogeneity_summary(self.resolved_params, mask)
            rows.append(
                np.r_[
                    self.state[mask].mean(axis=0),
                    float(pressure[mask].mean()),
                    float(infected.mean()),
                    float(np.mean(self.degree[mask])),
                    1.0 - self.k / max(1, self.cfg.horizon),
                    risk_summary,
                ]
            )
        return np.asarray(rows, dtype=np.float32)

    def _rates(
        self,
        defender_communities: np.ndarray,
        attacker_communities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        patch = np.zeros(self.cfg.nodes, dtype=np.float64)
        clean = np.zeros(self.cfg.nodes, dtype=np.float64)
        boost = np.zeros(self.cfg.nodes, dtype=np.float64)
        defense_cost = 0.0
        attack_cost = 0.0
        for c in defender_communities:
            mask = self.community == int(c)
            local_i = float(np.average(self.state[mask, 1], weights=self.resolved_params.criticality[mask]))
            if local_i > self.cfg.initial_infected * 1.25:
                clean[mask] = np.minimum(self.cfg.clean_rate, self.resolved_params.clean_bound[mask])
                defense_cost += self.cfg.defense_cost * float(np.mean(self.resolved_params.clean_cost[mask]))
            else:
                patch[mask] = np.minimum(self.cfg.patch_rate, self.resolved_params.patch_bound[mask])
                defense_cost += self.cfg.defense_cost * float(np.mean(self.resolved_params.patch_cost[mask]))
        for c in attacker_communities:
            mask = self.community == int(c)
            boost[mask] = self.cfg.attack_boost
            attack_cost += self.cfg.attack_cost
        return patch, clean, boost, defense_cost, attack_cost

    def step(self, defender_communities: np.ndarray, attacker_communities: np.ndarray):
        """Apply simultaneous community actions and integrate one decision interval."""

        patch, clean, boost, defense_action_cost, attack_action_cost = self._rates(
            defender_communities,
            attacker_communities,
        )

        def rhs_flat(y, _t):
            x = y.reshape(self.cfg.nodes, 4)
            return node_siprs_rhs_numpy(
                x,
                self.adjacency,
                self.resolved_params,
                patch=patch,
                clean=clean,
                beta_boost=boost,
            ).reshape(-1)

        y_next, _ = rk4_integrate(
            rhs_flat,
            self.state.reshape(-1),
            t0=self.k * self.cfg.dt,
            dt=self.cfg.dt,
            substeps=self.cfg.substeps,
            project=lambda y: project_compartments(y.reshape(self.cfg.nodes, 4)).reshape(-1),
        )
        self.state = y_next.reshape(self.cfg.nodes, 4)
        infected = self.state[:, 1]
        weighted_i = float(np.mean(self.resolved_params.criticality * infected))
        global_i = float(np.mean(infected))
        exposure = self.cfg.dt * (self.cfg.local_weight * weighted_i + self.cfg.global_weight * global_i)
        defender_payoff = -(exposure + defense_action_cost)
        attacker_payoff = exposure - attack_action_cost
        self.k += 1
        done = self.k >= self.cfg.horizon
        info = {
            "global_infected": global_i,
            "weighted_infected": weighted_i,
            "defender_payoff": defender_payoff,
            "attacker_payoff": attacker_payoff,
            "defense_action_cost": defense_action_cost,
            "attack_action_cost": attack_action_cost,
            "mass_error": float(np.max(np.abs(self.state.sum(axis=1) - 1.0))),
        }
        return self.observation(), defender_payoff, attacker_payoff, done, info


def _policy_scores(env: LargeAdversarialSIPRSEnv, role: str, policy: str) -> np.ndarray:
    infected = env.state[:, 1]
    susceptible = env.state[:, 0]
    risk = env.resolved_params.risk_score()
    if policy == "degree":
        return env.community_mean(env.degree)
    if policy == "risk":
        return env.community_mean(risk)
    if policy == "oracle" and role == "defender":
        return env.community_mean(env.resolved_params.criticality * infected)
    if policy == "oracle" and role == "attacker":
        return env.community_mean(env.resolved_params.criticality * susceptible * risk)
    raise ValueError(f"policy {policy!r} does not use deterministic scores")


def choose_communities(
    env: LargeAdversarialSIPRSEnv,
    role: str,
    policy: str,
    rng: np.random.Generator,
    *,
    logits: np.ndarray | None = None,
    sample: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose communities for one player and return a policy-gradient vector."""

    budget = env.cfg.defender_budget if role == "defender" else env.cfg.attacker_budget
    if policy == "none":
        return np.asarray([], dtype=int), np.zeros(env.cfg.communities, dtype=np.float64)
    if policy == "uniform":
        start = env.k % env.cfg.communities
        chosen = (start + np.arange(min(budget, env.cfg.communities))) % env.cfg.communities
        return chosen.astype(int), np.zeros(env.cfg.communities, dtype=np.float64)
    if policy == "budget_random":
        chosen = rng.choice(env.cfg.communities, size=min(budget, env.cfg.communities), replace=False)
        return np.asarray(chosen, dtype=int), np.zeros(env.cfg.communities, dtype=np.float64)
    if policy == "learned":
        if logits is None:
            raise ValueError("learned policy requires logits")
        if sample:
            return _sample_without_replacement(logits, budget, rng)
        return _top_communities(np.asarray(logits, dtype=np.float64), budget), np.zeros_like(logits)
    return _top_communities(_policy_scores(env, role, policy), budget), np.zeros(env.cfg.communities)


def rollout_game(
    defender_policy: str,
    attacker_policy: str,
    cfg: LargeAdversarialSIPRSConfig,
    *,
    seed: int | None = None,
    defender_logits: np.ndarray | None = None,
    attacker_logits: np.ndarray | None = None,
    sample_learned: bool = False,
) -> tuple[dict[str, float | int | str], np.ndarray, np.ndarray]:
    """Roll out one attacker-defender game and return metrics plus score gradients."""

    run_seed = cfg.seed if seed is None else int(seed)
    env = LargeAdversarialSIPRSEnv(replace(cfg, seed=run_seed))
    rng = np.random.default_rng(run_seed + 17_017)
    defender_grad = np.zeros(cfg.communities, dtype=np.float64)
    attacker_grad = np.zeros(cfg.communities, dtype=np.float64)
    defender_return = 0.0
    attacker_return = 0.0
    infected_exposure = 0.0
    peak_infected = float(env.state[:, 1].mean())
    last_info = {"mass_error": 0.0, "global_infected": peak_infected}
    done = False
    while not done:
        defender_action, dg = choose_communities(
            env,
            "defender",
            defender_policy,
            rng,
            logits=defender_logits,
            sample=sample_learned,
        )
        attacker_action, ag = choose_communities(
            env,
            "attacker",
            attacker_policy,
            rng,
            logits=attacker_logits,
            sample=sample_learned,
        )
        _, rd, ra, done, last_info = env.step(defender_action, attacker_action)
        defender_grad += dg
        attacker_grad += ag
        defender_return += rd
        attacker_return += ra
        global_i = float(last_info["global_infected"])
        infected_exposure += cfg.dt * global_i
        peak_infected = max(peak_infected, global_i)
    row = {
        "defender_policy": defender_policy,
        "attacker_policy": attacker_policy,
        "seed": run_seed,
        "nodes": cfg.nodes,
        "communities": cfg.communities,
        "heterogeneity_strength": cfg.heterogeneity_strength,
        "defender_payoff": defender_return,
        "attacker_payoff": attacker_return,
        "cumulative_infected_exposure": infected_exposure,
        "peak_global_infected": peak_infected,
        "final_global_infected": float(last_info["global_infected"]),
        "mass_error": float(last_info["mass_error"]),
    }
    return row, defender_grad, attacker_grad


def _center_clip(logits: np.ndarray, bound: float = 6.0) -> np.ndarray:
    centered = np.asarray(logits, dtype=np.float64) - float(np.mean(logits))
    return np.clip(centered, -bound, bound)


def train_self_play(
    cfg: LargeAdversarialSIPRSConfig,
    *,
    episodes: int = 40,
    lr: float = 0.08,
    seed: int | None = None,
) -> tuple[list[dict[str, float | int | str]], np.ndarray, np.ndarray]:
    """Train defender and attacker community logits with bounded self-play."""

    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    defender_logits = np.zeros(cfg.communities, dtype=np.float64)
    attacker_logits = np.zeros(cfg.communities, dtype=np.float64)
    defender_baseline = 0.0
    attacker_baseline = 0.0
    history: list[dict[str, float | int | str]] = []
    for episode in range(int(episodes)):
        run_seed = int(rng.integers(0, 2**31 - 1))
        row, dg, ag = rollout_game(
            "learned",
            "learned",
            cfg,
            seed=run_seed,
            defender_logits=defender_logits,
            attacker_logits=attacker_logits,
            sample_learned=True,
        )
        defender_payoff = float(row["defender_payoff"])
        attacker_payoff = float(row["attacker_payoff"])
        if episode == 0:
            defender_baseline = defender_payoff
            attacker_baseline = attacker_payoff
        defender_adv = defender_payoff - defender_baseline
        attacker_adv = attacker_payoff - attacker_baseline
        scale = 1.0 / max(1, cfg.horizon)
        defender_logits = _center_clip(defender_logits + lr * defender_adv * scale * dg)
        attacker_logits = _center_clip(attacker_logits + lr * attacker_adv * scale * ag)
        defender_baseline = 0.9 * defender_baseline + 0.1 * defender_payoff
        attacker_baseline = 0.9 * attacker_baseline + 0.1 * attacker_payoff
        history.append(
            {
                "episode": episode,
                **row,
                "defender_logit_max": float(np.max(defender_logits)),
                "attacker_logit_max": float(np.max(attacker_logits)),
            }
        )
    return history, defender_logits, attacker_logits


def evaluate_response_matrix(
    cfg: LargeAdversarialSIPRSConfig,
    defender_logits: np.ndarray | None = None,
    attacker_logits: np.ndarray | None = None,
    *,
    seeds: tuple[int, ...] = (101, 102, 103),
) -> list[dict[str, float | int | str]]:
    """Evaluate baseline and learned policies in a same-model response matrix."""

    defender_policies = ["none", "uniform", "degree", "risk", "oracle", "budget_random"]
    attacker_policies = ["none", "uniform", "degree", "risk", "oracle", "budget_random"]
    if defender_logits is not None:
        defender_policies.append("learned")
    if attacker_logits is not None:
        attacker_policies.append("learned")
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for defender in defender_policies:
            for attacker in attacker_policies:
                row, _, _ = rollout_game(
                    defender,
                    attacker,
                    replace(cfg, seed=seed),
                    defender_logits=defender_logits,
                    attacker_logits=attacker_logits,
                    sample_learned=False,
                )
                rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a large heterogeneous node-SIPRS attacker-defender benchmark.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--nodes", type=int, default=512)
    parser.add_argument("--communities", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=18)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--mean-degree", type=float, default=8.0)
    parser.add_argument("--heterogeneity-strength", type=float, default=0.40)
    parser.add_argument("--defender-budget", type=int, default=2)
    parser.add_argument("--attacker-budget", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--response-csv", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.nodes = 96
        args.communities = 6
        args.horizon = 6
        args.episodes = 4
    cfg = LargeAdversarialSIPRSConfig(
        nodes=args.nodes,
        communities=args.communities,
        horizon=args.horizon,
        mean_degree=args.mean_degree,
        heterogeneity_strength=args.heterogeneity_strength,
        defender_budget=args.defender_budget,
        attacker_budget=args.attacker_budget,
        seed=args.seed,
    )
    history, defender_logits, attacker_logits = train_self_play(cfg, episodes=args.episodes, lr=args.lr, seed=args.seed)
    if args.output_csv is not None:
        _write_csv(args.output_csv, history)
    if args.response_csv is not None:
        rows = evaluate_response_matrix(cfg, defender_logits, attacker_logits)
        _write_csv(args.response_csv, rows)
    final = history[-1]
    print(
        "final "
        f"defender_payoff={final['defender_payoff']:.3f}, "
        f"attacker_payoff={final['attacker_payoff']:.3f}, "
        f"infected_exposure={final['cumulative_infected_exposure']:.3f}, "
        f"mass_error={final['mass_error']:.1e}"
    )


if __name__ == "__main__":
    main()
