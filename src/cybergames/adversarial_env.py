"""Sparse heterogeneous node-SIPS attacker-defender environment.

The state has shape ``(nodes, 3)`` and order ``[S, I, P]``. Each row remains
on the probability simplex. Defender actions select communities for patching
or cleaning; attacker actions select communities receiving a temporary
receiver-side transmission multiplier during the next sampled-data interval.
Source-node infectivity is unchanged. Neither action resets the state, so this
is a sampled-flow Markov game rather than an impulse game.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from cybercontrol.graph_utils import community_mean, sparse_scale_free_graph
from cybercontrol.heterogeneity import node_heterogeneity_summary
from cybercontrol.network_models import (
    community_correlated_node_sips_params,
    contiguous_community_index,
    node_sips_rhs_numpy,
)
from cybercontrol.numerics import project_compartments, rk4_integrate

from .configs import AdversarialSIPSConfig


def build_sparse_scale_free_graph(cfg: AdversarialSIPSConfig):
    """Return a normalized sparse scale-free graph in model convention."""

    return sparse_scale_free_graph(
        cfg.nodes,
        mean_degree=cfg.mean_degree,
        seed=cfg.seed,
    ).tocsr()


class AdversarialSIPSEnv:
    """Node-SIPS Markov game with community attacker and defender actions.

    For node ``i``, infection pressure is
    ``susceptibility_i * A @ (infectivity * I)``. Patching moves ``S -> P``;
    recovery and cleaning move ``I -> P``; waning moves ``P -> S``. The
    defender minimizes weighted infection exposure plus action cost, while the
    attacker maximizes exposure minus its action cost.
    """

    DEFENDER_POLICIES = (
        "none",
        "uniform",
        "degree",
        "risk",
        "oracle",
        "budget_random",
        "static_logit",
        "learned",
    )
    ATTACKER_POLICIES = DEFENDER_POLICIES

    def __init__(self, cfg: AdversarialSIPSConfig | None = None):
        self.cfg = cfg or AdversarialSIPSConfig()
        self.cfg.validate()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.community = contiguous_community_index(self.cfg.nodes, self.cfg.communities)
        self.adjacency = build_sparse_scale_free_graph(self.cfg)
        self.degree = np.asarray(self.adjacency.getnnz(axis=1), dtype=np.float64)
        self.params = community_correlated_node_sips_params(
            self.community,
            strength=self.cfg.heterogeneity_strength,
            beta=self.cfg.beta,
            gamma=self.cfg.gamma,
            omega=self.cfg.omega,
        )
        self.resolved_params = self.params.resolve(self.cfg.nodes)
        self.k = 0
        self.state = self._initial_state()

    def _initial_state(self) -> np.ndarray:
        state = np.zeros((self.cfg.nodes, 3), dtype=np.float64)
        state[:, 0] = 1.0 - self.cfg.initial_infected
        state[:, 1] = self.cfg.initial_infected
        high_degree = np.argsort(self.degree)[-max(1, self.cfg.nodes // 25) :]
        state[high_degree, 1] = np.minimum(0.30, state[high_degree, 1] + 0.08)
        state[high_degree, 0] = 1.0 - state[high_degree, 1]
        jitter = self.rng.normal(0.0, 0.008, size=self.cfg.nodes)
        state[:, 1] = np.clip(state[:, 1] + jitter, 0.005, 0.35)
        state[:, 0] = 1.0 - state[:, 1]
        return project_compartments(state)

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

        return community_mean(values, self.community, self.cfg.communities)

    def observation(self) -> np.ndarray:
        """Return community summaries with shape ``(communities, features)``."""

        infected = self.state[:, 1]
        pressure = np.asarray(
            self.adjacency @ (self.resolved_params.infectivity * infected)
        ).reshape(-1)
        rows = []
        for community in range(self.cfg.communities):
            mask = self.community == community
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
        receiver_boost = np.zeros(self.cfg.nodes, dtype=np.float64)
        defense_cost = 0.0
        attack_cost = 0.0
        for community in defender_communities:
            mask = self.community == int(community)
            local_infection = float(
                np.average(self.state[mask, 1], weights=self.resolved_params.criticality[mask])
            )
            if local_infection > self.cfg.initial_infected * 1.25:
                clean[mask] = np.minimum(
                    self.cfg.clean_rate, self.resolved_params.clean_bound[mask]
                )
                defense_cost += self.cfg.defense_cost * float(
                    np.mean(self.resolved_params.clean_cost[mask])
                )
            else:
                patch[mask] = np.minimum(
                    self.cfg.patch_rate, self.resolved_params.patch_bound[mask]
                )
                defense_cost += self.cfg.defense_cost * float(
                    np.mean(self.resolved_params.patch_cost[mask])
                )
        for community in attacker_communities:
            mask = self.community == int(community)
            receiver_boost[mask] = self.cfg.attack_boost
            attack_cost += self.cfg.attack_cost
        return patch, clean, receiver_boost, defense_cost, attack_cost

    def step(
        self,
        defender_communities: np.ndarray,
        attacker_communities: np.ndarray,
    ):
        """Integrate one interval under simultaneous sampled community actions."""

        patch, clean, receiver_boost, defense_action_cost, attack_action_cost = self._rates(
            defender_communities,
            attacker_communities,
        )

        def rhs_flat(values, _time):
            state = values.reshape(self.cfg.nodes, 3)
            return node_sips_rhs_numpy(
                state,
                self.adjacency,
                self.resolved_params,
                patch=patch,
                clean=clean,
                receiver_transmission_boost=receiver_boost,
            ).reshape(-1)

        values, _ = rk4_integrate(
            rhs_flat,
            self.state.reshape(-1),
            t0=self.k * self.cfg.dt,
            dt=self.cfg.dt,
            substeps=self.cfg.substeps,
            project=lambda x: project_compartments(x.reshape(self.cfg.nodes, 3)).reshape(-1),
        )
        self.state = values.reshape(self.cfg.nodes, 3)
        infected = self.state[:, 1]
        weighted_infected = float(np.mean(self.resolved_params.criticality * infected))
        global_infected = float(np.mean(infected))
        exposure = self.cfg.dt * (
            self.cfg.local_weight * weighted_infected + self.cfg.global_weight * global_infected
        )
        defender_payoff = -(exposure + defense_action_cost)
        attacker_payoff = exposure - attack_action_cost
        self.k += 1
        done = self.k >= self.cfg.horizon
        info = {
            "global_infected": global_infected,
            "weighted_infected": weighted_infected,
            "defender_payoff": defender_payoff,
            "attacker_payoff": attacker_payoff,
            "defense_action_cost": defense_action_cost,
            "attack_action_cost": attack_action_cost,
            "mass_error": float(np.max(np.abs(self.state.sum(axis=1) - 1.0))),
        }
        return self.observation(), defender_payoff, attacker_payoff, done, info
