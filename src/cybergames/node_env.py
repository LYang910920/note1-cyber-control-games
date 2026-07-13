"""Heterogeneous node-SIPS environment for community-level defenders.

The node state is ``[S, I, P]`` with one row per node and invariant
``S_i + I_i + P_i = 1``. At each decision epoch, community actions select
no intervention, patching ``S -> P``, or cleaning ``I -> P``. Rates are held
constant over the interval; this module contains no impulse reset.
"""

from __future__ import annotations

import numpy as np

from cybercontrol.graph_utils import community_graph
from cybercontrol.heterogeneity import node_heterogeneity_summary
from cybercontrol.network_models import (
    community_correlated_node_sips_params,
    contiguous_community_index,
    node_sips_rhs_numpy,
    normalize_adjacency,
)
from cybercontrol.numerics import project_compartments, rk4_integrate

from .configs import NodeSIPSEnvConfig


def build_community_graph(cfg: NodeSIPSEnvConfig, rng: np.random.Generator) -> np.ndarray:
    """Create one canonical seeded community graph for an environment reset."""

    seed = int(rng.integers(0, 2**31 - 1))
    adjacency, _ = community_graph(
        cfg.nodes,
        cfg.communities,
        mean_degree=cfg.mean_degree,
        seed=seed,
    )
    return adjacency


class NodeSIPSEnv:
    """Node-probability SIPS flow environment with explicit observation shape.

    ``observation()`` returns ``[communities, 12]``. Each row contains local
    SIPS means, boundary pressure, global infection, a budget indicator,
    normalized time-to-go, previous action, and four known parameter-risk
    summaries. The information structure is therefore fully observed at the
    community-summary level rather than a hidden-parameter POMDP.
    """

    ACTIONS = ("none", "patch", "clean")

    def __init__(self, cfg: NodeSIPSEnvConfig | None = None):
        self.cfg = cfg or NodeSIPSEnvConfig()
        self.cfg.validate()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.community = contiguous_community_index(self.cfg.nodes, self.cfg.communities)
        self.adjacency = build_community_graph(self.cfg, self.rng)
        self.params = community_correlated_node_sips_params(
            self.community,
            strength=self.cfg.heterogeneity_strength,
            beta=self.cfg.beta,
            gamma=self.cfg.gamma,
            omega=self.cfg.omega,
        )
        self.resolved_params = self.params.resolve(self.cfg.nodes)
        self.obs_dim = 12
        self.n_agents = self.cfg.communities
        self.n_actions = len(self.ACTIONS)
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset graph and state using only the environment's local generator."""

        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.adjacency = build_community_graph(self.cfg, self.rng)
        self.k = 0
        self.prev_actions = np.zeros(self.n_agents, dtype=np.float64)
        state = np.zeros((self.cfg.nodes, 3), dtype=np.float64)
        state[:, 0] = 1.0 - self.cfg.initial_infected
        state[:, 1] = self.cfg.initial_infected
        jitter = self.rng.normal(0.0, 0.01, size=self.cfg.nodes)
        state[:, 1] = np.clip(state[:, 1] + jitter, 0.01, 0.35)
        state[:, 0] = 1.0 - state[:, 1]
        self.state = project_compartments(state)
        return self.observation()

    def observation(self) -> np.ndarray:
        """Return one explicit 12-feature vector per defender community."""

        observations = []
        infected = self.state[:, 1]
        global_infected = float(infected.mean())
        pressure = self.adjacency @ infected
        time_to_go = 1.0 - self.k / max(1, self.cfg.horizon)
        for community in range(self.n_agents):
            mask = self.community == community
            local = self.state[mask].mean(axis=0)
            boundary_pressure = float(pressure[mask].mean())
            risk_summary = node_heterogeneity_summary(self.resolved_params, mask)
            observations.append(
                np.r_[
                    local,
                    boundary_pressure,
                    global_infected,
                    1.0,
                    time_to_go,
                    self.prev_actions[community],
                    risk_summary,
                ]
            )
        return np.asarray(observations, dtype=np.float32)

    def community_adjacency(self) -> np.ndarray:
        """Aggregate the current node graph into a normalized community graph."""

        membership = np.eye(self.n_agents, dtype=np.float64)[self.community]
        matrix = membership.T @ self.adjacency @ membership
        matrix += np.eye(self.n_agents, dtype=np.float64)
        return normalize_adjacency(matrix)

    def _action_rates(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        patch = np.zeros(self.cfg.nodes, dtype=np.float64)
        clean = np.zeros(self.cfg.nodes, dtype=np.float64)
        for community, action in enumerate(actions):
            mask = self.community == community
            if int(action) == 1:
                patch[mask] = np.minimum(
                    self.cfg.patch_rate, self.resolved_params.patch_bound[mask]
                )
            elif int(action) == 2:
                clean[mask] = np.minimum(
                    self.cfg.clean_rate, self.resolved_params.clean_bound[mask]
                )
        return patch, clean

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict[str, float]]:
        """Apply ZOH community rates and integrate one SIPS decision interval."""

        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.n_agents,):
            raise ValueError(f"actions must have shape ({self.n_agents},)")
        if np.any((actions < 0) | (actions >= self.n_actions)):
            raise ValueError(f"actions must be integers in [0, {self.n_actions - 1}]")
        patch, clean = self._action_rates(actions)

        def rhs_flat(state, _time):
            node_state = state.reshape(self.cfg.nodes, 3)
            return node_sips_rhs_numpy(
                node_state,
                self.adjacency,
                self.params,
                patch=patch,
                clean=clean,
            ).reshape(-1)

        next_state, _ = rk4_integrate(
            rhs_flat,
            self.state.reshape(-1),
            t0=self.k * self.cfg.dt,
            dt=self.cfg.dt,
            substeps=self.cfg.substeps,
            project=lambda state: project_compartments(state.reshape(self.cfg.nodes, 3)).reshape(
                -1
            ),
        )
        self.state = next_state.reshape(self.cfg.nodes, 3)
        global_infected = float(self.state[:, 1].mean())
        rewards = []
        for community, action in enumerate(actions):
            mask = self.community == community
            local_infected = float(
                np.average(
                    self.state[mask, 1],
                    weights=self.resolved_params.criticality[mask],
                )
            )
            if int(action) == 1:
                cost = self.cfg.action_cost * float(np.mean(self.resolved_params.patch_cost[mask]))
            elif int(action) == 2:
                cost = self.cfg.action_cost * float(np.mean(self.resolved_params.clean_cost[mask]))
            else:
                cost = 0.0
            rewards.append(
                -self.cfg.dt
                * (
                    self.cfg.local_weight * local_infected
                    + self.cfg.global_weight * global_infected
                    + cost
                )
            )
        self.prev_actions = actions.astype(np.float64) / max(1, self.n_actions - 1)
        self.k += 1
        done = self.k >= self.cfg.horizon
        info = {
            "global_infected": global_infected,
            "mean_patch_rate": float(patch.mean()),
            "mean_clean_rate": float(clean.mean()),
            "mass_error": float(np.max(np.abs(self.state.sum(axis=1) - 1.0))),
            "mean_risk_score": float(self.resolved_params.risk_score().mean()),
            "heterogeneity_strength": float(self.cfg.heterogeneity_strength),
        }
        return self.observation(), np.asarray(rewards, dtype=np.float32), done, info
