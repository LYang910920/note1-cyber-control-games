"""
Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

Node-level epidemic-model robustness experiment for Note 1.

The experiment is small and deterministic enough for a laptop. It
illustrates a setting where a low-dimensional FBSM policy is a useful theory
baseline but is no longer the best operational controller:

* each graph node has a local S/I/R state and node-level infection pressure;
* the FBSM policy is open-loop and solved with a nominal, underestimated beta;
* the feedback policy observes the current aggregate state and can react with
  patch, clean, deceive, or isolate actions.

Here "robustness" means performance under this parameter mismatch and burst
disturbance.  It is measured by cumulative/peak/final infected-node share across
random graph seeds, not by a formal adversarial-robustness certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Union
import numpy as np


ActionDict = Dict[str, Union[float, str]]
NodePolicy = Callable[[int, np.ndarray], ActionDict]


@dataclass
class NodeSimConfig:
    """Parameters for the node-level epidemic malware simulator."""

    nodes: int = 160
    horizon: int = 45
    dt: float = 1.0
    mean_degree: float = 8.0
    initial_infected: float = 0.05
    beta_true: float = 1.25
    gamma: float = 0.035
    import_rate: float = 0.002
    burst_start: int = 14
    burst_end: int = 30
    burst_multiplier: float = 1.35
    protection_loss: float = 0.004
    patch_scale: float = 0.20
    clean_scale: float = 0.32
    isolate_scale: float = 0.48
    deception_strength: float = 0.55


def build_erdos_graph(nodes: int, mean_degree: float, rng: np.random.Generator) -> np.ndarray:
    """Build an undirected random graph adjacency matrix."""
    p = min(0.35, mean_degree / max(1, nodes - 1))
    upper = rng.random((nodes, nodes)) < p
    upper = np.triu(upper, 1)
    graph = upper | upper.T
    return graph.astype(np.float64)


def aggregate_observation(state: np.ndarray, deception_level: float = 0.0) -> np.ndarray:
    """Return the aggregate [S, I, R, z] observation used by feedback policies."""
    n = len(state)
    susceptible = float(np.count_nonzero(state == 0) / n)
    infected = float(np.count_nonzero(state == 1) / n)
    protected = float(np.count_nonzero(state == 2) / n)
    return np.array([susceptible, infected, protected, deception_level], dtype=np.float64)


def action_from_defender_mode(mode: int) -> ActionDict:
    """Map Note 1 defender modes to node-level intervention probabilities."""
    if mode == 1:
        return {"label": "patch", "patch": 1.0, "clean": 0.0, "isolate": 0.0, "deceive": 0.0}
    if mode == 2:
        return {"label": "clean", "patch": 0.0, "clean": 1.0, "isolate": 0.0, "deceive": 0.0}
    if mode == 3:
        return {"label": "deceive", "patch": 0.0, "clean": 0.0, "isolate": 0.0, "deceive": 1.0}
    if mode == 4:
        return {"label": "isolate", "patch": 0.0, "clean": 0.0, "isolate": 1.0, "deceive": 0.0}
    return {"label": "none", "patch": 0.0, "clean": 0.0, "isolate": 0.0, "deceive": 0.0}


def fbsm_open_loop_policy(control: np.ndarray) -> NodePolicy:
    """Use an FBSM open-loop curve as a node-level patching schedule."""
    def policy(k: int, obs: np.ndarray) -> ActionDict:
        u = float(control[min(k, len(control) - 1)])
        return {"label": "fbsm_patch", "patch": u, "clean": 0.0, "isolate": 0.0, "deceive": 0.0}

    return policy


def no_defense_node_policy(k: int, obs: np.ndarray) -> ActionDict:
    return action_from_defender_mode(0)


def node_step(
    state: np.ndarray,
    graph: np.ndarray,
    action: ActionDict,
    k: int,
    cfg: NodeSimConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Advance one node-level transition."""
    next_state = state.copy()
    n = len(state)

    patch = float(action.get("patch", 0.0))
    clean = float(action.get("clean", 0.0))
    isolate = float(action.get("isolate", 0.0))
    deceive = float(action.get("deceive", 0.0))

    susceptible = next_state == 0
    infected = next_state == 1

    patch_draw = susceptible & (rng.random(n) < cfg.patch_scale * patch)
    next_state[patch_draw] = 2

    clean_draw = infected & (rng.random(n) < cfg.clean_scale * clean)
    next_state[clean_draw] = 2

    infected = next_state == 1
    isolate_draw = infected & (rng.random(n) < cfg.isolate_scale * isolate)
    next_state[isolate_draw] = 2

    infected = next_state == 1
    recover_draw = infected & (rng.random(n) < cfg.gamma * cfg.dt)
    next_state[recover_draw] = 2

    protected = next_state == 2
    lose_draw = protected & (rng.random(n) < cfg.protection_loss * cfg.dt)
    next_state[lose_draw] = 0

    infected_vector = (next_state == 1).astype(np.float64)
    degree = np.maximum(graph.sum(axis=1), 1.0)
    neighbor_pressure = graph @ infected_vector / degree
    burst = cfg.burst_multiplier if cfg.burst_start <= k < cfg.burst_end else 1.0
    effective_beta = cfg.beta_true * burst * max(0.0, 1.0 - cfg.deception_strength * deceive)
    infection_rate = effective_beta * neighbor_pressure + cfg.import_rate * burst
    infection_prob = 1.0 - np.exp(-infection_rate * cfg.dt)
    susceptible = next_state == 0
    infection_draw = susceptible & (rng.random(n) < infection_prob)
    next_state[infection_draw] = 1

    action_cost = (
        0.7 * np.count_nonzero(patch_draw)
        + 1.0 * np.count_nonzero(clean_draw)
        + 2.0 * np.count_nonzero(isolate_draw)
        + 0.25 * deceive * n
    ) / n
    info = {
        "new_infections": float(np.count_nonzero(infection_draw) / n),
        "patched": float(np.count_nonzero(patch_draw) / n),
        "cleaned": float(np.count_nonzero(clean_draw) / n),
        "isolated": float(np.count_nonzero(isolate_draw) / n),
        "action_cost": float(action_cost),
        "effective_beta": float(effective_beta),
    }
    return next_state, info


def rollout_node_policy(
    label: str,
    policy: NodePolicy,
    seed: int,
    cfg: NodeSimConfig | None = None,
) -> Dict[str, object]:
    """Roll out one node-level policy.

    A rollout is one forward simulation on one random graph seed.  The returned
    observations are aggregate S/I/R/z summaries over the node states at each
    action epoch.
    """
    cfg = cfg or NodeSimConfig()
    rng = np.random.default_rng(seed)
    graph = build_erdos_graph(cfg.nodes, cfg.mean_degree, rng)
    state = np.zeros(cfg.nodes, dtype=np.int8)
    infected_count = max(1, int(round(cfg.initial_infected * cfg.nodes)))
    initial_infected = rng.choice(cfg.nodes, size=infected_count, replace=False)
    state[initial_infected] = 1

    observations = [aggregate_observation(state)]
    costs: List[float] = []
    actions: List[str] = []
    beta_values: List[float] = []

    for k in range(cfg.horizon):
        obs = observations[-1]
        action = policy(k, obs)
        state, info = node_step(state, graph, action, k, cfg, rng)
        observations.append(aggregate_observation(state, float(action.get("deceive", 0.0))))
        costs.append(info["action_cost"])
        actions.append(str(action.get("label", "unknown")))
        beta_values.append(info["effective_beta"])

    return {
        "label": label,
        "seed": seed,
        "cfg": cfg,
        "observations": np.asarray(observations),
        "costs": np.asarray(costs, dtype=np.float64),
        "actions": actions,
        "effective_beta": np.asarray(beta_values, dtype=np.float64),
    }


def summarize_node_rollout(rollout: Dict[str, object], beta_assumed: float) -> Dict[str, float | int | str]:
    """Return compact metrics for a node-level rollout.

    ``node_pmp_unknown_proxy`` is a scale indicator for full node-level FBSM:
    it counts state and costate variables over the time grid.  It is not a
    measured runtime or a learned loss.
    """
    cfg: NodeSimConfig = rollout["cfg"]  # type: ignore[assignment]
    observations = rollout["observations"]
    infected = observations[:, 1]
    costs = rollout["costs"]
    state_dim = 3 * cfg.nodes
    pmp_unknown_proxy = 2 * state_dim * (cfg.horizon + 1)
    return {
        "policy": str(rollout["label"]),
        "seed": int(rollout["seed"]),
        "nodes": cfg.nodes,
        "horizon": cfg.horizon,
        "beta_assumed_by_fbsm": beta_assumed,
        "beta_true_base": cfg.beta_true,
        "burst_multiplier": cfg.burst_multiplier,
        "cumulative_compromised": float(cfg.dt * infected[:-1].sum()),
        "peak_compromised": float(infected.max()),
        "final_compromised": float(infected[-1]),
        "total_defender_cost": float(costs.sum()),
        "state_dimension": state_dim,
        "node_pmp_unknown_proxy": pmp_unknown_proxy,
    }
