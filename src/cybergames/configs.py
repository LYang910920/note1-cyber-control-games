"""Typed learning configurations used by CLI and experiment runners."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NodeSIPSEnvConfig:
    """Deterministic node-SIPS profile for bounded cooperative experiments."""

    nodes: int = 48
    communities: int = 3
    horizon: int = 18
    dt: float = 0.5
    substeps: int = 4
    mean_degree: float = 5.0
    initial_infected: float = 0.08
    beta: float = 0.85
    gamma: float = 0.16
    omega: float = 0.035
    patch_rate: float = 0.35
    clean_rate: float = 0.45
    heterogeneity_strength: float = 0.35
    local_weight: float = 1.0
    global_weight: float = 0.5
    action_cost: float = 0.03
    seed: int = 17

    def validate(self) -> None:
        if self.nodes <= 1:
            raise ValueError("nodes must exceed one")
        if not 1 <= self.communities <= self.nodes:
            raise ValueError("communities must be between one and nodes")
        if min(self.horizon, self.substeps) <= 0:
            raise ValueError("horizon and substeps must be positive")


@dataclass(frozen=True)
class AdversarialSIPSConfig:
    """Sparse heterogeneous node-SIPS attacker-defender configuration."""

    nodes: int = 512
    communities: int = 8
    horizon: int = 18
    dt: float = 0.5
    substeps: int = 3
    mean_degree: float = 8.0
    initial_infected: float = 0.06
    beta: float = 0.86
    gamma: float = 0.15
    omega: float = 0.03
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

    def validate(self) -> None:
        if self.nodes <= 1:
            raise ValueError("nodes must exceed one")
        if not 1 <= self.communities <= self.nodes:
            raise ValueError("communities must be between one and nodes")
        if min(self.horizon, self.substeps, self.defender_budget, self.attacker_budget) <= 0:
            raise ValueError("horizon, substeps, and action budgets must be positive")


@dataclass
class DDQNConfig:
    smoke: bool = False
    episodes: int = 300
    horizon: int | None = None
    eval_horizon: int | None = None
    eval_episodes: int = 2
    batch_size: int = 128
    hidden: int = 128
    depth: int = 2
    lr: float = 1e-3
    gamma: float = 0.99
    buffer_size: int = 50_000
    target_update: int = 500
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 20_000.0
    log_every: int = 25
    seed: int = 0
    device: str = "auto"
    threads: int = 1
    return_history: bool = True


@dataclass
class CTDEConfig:
    smoke: bool = False
    episodes: int = 200
    horizon: int | None = None
    hidden: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    log_every: int = 10
    seed: int = 0
    device: str = "auto"
    threads: int = 1
    return_history: bool = True


@dataclass
class MAPPOConfig:
    nodes: int = 48
    communities: int = 3
    horizon: int = 18
    updates: int = 12
    rollout_steps: int = 18
    ppo_epochs: int = 3
    minibatch_size: int = 6
    hidden: int = 64
    lr: float = 3e-4
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "auto"
    threads: int = 1
    seed: int = 17
    heterogeneity_strength: float = 0.35
    log_every: int = 1
    action_budget: int = 1
    architecture: str = "summary_mlp"
    graph_layers: int = 1

    def validate(self) -> None:
        if self.nodes <= 1:
            raise ValueError("nodes must exceed one")
        if not 1 <= self.communities <= self.nodes:
            raise ValueError("communities must be between one and nodes")
        if self.action_budget != 1:
            raise ValueError("the current budgeted MAPPO actor supports action_budget=1")
        if self.architecture not in {"summary_mlp", "graph_context"}:
            raise ValueError("architecture must be summary_mlp or graph_context")
        if self.graph_layers <= 0:
            raise ValueError("graph_layers must be positive")
        if min(self.horizon, self.updates, self.rollout_steps, self.minibatch_size) <= 0:
            raise ValueError("horizon, updates, rollout_steps, and minibatch_size must be positive")
        if self.rollout_steps > self.horizon:
            raise ValueError("rollout_steps cannot exceed horizon in the current graph batch")
