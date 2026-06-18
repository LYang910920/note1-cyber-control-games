# Copyright (c) 2026 Luxing Yang.
# Licensed under the MIT License. See LICENSE in the repository root.

"""Named scenario profiles for adapting Note 1 code.

Students usually need two questions answered before editing a model:

1. Which file should I change first?
2. Which parameters define the scenario I am trying to study?

This module keeps those answers in one readable place.  It does not replace
the environment or learners; it provides small, named starting points that can
be copied into scripts, tests, or paper-specific experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from cyber_dynamics import HybridParams
from cyber_hybrid_env import EnvConfig


@dataclass(frozen=True)
class CyberScenarioProfile:
    """A readable scenario contract for ODE-RL and Markov-game experiments."""

    name: str
    question: str
    state_level: str
    timing: str
    control_type: str
    first_files_to_edit: tuple[str, ...]
    paper_extension: str
    config_factory: Callable[[], EnvConfig]

    def make_config(self) -> EnvConfig:
        """Return a fresh environment config so experiments can mutate safely."""
        return self.config_factory()


@dataclass(frozen=True)
class TrainingHyperparameterProfile:
    """Named neural/optimization hyperparameters used by diagnostics."""

    name: str
    method: str
    hyperparameters: tuple[tuple[str, str], ...]
    source: str


def tutorial_hybrid_small() -> EnvConfig:
    """Default compact hybrid cyber-defense setting."""
    return EnvConfig(
        dt=1.0,
        substeps=10,
        horizon=100,
        params=HybridParams(beta0=0.65, gamma=0.05, omega=0.01, chi=0.70, xi=0.04, zeta=0.08),
        randomize_initial_state=False,
    )


def impulse_visible_defense() -> EnvConfig:
    """Scenario where isolation jumps are intentionally easy to see."""
    return EnvConfig(
        dt=1.0,
        substeps=12,
        horizon=80,
        params=HybridParams(beta0=0.80, gamma=0.04, omega=0.015, chi=0.65, xi=0.04, zeta=0.10),
        c_isolate=1.4,
        usability_cost=1.8,
        randomize_initial_state=True,
    )


def paper_network_bridge() -> EnvConfig:
    """Longer-horizon profile used before moving to node-level or graph models."""
    return EnvConfig(
        dt=0.5,
        substeps=8,
        horizon=160,
        params=HybridParams(beta0=0.75, gamma=0.06, omega=0.02, chi=0.55, xi=0.03, zeta=0.12),
        w_I=12.0,
        c_deceive=1.2,
        c_isolate=2.4,
        randomize_initial_state=True,
    )


SCENARIOS: dict[str, CyberScenarioProfile] = {
    "tutorial-hybrid-small": CyberScenarioProfile(
        name="tutorial-hybrid-small",
        question="Can the full continuous/impulse/hybrid loop run end to end?",
        state_level="aggregate S/I/R/z compartments",
        timing="fixed sampled-data decisions t_k with RK4 substeps inside each interval",
        control_type="hybrid: continuous rates plus optional impulse isolation",
        first_files_to_edit=("src/cyber_dynamics.py", "src/cyber_hybrid_env.py"),
        paper_extension="Add compartments, budgets, attacker knowledge states, or richer jump maps.",
        config_factory=tutorial_hybrid_small,
    ),
    "impulse-visible-defense": CyberScenarioProfile(
        name="impulse-visible-defense",
        question="What happens when impulsive isolation has a visible state jump?",
        state_level="aggregate S/I/R/z compartments",
        timing="fixed t_k decisions; impulse is applied before continuous ODE flow",
        control_type="impulse-dominant hybrid control",
        first_files_to_edit=("src/cyber_hybrid_env.py", "src/evaluation_metrics.py"),
        paper_extension="Replace the scalar jump with node-local, edge-local, or event-triggered jumps.",
        config_factory=impulse_visible_defense,
    ),
    "paper-network-bridge": CyberScenarioProfile(
        name="paper-network-bridge",
        question="How should I prepare the teaching environment for a larger paper-style model?",
        state_level="aggregate state now; transition point toward node-level graph states",
        timing="shorter decision interval with more policy steps",
        control_type="hybrid control with stochastic initial states",
        first_files_to_edit=("src/node_level_robustness.py", "src/madrl_ctde_hybrid_game.py"),
        paper_extension="Move the state from aggregate compartments to graph/node features and run multi-seed stress tests.",
        config_factory=paper_network_bridge,
    ),
}

TRAINING_HYPERPARAMETERS: tuple[TrainingHyperparameterProfile, ...] = (
    TrainingHyperparameterProfile(
        name="fbsm-diagnostics",
        method="forward-backward sweep baseline",
        hyperparameters=(
            ("T", "24.0"),
            ("grid intervals n", "100"),
            ("max_iter", "35"),
            ("relax", "solver default 0.5"),
            ("tol", "solver default 1e-5"),
        ),
        source="scripts/run_training_iterations.py::run_fbsm",
    ),
    TrainingHyperparameterProfile(
        name="ddqn-defender",
        method="DDQN neural defender",
        hyperparameters=(
            ("episodes", "180 by default in scripts/run_training_iterations.py"),
            ("horizon/eval_horizon", "24 / 24"),
            ("eval_episodes", "4"),
            ("hidden width", "64"),
            ("batch_size", "32"),
            ("learning rate", "1e-3"),
            ("gamma", "0.99"),
            ("replay buffer", "10000"),
            ("target_update", "80"),
            ("epsilon schedule", "1.0 -> 0.02, decay 450"),
            ("seed", "11"),
        ),
        source="scripts/run_training_iterations.py::run_ddqn",
    ),
    TrainingHyperparameterProfile(
        name="ctde-madrl-game",
        method="compact CTDE/MADRL attacker-defender game",
        hyperparameters=(
            ("episodes", "180 by default in scripts/run_training_iterations.py"),
            ("horizon", "18"),
            ("hidden width", "48"),
            ("learning rate", "5e-4"),
            ("gamma", "0.97"),
            ("entropy_coef", "0.02"),
            ("seed", "13"),
        ),
        source="scripts/run_training_iterations.py::run_madrl",
    ),
    TrainingHyperparameterProfile(
        name="node-level-robustness",
        method="node-level epidemic stress test",
        hyperparameters=(
            ("nodes", "160"),
            ("horizon", "45"),
            ("mean_degree", "8.0"),
            ("beta_true", "1.25"),
            ("burst interval", "14 to 30"),
            ("burst_multiplier", "1.35"),
            ("nominal beta for FBSM", "0.45"),
            ("graph seeds", "21 to 28"),
        ),
        source="src/node_level_robustness.py and scripts/run_training_iterations.py::run_node_level_robustness",
    ),
)


def get_scenario(name: str) -> CyberScenarioProfile:
    """Return a named scenario profile with a helpful error message."""
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        available = ", ".join(sorted(SCENARIOS))
        raise KeyError(f"unknown scenario {name!r}; available: {available}") from exc


def describe_scenarios() -> list[dict[str, str]]:
    """Small table-friendly summary used by docs and tests."""
    rows = []
    for profile in SCENARIOS.values():
        cfg = profile.make_config()
        rows.append(
            {
                "name": profile.name,
                "state_level": profile.state_level,
                "control_type": profile.control_type,
                "horizon": str(cfg.horizon),
                "dt": str(cfg.dt),
                "beta0": f"{cfg.params.beta0:.3f}",
                "first_files_to_edit": ", ".join(profile.first_files_to_edit),
            }
        )
    return rows


def describe_training_hyperparameters() -> list[dict[str, str]]:
    """Return neural/optimization hyperparameters in a compact printable form."""
    return [
        {
            "name": profile.name,
            "method": profile.method,
            "hyperparameters": "; ".join(f"{key}={value}" for key, value in profile.hyperparameters),
            "source": profile.source,
        }
        for profile in TRAINING_HYPERPARAMETERS
    ]


if __name__ == "__main__":
    print("Scenario parameters:")
    for row in describe_scenarios():
        print(
            f"{row['name']}: state={row['state_level']}; control={row['control_type']}; "
            f"horizon={row['horizon']}; dt={row['dt']}; beta0={row['beta0']}; "
            f"edit={row['first_files_to_edit']}"
        )
    print("\nTraining and neural hyperparameters:")
    for row in describe_training_hyperparameters():
        print(f"{row['name']}: method={row['method']}; {row['hyperparameters']}; source={row['source']}")
