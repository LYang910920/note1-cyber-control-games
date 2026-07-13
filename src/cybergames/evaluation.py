"""
Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

Policy-rollout metrics for the Note 1 sampled-flow and impulse examples.

The helpers keep method-comparison plots and experiment CSV files consistent.
They also make the timing convention explicit: policies observe at decision
epochs, while RK4 substeps remain internal simulator steps.
"""

from __future__ import annotations

from dataclasses import replace
from statistics import fmean
from typing import Callable, Dict, List, Tuple
import numpy as np

from .actions import Action, mode_intensity
from .envs import (
    EnvConfig,
    SampledContinuousImpulseCyberEnv,
    scripted_attacker,
)

PolicyFn = Callable[[SampledContinuousImpulseCyberEnv, int, np.ndarray], Action]
AttackerFn = Callable[[SampledContinuousImpulseCyberEnv, int, np.ndarray], Action]


def no_defense_policy(env: SampledContinuousImpulseCyberEnv, k: int, obs: np.ndarray) -> Action:
    return env.DEF_NONE


def always_patch_policy(env: SampledContinuousImpulseCyberEnv, k: int, obs: np.ndarray) -> Action:
    return mode_intensity(env.DEF_PATCH, 0.8)


def always_clean_policy(env: SampledContinuousImpulseCyberEnv, k: int, obs: np.ndarray) -> Action:
    return mode_intensity(env.DEF_CLEAN, 0.8)


def adaptive_sampled_impulse_policy(
    env: SampledContinuousImpulseCyberEnv, k: int, obs: np.ndarray
) -> Action:
    """Threshold rule using deception, patching, and isolate impulses."""
    if obs[1] > 0.20:
        return mode_intensity(env.DEF_ISOLATE, 0.8)
    if k < 20:
        return mode_intensity(env.DEF_DECEIVE, 0.7)
    return mode_intensity(env.DEF_PATCH, 0.7)


def scripted_attacker_policy(
    env: SampledContinuousImpulseCyberEnv, k: int, obs: np.ndarray
) -> Action:
    """Scenario attacker used in the single-defender tutorial experiments."""
    return scripted_attacker(env, k)


def fixed_scan_attacker(env: SampledContinuousImpulseCyberEnv, k: int, obs: np.ndarray) -> Action:
    return env.ATK_SCAN


def fixed_exploit_attacker(
    env: SampledContinuousImpulseCyberEnv, k: int, obs: np.ndarray
) -> Action:
    return env.ATK_EXPLOIT


def fixed_lateral_attacker(
    env: SampledContinuousImpulseCyberEnv, k: int, obs: np.ndarray
) -> Action:
    return env.ATK_LATERAL


def fixed_stealth_attacker(
    env: SampledContinuousImpulseCyberEnv, k: int, obs: np.ndarray
) -> Action:
    return env.ATK_STEALTH


def policy_suite() -> List[Tuple[str, PolicyFn]]:
    """Return deterministic comparison policies used by figures and CSVs."""
    return [
        ("No defense", no_defense_policy),
        ("Fixed high patch", always_patch_policy),
        ("Fixed high clean", always_clean_policy),
        ("Rule threshold isolate/deceive/patch", adaptive_sampled_impulse_policy),
    ]


def attacker_suite() -> List[Tuple[str, AttackerFn]]:
    """Return attacker policies for game-style response matrices."""
    return [
        ("Scripted scan -> exploit -> lateral attacker", scripted_attacker_policy),
        ("Fixed exploit attacker", fixed_exploit_attacker),
        ("Fixed lateral attacker", fixed_lateral_attacker),
        ("Fixed stealth attacker", fixed_stealth_attacker),
    ]


def rollout_policy(
    label: str,
    policy: PolicyFn,
    horizon: int = 50,
    seed: int = 7,
    config: EnvConfig | None = None,
    attacker_policy: AttackerFn | None = None,
) -> Dict[str, object]:
    """Roll out one policy and retain states, rewards, actions, and timing."""
    env = (
        SampledContinuousImpulseCyberEnv(config=config, seed=seed)
        if config
        else SampledContinuousImpulseCyberEnv(seed=seed)
    )
    env.cfg.horizon = horizon
    if attacker_policy is None:
        attacker_policy = scripted_attacker_policy
    obs = env.reset()
    states = [obs.copy()]
    defender_modes: List[int] = []
    attacker_modes: List[int] = []
    defender_rewards: List[float] = []
    attacker_rewards: List[float] = []
    interval_compromised: List[float] = []
    jump_removed: List[float] = []

    for k in range(horizon):
        defender_action = policy(env, k, obs)
        attacker_action = attacker_policy(env, k, obs)
        next_obs, rewards, done, info = env.step(defender_action, attacker_action)
        defender_mode, _ = env.decode_action(defender_action)
        attacker_mode, _ = env.decode_action(attacker_action)
        defender_modes.append(defender_mode)
        attacker_modes.append(attacker_mode)
        defender_rewards.append(float(rewards["defender"]))
        attacker_rewards.append(float(rewards["attacker"]))
        interval_compromised.append(float(info["path"][:, 1].mean()))
        jump_removed.append(max(0.0, float(info["pre_jump"][1] - info["post_jump"][1])))
        obs = next_obs
        states.append(obs.copy())
        if done:
            break

    return {
        "label": label,
        "states": np.asarray(states),
        "defender_modes": np.asarray(defender_modes, dtype=np.int64),
        "attacker_modes": np.asarray(attacker_modes, dtype=np.int64),
        "defender_rewards": np.asarray(defender_rewards, dtype=np.float64),
        "attacker_rewards": np.asarray(attacker_rewards, dtype=np.float64),
        "interval_compromised": np.asarray(interval_compromised, dtype=np.float64),
        "jump_removed": np.asarray(jump_removed, dtype=np.float64),
        "decision_dt": float(env.cfg.dt),
        "rk4_substeps": int(env.cfg.substeps),
        "horizon": int(horizon),
    }


def summarize_rollout(rollout: Dict[str, object]) -> Dict[str, float | int | str]:
    """Summarize one rollout with metrics used in tables and comparison plots."""
    states = rollout["states"]
    defender_modes = rollout["defender_modes"]
    interval_compromised = rollout["interval_compromised"]
    defender_rewards = rollout["defender_rewards"]
    attacker_rewards = rollout["attacker_rewards"]
    jump_removed = rollout["jump_removed"]
    dt = float(rollout["decision_dt"])
    deception_steps = int(
        np.count_nonzero(defender_modes == SampledContinuousImpulseCyberEnv.DEF_DECEIVE)
    )
    action_switches = (
        int(np.sum(defender_modes[1:] != defender_modes[:-1])) if len(defender_modes) > 1 else 0
    )
    total_defender_reward = float(np.sum(defender_rewards))
    return {
        "policy": str(rollout["label"]),
        "decision_dt": dt,
        "decision_epochs": int(len(defender_modes)),
        "rk4_substeps": int(rollout["rk4_substeps"]),
        "total_defender_reward": total_defender_reward,
        "total_defender_cost": -total_defender_reward,
        "total_attacker_reward": float(np.sum(attacker_rewards)),
        "cumulative_compromised": float(dt * np.sum(interval_compromised)),
        "peak_compromised": float(np.max(states[:, 1])),
        "final_compromised": float(states[-1, 1]),
        "final_protected": float(states[-1, 2]),
        "deception_action_fraction": float(deception_steps / max(1, len(defender_modes))),
        "impulse_events": int(np.count_nonzero(jump_removed > 1e-12)),
        "total_impulse_removed": float(np.sum(jump_removed)),
        "action_switches": action_switches,
    }


def evaluate_policy_suite(
    horizon: int = 50,
    seed: int = 7,
    config: EnvConfig | None = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, float | int | str]]]:
    """Roll out all representative policies and return raw and summary data."""
    rollouts = [
        rollout_policy(
            label,
            policy,
            horizon=horizon,
            seed=seed,
            config=replace(config) if config is not None else None,
        )
        for label, policy in policy_suite()
    ]
    rows = [summarize_rollout(rollout) for rollout in rollouts]
    return rollouts, rows


def evaluate_game_response_matrix(
    defender_policies: List[Tuple[str, PolicyFn]] | None = None,
    attacker_policies: List[Tuple[str, AttackerFn]] | None = None,
    horizon: int = 40,
    seed: int = 17,
) -> List[Dict[str, float | int | str]]:
    """Evaluate defender policies against several attacker strategies."""
    defender_policies = defender_policies or policy_suite()
    attacker_policies = attacker_policies or attacker_suite()
    rows: List[Dict[str, float | int | str]] = []
    for defender_label, defender_policy in defender_policies:
        for attacker_label, attacker_policy in attacker_policies:
            rollout = rollout_policy(
                defender_label,
                defender_policy,
                horizon=horizon,
                seed=seed,
                attacker_policy=attacker_policy,
            )
            summary = summarize_rollout(rollout)
            summary["defender_policy"] = defender_label
            summary["attacker_policy"] = attacker_label
            rows.append(summary)
    return rows


def evaluate_large_response_matrix(
    cfg,
    defender_logits: np.ndarray | None = None,
    attacker_logits: np.ndarray | None = None,
    *,
    seeds: tuple[int, ...] = (101, 102, 103),
) -> list[dict[str, float | int | str]]:
    """Evaluate static-logit and transparent policies on the same graph game."""

    from .adversarial import rollout_game

    defenders = ["none", "uniform", "degree", "risk", "oracle", "budget_random"]
    attackers = ["none", "uniform", "degree", "risk", "oracle", "budget_random"]
    if defender_logits is not None:
        defenders.append("static_logit")
    if attacker_logits is not None:
        attackers.append("static_logit")
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for defender in defenders:
            for attacker in attackers:
                row, _, _ = rollout_game(
                    defender,
                    attacker,
                    replace(cfg, seed=seed),
                    defender_logits=defender_logits,
                    attacker_logits=attacker_logits,
                    sample_learned=False,
                )
                row["defender_policy"] = defender
                row["attacker_policy"] = attacker
                rows.append(row)
    return rows


def evaluate_large_response_sweep(
    cfg,
    defender_logits: np.ndarray | None = None,
    attacker_logits: np.ndarray | None = None,
    *,
    seeds: tuple[int, ...] = (101, 102, 103),
    strengths: tuple[float, ...] | None = None,
    sizes: tuple[int, ...] | None = None,
) -> list[dict[str, float | int | str]]:
    """Evaluate response matrices across held-out sizes and heterogeneity."""

    strengths = strengths or (cfg.heterogeneity_strength,)
    sizes = sizes or (cfg.nodes,)
    rows: list[dict[str, float | int | str]] = []
    for nodes in sizes:
        if nodes < cfg.communities:
            raise ValueError(f"node count {nodes} is smaller than communities={cfg.communities}")
        for strength in strengths:
            rows.extend(
                evaluate_large_response_matrix(
                    replace(cfg, nodes=int(nodes), heterogeneity_strength=float(strength)),
                    defender_logits,
                    attacker_logits,
                    seeds=seeds,
                )
            )
    return rows


def summarize_large_response_rows(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    """Aggregate response rows by size, strength, and policy pair."""

    groups: dict[tuple[int, int, float, str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = (
            int(row["nodes"]),
            int(row["communities"]),
            float(row["heterogeneity_strength"]),
            str(row["defender_policy"]),
            str(row["attacker_policy"]),
        )
        groups.setdefault(key, []).append(row)
    metrics = (
        "defender_payoff",
        "attacker_payoff",
        "cumulative_infected_exposure",
        "peak_global_infected",
        "final_global_infected",
    )
    summary: list[dict[str, float | int | str]] = []
    for (nodes, communities, strength, defender, attacker), group in sorted(groups.items()):
        output: dict[str, float | int | str] = {
            "nodes": nodes,
            "communities": communities,
            "heterogeneity_strength": strength,
            "defender_policy": defender,
            "attacker_policy": attacker,
            "rollouts": len(group),
            "seeds": ";".join(str(seed) for seed in sorted({int(row["seed"]) for row in group})),
            "horizon": int(group[0]["horizon"]),
            "mean_degree": float(group[0]["mean_degree"]),
            "defender_budget": int(group[0]["defender_budget"]),
            "attacker_budget": int(group[0]["attacker_budget"]),
            "mass_error_max": max(float(row["mass_error"]) for row in group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            output[f"{metric}_mean"] = fmean(values)
            output[f"{metric}_std"] = float(np.std(values, ddof=0))
        summary.append(output)
    return summary
