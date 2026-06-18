"""
Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

Policy-rollout metrics for the Note 1 hybrid cyber-control examples.

The helpers keep method-comparison plots and experiment CSV files consistent.
They also make the timing convention explicit: policies observe at decision
epochs, while RK4 substeps remain internal simulator steps.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple
import numpy as np

from cyber_hybrid_env import Action, EnvConfig, HybridCyberDefenseEnv, scripted_attacker

PolicyFn = Callable[[HybridCyberDefenseEnv, int, np.ndarray], Action]


def no_defense_policy(env: HybridCyberDefenseEnv, k: int, obs: np.ndarray) -> Action:
    return env.DEF_NONE


def always_patch_policy(env: HybridCyberDefenseEnv, k: int, obs: np.ndarray) -> Action:
    return env.DEF_PATCH, 0.8


def always_clean_policy(env: HybridCyberDefenseEnv, k: int, obs: np.ndarray) -> Action:
    return env.DEF_CLEAN, 0.8


def adaptive_hybrid_policy(env: HybridCyberDefenseEnv, k: int, obs: np.ndarray) -> Action:
    """Simple readable baseline using rate controls and an impulse action."""
    if obs[1] > 0.20:
        return env.DEF_ISOLATE, 0.8
    if k < 20:
        return env.DEF_DECEIVE, 0.7
    return env.DEF_PATCH, 0.7


def policy_suite() -> List[Tuple[str, PolicyFn]]:
    """Return deterministic comparison policies used by figures and CSVs."""
    return [
        ("No defense", no_defense_policy),
        ("Always patch", always_patch_policy),
        ("Always clean", always_clean_policy),
        ("Adaptive hybrid", adaptive_hybrid_policy),
    ]


def rollout_policy(
    label: str,
    policy: PolicyFn,
    horizon: int = 50,
    seed: int = 7,
    config: EnvConfig | None = None,
) -> Dict[str, object]:
    """Roll out one policy and retain states, rewards, actions, and timing."""
    env = HybridCyberDefenseEnv(config=config, seed=seed) if config else HybridCyberDefenseEnv(seed=seed)
    env.cfg.horizon = horizon
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
        attacker_action = scripted_attacker(env, k)
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
    action_switches = int(np.sum(defender_modes[1:] != defender_modes[:-1])) if len(defender_modes) > 1 else 0
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
        "mean_deception": float(np.mean(states[:, 3])),
        "impulse_events": int(np.count_nonzero(jump_removed > 1e-12)),
        "total_impulse_removed": float(np.sum(jump_removed)),
        "action_switches": action_switches,
    }


def evaluate_policy_suite(horizon: int = 50, seed: int = 7) -> Tuple[List[Dict[str, object]], List[Dict[str, float | int | str]]]:
    """Roll out all representative policies and return raw and summary data."""
    rollouts = [rollout_policy(label, policy, horizon=horizon, seed=seed) for label, policy in policy_suite()]
    rows = [summarize_rollout(rollout) for rollout in rollouts]
    return rollouts, rows
