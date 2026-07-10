"""Policies and bounded baselines for the node-SIPS attacker-defender game.

The environment lives in :mod:`cybergames.adversarial_env`. This module keeps
policy selection, rollout metrics, and the intentionally small static-logit
self-play baseline separate from the simulator. The baseline is diagnostic;
it is not an equilibrium solver.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .adversarial_env import AdversarialSIPSEnv
from .configs import AdversarialSIPSConfig


def _top_communities(scores: np.ndarray, budget: int) -> np.ndarray:
    if budget <= 0:
        return np.asarray([], dtype=int)
    count = min(int(budget), len(scores))
    return np.argsort(scores)[-count:][::-1].astype(int)


def _softmax(logits: np.ndarray, available: np.ndarray) -> np.ndarray:
    masked = np.asarray(logits, dtype=np.float64).copy()
    masked[~available] = -1e9
    shifted = masked - float(np.max(masked))
    probabilities = np.exp(shifted)
    probabilities[~available] = 0.0
    total = float(probabilities.sum())
    if total <= 0.0:
        probabilities = available.astype(np.float64)
        total = float(probabilities.sum())
    return probabilities / total


def _sample_without_replacement(
    logits: np.ndarray,
    budget: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample communities and return the score-function gradient."""

    available = np.ones(len(logits), dtype=bool)
    gradient = np.zeros(len(logits), dtype=np.float64)
    chosen: list[int] = []
    for _ in range(min(int(budget), len(logits))):
        probabilities = _softmax(logits, available)
        index = int(rng.choice(len(logits), p=probabilities))
        chosen.append(index)
        gradient[index] += 1.0
        gradient -= probabilities
        available[index] = False
    return np.asarray(chosen, dtype=int), gradient


def _policy_scores(env: AdversarialSIPSEnv, role: str, policy: str) -> np.ndarray:
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
    env: AdversarialSIPSEnv,
    role: str,
    policy: str,
    rng: np.random.Generator,
    *,
    logits: np.ndarray | None = None,
    sample: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose communities for one player and return a score gradient."""

    if role not in {"defender", "attacker"}:
        raise ValueError("role must be defender or attacker")
    budget = env.cfg.defender_budget if role == "defender" else env.cfg.attacker_budget
    if policy == "none":
        return np.asarray([], dtype=int), np.zeros(env.cfg.communities, dtype=np.float64)
    if policy == "uniform":
        start = env.k % env.cfg.communities
        chosen = (start + np.arange(min(budget, env.cfg.communities))) % env.cfg.communities
        return chosen.astype(int), np.zeros(env.cfg.communities, dtype=np.float64)
    if policy == "budget_random":
        chosen = rng.choice(
            env.cfg.communities,
            size=min(budget, env.cfg.communities),
            replace=False,
        )
        return np.asarray(chosen, dtype=int), np.zeros(env.cfg.communities, dtype=np.float64)
    if policy == "learned":
        if logits is None:
            raise ValueError("learned policy requires logits")
        if sample:
            return _sample_without_replacement(logits, budget, rng)
        return _top_communities(np.asarray(logits, dtype=np.float64), budget), np.zeros_like(logits)
    return _top_communities(_policy_scores(env, role, policy), budget), np.zeros(
        env.cfg.communities
    )


def rollout_game(
    defender_policy: str,
    attacker_policy: str,
    cfg: AdversarialSIPSConfig,
    *,
    seed: int | None = None,
    defender_logits: np.ndarray | None = None,
    attacker_logits: np.ndarray | None = None,
    sample_learned: bool = False,
) -> tuple[dict[str, float | int | str], np.ndarray, np.ndarray]:
    """Roll out one game and return metrics plus score gradients."""

    run_seed = cfg.seed if seed is None else int(seed)
    env = AdversarialSIPSEnv(replace(cfg, seed=run_seed))
    rng = np.random.default_rng(run_seed + 17_017)
    defender_gradient = np.zeros(cfg.communities, dtype=np.float64)
    attacker_gradient = np.zeros(cfg.communities, dtype=np.float64)
    defender_return = 0.0
    attacker_return = 0.0
    infected_exposure = 0.0
    peak_infected = float(env.state[:, 1].mean())
    last_info = {"mass_error": 0.0, "global_infected": peak_infected}
    done = False
    while not done:
        defender_action, defender_score = choose_communities(
            env,
            "defender",
            defender_policy,
            rng,
            logits=defender_logits,
            sample=sample_learned,
        )
        attacker_action, attacker_score = choose_communities(
            env,
            "attacker",
            attacker_policy,
            rng,
            logits=attacker_logits,
            sample=sample_learned,
        )
        _, defender_reward, attacker_reward, done, last_info = env.step(
            defender_action,
            attacker_action,
        )
        defender_gradient += defender_score
        attacker_gradient += attacker_score
        defender_return += defender_reward
        attacker_return += attacker_reward
        global_infected = float(last_info["global_infected"])
        infected_exposure += cfg.dt * global_infected
        peak_infected = max(peak_infected, global_infected)
    row = {
        "defender_policy": defender_policy,
        "attacker_policy": attacker_policy,
        "seed": run_seed,
        "nodes": cfg.nodes,
        "communities": cfg.communities,
        "horizon": cfg.horizon,
        "mean_degree": cfg.mean_degree,
        "heterogeneity_strength": cfg.heterogeneity_strength,
        "defender_budget": cfg.defender_budget,
        "attacker_budget": cfg.attacker_budget,
        "defender_payoff": defender_return,
        "attacker_payoff": attacker_return,
        "cumulative_infected_exposure": infected_exposure,
        "peak_global_infected": peak_infected,
        "final_global_infected": float(last_info["global_infected"]),
        "mass_error": float(last_info["mass_error"]),
    }
    return row, defender_gradient, attacker_gradient


def _center_clip(logits: np.ndarray, bound: float = 6.0) -> np.ndarray:
    centered = np.asarray(logits, dtype=np.float64) - float(np.mean(logits))
    return np.clip(centered, -bound, bound)


def train_static_logit_self_play(
    cfg: AdversarialSIPSConfig,
    *,
    episodes: int = 40,
    lr: float = 0.08,
    seed: int | None = None,
) -> tuple[list[dict[str, float | int | str]], np.ndarray, np.ndarray]:
    """Fit role-specific static community logits as a diagnostic baseline."""

    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    defender_logits = np.zeros(cfg.communities, dtype=np.float64)
    attacker_logits = np.zeros(cfg.communities, dtype=np.float64)
    defender_baseline = 0.0
    attacker_baseline = 0.0
    history: list[dict[str, float | int | str]] = []
    for episode in range(int(episodes)):
        run_seed = int(rng.integers(0, 2**31 - 1))
        row, defender_score, attacker_score = rollout_game(
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
        scale = 1.0 / max(1, cfg.horizon)
        defender_logits = _center_clip(
            defender_logits + lr * (defender_payoff - defender_baseline) * scale * defender_score
        )
        attacker_logits = _center_clip(
            attacker_logits + lr * (attacker_payoff - attacker_baseline) * scale * attacker_score
        )
        defender_baseline = 0.9 * defender_baseline + 0.1 * defender_payoff
        attacker_baseline = 0.9 * attacker_baseline + 0.1 * attacker_payoff
        history.append(
            {
                "episode": episode,
                **row,
                "defender_logit_max": float(np.max(defender_logits)),
                "attacker_logit_max": float(np.max(attacker_logits)),
                "method": "static_logit_score_function_baseline",
            }
        )
    return history, defender_logits, attacker_logits
