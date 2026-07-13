"""State-conditioned attacker-defender actor-critic self-play."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import torch
import torch.nn.functional as functional
from torch.distributions import Categorical

from cybercontrol.nn import parameter_count
from cybercontrol.experiments import configure_torch

from .adversarial import choose_communities
from .adversarial_env import AdversarialSIPSEnv
from .architectures import PooledStateCritic, StateConditionedCommunityPolicy
from .configs import AdversarialSIPSConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelfPlayConfig:
    """Bounded actor-critic self-play settings."""

    episodes: int = 24
    hidden: int = 64
    lr: float = 3e-4
    gamma: float = 0.97
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    seed: int = 0
    device: str = "auto"
    threads: int = 1


@dataclass
class SelfPlayResult:
    """Trained role-specific policies, critics, and diagnostics."""

    defender_actor: StateConditionedCommunityPolicy
    attacker_actor: StateConditionedCommunityPolicy
    defender_critic: PooledStateCritic
    attacker_critic: PooledStateCritic
    history: list[dict[str, float | int | str]]
    device: str


def _sample_budgeted(logits, budget: int, *, deterministic: bool = False):
    """Sample or greedily select distinct communities and return log probability."""

    available = torch.ones_like(logits, dtype=torch.bool)
    choices = []
    log_probability = torch.zeros((), dtype=logits.dtype, device=logits.device)
    entropy = torch.zeros_like(log_probability)
    for _ in range(min(int(budget), logits.numel())):
        masked = logits.masked_fill(~available, -torch.inf)
        distribution = Categorical(logits=masked)
        choice = torch.argmax(masked) if deterministic else distribution.sample()
        choices.append(choice)
        log_probability = log_probability + distribution.log_prob(choice)
        entropy = entropy + distribution.entropy()
        available = available.clone()
        available[choice] = False
    return torch.stack(choices), log_probability, entropy


def _discounted_returns(rewards: list[float], gamma: float, device: str):
    running = torch.zeros((), dtype=torch.float32, device=device)
    returns = []
    for reward in reversed(rewards):
        running = torch.as_tensor(reward, dtype=torch.float32, device=device) + gamma * running
        returns.append(running)
    return torch.stack(list(reversed(returns)))


def train_state_conditioned_self_play(
    game: AdversarialSIPSConfig,
    training: SelfPlayConfig | None = None,
) -> SelfPlayResult:
    """Train separate state-conditioned actors and state-value critics.

    This bounded baseline uses alternating role-specific policy gradients on
    complete trajectories. It is not presented as an equilibrium solver;
    fixed-policy cross-play is supplied here; best-response retraining remains
    required before making an exploitability or equilibrium claim.
    """

    game.validate()
    training = training or SelfPlayConfig(seed=game.seed)
    torch_module, device, _ = configure_torch(
        seed=training.seed,
        device=training.device,
        threads=training.threads,
    )
    env = AdversarialSIPSEnv(game)
    observation_dim = env.observation().shape[1]
    defender_actor = StateConditionedCommunityPolicy(observation_dim, training.hidden).to(device)
    attacker_actor = StateConditionedCommunityPolicy(observation_dim, training.hidden).to(device)
    defender_critic = PooledStateCritic(observation_dim, training.hidden).to(device)
    attacker_critic = PooledStateCritic(observation_dim, training.hidden).to(device)
    parameters = list(defender_actor.parameters()) + list(attacker_actor.parameters())
    critic_parameters = list(defender_critic.parameters()) + list(attacker_critic.parameters())
    optimizer = torch_module.optim.Adam(parameters + critic_parameters, lr=training.lr)
    history: list[dict[str, float | int | str]] = []

    for episode in range(training.episodes):
        observation = env.reset(seed=training.seed + episode)
        defender_logp = []
        attacker_logp = []
        defender_entropy = []
        attacker_entropy = []
        defender_values = []
        attacker_values = []
        defender_rewards: list[float] = []
        attacker_rewards: list[float] = []
        done = False
        last_info = {"mass_error": 0.0, "global_infected": float(env.state[:, 1].mean())}
        while not done:
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
            d_choice, d_logp, d_entropy = _sample_budgeted(
                defender_actor(obs_tensor), game.defender_budget
            )
            a_choice, a_logp, a_entropy = _sample_budgeted(
                attacker_actor(obs_tensor), game.attacker_budget
            )
            defender_values.append(defender_critic(obs_tensor).squeeze(0))
            attacker_values.append(attacker_critic(obs_tensor).squeeze(0))
            observation, defender_reward, attacker_reward, done, last_info = env.step(
                d_choice.detach().cpu().numpy(),
                a_choice.detach().cpu().numpy(),
            )
            defender_logp.append(d_logp)
            attacker_logp.append(a_logp)
            defender_entropy.append(d_entropy)
            attacker_entropy.append(a_entropy)
            defender_rewards.append(float(defender_reward))
            attacker_rewards.append(float(attacker_reward))

        d_return = _discounted_returns(defender_rewards, training.gamma, device)
        a_return = _discounted_returns(attacker_rewards, training.gamma, device)
        d_value = torch.stack(defender_values)
        a_value = torch.stack(attacker_values)
        d_advantage = d_return - d_value.detach()
        a_advantage = a_return - a_value.detach()
        actor_loss = (
            -(torch.stack(defender_logp) * d_advantage).mean()
            - (torch.stack(attacker_logp) * a_advantage).mean()
        )
        critic_loss = functional.mse_loss(d_value, d_return) + functional.mse_loss(
            a_value, a_return
        )
        entropy = torch.stack(defender_entropy).mean() + torch.stack(attacker_entropy).mean()
        loss = actor_loss + training.value_coef * critic_loss - training.entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters + critic_parameters, training.max_grad_norm)
        optimizer.step()
        history.append(
            {
                "episode": episode,
                "defender_return": float(sum(defender_rewards)),
                "attacker_return": float(sum(attacker_rewards)),
                "actor_loss": float(actor_loss.detach().cpu()),
                "critic_loss": float(critic_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "final_global_infected": float(last_info["global_infected"]),
                "mass_error": float(last_info["mass_error"]),
                "defender_actor_parameters": parameter_count(defender_actor),
                "attacker_actor_parameters": parameter_count(attacker_actor),
                "policy_type": "state-conditioned actor-critic",
            }
        )
        LOGGER.info(
            "episode=%03d defender=%.3f attacker=%.3f mass_error=%.1e",
            episode,
            history[-1]["defender_return"],
            history[-1]["attacker_return"],
            history[-1]["mass_error"],
        )
    return SelfPlayResult(
        defender_actor,
        attacker_actor,
        defender_critic,
        attacker_critic,
        history,
        device,
    )


def evaluate_fixed_policy_cross_play(
    result: SelfPlayResult,
    game: AdversarialSIPSConfig,
    *,
    seeds: tuple[int, ...] = (101, 102, 103),
) -> list[dict[str, float | int | str]]:
    """Compare fixed learned policies with each other and heuristic opponents.

    These are held-out cross-play rollouts, not best-response retraining or an
    exploitability estimate.
    """

    baselines = ("uniform", "degree", "risk", "oracle", "budget_random")
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for learned_role, opponents in (
            ("both", ("learned",)),
            ("defender", baselines),
            ("attacker", baselines),
        ):
            for opponent in opponents:
                env = AdversarialSIPSEnv(game)
                observation = env.reset(seed=seed)
                rng = np.random.default_rng(seed + 6_001)
                defender_return = 0.0
                attacker_return = 0.0
                done = False
                last_info = {"mass_error": 0.0, "global_infected": float(env.state[:, 1].mean())}
                while not done:
                    obs_tensor = torch.as_tensor(
                        observation, dtype=torch.float32, device=result.device
                    )
                    if learned_role in {"both", "defender"}:
                        defender, _, _ = _sample_budgeted(
                            result.defender_actor(obs_tensor),
                            game.defender_budget,
                            deterministic=True,
                        )
                        defender_np = defender.cpu().numpy()
                    else:
                        defender, _ = choose_communities(env, "defender", opponent, rng)
                        defender_np = defender

                    if learned_role in {"both", "attacker"}:
                        attacker, _, _ = _sample_budgeted(
                            result.attacker_actor(obs_tensor),
                            game.attacker_budget,
                            deterministic=True,
                        )
                        attacker_np = attacker.cpu().numpy()
                    else:
                        attacker, _ = choose_communities(env, "attacker", opponent, rng)
                        attacker_np = attacker
                    observation, defender_reward, attacker_reward, done, last_info = env.step(
                        defender_np, attacker_np
                    )
                    defender_return += defender_reward
                    attacker_return += attacker_reward
                rows.append(
                    {
                        "seed": seed,
                        "fixed_learned_role": learned_role,
                        "opponent_policy": opponent,
                        "evaluation_type": (
                            "learned_profile_reference"
                            if learned_role == "both"
                            else "fixed_policy_cross_play"
                        ),
                        "best_response_retraining": False,
                        "defender_payoff": defender_return,
                        "attacker_payoff": attacker_return,
                        "final_global_infected": float(last_info["global_infected"]),
                        "mass_error": float(last_info["mass_error"]),
                    }
                )
    return rows
