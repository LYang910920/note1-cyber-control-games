"""Cooperative budgeted PPO update with a centralized state-value critic.

Community encoders produce local mode scores. A shared budget allocator forms
one joint categorical action, so execution is coordinated rather than strictly
decentralized. The graph-context variant adds community message passing before
the same allocator. This boundary is stated explicitly in the guide.
"""

from __future__ import annotations

import logging

import numpy as np

from cybercontrol.nn import parameter_count
from cybercontrol.rl import RolloutBuffer, compute_gae
from cybercontrol.torch_utils import configure_torch

from .architectures import ARCHITECTURE_REGISTRY
from .configs import MAPPOConfig, NodeSIPSEnvConfig
from .node_env import NodeSIPSEnv

LOGGER = logging.getLogger(__name__)


def train_mappo(args: MAPPOConfig):
    """Train a budgeted cooperative actor and centralized value critic."""

    args.validate()
    torch, device, _ = configure_torch(
        seed=args.seed,
        device=args.device,
        threads=args.threads,
    )
    import torch.nn.functional as F
    from torch.distributions import Categorical

    rng = np.random.default_rng(args.seed)
    env = NodeSIPSEnv(
        NodeSIPSEnvConfig(
            nodes=args.nodes,
            communities=args.communities,
            horizon=args.horizon,
            seed=args.seed,
            heterogeneity_strength=args.heterogeneity_strength,
        )
    )
    components = ARCHITECTURE_REGISTRY.build(
        args.architecture,
        env.obs_dim,
        args.hidden,
        args.graph_layers,
    )
    actor = components["actor"].to(device)
    critic = components["critic"].to(device)
    architecture = ARCHITECTURE_REGISTRY.describe(
        args.architecture,
        {"actor": actor, "critic": critic},
    )
    actor._cybercontrol_device = device
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)
    history: list[dict[str, float | int | str]] = []

    for update in range(args.updates):
        observation = env.reset(seed=args.seed + update)
        community_adjacency = env.community_adjacency()
        buffer = RolloutBuffer()
        for _ in range(args.rollout_steps):
            observation_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=device,
            )
            distribution = Categorical(logits=actor(observation_tensor, community_adjacency))
            joint_action = distribution.sample()
            log_probability = distribution.log_prob(joint_action)
            actions = actor.decode(int(joint_action.item()), env.n_agents)
            value = critic(observation_tensor, community_adjacency).squeeze(0)
            next_observation, rewards, done, info = env.step(actions)
            buffer.add(
                observation,
                int(joint_action.item()),
                float(log_probability.detach().cpu()),
                float(np.mean(rewards)),
                done,
                float(value.detach().cpu()),
            )
            observation = next_observation

        if buffer.dones[-1]:
            last_value = 0.0
        else:
            with torch.no_grad():
                last_value = float(
                    critic(
                        torch.as_tensor(
                            observation,
                            dtype=torch.float32,
                            device=device,
                        ),
                        community_adjacency,
                    )
                    .cpu()
                    .item()
                )
        rewards = np.asarray(buffer.rewards, dtype=np.float32)
        values = np.asarray(buffer.values, dtype=np.float32)
        advantages, returns = compute_gae(
            rewards,
            values,
            np.asarray(buffer.dones, dtype=np.float32),
            last_value,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        observations = torch.as_tensor(
            np.asarray(buffer.observations),
            dtype=torch.float32,
            device=device,
        )
        sampled_actions = torch.as_tensor(
            np.asarray(buffer.actions),
            dtype=torch.int64,
            device=device,
        )
        old_log_probabilities = torch.as_tensor(
            np.asarray(buffer.log_probabilities),
            dtype=torch.float32,
            device=device,
        )
        advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
        indices = np.arange(args.rollout_steps)
        for _ in range(args.ppo_epochs):
            rng.shuffle(indices)
            for start in range(0, args.rollout_steps, args.minibatch_size):
                index = indices[start : start + args.minibatch_size]
                batch_observations = observations[index]
                distribution = Categorical(logits=actor(batch_observations, community_adjacency))
                log_probability = distribution.log_prob(sampled_actions[index])
                entropy = distribution.entropy().mean()
                ratio = torch.exp(log_probability - old_log_probabilities[index])
                unclipped = ratio * advantage_tensor[index]
                clipped = (
                    torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
                    * advantage_tensor[index]
                )
                policy_loss = -torch.min(unclipped, clipped).mean()
                value = critic(batch_observations, community_adjacency)
                value_loss = F.mse_loss(value, return_tensor[index])

                actor_opt.zero_grad()
                critic_opt.zero_grad()
                (
                    policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
                ).backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    args.max_grad_norm,
                )
                actor_opt.step()
                critic_opt.step()

        history.append(
            {
                "update": update,
                "mean_reward": float(rewards.mean()),
                "final_global_infected": float(info["global_infected"]),
                "mass_error": float(info["mass_error"]),
                "active_actions": int(np.count_nonzero(actions)),
                "action_budget": args.action_budget,
                "actor_parameters": parameter_count(actor),
                "critic_parameters": parameter_count(critic),
                "architecture": args.architecture,
                "architecture_activation": str(architecture["activation"]),
                "architecture_normalization": str(architecture["normalization"]),
                "architecture_encoder": str(architecture["encoder"]),
                "architecture_pooling": str(architecture["pooling"]),
                "architecture_decoder": str(architecture["decoder"]),
                "architecture_input_shape": str(architecture["input_shape"]),
                "architecture_output_shape": str(architecture["output_shape"]),
                "architecture_parameters": int(architecture["parameters"]),
            }
        )
        if update % args.log_every == 0:
            LOGGER.info(
                "update=%03d reward=%.4f global_I=%.4f mass_error=%.1e",
                update,
                history[-1]["mean_reward"],
                history[-1]["final_global_infected"],
                history[-1]["mass_error"],
            )
    return actor, critic, history
