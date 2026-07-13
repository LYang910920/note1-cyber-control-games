"""Bounded, reproducible experiment profiles for the game-learning repository."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

from cybercontrol.experiments import run_provenance
from cybercontrol.io import write_csv, write_json
from cybercontrol.nn import parameter_count

from .adversarial import rollout_game, train_static_logit_self_play
from .architectures import matched_graph_mappo_width
from .configs import AdversarialSIPSConfig, DDQNConfig, MAPPOConfig, NodeSIPSEnvConfig
from .ddqn import evaluate as evaluate_ddqn
from .ddqn import train as train_ddqn
from .envs import EnvConfig
from .evaluation import evaluate_policy_suite
from .mappo import train_mappo
from .node_evaluation import evaluate_policy_baselines
from .self_play import (
    SelfPlayConfig,
    evaluate_fixed_policy_cross_play,
    train_state_conditioned_self_play,
)

ROOT = Path(__file__).resolve().parents[2]
ExperimentRow = dict[str, Any]


@dataclass(frozen=True)
class MediumProfile:
    """Work limits and evaluation seeds for the documented medium run."""

    seeds: tuple[int, ...] = (11, 23, 37, 51, 73)
    ddqn_episodes: int = 30
    ddqn_horizon: int = 20
    mappo_nodes: int = 36
    mappo_updates: int = 4
    held_out_nodes: int = 40
    held_out_strengths: tuple[float, ...] = (0.2, 0.5)
    game_nodes: int = 64
    self_play_episodes: int = 4


def _config_without_seed(config: object) -> dict[str, Any]:
    values = asdict(config)
    values.pop("seed", None)
    return values


def _run_ddqn(
    seed: int, device: str, profile: MediumProfile
) -> tuple[list[ExperimentRow], DDQNConfig]:
    config = DDQNConfig(
        episodes=profile.ddqn_episodes,
        horizon=profile.ddqn_horizon,
        eval_horizon=profile.ddqn_horizon,
        eval_episodes=3,
        batch_size=32,
        buffer_size=2_000,
        target_update=80,
        eps_decay=400.0,
        hidden=64,
        log_every=10,
        seed=seed,
        device=device,
        return_history=True,
    )
    started = time.perf_counter()
    q_network, history = train_ddqn(config)
    runtime_seconds = time.perf_counter() - started
    evaluation_seed = 10_000 + seed
    rows: list[ExperimentRow] = [
        {
            "method": "ddqn_mlp",
            "seed": seed,
            "evaluation_return": evaluate_ddqn(
                q_network,
                episodes=3,
                seed=evaluation_seed,
                horizon=profile.ddqn_horizon,
            ),
            "training_rows": len(history),
            "training_runtime_seconds": runtime_seconds,
            "network_parameters": parameter_count(q_network),
            "resolved_device": str(next(q_network.parameters()).device),
            "evaluation_episodes": 3,
            "evaluation_horizon": profile.ddqn_horizon,
            "evaluation_initial_state": "seeded random",
        }
    ]
    _, baselines = evaluate_policy_suite(
        horizon=profile.ddqn_horizon,
        seed=evaluation_seed,
        config=EnvConfig(randomize_initial_state=True),
    )
    for baseline in baselines:
        row = dict(baseline)
        row.update(
            {
                "method": "ddqn_rule_baseline",
                "seed": seed,
                "evaluation_return": float(baseline["total_defender_reward"]),
                "evaluation_initial_state": "seeded random",
                "training_runtime_seconds": 0.0,
                "resolved_device": "cpu",
            }
        )
        rows.append(row)
    return rows, config


def _run_mappo(
    seed: int,
    device: str,
    profile: MediumProfile,
) -> tuple[list[ExperimentRow], dict[str, MAPPOConfig], int, int]:
    graph_width, baseline_budget, graph_budget = matched_graph_mappo_width(12, 48)
    configs: dict[str, MAPPOConfig] = {}
    rows: list[ExperimentRow] = []
    for architecture, hidden in (("summary_mlp", 48), ("graph_context", graph_width)):
        config = MAPPOConfig(
            nodes=profile.mappo_nodes,
            communities=4,
            horizon=10,
            updates=profile.mappo_updates,
            rollout_steps=10,
            ppo_epochs=2,
            minibatch_size=5,
            hidden=hidden,
            seed=seed,
            device=device,
            architecture=architecture,
        )
        configs[architecture] = config
        started = time.perf_counter()
        actor, critic, history = train_mappo(config)
        runtime_seconds = time.perf_counter() - started
        diagnostics = {
            key: history[-1][key]
            for key in (
                "architecture_activation",
                "architecture_normalization",
                "architecture_encoder",
                "architecture_pooling",
                "architecture_decoder",
                "architecture_input_shape",
                "architecture_output_shape",
                "architecture_parameters",
                "architecture_hidden",
                "architecture_graph_layers",
            )
        }
        held_out = evaluate_policy_baselines(
            actor=actor,
            base_cfg=NodeSIPSEnvConfig(
                nodes=profile.held_out_nodes,
                communities=4,
                horizon=10,
                seed=seed,
            ),
            seeds=(1_000 + seed,),
            strengths=profile.held_out_strengths,
            device=getattr(actor, "_cybercontrol_device", "cpu"),
        )
        for record in held_out:
            row = dict(record)
            row.update(
                {
                    "method": "budgeted_mappo_style_ppo",
                    "architecture": architecture,
                    "training_seed": seed,
                    "training_runtime_seconds": runtime_seconds,
                    "actor_parameters": parameter_count(actor),
                    "critic_parameters": parameter_count(critic),
                    "matched_parameter_target": baseline_budget,
                    "matched_graph_parameter_count": graph_budget,
                    "resolved_device": getattr(actor, "_cybercontrol_device", "cpu"),
                    **diagnostics,
                }
            )
            rows.append(row)
        rows.append(
            {
                "method": "budgeted_ppo_training",
                "architecture": architecture,
                "seed": seed,
                "evaluation_return": float(history[-1]["mean_reward"]),
                "training_rows": len(history),
                "training_runtime_seconds": runtime_seconds,
                "actor_parameters": parameter_count(actor),
                "critic_parameters": parameter_count(critic),
                "matched_parameter_target": baseline_budget,
                "matched_graph_parameter_count": graph_budget,
                "resolved_device": getattr(actor, "_cybercontrol_device", "cpu"),
                **diagnostics,
            }
        )
    return rows, configs, baseline_budget, graph_budget


def _run_adversarial(
    seed: int,
    device: str,
    profile: MediumProfile,
) -> tuple[list[ExperimentRow], AdversarialSIPSConfig, SelfPlayConfig]:
    game = AdversarialSIPSConfig(
        nodes=profile.game_nodes,
        communities=4,
        horizon=6,
        defender_budget=1,
        attacker_budget=1,
        seed=seed,
    )
    started = time.perf_counter()
    static_history, defender_logits, attacker_logits = train_static_logit_self_play(
        game,
        episodes=profile.self_play_episodes,
        lr=0.05,
        seed=seed,
    )
    static_row, _, _ = rollout_game(
        "static_logit",
        "static_logit",
        game,
        seed=2_000 + seed,
        defender_logits=defender_logits,
        attacker_logits=attacker_logits,
    )
    static_row.update(
        {
            "method": "static_logit_self_play_baseline",
            "training_seed": seed,
            "training_rows": len(static_history),
            "training_runtime_seconds": time.perf_counter() - started,
            "network_parameters": int(defender_logits.size + attacker_logits.size),
            "resolved_device": "cpu",
            "nodes": game.nodes,
            "communities": game.communities,
            "horizon": game.horizon,
            "defender_budget": game.defender_budget,
            "attacker_budget": game.attacker_budget,
            "heterogeneity_strength": game.heterogeneity_strength,
            "training_episodes": profile.self_play_episodes,
        }
    )

    training = SelfPlayConfig(
        episodes=profile.self_play_episodes,
        hidden=32,
        seed=seed,
        device=device,
    )
    started = time.perf_counter()
    result = train_state_conditioned_self_play(game, training)
    runtime_seconds = time.perf_counter() - started
    rows: list[ExperimentRow] = [static_row]
    for response in evaluate_fixed_policy_cross_play(result, game, seeds=(2_000 + seed,)):
        row = dict(response)
        row.update(
            {
                "method": "state_conditioned_actor_critic",
                "training_seed": seed,
                "training_rows": len(result.history),
                "training_runtime_seconds": runtime_seconds,
                "actor_parameters": parameter_count(result.defender_actor)
                + parameter_count(result.attacker_actor),
                "critic_parameters": parameter_count(result.defender_critic)
                + parameter_count(result.attacker_critic),
                "resolved_device": result.device,
                "nodes": game.nodes,
                "communities": game.communities,
                "horizon": game.horizon,
                "defender_budget": game.defender_budget,
                "attacker_budget": game.attacker_budget,
                "heterogeneity_strength": game.heterogeneity_strength,
                "training_episodes": training.episodes,
                "training_hidden": training.hidden,
            }
        )
        rows.append(row)
    return rows, game, training


def run_medium(
    output_dir: Path, device: str, profile: MediumProfile | None = None
) -> list[ExperimentRow]:
    """Run the documented five-seed study and write metrics plus provenance."""

    profile = profile or MediumProfile()
    output_dir.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()
    rows: list[ExperimentRow] = []
    last_ddqn: DDQNConfig | None = None
    last_mappo: dict[str, MAPPOConfig] = {}
    last_game: AdversarialSIPSConfig | None = None
    last_self_play: SelfPlayConfig | None = None
    baseline_budget = graph_budget = 0

    for seed in profile.seeds:
        ddqn_rows, last_ddqn = _run_ddqn(seed, device, profile)
        mappo_rows, last_mappo, baseline_budget, graph_budget = _run_mappo(
            seed,
            device,
            profile,
        )
        game_rows, last_game, last_self_play = _run_adversarial(seed, device, profile)
        rows.extend(ddqn_rows)
        rows.extend(mappo_rows)
        rows.extend(game_rows)

    assert last_ddqn is not None and last_game is not None and last_self_play is not None
    write_csv(output_dir / "medium_metrics.csv", rows)
    write_json(
        output_dir / "medium_config.json",
        {
            "seeds": list(profile.seeds),
            "device": device,
            **run_provenance(ROOT),
            "total_runtime_seconds": time.perf_counter() - total_started,
            "ddqn": _config_without_seed(last_ddqn),
            "ddqn_evaluation": {
                "episodes": 3,
                "horizon": profile.ddqn_horizon,
                "initial_state": "seeded random",
                "baselines": [
                    "No defense",
                    "Fixed high patch",
                    "Fixed high clean",
                    "Rule threshold isolate/deceive/patch",
                ],
            },
            "mappo": {name: _config_without_seed(cfg) for name, cfg in last_mappo.items()},
            "mappo_parameter_budget": {
                "summary_target": baseline_budget,
                "graph_count": graph_budget,
                "relative_difference": abs(graph_budget - baseline_budget) / baseline_budget,
            },
            "held_out_strengths": list(profile.held_out_strengths),
            "held_out_nodes": profile.held_out_nodes,
            "adversarial_game": _config_without_seed(last_game),
            "adversarial_seed_rule": "use each listed training seed",
            "static_logit_self_play": {
                "episodes": profile.self_play_episodes,
                "learning_rate": 0.05,
            },
            "state_conditioned_self_play": _config_without_seed(last_self_play),
            "fixed_policy_cross_play": {
                "evaluation_seed_rule": "2000 + training seed",
                "opponents": ["uniform", "degree", "risk", "oracle", "budget_random"],
                "includes_learned_vs_learned_reference": True,
                "best_response_retraining": False,
            },
        },
    )
    return rows
