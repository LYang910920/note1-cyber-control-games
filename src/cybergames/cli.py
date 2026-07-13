"""Single public command-line interface for Note 1."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from cybercontrol.experiments import run_provenance
from cybercontrol.io import write_csv, write_json
from cybercontrol.nn import parameter_count

from .adversarial import rollout_game, train_static_logit_self_play
from .architectures import matched_graph_mappo_width
from .configs import (
    AdversarialSIPSConfig,
    CTDEConfig,
    DDQNConfig,
    MAPPOConfig,
    NodeSIPSEnvConfig,
)
from .ctde import train as train_ctde
from .ddqn import evaluate as evaluate_ddqn
from .ddqn import train as train_ddqn
from .envs import EnvConfig
from .evaluation import evaluate_policy_suite
from .fbsm import solve_fbsm
from .mappo import train_mappo
from .node_evaluation import evaluate_policy_baselines
from .self_play import (
    SelfPlayConfig,
    evaluate_fixed_policy_cross_play,
    train_state_conditioned_self_play,
)

ROOT = Path(__file__).resolve().parents[2]


def _source_root() -> Path:
    """Return the source checkout required for figures and LaTeX commands."""

    if (
        not (ROOT / "scripts" / "generate_figures.py").exists()
        or not (ROOT / "docs" / "source").exists()
    ):
        raise RuntimeError(
            "This command needs a source checkout containing scripts/ and docs/source/. "
            "The installed cybergames package can still be imported normally."
        )
    return ROOT


def _smoke() -> dict[str, float | int]:
    """Run bounded execution and invariant checks without writing artifacts."""

    _, state, control, _, objective = solve_fbsm(n=40, max_iter=4)
    ddqn_cfg = DDQNConfig(
        smoke=True,
        episodes=2,
        horizon=8,
        eval_horizon=8,
        batch_size=4,
        hidden=24,
        log_every=1,
        target_update=8,
        device="cpu",
        return_history=True,
    )
    q_network, ddqn_history = train_ddqn(ddqn_cfg)
    ctde_cfg = CTDEConfig(
        smoke=True,
        episodes=2,
        horizon=6,
        hidden=24,
        log_every=1,
        device="cpu",
        return_history=True,
    )
    _, _, ctde_history = train_ctde(ctde_cfg)
    mappo_cfg = MAPPOConfig(
        nodes=18,
        communities=3,
        horizon=4,
        updates=1,
        rollout_steps=4,
        ppo_epochs=1,
        minibatch_size=2,
        hidden=16,
        device="cpu",
        log_every=1,
    )
    _, _, mappo_history = train_mappo(mappo_cfg)
    return {
        "fbsm_objective": float(objective),
        "fbsm_mass_error": float(np.max(np.abs(state.sum(axis=1) - 1.0))),
        "fbsm_mean_control": float(control.mean()),
        "ddqn_history_rows": len(ddqn_history),
        "ddqn_eval": float(evaluate_ddqn(q_network, episodes=1, horizon=8)),
        "ctde_history_rows": len(ctde_history),
        "mappo_history_rows": len(mappo_history),
        "mappo_active_actions": int(mappo_history[-1]["active_actions"]),
    }


def _medium(output_dir: Path, device: str) -> list[dict[str, float | int | str]]:
    """Run five bounded seeds with held-out MAPPO profiles and DDQN evaluation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()
    rows: list[dict[str, float | int | str]] = []
    for seed in (11, 23, 37, 51, 73):
        ddqn_cfg = DDQNConfig(
            episodes=30,
            horizon=20,
            eval_horizon=20,
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
        q_network, history = train_ddqn(ddqn_cfg)
        runtime_seconds = time.perf_counter() - started
        rows.append(
            {
                "method": "ddqn_mlp",
                "seed": seed,
                "evaluation_return": evaluate_ddqn(
                    q_network, episodes=3, seed=10_000 + seed, horizon=20
                ),
                "training_rows": len(history),
                "training_runtime_seconds": runtime_seconds,
                "network_parameters": parameter_count(q_network),
                "resolved_device": str(next(q_network.parameters()).device),
                "evaluation_episodes": 3,
                "evaluation_horizon": 20,
                "evaluation_initial_state": "seeded random",
            }
        )
        _, ddqn_baselines = evaluate_policy_suite(
            horizon=20,
            seed=10_000 + seed,
            config=EnvConfig(randomize_initial_state=True),
        )
        for baseline in ddqn_baselines:
            baseline_row = dict(baseline)
            baseline_row.update(
                {
                    "method": "ddqn_rule_baseline",
                    "seed": seed,
                    "evaluation_return": float(baseline["total_defender_reward"]),
                    "evaluation_initial_state": "seeded random",
                    "training_runtime_seconds": 0.0,
                    "resolved_device": "cpu",
                }
            )
            rows.append(baseline_row)

        graph_width, baseline_budget, graph_budget = matched_graph_mappo_width(12, 48)
        mappo_configs: dict[str, MAPPOConfig] = {}
        for architecture, hidden in (("summary_mlp", 48), ("graph_context", graph_width)):
            mappo_cfg = MAPPOConfig(
                nodes=36,
                communities=4,
                horizon=10,
                updates=4,
                rollout_steps=10,
                ppo_epochs=2,
                minibatch_size=5,
                hidden=hidden,
                seed=seed,
                device=device,
                architecture=architecture,
            )
            mappo_configs[architecture] = mappo_cfg
            started = time.perf_counter()
            actor, critic, mappo_history = train_mappo(mappo_cfg)
            runtime_seconds = time.perf_counter() - started
            architecture_diagnostics = {
                key: mappo_history[-1][key]
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
                base_cfg=NodeSIPSEnvConfig(nodes=40, communities=4, horizon=10, seed=seed),
                seeds=(1_000 + seed,),
                strengths=(0.2, 0.5),
                device=getattr(actor, "_cybercontrol_device", "cpu"),
            )
            for record in held_out:
                record = dict(record)
                record.update(
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
                        **architecture_diagnostics,
                    }
                )
                rows.append(record)
            rows.append(
                {
                    "method": "budgeted_ppo_training",
                    "architecture": architecture,
                    "seed": seed,
                    "evaluation_return": float(mappo_history[-1]["mean_reward"]),
                    "training_rows": len(mappo_history),
                    "training_runtime_seconds": runtime_seconds,
                    "actor_parameters": parameter_count(actor),
                    "critic_parameters": parameter_count(critic),
                    "matched_parameter_target": baseline_budget,
                    "matched_graph_parameter_count": graph_budget,
                    "resolved_device": getattr(actor, "_cybercontrol_device", "cpu"),
                    **architecture_diagnostics,
                }
            )

        game = AdversarialSIPSConfig(
            nodes=64,
            communities=4,
            horizon=6,
            defender_budget=1,
            attacker_budget=1,
            seed=seed,
        )
        started = time.perf_counter()
        static_history, defender_logits, attacker_logits = train_static_logit_self_play(
            game,
            episodes=4,
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
                "training_episodes": 4,
            }
        )
        rows.append(static_row)
        started = time.perf_counter()
        state_training = SelfPlayConfig(episodes=4, hidden=32, seed=seed, device=device)
        state_result = train_state_conditioned_self_play(game, state_training)
        state_training_runtime = time.perf_counter() - started
        for response in evaluate_fixed_policy_cross_play(
            state_result,
            game,
            seeds=(2_000 + seed,),
        ):
            response = dict(response)
            response.update(
                {
                    "method": "state_conditioned_actor_critic",
                    "training_seed": seed,
                    "training_rows": len(state_result.history),
                    "training_runtime_seconds": state_training_runtime,
                    "actor_parameters": parameter_count(state_result.defender_actor)
                    + parameter_count(state_result.attacker_actor),
                    "critic_parameters": parameter_count(state_result.defender_critic)
                    + parameter_count(state_result.attacker_critic),
                    "resolved_device": state_result.device,
                    "nodes": game.nodes,
                    "communities": game.communities,
                    "horizon": game.horizon,
                    "defender_budget": game.defender_budget,
                    "attacker_budget": game.attacker_budget,
                    "heterogeneity_strength": game.heterogeneity_strength,
                    "training_episodes": state_training.episodes,
                    "training_hidden": state_training.hidden,
                }
            )
            rows.append(response)

    write_csv(output_dir / "medium_metrics.csv", rows)
    write_json(
        output_dir / "medium_config.json",
        {
            "seeds": [11, 23, 37, 51, 73],
            "device": device,
            **run_provenance(ROOT),
            "total_runtime_seconds": time.perf_counter() - total_started,
            "ddqn": asdict(ddqn_cfg),
            "ddqn_evaluation": {
                "episodes": 3,
                "horizon": 20,
                "initial_state": "seeded random",
                "baselines": [
                    "No defense",
                    "Fixed high patch",
                    "Fixed high clean",
                    "Rule threshold isolate/deceive/patch",
                ],
            },
            "mappo": {name: asdict(cfg) for name, cfg in mappo_configs.items()},
            "mappo_parameter_budget": {
                "summary_target": baseline_budget,
                "graph_count": graph_budget,
                "relative_difference": abs(graph_budget - baseline_budget) / baseline_budget,
            },
            "held_out_strengths": [0.2, 0.5],
            "held_out_nodes": 40,
            "adversarial_game": asdict(game),
            "adversarial_seed_rule": "replace config seed with each training seed",
            "static_logit_self_play": {"episodes": 4, "learning_rate": 0.05},
            "state_conditioned_self_play": asdict(state_training),
            "fixed_policy_cross_play": {
                "evaluation_seed_rule": "2000 + training seed",
                "opponents": ["uniform", "degree", "risk", "oracle", "budget_random"],
                "includes_learned_vs_learned_reference": True,
                "best_response_retraining": False,
            },
        },
    )
    return rows


def _run_script(path: str, *arguments: str) -> None:
    root = _source_root()
    subprocess.run([sys.executable, path, *arguments], cwd=root, check=True)


def _build_docs() -> None:
    root = _source_root()
    source = root / "docs" / "source"
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "note1_game_learning_cyber_control.tex"],
        cwd=source,
        check=True,
    )
    built = source / "note1_game_learning_cyber_control.pdf"
    if built.exists():
        (root / "docs" / built.name).write_bytes(built.read_bytes())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cybergames", description="Cyber control and game-learning experiments."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke", help="Fast execution and invariant checks.")
    medium = sub.add_parser("medium", help="Bounded five-seed DDQN/MAPPO evaluation.")
    medium.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    medium.add_argument("--output-dir", type=Path, default=Path("artifacts/medium"))
    sub.add_parser("figures", help="Regenerate curated guide figures.")
    sub.add_parser("docs", help="Build the current tutorial PDF.")
    sub.add_parser("all", help="Run smoke, figures, and docs.")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        print(
            write_json(
                Path("artifacts/smoke_summary.json"),
                {"command": "smoke", **run_provenance(ROOT), "metrics": _smoke()},
            )
        )
    elif args.command == "medium":
        rows = _medium(args.output_dir, args.device)
        print(f"wrote {len(rows)} rows to {args.output_dir}")
    elif args.command == "figures":
        _run_script("scripts/generate_figures.py")
    elif args.command == "docs":
        _build_docs()
    elif args.command == "all":
        print(
            write_json(
                Path("artifacts/smoke_summary.json"),
                {"command": "smoke", **run_provenance(ROOT), "metrics": _smoke()},
            )
        )
        _run_script("scripts/generate_figures.py")
        _build_docs()


if __name__ == "__main__":
    main()
