"""Train one defender, compare held-out cases, then explore multi-agent games."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from cybercontrol.experiments import configure_torch
from cybercontrol.numerics import trapezoid_integral

from .configs import DDQNConfig
from .ddqn import make_q_network, train
from .envs import EnvConfig, SampledFlowImpulseEnv, scripted_attacker
from .fbsm import solve_fbsm

ROOT = Path(__file__).resolve().parents[2]
ACTION_NAMES = ("No intervention", "Patch", "Clean", "Deceive", "Isolate")


def greedy_policy(network):
    """Convert Q values into the action actually executed by the environment."""
    device = next(network.parameters()).device

    def action(observation):
        with torch.no_grad():
            state = torch.as_tensor(observation, dtype=torch.float32, device=device)[None, :]
            return int(network(state).argmax(dim=1).item())

    return action


def compare_policies(policies, *, horizon: int, episodes: int, seed: int):
    """Use identical held-out initial states for every policy; retain case-wise costs."""
    rows, trajectories = [], []
    for name, policy in policies.items():
        env = SampledFlowImpulseEnv(
            EnvConfig(horizon=horizon, randomize_initial_state=True), seed=seed
        )
        for case in range(episodes):
            observation = env.reset()
            initial = observation[:3].copy()
            cost, infected_time = 0.0, 0.0
            for decision in range(horizon):
                action = policy(observation)
                if case == 0:
                    trajectories.append(
                        {
                            "policy": name,
                            "time": decision * env.cfg.dt,
                            "infected": observation[1],
                            "action": action,
                        }
                    )
                observation, rewards, done, info = env.step(
                    action, scripted_attacker(env, decision)
                )
                cost -= rewards["defender"]
                grid = np.linspace(0.0, env.cfg.dt, len(info["path"]))
                infected_time += trapezoid_integral(info["path"][:, 1], grid)
                if done:
                    break
            rows.append(
                {
                    "policy": name,
                    "case": case,
                    "initial_S": initial[0],
                    "initial_I": initial[1],
                    "initial_R": initial[2],
                    "cost": cost,
                    "infected_time": infected_time,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(trajectories)


def run_example(args: argparse.Namespace) -> pd.DataFrame:
    """Observation -> DDQN training -> frozen-policy comparisons -> plots."""
    import matplotlib.pyplot as plt

    config = DDQNConfig(
        episodes=args.episodes,
        horizon=args.horizon,
        eval_horizon=args.horizon,
        batch_size=min(32, args.horizon),
        hidden=32,
        gamma=1.0,
        target_update=64,
        eps_decay=max(100.0, args.episodes * args.horizon / 3.0),
        log_every=max(1, args.episodes // 10),
        seed=args.seed,
        device=args.device,
    )
    # train() resets the same seed before creating its Q network.
    _, device, _ = configure_torch(seed=args.seed, device=args.device, threads=1)
    untrained = make_q_network(4, 5, hidden=config.hidden, depth=config.depth).to(device)
    trained, history = train(config)
    policies = {name: (lambda observation, mode=mode: mode) for mode, name in enumerate(ACTION_NAMES)}
    policies["Untrained DDQN"] = greedy_policy(untrained)
    policies["Trained DDQN"] = greedy_policy(trained)
    cases, trajectories = compare_policies(
        policies, horizon=args.horizon, episodes=args.test_cases, seed=10_000 + args.seed
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cases.to_csv(out / "comparison.csv", index=False)
    trajectories.to_csv(out / "trajectories.csv", index=False)
    training = pd.DataFrame(history)
    training.to_csv(out / "training.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(training["episode"], training["evaluation_return"])
    ax.set(xlabel="Training episode", ylabel="Validation return (higher is better)")
    fig.tight_layout()
    fig.savefig(out / "learning.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, group in trajectories.groupby("policy", sort=False):
        ax.plot(group["time"], group["infected"], label=name)
    ax.set(xlabel="Time", ylabel="Infected fraction", title="Held-out initial condition 1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "infection.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for name in ("Untrained DDQN", "Trained DDQN"):
        group = trajectories[trajectories["policy"] == name]
        ax.step(group["time"], group["action"], where="post", label=name)
    ax.set(xlabel="Time", ylabel="Executed action", yticks=range(5), yticklabels=ACTION_NAMES)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "actions.png", dpi=160)
    plt.close(fig)

    summary = cases.groupby("policy", sort=False)[["cost", "infected_time"]].mean()
    print(summary.to_string(float_format=lambda value: f"{value:.4g}"))
    print(f"Paired evaluation cases: {args.test_cases}; figures and tables: {out}")
    print("This is a defender-learning example against a scripted attacker, not a Nash solver.")
    return cases


def _check() -> None:
    _, state, _, _, objective = solve_fbsm(n=40, max_iter=4)
    assert np.isfinite(objective)
    assert np.allclose(state.sum(axis=1), 1.0)
    env = SampledFlowImpulseEnv(seed=1)
    env.reset()
    observation, _, _, info = env.step(env.DEF_ISOLATE)
    assert np.isclose(observation[:3].sum(), 1.0)
    assert info["jump_applied"]
    print("State integration, adjoint sweep and impulse reset passed.")


def _build_docs() -> None:
    source = ROOT / "docs" / "source"
    if not source.is_dir():
        raise RuntimeError("Building the lecture PDF requires a repository checkout.")
    name = "note1_game_learning_cyber_control"
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
        cwd=source,
        check=True,
    )
    (source.parent / f"{name}.pdf").write_bytes((source / f"{name}.pdf").read_bytes())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Learn and compare a sampled-data defender.")
    parser.add_argument(
        "command", nargs="?", choices=["example", "check", "study", "docs"], default="example"
    )
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--test-cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "auto", "cuda", "mps"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    aliases = {"smoke": "check", "figures": "example", "medium": "study"}
    if argv:
        argv[0] = aliases.get(argv[0], argv[0])
    parser = build_parser()
    args = parser.parse_args(argv)
    if min(args.episodes, args.horizon, args.test_cases) < 1:
        parser.error("Episodes, horizon and test cases must be positive.")
    if args.command == "check":
        _check()
    elif args.command == "docs":
        _build_docs()
    elif args.command == "study":
        from .experiments import run_medium

        run_medium(args.output_dir, args.device)
    else:
        run_example(args)


if __name__ == "__main__":
    main()
