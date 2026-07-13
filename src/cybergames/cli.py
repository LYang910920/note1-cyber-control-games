"""Single public command-line interface for Note 1."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

from cybercontrol.experiments import write_run_manifest

from .configs import CTDEConfig, DDQNConfig, MAPPOConfig
from .ctde import train as train_ctde
from .ddqn import evaluate as evaluate_ddqn
from .ddqn import train as train_ddqn
from .experiments import run_medium
from .fbsm import solve_fbsm
from .mappo import train_mappo

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
            write_run_manifest(
                Path("artifacts/smoke_summary.json"),
                command="smoke",
                metrics=_smoke(),
                repository_root=ROOT,
            )
        )
    elif args.command == "medium":
        rows = run_medium(args.output_dir, args.device)
        print(f"wrote {len(rows)} rows to {args.output_dir}")
    elif args.command == "figures":
        _run_script("scripts/generate_figures.py")
    elif args.command == "docs":
        _build_docs()
    elif args.command == "all":
        print(
            write_run_manifest(
                Path("artifacts/smoke_summary.json"),
                command="smoke",
                metrics=_smoke(),
                repository_root=ROOT,
            )
        )
        _run_script("scripts/generate_figures.py")
        _build_docs()


if __name__ == "__main__":
    main()
