from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddqn_cyber_defense import train as train_ddqn
from fbsm_malware_baseline import solve_fbsm
from madrl_ctde_hybrid_game import train as train_madrl


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_fbsm() -> list[dict]:
    _, _, _, _, _, history = solve_fbsm(T=24.0, n=100, max_iter=35, return_history=True)
    return history


def run_ddqn(episodes: int) -> list[dict]:
    args = SimpleNamespace(
        smoke=True,
        episodes=episodes,
        batch_size=8,
        hidden=48,
        lr=1e-3,
        gamma=0.99,
        buffer_size=5000,
        target_update=20,
        eps_start=1.0,
        eps_end=0.10,
        eps_decay=250.0,
        log_every=1,
        seed=11,
        return_history=True,
    )
    _, history = train_ddqn(args)
    return history


def run_madrl(episodes: int) -> list[dict]:
    args = SimpleNamespace(
        smoke=True,
        episodes=episodes,
        hidden=48,
        lr=5e-4,
        gamma=0.97,
        entropy_coef=0.02,
        log_every=1,
        seed=13,
        return_history=True,
    )
    _, _, history = train_madrl(args)
    return history


def plot_training_diagnostics(output_path: Path, fbsm: list[dict], ddqn: list[dict], madrl: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].semilogy([r["iteration"] for r in fbsm], [r["max_control_change"] for r in fbsm], color="black")
    axes[0].set_title("FBSM convergence")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Max control change")
    axes[0].grid(alpha=0.25)

    axes[1].plot([r["episode"] for r in ddqn], [r["training_return"] for r in ddqn], label="train")
    axes[1].plot([r["episode"] for r in ddqn], [r["evaluation_return"] for r in ddqn], label="eval")
    axes[1].set_title("DDQN defender")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Defender return")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot([r["episode"] for r in madrl], [r["loss"] for r in madrl], label="joint loss")
    axes[2].plot([r["episode"] for r in madrl], [r["defender_return"] for r in madrl], label="defender return")
    axes[2].set_title("CTDE/MADRL diagnostics")
    axes[2].set_xlabel("Episode")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run short training-iteration experiments for Note 1.")
    parser.add_argument("--episodes", type=int, default=14, help="Small episode count for DDQN and MADRL diagnostics.")
    args = parser.parse_args()

    exp_dir = ROOT / "experiments"
    fig_dir = ROOT / "figures"
    fbsm = run_fbsm()
    ddqn = run_ddqn(args.episodes)
    madrl = run_madrl(args.episodes)

    write_csv(exp_dir / "fbsm_iteration_history.csv", fbsm)
    write_csv(exp_dir / "ddqn_training_history.csv", ddqn)
    write_csv(exp_dir / "madrl_training_history.csv", madrl)
    plot_training_diagnostics(fig_dir / "training_iteration_diagnostics.png", fbsm, ddqn, madrl)

    print(f"Wrote experiment CSV files to {exp_dir}")
    print(f"Wrote training diagnostic figure to {fig_dir / 'training_iteration_diagnostics.png'}")


if __name__ == "__main__":
    main()
