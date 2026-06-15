from __future__ import annotations

import argparse
import csv
import math
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
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_fbsm() -> list[dict]:
    _, _, _, _, _, history = solve_fbsm(T=24.0, n=100, max_iter=35, return_history=True)
    return history


def run_ddqn(episodes: int) -> list[dict]:
    args = SimpleNamespace(
        smoke=False,
        episodes=episodes,
        horizon=24,
        eval_horizon=24,
        eval_episodes=4,
        batch_size=32,
        hidden=64,
        lr=1e-3,
        gamma=0.99,
        buffer_size=10000,
        target_update=80,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=500.0,
        log_every=max(1, episodes // 30),
        seed=11,
        return_history=True,
    )
    _, history = train_ddqn(args)
    return history


def run_madrl(episodes: int) -> list[dict]:
    args = SimpleNamespace(
        smoke=False,
        episodes=episodes,
        horizon=18,
        hidden=48,
        lr=5e-4,
        gamma=0.97,
        entropy_coef=0.02,
        log_every=max(1, episodes // 30),
        seed=13,
        return_history=True,
    )
    _, _, history = train_madrl(args)
    return history


def rolling_mean(values: list[float], window: int = 5) -> list[float]:
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start:idx + 1]
        finite = [x for x in chunk if not math.isnan(x)]
        out.append(sum(finite) / len(finite) if finite else float("nan"))
    return out


def write_summary(path: Path, fbsm: list[dict], ddqn: list[dict], madrl: list[dict]) -> None:
    fbsm_ratio = fbsm[-1]["max_control_change"] / max(fbsm[0]["max_control_change"], 1e-12)
    ddqn_eval = [r["evaluation_return"] for r in ddqn]
    madrl_loss = [r["loss"] for r in madrl]
    text = f"""# Training Summary

These runs are intentionally small enough for a laptop, but long enough to show the main convergence or stabilization signals.

| Diagnostic | Start | End | Interpretation |
|---|---:|---:|---|
| FBSM max control change | {fbsm[0]["max_control_change"]:.3e} | {fbsm[-1]["max_control_change"]:.3e} | Control updates shrink to {fbsm_ratio:.3e} of the initial change. |
| DDQN evaluation return | {ddqn_eval[0]:.3f} | {ddqn_eval[-1]:.3f} | Stochastic policy learning is noisy, so inspect the rolling trend rather than one episode. |
| MADRL joint loss | {madrl_loss[0]:.3f} | {madrl_loss[-1]:.3f} | Compact CTDE runs are stability diagnostics, not equilibrium proofs. |

For longer research runs, increase `--episodes`, run multiple seeds, and compare against no-defense and rule-based baselines.
"""
    path.write_text(text)


def plot_training_diagnostics(output_path: Path, fbsm: list[dict], ddqn: list[dict], madrl: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].semilogy([r["iteration"] for r in fbsm], [r["max_control_change"] for r in fbsm], color="black")
    axes[0].set_title("FBSM convergence")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Max control change")
    axes[0].grid(alpha=0.25)

    ddqn_episode = [r["episode"] for r in ddqn]
    ddqn_train = [r["training_return"] for r in ddqn]
    ddqn_eval = [r["evaluation_return"] for r in ddqn]
    axes[1].plot(ddqn_episode, ddqn_train, alpha=0.35, label="train")
    axes[1].plot(ddqn_episode, ddqn_eval, alpha=0.35, label="eval")
    axes[1].plot(ddqn_episode, rolling_mean(ddqn_eval, window=5), color="black", linewidth=2, label="eval rolling mean")
    axes[1].set_title("DDQN defender")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Defender return")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    madrl_episode = [r["episode"] for r in madrl]
    madrl_loss = [r["loss"] for r in madrl]
    madrl_def = [r["defender_return"] for r in madrl]
    axes[2].plot(madrl_episode, madrl_loss, alpha=0.35, label="joint loss")
    axes[2].plot(madrl_episode, rolling_mean(madrl_loss, window=5), color="black", linewidth=2, label="loss rolling mean")
    axes[2].plot(madrl_episode, madrl_def, alpha=0.35, label="defender return")
    axes[2].set_title("CTDE/MADRL diagnostics")
    axes[2].set_xlabel("Episode")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training-iteration experiments for Note 1.")
    parser.add_argument("--episodes", type=int, default=120, help="Episode count for DDQN and MADRL diagnostics.")
    args = parser.parse_args()

    exp_dir = ROOT / "experiments"
    fig_dir = ROOT / "figures"
    fbsm = run_fbsm()
    ddqn = run_ddqn(args.episodes)
    madrl = run_madrl(args.episodes)

    write_csv(exp_dir / "fbsm_iteration_history.csv", fbsm)
    write_csv(exp_dir / "ddqn_training_history.csv", ddqn)
    write_csv(exp_dir / "madrl_training_history.csv", madrl)
    write_summary(exp_dir / "training_summary.md", fbsm, ddqn, madrl)
    plot_training_diagnostics(fig_dir / "training_iteration_diagnostics.png", fbsm, ddqn, madrl)

    print(f"Wrote experiment CSV files to {exp_dir}")
    print(f"Wrote training diagnostic figure to {fig_dir / 'training_iteration_diagnostics.png'}")


if __name__ == "__main__":
    main()
