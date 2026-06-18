"""Run longer Note 1 diagnostics and save reader-friendly artifacts.

Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

This script is separate from smoke tests.  Smoke tests answer "does the code
run?" while this script answers "do the teaching metrics move in a sensible
direction over time?"
"""

from __future__ import annotations

import argparse
import csv
import math
import textwrap
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddqn_cyber_defense import train as train_ddqn
from evaluation_metrics import (
    evaluate_game_response_matrix,
    evaluate_policy_suite,
    policy_suite,
    rollout_policy,
    summarize_rollout,
)
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
    """Run enough FBSM iterations to expose the control-update decay curve."""
    _, _, _, _, _, history = solve_fbsm(T=24.0, n=100, max_iter=35, return_history=True)
    return history


def run_ddqn(episodes: int):
    """Train the DDQN defender with small, stable settings for a laptop run."""
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
        eps_end=0.02,
        eps_decay=450.0,
        log_every=max(1, episodes // 30),
        seed=11,
        return_history=True,
    )
    qnet, history = train_ddqn(args)
    return qnet, history


def run_madrl(episodes: int) -> list[dict]:
    """Run a compact CTDE/MADRL stability diagnostic."""
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


def make_ddqn_policy(qnet):
    """Wrap a trained Q-network as a greedy policy function."""
    def policy(env, k, obs):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            return int(qnet(obs_t).argmax(1).item())
    return policy


def run_policy_comparison(qnet) -> list[dict]:
    """Compare fixed, rule-based, and learned policies with shared metrics."""
    rollouts, rows = evaluate_policy_suite(horizon=50, seed=7)
    ddqn_policy = make_ddqn_policy(qnet)
    ddqn_rollout = rollout_policy("DDQN learned defender (greedy)", ddqn_policy, horizon=50, seed=7)
    rows.append(summarize_rollout(ddqn_rollout))
    return rows


def run_game_response(qnet) -> list[dict]:
    """Evaluate defender policies against several attacker strategies."""
    defender_policies = policy_suite() + [("DDQN learned defender (greedy)", make_ddqn_policy(qnet))]
    return evaluate_game_response_matrix(defender_policies=defender_policies, horizon=40, seed=17)


def rolling_mean(values: list[float], window: int = 5) -> list[float]:
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start:idx + 1]
        finite = [x for x in chunk if not math.isnan(x)]
        out.append(sum(finite) / len(finite) if finite else float("nan"))
    return out


def write_summary(
    path: Path,
    fbsm: list[dict],
    ddqn: list[dict],
    madrl: list[dict],
    policy_metrics: list[dict],
    game_metrics: list[dict],
    episodes: int,
) -> None:
    fbsm_ratio = fbsm[-1]["max_control_change"] / max(fbsm[0]["max_control_change"], 1e-12)
    ddqn_eval = [r["evaluation_return"] for r in ddqn]
    madrl_loss = [r["loss"] for r in madrl]
    ddqn_roll = rolling_mean(ddqn_eval, window=5)
    ddqn_gain = ddqn_roll[-1] - ddqn_roll[0]
    timing = policy_metrics[0]
    policy_rows = "\n".join(
        f'| {row["policy"]} | {row["cumulative_compromised"]:.3f} | {row["peak_compromised"]:.3f} | '
        f'{row["final_compromised"]:.3f} | {row["total_defender_cost"]:.2f} | {row["impulse_events"]} |'
        for row in policy_metrics
    )
    ddqn_policy = next(row for row in policy_metrics if row["policy"].startswith("DDQN"))
    non_learning_rows = [
        row for row in policy_metrics if not row["policy"].startswith("DDQN")
    ]
    best_fixed = min(non_learning_rows, key=lambda row: row["cumulative_compromised"])
    game_best = min(game_metrics, key=lambda row: row["cumulative_compromised"])
    text = f"""# Training Summary

These runs are intentionally small enough for a laptop, but long enough to show the main convergence or stabilization signals.

## Experiment Configuration

| Item | Setting |
|---|---|
| Model | Hybrid malware/deception state `[S,I,R,z]` |
| Decision timing | observe at `t_k`, apply impulse jump if selected, integrate ODE to `t_{{k+1}}` |
| Defender actions | none, patch, clean, deceive, isolate |
| Attacker actions | scan, exploit, lateral, stealth |
| DDQN setting | {episodes} episodes, horizon 24, hidden width 64, learning rate 1e-3, gamma 0.99 |
| CTDE/MADRL setting | {episodes} episodes, horizon 18, hidden width 48, learning rate 5e-4, gamma 0.97 |

## Timing Parameters

| Parameter | Value | Meaning |
|---|---:|---|
| Decision interval `Delta t` | {timing["decision_dt"]:.2f} | Policies observe and act once per interval. |
| RK4 substeps per interval | {timing["rk4_substeps"]} | Internal ODE solver steps, not extra MDP/MG decisions. |
| Policy-comparison horizon | {timing["decision_epochs"]} | Number of sampled decision epochs in each rollout. |

| Diagnostic | Start | End | Interpretation |
|---|---:|---:|---|
| FBSM max control change | {fbsm[0]["max_control_change"]:.3e} | {fbsm[-1]["max_control_change"]:.3e} | Control updates shrink to {fbsm_ratio:.3e} of the initial change. |
| DDQN evaluation return | {ddqn_eval[0]:.3f} | {ddqn_eval[-1]:.3f} | Rolling evaluation improves by {ddqn_gain:.3f}; inspect the rolling trend rather than one episode. |
| MADRL joint loss | {madrl_loss[0]:.3f} | {madrl_loss[-1]:.3f} | Compact CTDE runs are stability diagnostics, not equilibrium proofs. |

## Representative Policy Comparison

Lower cumulative/peak/final compromised values and lower defender cost are better.

| Policy | Cumulative compromised | Peak compromised | Final compromised | Defender cost | Impulse events |
|---|---:|---:|---:|---:|---:|
{policy_rows}

The learned DDQN policy has cumulative compromised exposure {ddqn_policy["cumulative_compromised"]:.3f}, compared with {best_fixed["cumulative_compromised"]:.3f} for the best non-learning baseline in this run.

## Game Response Snapshot

`game_response_metrics.csv` evaluates defender policies against several attacker strategies.  The lowest cumulative compromised exposure in the matrix is {game_best["cumulative_compromised"]:.3f}, achieved by `{game_best["defender_policy"]}` against `{game_best["attacker_policy"]}`.

For longer research runs, increase `--episodes`, run multiple seeds, and compare against no-defense and rule-based baselines.
"""
    path.write_text(text)


def write_output_preview(path: Path, policy_metrics: list[dict], game_metrics: list[dict]) -> None:
    """Write a categorized preview of the main generated outputs."""
    ddqn_policy = next(row for row in policy_metrics if row["policy"].startswith("DDQN"))
    non_learning_rows = [
        row for row in policy_metrics if not row["policy"].startswith("DDQN")
    ]
    best_fixed = min(non_learning_rows, key=lambda row: row["cumulative_compromised"])
    game_best = min(game_metrics, key=lambda row: row["cumulative_compromised"])
    text = f"""# Output Preview

Use this page as the first stop after running `python scripts/run_training_iterations.py`.

## 1. Model And Timing

| Item | Value |
|---|---|
| Model | Hybrid malware/deception `[S,I,R,z]` |
| Decision interval | `Delta t = {policy_metrics[0]["decision_dt"]:.2f}` |
| Solver substeps | `{policy_metrics[0]["rk4_substeps"]}` RK4 substeps per decision interval |
| Observation convention | policy sees pre-jump `x(t_k^-)`; next observation is `x(t_{{k+1}}^-)` |

## 2. Training Convergence

Open `figures/training_iteration_diagnostics.png`.

| Panel | What to check |
|---|---|
| FBSM convergence | max control change should decay toward zero |
| DDQN defender | rolling evaluation return should improve and stabilize |
| CTDE/MADRL diagnostics | loss and defender return should remain finite and interpretable |
| Learning vs baselines | DDQN should be competitive with or better than fixed policies |

## 3. Learning-Versus-Baseline Result

| Policy | Cumulative compromised | Defender cost | Peak compromised | Impulse events |
|---|---:|---:|---:|---:|
| DDQN learned defender (greedy) | {ddqn_policy["cumulative_compromised"]:.3f} | {ddqn_policy["total_defender_cost"]:.2f} | {ddqn_policy["peak_compromised"]:.3f} | {ddqn_policy["impulse_events"]} |
| Best non-learning baseline: {best_fixed["policy"]} | {best_fixed["cumulative_compromised"]:.3f} | {best_fixed["total_defender_cost"]:.2f} | {best_fixed["peak_compromised"]:.3f} | {best_fixed["impulse_events"]} |

## 4. Game Response

Open `figures/game_response_matrix.png` and `experiments/game_response_metrics.csv`.

Best cell in this deterministic response matrix:

| Defender policy | Attacker strategy | Cumulative compromised |
|---|---|---:|
| {game_best["defender_policy"]} | {game_best["attacker_policy"]} | {game_best["cumulative_compromised"]:.3f} |

## 5. Files To Open First

| Category | File |
|---|---|
| Summary | `experiments/training_summary.md` |
| Learning curves | `figures/training_iteration_diagnostics.png` |
| Policy comparison CSV | `experiments/policy_comparison_metrics.csv` |
| Game matrix CSV | `experiments/game_response_metrics.csv` |
| Timing explanation | `docs/MODEL_TO_MDP.md` |
"""
    path.write_text(text)


def plot_training_diagnostics(output_path: Path, fbsm: list[dict], ddqn: list[dict], madrl: list[dict], policy_metrics: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    axes = axes.ravel()

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

    labels = [textwrap.fill(row["policy"], width=24) for row in policy_metrics]
    y = np.arange(len(labels))
    axes[3].barh(y, [row["cumulative_compromised"] for row in policy_metrics], color="#4c78a8", alpha=0.85)
    axes[3].set_yticks(y, labels)
    axes[3].invert_yaxis()
    axes[3].set_title("Learning vs baselines")
    axes[3].set_xlabel("Cumulative compromised exposure")
    axes[3].grid(axis="x", alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_game_response_matrix(output_path: Path, rows: list[dict]) -> None:
    defenders = list(dict.fromkeys(row["defender_policy"] for row in rows))
    attackers = list(dict.fromkeys(row["attacker_policy"] for row in rows))
    matrix = np.zeros((len(defenders), len(attackers)), dtype=float)
    for row in rows:
        i = defenders.index(row["defender_policy"])
        j = attackers.index(row["attacker_policy"])
        matrix[i, j] = row["cumulative_compromised"]

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(attackers)), [textwrap.fill(label, width=18) for label in attackers], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(defenders)), [textwrap.fill(label, width=24) for label in defenders])
    ax.set_title("Attacker-defender response matrix: cumulative compromised exposure")
    ax.set_xlabel("Attacker strategy")
    ax.set_ylabel("Defender policy")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Lower is better")
    fig.tight_layout()
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training-iteration experiments for Note 1.")
    parser.add_argument("--episodes", type=int, default=180, help="Episode count for DDQN and MADRL diagnostics.")
    args = parser.parse_args()

    exp_dir = ROOT / "experiments"
    fig_dir = ROOT / "figures"
    fbsm = run_fbsm()
    qnet, ddqn = run_ddqn(args.episodes)
    madrl = run_madrl(args.episodes)
    policy_metrics = run_policy_comparison(qnet)
    game_metrics = run_game_response(qnet)

    write_csv(exp_dir / "fbsm_iteration_history.csv", fbsm)
    write_csv(exp_dir / "ddqn_training_history.csv", ddqn)
    write_csv(exp_dir / "madrl_training_history.csv", madrl)
    write_csv(exp_dir / "policy_comparison_metrics.csv", policy_metrics)
    write_csv(exp_dir / "game_response_metrics.csv", game_metrics)
    write_summary(exp_dir / "training_summary.md", fbsm, ddqn, madrl, policy_metrics, game_metrics, args.episodes)
    write_output_preview(exp_dir / "OUTPUT_PREVIEW.md", policy_metrics, game_metrics)
    plot_training_diagnostics(fig_dir / "training_iteration_diagnostics.png", fbsm, ddqn, madrl, policy_metrics)
    plot_game_response_matrix(fig_dir / "game_response_matrix.png", game_metrics)

    print(f"Wrote experiment CSV files to {exp_dir}")
    print(f"Wrote training diagnostic figure to {fig_dir / 'training_iteration_diagnostics.png'}")
    print(f"Wrote game response matrix figure to {fig_dir / 'game_response_matrix.png'}")


if __name__ == "__main__":
    main()
