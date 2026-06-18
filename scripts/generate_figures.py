"""Generate static figures for the Note 1 README and teaching notes.

Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

The script is intentionally deterministic and lightweight.  It does not train
neural policies; it visualizes model behavior, hand-coded policies, and the
architecture diagrams used for orientation.
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cyber_hybrid_env import HybridCyberDefenseEnv, scripted_attacker
from evaluation_metrics import evaluate_policy_suite
from fbsm_malware_baseline import solve_fbsm


def plot_fbsm(output_dir: Path) -> None:
    t, x, u, _, objective = solve_fbsm(T=24.0, n=120, max_iter=30)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, x[:, 0], label="Susceptible")
    axes[0].plot(t, x[:, 1], label="Compromised")
    axes[0].plot(t, x[:, 2], label="Recovered/protected")
    axes[0].set_ylabel("Population share")
    axes[0].set_title(f"FBSM malware-control baseline, objective={objective:.2f}")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, u, color="black", label="Patch control u(t)")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Control intensity")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fbsm_malware_control.png", dpi=180)
    plt.close(fig)


def plot_hybrid_rollout(output_dir: Path) -> None:
    env = HybridCyberDefenseEnv(seed=4)
    obs = env.reset()
    states = [obs.copy()]
    actions = []
    for k in range(40):
        if obs[1] > 0.20:
            action = (env.DEF_ISOLATE, 0.8)
        elif obs[1] > 0.08:
            action = (env.DEF_CLEAN, 0.7)
        else:
            action = (env.DEF_PATCH, 0.5)
        obs, _, done, _ = env.step(action, scripted_attacker(env, k))
        actions.append(action[0])
        states.append(obs.copy())
        if done:
            break

    states = np.asarray(states)
    t = np.arange(states.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, states[:, 0], label="Susceptible")
    axes[0].plot(t, states[:, 1], label="Compromised")
    axes[0].plot(t, states[:, 2], label="Recovered/protected")
    axes[0].plot(t, states[:, 3], label="Deception level")
    axes[0].set_ylabel("State")
    axes[0].set_title("Hybrid rollout: observe, act, jump if needed, then flow")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].step(np.arange(len(actions)), actions, where="post", color="black")
    axes[1].set_yticks([env.DEF_PATCH, env.DEF_CLEAN, env.DEF_ISOLATE])
    axes[1].set_yticklabels(["patch", "clean", "isolate"])
    axes[1].set_xlabel("Decision epoch k, observation at t_k")
    axes[1].set_ylabel("Defender mode")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "hybrid_policy_rollout.png", dpi=180)
    plt.close(fig)


def plot_hybrid_policy_comparison(output_dir: Path) -> None:
    rollouts, metrics = evaluate_policy_suite(horizon=50, seed=7)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]
    labels = [row["policy"] for row in metrics]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    ax = axes[0, 0]
    for idx, rollout in enumerate(rollouts):
        states = rollout["states"]
        t = np.arange(states.shape[0])
        markevery = max(1, len(t) // 8)
        ax.plot(
            t,
            states[:, 1],
            label=rollout["label"],
            color=colors[idx],
            linestyle=linestyles[idx],
            marker=markers[idx],
            markevery=markevery,
            linewidth=2.0,
            markersize=4,
        )
    ax.set_title("Compromised trajectory at observation points")
    ax.set_xlabel("Decision epoch k")
    ax.set_ylabel("Compromised share I")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.bar(x, [row["cumulative_compromised"] for row in metrics], color=colors, alpha=0.85)
    ax.set_title("Cumulative compromised exposure")
    ax.set_ylabel("sum mean(I) * Delta t")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)

    width = 0.36
    ax = axes[1, 0]
    ax.bar(x - width / 2, [row["peak_compromised"] for row in metrics], width, label="peak I", color="#e45756", alpha=0.80)
    ax.bar(x + width / 2, [row["final_compromised"] for row in metrics], width, label="final I", color="#72b7b2", alpha=0.85)
    ax.set_title("Peak and final compromised share")
    ax.set_ylabel("Compromised share")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    costs = [row["total_defender_cost"] for row in metrics]
    ax.barh(labels, costs, color=colors, alpha=0.85)
    for idx, row in enumerate(metrics):
        note = f'impulses={row["impulse_events"]}'
        ax.text(costs[idx], idx, f" {note}", va="center", fontsize=8)
    ax.set_title("Defender cost and impulse count")
    ax.set_xlabel("Total defender cost (lower is better)")
    ax.set_xlim(0, max(costs) * 1.18)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "hybrid_policy_comparison.png", dpi=180)
    plt.close(fig)


def _box(ax, xy, text, width=1.8, height=0.55, fc="#f7f7f7", ec="#333333", fontsize=8.5):
    rect = plt.Rectangle(xy, width, height, facecolor=fc, edgecolor=ec, linewidth=1.4)
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return rect


def _arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#333333"})


def plot_neural_architectures(output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    ax = axes[0]
    _box(ax, (0.1, 2.8), "state x(t_k)\n+ time/context", fc="#e8f1ff")
    _box(ax, (2.4, 2.8), "Q-network\nMLP", fc="#fff4df")
    _box(ax, (4.7, 2.8), "Q-values\nfor actions", fc="#e9f7ef")
    _box(ax, (7.0, 2.8), "argmax\nor epsilon", fc="#f4ecff")
    _box(ax, (2.4, 1.55), "target\nQ-network", fc="#fff4df")
    _box(ax, (4.7, 1.55), "DDQN target\nr + gamma Q'", width=2.1, fc="#ffecec")
    _arrow(ax, (1.9, 3.08), (2.4, 3.08))
    _arrow(ax, (4.2, 3.08), (4.7, 3.08))
    _arrow(ax, (6.5, 3.08), (7.0, 3.08))
    _arrow(ax, (5.75, 2.8), (5.75, 2.1))
    _arrow(ax, (4.2, 1.82), (4.7, 1.82))
    ax.set_title("DDQN defender")
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0.8, 4.0)
    ax.axis("off")

    ax = axes[1]
    _box(ax, (0.1, 2.9), "defender obs", fc="#e8f1ff")
    _box(ax, (0.1, 1.75), "attacker obs", fc="#ffecec")
    _box(ax, (2.3, 2.9), "defender actor\npi_D(a_D|o_D)", fc="#fff4df")
    _box(ax, (2.3, 1.75), "attacker actor\npi_A(a_A|o_A)", fc="#fff4df")
    _box(ax, (4.8, 2.25), "central critic\nQ(s, a_D, a_A)", width=2.1, fc="#e9f7ef")
    _box(ax, (7.4, 2.25), "policy-gradient\nupdates", fc="#f4ecff")
    _arrow(ax, (1.9, 3.18), (2.3, 3.18))
    _arrow(ax, (1.9, 2.03), (2.3, 2.03))
    _arrow(ax, (4.1, 3.18), (4.8, 2.8))
    _arrow(ax, (4.1, 2.03), (4.8, 2.45))
    _arrow(ax, (6.9, 2.53), (7.4, 2.53))
    ax.set_title("CTDE attacker-defender learning")
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0.8, 4.0)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "neural_architectures.png", dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = ROOT / "figures"
    output_dir.mkdir(exist_ok=True)
    plot_fbsm(output_dir)
    plot_hybrid_rollout(output_dir)
    plot_hybrid_policy_comparison(output_dir)
    plot_neural_architectures(output_dir)
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
