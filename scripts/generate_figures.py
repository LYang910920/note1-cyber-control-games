"""Generate static figures for the Note 1 README and guide notes.

Copyright (c) 2026 Luxing Yang.
Licensed under the MIT License. See LICENSE in the repository root.

The script is deterministic and lightweight.  It does not train
neural policies; it visualizes model behavior, hand-coded policies, and the
architecture diagrams used for orientation.
"""

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from cybercontrol.plotting import (
    PUBLICATION_COLORS,
    PUBLICATION_LINESTYLES,
    PUBLICATION_MARKERS,
    add_arrow,
    add_box,
    guide_style,
    panel_label,
    publication_style,
    save_guide_figure,
    save_publication_figure,
    style_axis,
)
from cyber_hybrid_env import HybridCyberDefenseEnv, scripted_attacker
from evaluation_metrics import evaluate_policy_suite
from fbsm_malware_baseline import solve_fbsm


def plot_fbsm(output_dir: Path) -> None:
    t, x, u, _, objective = solve_fbsm(T=24.0, n=120, max_iter=30)
    with publication_style():
        fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.8), sharex=True)
    axes[0].plot(t, x[:, 0], label="Susceptible", linestyle="-")
    axes[0].plot(t, x[:, 1], label="Compromised", linestyle="--")
    axes[0].plot(t, x[:, 2], label="Recovered/protected", linestyle="-.")
    panel_label(axes[0], "(a) continuous-control state trajectory")
    style_axis(axes[0], ylabel="Population share", legend=True)

    axes[1].plot(t, u, color="black", linestyle="-", label="Patch control u(t)")
    panel_label(axes[1], f"(b) continuous control, objective={objective:.2f}")
    style_axis(axes[1], xlabel="Time", ylabel="Control intensity", legend=True)
    axes[1].set_ylim(-0.05, 1.05)
    fig.tight_layout()
    save_publication_figure(
        fig,
        output_dir / "fbsm_malware_control",
        metadata={
            "model": "malware SIR optimal control",
            "control_type": "continuous",
            "caption_hint": "FBSM state and continuous control trajectory.",
        },
    )
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
    with publication_style():
        fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.8), sharex=True)
    axes[0].plot(t, states[:, 0], label="Susceptible", linestyle="-")
    axes[0].plot(t, states[:, 1], label="Compromised", linestyle="--")
    axes[0].plot(t, states[:, 2], label="Recovered/protected", linestyle="-.")
    panel_label(axes[0], "(a) hybrid state trajectory")
    style_axis(axes[0], ylabel="State", legend=True)

    axes[1].step(np.arange(len(actions)), actions, where="post", color="black")
    axes[1].set_yticks([env.DEF_PATCH, env.DEF_CLEAN, env.DEF_ISOLATE])
    axes[1].set_yticklabels(["patch", "clean", "isolate"])
    panel_label(axes[1], "(b) sampled defender action")
    style_axis(axes[1], xlabel="Action epoch k, observation at t_k", ylabel="Defender mode")
    fig.tight_layout()
    save_publication_figure(
        fig,
        output_dir / "hybrid_policy_rollout",
        metadata={
            "model": "sampled SIR malware environment with action-dependent deception",
            "control_type": "hybrid sampled continuous-plus-impulse policy",
            "caption_hint": "Hybrid rollout under sampled defender actions.",
        },
    )
    plt.close(fig)


def plot_hybrid_policy_comparison(output_dir: Path) -> None:
    rollouts, metrics = evaluate_policy_suite(horizon=50, seed=7)
    colors = list(PUBLICATION_COLORS[:4])
    linestyles = list(PUBLICATION_LINESTYLES[:4])
    markers = list(PUBLICATION_MARKERS[:4])
    labels = [textwrap.fill(row["policy"], width=18) for row in metrics]
    x = np.arange(len(labels))

    with publication_style():
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.2))
    ax = axes[0, 0]
    for idx, rollout in enumerate(rollouts):
        states = rollout["states"]
        t = np.arange(states.shape[0])
        markevery = max(1, len(t) // 8)
        ax.plot(
            t,
            states[:, 1],
            label=textwrap.fill(rollout["label"], width=18),
            color=colors[idx],
            linestyle=linestyles[idx],
            marker=markers[idx],
            markevery=markevery,
            linewidth=2.0,
            markersize=4,
        )
    panel_label(ax, "(a) compromised state")
    style_axis(ax, xlabel="Action/observation epoch k", ylabel="Compromised share I", legend=True)

    ax = axes[0, 1]
    ax.bar(x, [row["cumulative_compromised"] for row in metrics], color=colors, alpha=0.85)
    panel_label(ax, "(b) exposure")
    ax.set_ylabel("sum I_k * Delta t (lower is better)")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)

    width = 0.36
    ax = axes[1, 0]
    ax.bar(x - width / 2, [row["peak_compromised"] for row in metrics], width, label="peak I", color="#e45756", alpha=0.80)
    ax.bar(x + width / 2, [row["final_compromised"] for row in metrics], width, label="final I", color="#72b7b2", alpha=0.85)
    panel_label(ax, "(c) peak and final state")
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
    panel_label(ax, "(d) cost and impulses")
    ax.set_xlabel("Total defender cost (lower is better)")
    ax.set_xlim(0, max(costs) * 1.18)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_publication_figure(
        fig,
        output_dir / "hybrid_policy_comparison",
        metadata={
            "model": "hybrid malware/deception environment",
            "control_type": "hybrid policy comparison",
            "caption_hint": "Same-model comparison of sampled hybrid defender policies.",
        },
    )
    plt.close(fig)


def plot_neural_architectures(output_dir: Path) -> None:
    with guide_style():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    ax = axes[0]
    add_box(ax, (0.1, 2.8), "state x(t_k)\n+ time/context", fc="#e8f1ff")
    add_box(ax, (2.4, 2.8), "Q-network\nMLP", fc="#fff4df")
    add_box(ax, (4.7, 2.8), "Q-values\nfor actions", fc="#e9f7ef")
    add_box(ax, (7.0, 2.8), "argmax\nor epsilon", fc="#f4ecff")
    add_box(ax, (2.4, 1.55), "target\nQ-network", fc="#fff4df")
    add_box(ax, (4.7, 1.55), "DDQN target\nr + gamma Q'", width=2.1, fc="#ffecec")
    add_arrow(ax, (1.9, 3.08), (2.4, 3.08))
    add_arrow(ax, (4.2, 3.08), (4.7, 3.08))
    add_arrow(ax, (6.5, 3.08), (7.0, 3.08))
    add_arrow(ax, (5.75, 2.8), (5.75, 2.1))
    add_arrow(ax, (4.2, 1.82), (4.7, 1.82))
    panel_label(ax, "(a) DDQN defender")
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0.8, 4.0)
    ax.axis("off")

    ax = axes[1]
    add_box(ax, (0.1, 2.9), "defender obs", fc="#e8f1ff")
    add_box(ax, (0.1, 1.75), "attacker obs", fc="#ffecec")
    add_box(ax, (2.3, 2.9), "defender actor\npi_D(a_D|o_D)", fc="#fff4df")
    add_box(ax, (2.3, 1.75), "attacker actor\npi_A(a_A|o_A)", fc="#fff4df")
    add_box(ax, (4.8, 2.25), "central critic\nQ(s, a_D, a_A)", width=2.1, fc="#e9f7ef")
    add_box(ax, (7.4, 2.25), "policy-gradient\nupdates", fc="#f4ecff")
    add_arrow(ax, (1.9, 3.18), (2.3, 3.18))
    add_arrow(ax, (1.9, 2.03), (2.3, 2.03))
    add_arrow(ax, (4.1, 3.18), (4.8, 2.8))
    add_arrow(ax, (4.1, 2.03), (4.8, 2.45))
    add_arrow(ax, (6.9, 2.53), (7.4, 2.53))
    panel_label(ax, "(b) CTDE attacker-defender learning")
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0.8, 4.0)
    ax.axis("off")

    fig.tight_layout()
    save_guide_figure(
        fig,
        output_dir / "neural_architectures",
        formats=("png", "pdf"),
        metadata={"figure_type": "guide diagram", "caption_hint": "Neural-control architecture overview."},
    )
    plt.close(fig)


def plot_action_timing(output_dir: Path) -> None:
    """Show policy action epochs, ODE substeps, and original impulse times."""
    with guide_style():
        fig, ax = plt.subplots(figsize=(11, 4.6))
    action_times = np.array([0.0, 1.15, 2.75, 4.0, 5.35])
    impulse_times = np.array([0.55, 2.75, 4.75])
    substep_counts = [3, 4, 3, 4]

    y_action, y_flow, y_impulse = 1.2, 0.0, -1.2
    x0, x1 = action_times[0] - 0.2, action_times[-1] + 0.35
    for y, color in ((y_action, "#4c78a8"), (y_flow, "#666666"), (y_impulse, "#e45756")):
        ax.hlines(y, x0, x1, color=color, linewidth=1.4, alpha=0.9)
        ax.annotate(
            "",
            xy=(action_times[-1] + 0.42, y),
            xytext=(action_times[-1] + 0.12, y),
            arrowprops={"arrowstyle": "->", "lw": 1.5, "color": color},
        )

    ax.text(x0 - 0.08, y_action, "decision epochs", ha="right", va="center", fontsize=9.5, color="#234f7f")
    ax.text(x0 - 0.08, y_flow, "ODE substeps", ha="right", va="center", fontsize=9.5, color="#444444")
    ax.text(x0 - 0.08, y_impulse, "impulse times", ha="right", va="center", fontsize=9.5, color="#a83232")

    for idx, t in enumerate(action_times):
        ax.vlines(t, y_action - 0.16, y_action + 0.16, color="#4c78a8", linewidth=2.4)
        ax.scatter([t], [y_action], s=60, color="#4c78a8", zorder=3)
        ax.text(t, y_action + 0.24, f"$t_{idx}$", ha="center", va="bottom", fontsize=11, color="#234f7f")
        ax.text(t, y_action - 0.28, "observe\nchoose action", ha="center", va="top", fontsize=8.2)

    for idx in range(len(action_times) - 1):
        left, right = action_times[idx], action_times[idx + 1]
        ax.annotate(
            "",
            xy=(right - 0.08, y_flow - 0.22),
            xytext=(left + 0.08, y_flow - 0.22),
            arrowprops={"arrowstyle": "<->", "lw": 1.1, "color": "#666666"},
        )
        ax.text((left + right) / 2, y_flow - 0.38, f"$\\Delta t_{idx}$", ha="center", va="top", fontsize=10)
        substeps = np.linspace(left, right, substep_counts[idx] + 2)[1:-1]
        ax.scatter(substeps, np.full_like(substeps, y_flow), s=24, color="#666666", zorder=3)
        ax.text((left + right) / 2, y_flow + 0.18, "RK4/internal flow", ha="center", va="bottom", fontsize=8.5, color="#444444")

    for j, tau in enumerate(impulse_times, start=1):
        ax.vlines(tau, y_impulse - 0.18, y_impulse + 0.18, color="#e45756", linewidth=2.0, linestyles="--")
        ax.scatter([tau], [y_impulse], marker="v", s=80, color="#e45756", zorder=4)
        label = f"$\\tau_{j}$"
        if np.any(np.isclose(tau, action_times)):
            label += " = action point"
        else:
            label += " inside transition"
        ax.text(tau, y_impulse - 0.26, label, ha="center", va="top", fontsize=9.5, color="#a83232")

    ax.text(action_times[0] - 0.1, 2.0, "$t_k$: MDP/MG observation and action points", color="#234f7f", fontsize=10)
    ax.text(action_times[0] - 0.1, 1.78, "$\\tau_j$: impulse/event times in the original hybrid model", color="#a83232", fontsize=10)
    ax.text(
        action_times[0] - 0.1,
        -1.92,
        "Actions are held or mapped over each interval. Internal ODE substeps are solver operations, not extra policy decisions.",
        fontsize=9.2,
        color="#333333",
    )

    panel_label(ax, "Three-lane timing: decisions, ODE substeps, and impulses", x=0.0, y=0.98)
    ax.set_xlim(action_times[0] - 0.95, action_times[-1] + 0.55)
    ax.set_ylim(-2.15, 2.25)
    ax.axis("off")
    fig.tight_layout()
    save_guide_figure(
        fig,
        output_dir / "action_timing",
        formats=("png", "pdf"),
        metadata={
            "figure_type": "guide diagram",
            "caption_hint": "Distinguishes sampled decision epochs from model impulse times.",
        },
    )
    plt.close(fig)


def main() -> None:
    output_dir = ROOT / "docs" / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_fbsm(output_dir)
    plot_hybrid_rollout(output_dir)
    plot_hybrid_policy_comparison(output_dir)
    plot_neural_architectures(output_dir)
    plot_action_timing(output_dir)
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
